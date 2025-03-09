use femto_gpt::gpt::{TrainingState, GPT};
use femto_gpt::graph::GraphError;
use femto_gpt::optimizer::AdamW;
use femto_gpt::tokenizer::{Tokenizer, TokenizerImpl};
use rand::seq::SliceRandom;
use structopt::StructOpt;

use femto_gpt::log::{
    initialize_log,
    write_log_dataset_file,
    write_log_save,
};

// I just figured out what `context_size` does when training. It splits the sample data into chunks where each chunk is `context_size` tokens long. Then it
// predicts the token that comes after the chunk.
// But it seems like we can change `context_size` without re-training the model.
#[derive(StructOpt, Debug)]
enum Cli {
    Train {
        #[structopt(long)]
        repo: String,
        #[structopt(long, default_value = "256")]
        context_size: usize,
    },
    Infer {
        #[structopt(long)]
        repo: String,
        #[structopt(long)]
        old_checkpoint: Option<usize>,
        #[structopt(long)]
        prompt: String,
        #[structopt(long, default_value = "100")]
        count: usize,
        #[structopt(long, default_value = "0.5")]
        temperature: f32,
        #[structopt(long, default_value = "256")]
        context_size: usize,
    },
}

fn main() -> Result<(), GraphError> {
    #[cfg(not(feature = "gpu"))]
    let graph = femto_gpt::graph::CpuGraph::new();
    #[cfg(not(feature = "gpu"))]
    let is_gpu = false;

    #[cfg(feature = "gpu")]
    let graph = femto_gpt::graph::gpu::GpuGraph::new()?;
    #[cfg(feature = "gpu")]
    let is_gpu = true;

    let batch_size = 32;
    let dropout = 0.0;
    let cli = Cli::from_args();

    match cli {
        Cli::Infer {
            repo,
            old_checkpoint,
            prompt,
            count,
            temperature,
            context_size,
        } => {
            // NOTE: Feed-Forward dimension is always `embedding_dimension * 4`.
            let (embedding_dimension, num_layers, num_heads, head_size) = load_hyper_params(&repo);
            let mut rng = rand::thread_rng();
            let tokenizer = Tokenizer::new("");

            assert_eq!(num_heads * head_size, embedding_dimension);

            let vocab_size = tokenizer.vocab_size();
            println!("Vocab-size: {} unique characters", vocab_size);
            let mut gpt = GPT::new(
                &mut rng,
                graph,
                is_gpu.then(|| batch_size), // Pre-allocate batches only when using GPUs
                vocab_size,
                embedding_dimension,
                context_size,
                num_layers,
                num_heads,
                head_size,
                dropout,
            )?;

            gpt.sync()?;

            let ts: TrainingState = get_checkpoint(&repo, old_checkpoint).unwrap();
            gpt.set_training_state(ts, true)?;

            println!("Generating text:");

            let inference = gpt.infer(
                &mut rng,
                &tokenizer.tokenize(&prompt),
                count,
                temperature,
                |_ch| {},
                true,  // show_options
            )?;

            // Generate 100 character with the currently trained model
            println!("{}", tokenizer.untokenize(&inference));

            Ok(())
        },
        Cli::Train { repo, context_size } => {
            // NOTE: Feed-Forward dimension is always `embedding_dimension * 4`.
            let (embedding_dimension, num_layers, num_heads, head_size) = load_hyper_params(&repo);
            initialize_log();
            let tokenizer = Tokenizer::new("");
            let mut rng = rand::thread_rng();
            let vocab_size = tokenizer.vocab_size();

            println!("Vocab-size: {} unique characters", vocab_size);
            let mut gpt = GPT::new(
                &mut rng,
                graph,
                is_gpu.then(|| batch_size), // Pre-allocate batches only when using GPUs
                vocab_size,
                embedding_dimension,
                context_size,
                num_layers,
                num_heads,
                head_size,
                dropout,
            )?;

            gpt.sync()?;

            println!("Number of parameters: {}", gpt.num_params());

            if let Some(ts) = get_checkpoint(&repo, None) {
                gpt.set_training_state(ts, true)?;
            }

            println!();
            println!(
                "Starting the training loop... (This make take hours to converge! be patient!)"
            );
            println!();

            let base_lr = 0.001;
            let min_lr = 0.00001;
            let warmup_steps = 100;
            let decay_steps = 300;

            let learning_rate = |step| {
                if step < warmup_steps {
                    (base_lr / warmup_steps as f32) * step as f32
                } else {
                    // Fancy LR tuning, thanks to https://github.com/cutoken!
                    f32::max(
                        min_lr,
                        base_lr
                            - (base_lr - min_lr) * (step - warmup_steps) as f32
                                / decay_steps as f32,
                    )
                }
            };

            let callback = |gpt: &mut GPT<_>, step: usize, loss: f32| {
                gpt.sync().unwrap();
                let ts = gpt.get_training_state().unwrap();
                save_checkpoint(&repo, step, loss, &ts);
                write_log_save();

                Ok(())
            };

            let dataset = ragit_fs::join(&repo, "data").unwrap();
            let mut dataset_files = ragit_fs::read_dir(&dataset, false).unwrap();
            dataset_files.shuffle(&mut rand::thread_rng());
            let mut file_index = 0;

            loop {
                let dataset_file = dataset_files[file_index % dataset_files.len()].clone();
                write_log_dataset_file(&dataset_file, context_size);
                file_index += 1;
                println!("Reading {dataset_file}...");
                let dataset_char = ragit_fs::read_string(&dataset_file).expect("Should have been able to read the file");
                let tokenizer = Tokenizer::new(&dataset_char);
                let dataset = tokenizer.tokenize(&dataset_char);

                // Training loop!
                #[cfg(not(feature = "gpu"))]
                gpt.train_cpu(
                    &dataset,
                    500,
                    batch_size,
                    None, // or Some(n), limit backward process to last n computations
                    &AdamW::new(),
                    learning_rate,
                    callback,
                )?;

                #[cfg(feature = "gpu")]
                gpt.train(
                    &dataset,
                    500,
                    batch_size,
                    None, // or Some(n), limit backward process to last n computations
                    &AdamW::new(),
                    learning_rate,
                    callback,
                )?;
            }
        }
    }
}

use lazy_static::lazy_static;
use regex::Regex;
use serde::Deserialize;

#[derive(Deserialize)]
struct HyperParameter {
    embedding_dimension: usize,
    num_layers: usize,
    num_heads: usize,
}

fn load_hyper_params(repo: &str) -> (usize, usize, usize, usize) {
    let path = ragit_fs::join(repo, "hyper_params.json").unwrap();
    let j = ragit_fs::read_string(&path).unwrap();
    let hyper_params = serde_json::from_str::<HyperParameter>(&j).unwrap();
    let head_size = hyper_params.embedding_dimension / hyper_params.num_heads;
    assert_eq!(hyper_params.num_heads * head_size, hyper_params.embedding_dimension);

    (
        hyper_params.embedding_dimension,
        hyper_params.num_layers,
        hyper_params.num_heads,
        head_size,
    )
}

fn get_checkpoint(repo: &str, index: Option<usize>) -> Option<TrainingState> {
    let checkpoints_at = ragit_fs::join(repo, "checkpoint").unwrap();

    match ragit_fs::read_dir(&checkpoints_at, true) {
        Ok(cp) if !cp.is_empty() => {
            let checkpoint = &cp[cp.len() - index.unwrap_or(0) - 1];
            let cpre = CHECKPOINT_RE.captures(&checkpoint).unwrap();
            println!("reading checkpoint... step: {}, loss: {}", &cpre[1], &cpre[2]);
            let bytes = ragit_fs::read_bytes(&checkpoint).unwrap();
            Some(bincode::deserialize(&bytes).unwrap())
        },
        _ => None,
    }
}

lazy_static! {
    static ref CHECKPOINT_RE: Regex = Regex::new(r".*cp-(\d{6})-(\d*\.?\d*).*").unwrap();
}

fn save_checkpoint(repo: &str, step: usize, loss: f32, ts: &TrainingState) {
    let checkpoints_at = ragit_fs::join(repo, "checkpoint").unwrap();

    if !ragit_fs::exists(&checkpoints_at) {
        ragit_fs::create_dir(&checkpoints_at).unwrap();
    }

    let mut checkpoints = ragit_fs::read_dir(&checkpoints_at, true).unwrap();
    checkpoints = checkpoints.into_iter().filter(|cp| CHECKPOINT_RE.is_match(&cp)).collect();

    let d_step = if checkpoints.len() < 12 {
        0
    } else if checkpoints.len() < 24 {
        let cp1 = CHECKPOINT_RE.captures(&checkpoints[0]).unwrap()[1].parse::<usize>().unwrap();
        let cp2 = CHECKPOINT_RE.captures(&checkpoints[1]).unwrap()[1].parse::<usize>().unwrap();

        cp2 - cp1
    } else {
        for (i, checkpoint) in checkpoints.iter().enumerate() {
            if i % 2 == 0 {
                ragit_fs::remove_file(&checkpoint).unwrap();
            }
        }

        return;
    };

    let last_step = if let Some(last_checkpoint) = checkpoints.last() {
        CHECKPOINT_RE.captures(last_checkpoint).unwrap()[1].parse::<usize>().unwrap()
    } else {
        0
    };

    if step - last_step >= d_step {
        let file_name = ragit_fs::join3(repo, "checkpoint", &format!("cp-{step:06}-{loss:.04}")).unwrap();
        println!("Saving {file_name}...");
        ragit_fs::write_bytes(
            &file_name,
            &bincode::serialize(&ts).unwrap(),
            ragit_fs::WriteMode::AlwaysCreate,
        ).unwrap();
    }
}
