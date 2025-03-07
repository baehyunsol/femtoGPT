use femto_gpt::gpt::{TrainingState, GPT};
use femto_gpt::graph::GraphError;
use femto_gpt::optimizer::AdamW;
use femto_gpt::tokenizer::{SimpleTokenizer, Tokenizer};
use rand::seq::SliceRandom;
use std::fs;
use std::io::prelude::*;
use std::path::PathBuf;
use structopt::StructOpt;

use femto_gpt::log::{
    initialize_log,
    write_log_dataset_file,
    write_log_save,
};

// femtoGPT reads the first `context_size` tokens of the sample data when training. It's a hard-coded context-size.
// But it seems like we can change `context_size` without re-training the model.
#[derive(StructOpt, Debug)]
enum Cli {
    Train {
        #[structopt(long, default_value = "dataset.txt")]
        dataset: String,
        #[structopt(long, default_value = "training_state.dat")]
        model: PathBuf,
        #[structopt(long, default_value = "64")]
        context_size: usize,
    },
    Infer {
        #[structopt(long, default_value = "training_state.dat")]
        model: PathBuf,
        #[structopt(long)]
        prompt: String,
        #[structopt(long, default_value = "100")]
        count: usize,
        #[structopt(long, default_value = "0.5")]
        temperature: f32,
        #[structopt(long, default_value = "64")]
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

    // Hyper Parameters
    // NOTE: Feed-Forward dimension is always `embedding_dimension * 4`.
    let batch_size = 32;
    let embedding_dimension = 64;
    let num_layers = 4;
    let num_heads = 4;
    let head_size = embedding_dimension / num_heads;
    let dropout = 0.0;
    assert_eq!(num_heads * head_size, embedding_dimension);

    let cli = Cli::from_args();
    match cli {
        Cli::Infer {
            model,
            prompt,
            count,
            temperature,
            context_size,
        } => {
            let training_state_path = &model.clone();
            let mut rng = rand::thread_rng();
            let tokenizer = SimpleTokenizer::new("");

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

            let mut ts_file = fs::File::open(&training_state_path).unwrap();
            let mut bytes = Vec::new();
            ts_file.read_to_end(&mut bytes).unwrap();
            let ts: TrainingState = bincode::deserialize(&bytes).unwrap();
            gpt.set_training_state(ts, true)?;

            println!("Generating text:");

            let inference = gpt.infer(
                &mut rng,
                &tokenizer.tokenize(&prompt),
                count,
                temperature,
                |_ch| {},
            )?;

            // Generate 100 character with the currently trained model
            println!("{}", tokenizer.untokenize(&inference));

            Ok(())
        }
        Cli::Train { dataset, model, context_size } => {
            initialize_log();
            let training_state_path = &model.clone();
            let tokenizer = SimpleTokenizer::new("");
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

            // Load training data from train_data directory (If exists)
            // If you want to reuse training_data of a smaller model in a bigger model, you may
            // first start again with a new optimizer by setting load_optimizer=false
            // WARN: YOU CAN ONLY REUSE THE WEIGHTS OF A MODEL WITH DIFFERENT NUM-LAYERS!
            // IT'S NOT POSSIBLE TO CHANGE OTHER PROPERTIES ONCE THE MODEL IS TRAINED!
            if training_state_path.is_file() {
                let mut ts_file = fs::File::open(&training_state_path).unwrap();
                let mut bytes = Vec::new();
                ts_file.read_to_end(&mut bytes).unwrap();
                let ts: TrainingState = bincode::deserialize(&bytes).unwrap();
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
            let decay_steps = 50000;

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

            let callback = |gpt: &mut GPT<_>| {
                println!("Saving the model...");
                gpt.sync().unwrap();
                let ts = gpt.get_training_state().unwrap();
                let bytes = bincode::serialize(&ts).unwrap();
                fs::write(training_state_path, &bytes).expect("Unable to write file");
                write_log_save();

                Ok(())
            };

            let mut dataset_files = if ragit_fs::is_dir(&dataset) {
                ragit_fs::read_dir(&dataset, false).unwrap()
            } else {
                vec![dataset]
            };
            dataset_files.shuffle(&mut rand::thread_rng());
            let mut file_index = 0;

            loop {
                let dataset_file = dataset_files[file_index % dataset_files.len()].clone();
                write_log_dataset_file(&dataset_file, context_size);
                file_index += 1;
                println!("Reading {dataset_file}...");
                let dataset_char = ragit_fs::read_string(&dataset_file).expect("Should have been able to read the file");
                let tokenizer = SimpleTokenizer::new(&dataset_char);
                let dataset = tokenizer.tokenize(&dataset_char);

                // Training loop!
                #[cfg(not(feature = "gpu"))]
                gpt.train_cpu(
                    &dataset,
                    40,
                    batch_size,
                    None, // or Some(n), limit backward process to last n computations
                    &AdamW::new(),
                    learning_rate,
                    callback,
                )?;

                #[cfg(feature = "gpu")]
                gpt.train(
                    &dataset,
                    40,
                    batch_size,
                    None, // or Some(n), limit backward process to last n computations
                    &AdamW::new(),
                    learning_rate,
                    callback,
                )?;
            }

            Ok(())
        }
    }
}
