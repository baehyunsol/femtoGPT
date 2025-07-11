use femto_gpt::error::Error;
use femto_gpt::gpt::GPT;
use femto_gpt::model::{Model, HyperParameters};
use femto_gpt::optimizer::AdamW;
use femto_gpt::tensor::TensorOps;
use femto_gpt::tokenizer::{ByteTokenizer, Tokenizer};
use ragit_cli::{
    ArgCount,
    ArgParser,
    ArgType,
};
use ragit_fs::{
    WriteMode,
    read_bytes,
    read_string,
    write_bytes,
};
use std::collections::HashMap;

fn main() {
    if let Err(e) = run() {
        match e {
            Error::CliError { message, span } => {
                eprintln!("cli error: {message}\n\n{}", ragit_cli::underline_span(
                    &span.0,
                    span.1,
                    span.2,
                ));
            },
            _ => panic!("{e:?}"),
        }
    }
}

fn run() -> Result<(), Error> {
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

    let tokenizers: HashMap<String, Box<dyn Tokenizer>> = vec![
        ("byte", Box::new(ByteTokenizer) as Box<dyn Tokenizer>),
    ].into_iter().map(|(name, tokenizer)| (name.to_string(), tokenizer)).collect();

    let args = std::env::args().collect::<Vec<_>>();

    match args.get(1).map(|arg| arg.as_str()) {
        Some("init") => {
            let parsed_args = ArgParser::new()
                .arg_flag_with_default("--model", "model.dat", ArgType::Path)
                .arg_flag_with_default("--tokenizer", "byte", ArgType::String)
                .arg_flag_with_default("--num-tokens", "80", ArgType::IntegerBetween { min: Some(0), max: None })
                .arg_flag_with_default("--embedding-degree", "80", ArgType::IntegerBetween { min: Some(0), max: None })
                .arg_flag_with_default("--num-layers", "4", ArgType::IntegerBetween { min: Some(0), max: None })
                .arg_flag_with_default("--num-heads", "4", ArgType::IntegerBetween { min: Some(0), max: None })
                .args(ArgType::Path, ArgCount::None)
                .parse(&args, 2)?;

            let model_path = parsed_args.arg_flags.get("--model").unwrap().to_string();
            let tokenizer = parsed_args.arg_flags.get("--tokenizer").unwrap().to_string();
            let num_tokens = parsed_args.arg_flags.get("--num-tokens").unwrap().parse::<usize>().unwrap();
            let embedding_degree = parsed_args.arg_flags.get("--embedding-degree").unwrap().parse::<usize>().unwrap();
            let num_layers = parsed_args.arg_flags.get("--num-layers").unwrap().parse::<usize>().unwrap();
            let num_heads = parsed_args.arg_flags.get("--num-heads").unwrap().parse::<usize>().unwrap();

            let head_size = embedding_degree / num_heads;
            assert_eq!(head_size * num_heads, embedding_degree);
            assert!(tokenizers.get(&tokenizer).is_some());

            let mut rng = rand::thread_rng();
            let tokenizer = tokenizers.get(&tokenizer).unwrap();
            let vocab_size = tokenizer.vocab_size();

            let mut gpt = GPT::new(
                &mut rng,
                graph,
                is_gpu.then(|| batch_size), // Pre-allocate batches only when using GPUs
                vocab_size,
                embedding_degree,
                num_tokens,
                num_layers,
                num_heads,
                head_size,
                dropout,
            )?;
            gpt.sync()?;

            let training_state = gpt.get_training_state().unwrap();
            let model = Model {
                tokenizer: tokenizer.name(),
                hyper_parameters: HyperParameters {
                    num_tokens,
                    embedding_degree,
                    num_layers,
                    num_heads,
                },
                training_state,
            };
            let bytes = bincode::serialize(&model).unwrap();
            write_bytes(
                &model_path,
                &bytes,
                WriteMode::CreateOrTruncate,
            )?;

            Ok(())
        },
        Some("infer" | "inference") => {
            let parsed_args = ArgParser::new()
                .arg_flag_with_default("--model", "model.dat", ArgType::Path)
                .arg_flag_with_default("--count", "100", ArgType::IntegerBetween { min: Some(1), max: None })
                // TODO: `ArgType::Float`
                // .arg_flag_with_default("--temperature", "0.5", ArgType::FloatBetween { min: Some(0.01), max: Some(0.99) })
                .args(ArgType::String, ArgCount::Exact(1))
                .parse(&args, 2)?;

            let model_path = parsed_args.arg_flags.get("--model").unwrap().to_string();
            let count = parsed_args.arg_flags.get("--count").unwrap().parse::<usize>().unwrap();
            let prompt = parsed_args.get_args_exact(1)?[0].to_string();
            let temperature = 0.5;  // TODO

            let bytes = read_bytes(&model_path)?;
            let model: Model = bincode::deserialize(&bytes).unwrap();
            let HyperParameters {
                num_tokens,
                embedding_degree,
                num_layers,
                num_heads,
            } = model.hyper_parameters;
            let head_size = embedding_degree / num_heads;

            let mut rng = rand::thread_rng();
            let tokenizer = tokenizers.get(&model.tokenizer).unwrap();
            let ts = model.training_state;

            assert_eq!(num_heads * head_size, embedding_degree);

            let vocab_size = tokenizer.vocab_size();
            let mut gpt = GPT::new(
                &mut rng,
                graph,
                is_gpu.then(|| batch_size), // Pre-allocate batches only when using GPUs
                vocab_size,
                embedding_degree,
                num_tokens,
                num_layers,
                num_heads,
                head_size,
                dropout,
            )?;
            println!("Successfully loaded a model with {} parameters", gpt.num_params());
            println!("Vocab-size: {} unique characters", vocab_size);

            gpt.sync()?;
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
        },
        Some("train") => {
            let parsed_args = ArgParser::new()
                .arg_flag_with_default("--dataset", "dataset.txt", ArgType::Path)
                .arg_flag_with_default("--model", "model.dat", ArgType::Path)
                // TODO: `ArgType::Float`
                // .arg_flag_with_default("--dropout", "0", ArgType::FloatBetween { min: Some(0.0), max: Some(0.99) })
                .optional_flag(&["--reset-optimizer"])
                .args(ArgType::Path, ArgCount::None)
                .parse(&args, 2)?;

            let model_path = parsed_args.arg_flags.get("--model").unwrap().to_string();
            let dataset = parsed_args.arg_flags.get("--dataset").unwrap().to_string();
            let reset_optimizer = parsed_args.get_flag(0).is_some();

            let bytes = read_bytes(&model_path)?;
            let model: Model = bincode::deserialize(&bytes).unwrap();
            let HyperParameters {
                num_tokens,
                embedding_degree,
                num_layers,
                num_heads,
            } = model.hyper_parameters;
            let head_size = embedding_degree / num_heads;

            let mut rng = rand::thread_rng();

            let dataset_char = read_string(&dataset)?;
            let tokenizer = tokenizers.get(&model.tokenizer).unwrap();
            let ts = model.training_state;
            let dataset = tokenizer.tokenize(&dataset_char);

            let vocab_size = tokenizer.vocab_size();
            let mut gpt = GPT::new(
                &mut rng,
                graph,
                is_gpu.then(|| batch_size), // Pre-allocate batches only when using GPUs
                vocab_size,
                embedding_degree,
                num_tokens,
                num_layers,
                num_heads,
                head_size,
                dropout,
            )?;

            println!("Successfully loaded a model with {} parameters", gpt.num_params());
            println!("Vocab-size: {} unique characters", vocab_size);
            gpt.sync()?;
            gpt.set_training_state(ts, !reset_optimizer)?;

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
                let mut rng = rand::thread_rng();
                let inference_temperature = 0.5; // How creative? 0.0 min 1.0 max

                // I'm calling it less often because it's too expensive to generate 100 tokens.
                if rand::random::<u32>() % 5 == 0 {
                    println!("Generating text:");

                    let inference = gpt.infer(
                        &mut rng,
                        &tokenizer.tokenize("\n"),
                        100,
                        inference_temperature,
                        |_ch| {},
                    )?;

                    // Generate 100 character with the currently trained model before
                    // starting the training loop.
                    println!("{}", tokenizer.untokenize(&inference));
                }

                println!("Saving the model...");
                gpt.sync().unwrap();
                let training_state = gpt.get_training_state().unwrap();
                let model = Model {
                    tokenizer: tokenizer.name(),
                    hyper_parameters: HyperParameters {
                        num_tokens,
                        embedding_degree,
                        num_layers,
                        num_heads,
                    },
                    training_state,
                };
                let bytes = bincode::serialize(&model).unwrap();
                write_bytes(
                    &model_path,
                    &bytes,
                    WriteMode::CreateOrTruncate,
                ).unwrap();

                Ok(())
            };

            // Training loop!
            #[cfg(not(feature = "gpu"))]
            gpt.train_cpu(
                &dataset,
                100000,
                batch_size,
                None, // or Some(n), limit backward process to last n computations
                &AdamW::new(),
                learning_rate,
                callback,
            )?;

            #[cfg(feature = "gpu")]
            gpt.train(
                &dataset,
                100000,
                batch_size,
                None, // or Some(n), limit backward process to last n computations
                &AdamW::new(),
                learning_rate,
                callback,
            )?;

            Ok(())
        },
        Some("info") => {
            let parsed_args = ArgParser::new()
                .arg_flag_with_default("--model", "model.dat", ArgType::Path)
                .args(ArgType::Path, ArgCount::None)
                .parse(&args, 2)?;

            let model_path = parsed_args.arg_flags.get("--model").unwrap().to_string();
            let bytes = read_bytes(&model_path)?;
            let model: Model = bincode::deserialize(&bytes).unwrap();
            let HyperParameters {
                embedding_degree,
                num_heads,
                ..
            } = model.hyper_parameters;
            let head_size = embedding_degree / num_heads;
            println!("{:?}", model.hyper_parameters);

            let embeddings = model.training_state.tensors.get("token_embedding").unwrap();
            let token_shape = embeddings.shape();
            let embedding_degree = token_shape[1];
            let blob = embeddings.blob();

            for token_i in 0..token_shape[0] {
                match model.tokenizer.as_str() {
                    "byte" => {
                        println!("{token_i} ({:?})", String::from_utf8_lossy(&[token_i as u8]));
                    },
                    _ => unreachable!(),
                }

                let token_embedding = &blob[(token_i * embedding_degree)..(token_i * embedding_degree + embedding_degree)];

                for i in 0..num_heads {
                    let curr_head = &token_embedding[(i * head_size)..(i * head_size + head_size)];
                    let rendered = format!(
                        "[{}]",
                        // If the numbers are too small, it's less readable. So I multiply them by `embedding_degree`.
                        curr_head.iter().map(|f| format!("{:.3}", *f * embedding_degree as f32)).collect::<Vec<_>>().join(", "),
                    );

                    println!("    head {i}: {rendered}");
                }
            }

            Ok(())
        },
        _ => todo!(),
    }
}
