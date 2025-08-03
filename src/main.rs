use femto_gpt::error::Error;
use femto_gpt::gpt::GPT;
use femto_gpt::model::{
    Hyperparameters,
    Log,
    LogKind,
    Model,
    ModelInfo,
    TokenInfo,
    f2s,
};
use femto_gpt::optimizer::AdamW;
use femto_gpt::tensor::TensorOps;
use femto_gpt::tokenizer::{
    BpeConfig,
    TokenizerInner,
    Tokenizer,
    count_chars,
    count_tokens,
};
use ragit_cli::{
    ArgCount,
    ArgParser,
    ArgType,
};
use ragit_fs::{
    WriteMode,
    file_size,
    is_dir,
    read_bytes,
    read_bytes_offset,
    read_dir,
    read_string,
    write_bytes,
    write_string,
};
use rand::seq::SliceRandom;
use serde_json::{Map, Value};
use std::collections::HashMap;
use std::io::Write;

fn main() {
    if let Err(e) = run() {
        match e {
            Error::CliError { message, span } => {
                eprintln!(
                    "cli error: {message}{}",
                    if let Some(span) = span {
                        format!("\n\n{}", ragit_cli::underline_span(&span))
                    } else {
                        String::new()
                    },
                );
                std::process::exit(1);
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

    let args = std::env::args().collect::<Vec<_>>();

    match args.get(1).map(|arg| arg.as_str()) {
        Some("init") => {
            let parsed_args = ArgParser::new()
                .optional_flag(&["--interactive"])
                .arg_flag_with_default("--model", "model.dat", ArgType::String)  // path
                .arg_flag_with_default("--tokenizer", "ascii", ArgType::enum_(&["ascii", "char", "bpe"]))
                .optional_arg_flag("--tokenizer-data", ArgType::String)  // path
                .flag_with_default(&["--case-sensitive", "--case-insensitive"])
                .arg_flag_with_default("--reserve-tokens", "0", ArgType::uinteger())
                .arg_flag_with_default("--positional-encoding", "absolute", ArgType::String)
                .arg_flag_with_default("--num-tokens", "80", ArgType::uinteger())
                .arg_flag_with_default("--embedding-degree", "80", ArgType::uinteger())
                .arg_flag_with_default("--num-layers", "4", ArgType::uinteger())
                .arg_flag_with_default("--num-heads", "4", ArgType::uinteger())
                .short_flag(&["--interactive"])
                .args(ArgType::String, ArgCount::None)
                .parse(&args, 2)?;

            if parsed_args.show_help() {
                println!("{}", include_str!("../docs/commands/init.txt"));
                return Ok(());
            }

            let model_path = parsed_args.arg_flags.get("--model").unwrap().to_string();
            let mut tokenizer = parsed_args.arg_flags.get("--tokenizer").unwrap().to_string();
            let reserve_tokens = parsed_args.arg_flags.get("--reserve-tokens").unwrap().parse().unwrap();
            let mut pos_enc = parsed_args.arg_flags.get("--positional-encoding").unwrap().to_string();
            let mut case_sensitive = parsed_args.get_flag(1).unwrap() == "--case-sensitive";
            let mut tokenizer_data = match parsed_args.arg_flags.get("--tokenizer-data") {
                Some(tokenizer_data) => tokenizer_data.to_string(),
                None => match tokenizer.as_str() {
                    "char" => String::from("dataset.txt"),
                    "bpe" => String::from("tokenizer.json"),
                    _ => String::new(),
                },
            };
            let mut num_tokens = parsed_args.arg_flags.get("--num-tokens").unwrap().parse::<usize>().unwrap();
            let mut embedding_degree = parsed_args.arg_flags.get("--embedding-degree").unwrap().parse::<usize>().unwrap();
            let mut num_layers = parsed_args.arg_flags.get("--num-layers").unwrap().parse::<usize>().unwrap();
            let mut num_heads = parsed_args.arg_flags.get("--num-heads").unwrap().parse::<usize>().unwrap();

            if parsed_args.get_flag(0).is_some() {
                let mut s = String::new();

                println!("Select tokenizer: ascii, char or bpe (default: ascii)");
                print!(">>> ");
                std::io::stdout().flush()?;
                std::io::stdin().read_line(&mut s)?;
                tokenizer = s.trim().to_string();
                s = String::new();

                println!("Case sensitive tokenizer? yes / no");
                print!(">>> ");
                std::io::stdout().flush()?;
                std::io::stdin().read_line(&mut s)?;
                case_sensitive = s.trim().to_ascii_lowercase().starts_with("y");
                s = String::new();

                if tokenizer == "char" {
                    println!("Select dataset file (default: dataset.txt)");
                    print!(">>> ");
                    std::io::stdout().flush()?;
                    std::io::stdin().read_line(&mut s)?;
                    tokenizer_data = s.trim().to_string();
                    s = String::new();
                }

                else if tokenizer == "bpe" {
                    println!("Select tokenizer data file (default: tokenizer.json)");
                    print!(">>> ");
                    std::io::stdout().flush()?;
                    std::io::stdin().read_line(&mut s)?;
                    tokenizer_data = s.trim().to_string();
                    s = String::new();
                }

                println!("Select positional encoding: none or absolute (default: absolute)");
                print!(">>> ");
                std::io::stdout().flush()?;
                std::io::stdin().read_line(&mut s)?;
                pos_enc = s.trim().to_string();
                s = String::new();

                println!("Set num tokens (default: 80)");
                print!(">>> ");
                std::io::stdout().flush()?;
                std::io::stdin().read_line(&mut s)?;
                num_tokens = s.trim().parse().unwrap();
                s = String::new();

                println!("Set embedding degree (default: 80)");
                print!(">>> ");
                std::io::stdout().flush()?;
                std::io::stdin().read_line(&mut s)?;
                embedding_degree = s.trim().parse().unwrap();
                s = String::new();

                println!("Set num layers (default: 4)");
                print!(">>> ");
                std::io::stdout().flush()?;
                std::io::stdin().read_line(&mut s)?;
                num_layers = s.trim().parse().unwrap();
                s = String::new();

                println!("Set num heads (default: 4)");
                print!(">>> ");
                std::io::stdout().flush()?;
                std::io::stdin().read_line(&mut s)?;
                num_heads = s.trim().parse().unwrap();
            }

            let head_size = embedding_degree / num_heads;
            assert_eq!(head_size * num_heads, embedding_degree);

            let mut rng = rand::thread_rng();

            let mut tokenizer = match tokenizer.as_str() {
                "ascii" => Tokenizer::ascii(case_sensitive),
                "char" => {
                    let mut config = BpeConfig::default();
                    config.vocab_size = 2048;
                    config.char_vocab_size = Some(2048);
                    config.case_sensitive = case_sensitive;
                    let char_count = count_chars(&tokenizer_data, config.unit)?;
                    Tokenizer::from_inner(TokenizerInner::from_char_count(&char_count, &config))
                },
                "bpe" => {
                    let data = read_string(&tokenizer_data)?;
                    let data: serde_json::Value = serde_json::from_str(&data)?;

                    match serde_json::from_value::<Vec<String>>(data.clone()) {
                        Ok(tokens) => Tokenizer::from_tokens(tokens, case_sensitive),
                        Err(_) => Tokenizer::from_inner(serde_json::from_value(data)?),
                    }
                },
                t => {
                    panic!("{t:?} is not a valid tokenizer.");
                },
            };

            if reserve_tokens > 0 {
                tokenizer.reserve_tokens(reserve_tokens);
            }

            let pos_enc = pos_enc.parse()?;
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
                0.0,  // dropout
                vec![],  // logs
                pos_enc,
            )?;
            gpt.sync()?;
            println!("Successfully initialized a model with {} parameters", gpt.num_params());

            let training_state = gpt.get_training_state().unwrap();
            let hyperparameters = Hyperparameters {
                num_tokens,
                vocab_size,
                embedding_degree,
                num_layers,
                num_heads,
                head_size,
            };
            let model = Model {
                tokenizer: tokenizer.inner.clone(),
                pos_enc,
                hyperparameters,
                training_state,
                logs: vec![Log::init(hyperparameters)],
            };
            let bytes = bincode::serialize(&model)?;
            write_bytes(
                &model_path,
                &bytes,
                WriteMode::Atomic,
            )?;

            Ok(())
        },
        Some("infer" | "inference") => {
            let parsed_args = ArgParser::new()
                .arg_flag_with_default("--model", "model.dat", ArgType::String)  // path
                .arg_flag_with_default("--count", "100", ArgType::integer_between(Some(1), None))
                .arg_flag_with_default("--temperature", "0.5", ArgType::float_between(Some(0.0), Some(1.0)))
                .args(ArgType::String, ArgCount::Exact(1))
                .parse(&args, 2)?;

            if parsed_args.show_help() {
                println!("{}", include_str!("../docs/commands/infer.txt"));
                return Ok(());
            }

            let model_path = parsed_args.arg_flags.get("--model").unwrap().to_string();
            let count = parsed_args.arg_flags.get("--count").unwrap().parse::<usize>().unwrap();
            let temperature = parsed_args.arg_flags.get("--temperature").unwrap().parse::<f32>().unwrap();
            let prompt = parsed_args.get_args_exact(1)?[0].to_string();

            let bytes = read_bytes(&model_path)?;
            let model: Model = bincode::deserialize(&bytes)?;
            let Hyperparameters {
                num_tokens,
                vocab_size,
                embedding_degree,
                num_layers,
                num_heads,
                head_size,
            } = model.hyperparameters;

            let mut rng = rand::thread_rng();
            let tokenizer = Tokenizer::from_inner(model.tokenizer.clone());

            let ts = model.training_state;

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

                // We don't need dropouts for inference, right?
                0.0,  // dropout

                // Do we have to log inferences?
                vec![],  // logs
                model.pos_enc,
            )?;
            println!("Successfully loaded a model with {} parameters", gpt.num_params());
            println!("Vocab-size: {} tokens", vocab_size);

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

            println!("{}", tokenizer.untokenize(&inference));
            Ok(())
        },
        Some("train") => {
            let parsed_args = ArgParser::new()
                .arg_flag_with_default("--model", "model.dat", ArgType::String)  // path
                .arg_flag_with_default("--dataset", "dataset.txt", ArgType::String)  // path
                .arg_flag_with_default("--dropout", "0", ArgType::float_between(Some(0.0), Some(1.0)))
                .arg_flag_with_default("--steps", "100000", ArgType::integer_between(Some(10), None))
                .optional_flag(&["--reset-optimizer"])
                .args(ArgType::String, ArgCount::None)
                .parse(&args, 2)?;

            if parsed_args.show_help() {
                println!("{}", include_str!("../docs/commands/train.txt"));
                return Ok(());
            }

            let model_path = parsed_args.arg_flags.get("--model").unwrap().to_string();
            let dataset_path = parsed_args.arg_flags.get("--dataset").unwrap().to_string();
            let dropout = parsed_args.arg_flags.get("--dropout").unwrap().parse::<f32>().unwrap();
            let steps = parsed_args.arg_flags.get("--steps").unwrap().parse::<usize>().unwrap();
            let reset_optimizer = parsed_args.get_flag(0).is_some();

            let bytes = read_bytes(&model_path)?;
            let mut model: Model = bincode::deserialize(&bytes).unwrap();

            if reset_optimizer {
                model.logs.push(Log::reset_optimizer());
            }

            let Hyperparameters {
                num_tokens,
                vocab_size,
                embedding_degree,
                num_layers,
                num_heads,
                head_size,
            } = model.hyperparameters;

            let mut rng = rand::thread_rng();

            let dataset = read_string(&dataset_path)?;
            model.logs.push(Log::train_session(dropout, &dataset_path, &dataset));
            let tokenizer = Tokenizer::from_inner(model.tokenizer.clone());

            let ts = model.training_state;
            let dataset = tokenizer.tokenize(&dataset);

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
                model.logs.clone(),
                model.pos_enc,
            )?;

            println!("Successfully loaded a model with {} parameters", gpt.num_params());
            println!("Vocab-size: {} tokens", vocab_size);
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
                    tokenizer: tokenizer.inner.clone(),
                    pos_enc: gpt.pos_enc,
                    hyperparameters: Hyperparameters {
                        num_tokens,
                        vocab_size,
                        embedding_degree,
                        num_layers,
                        num_heads,
                        head_size,
                    },
                    training_state,
                    logs: gpt.logs.clone(),
                };
                let bytes = bincode::serialize(&model).unwrap();
                write_bytes(
                    &model_path,
                    &bytes,
                    WriteMode::Atomic,
                ).unwrap();

                Ok(())
            };

            // Training loop!
            #[cfg(not(feature = "gpu"))]
            gpt.train_cpu(
                &dataset,
                steps,
                batch_size,
                None, // or Some(n), limit backward process to last n computations
                &AdamW::new(),
                learning_rate,
                callback,
            )?;

            #[cfg(feature = "gpu")]
            gpt.train(
                &dataset,
                steps,
                batch_size,
                None, // or Some(n), limit backward process to last n computations
                &AdamW::new(),
                learning_rate,
                callback,
            )?;

            Ok(())
        },
        Some("train-bpe") => {
            let parsed_args = ArgParser::new()
                .flag_with_default(&["--case-sensitive", "--case-insensitive"])
                .arg_flag_with_default("--dataset", "dataset.txt", ArgType::String)  // path
                .arg_flag_with_default("--tokenizer-data", "tokenizer.json", ArgType::String)  // path
                .arg_flag_with_default("--reserve-tokens", "0", ArgType::uinteger())
                .arg_flag_with_default("--vocab-size", "768", ArgType::integer_between(Some(256), None))
                .args(ArgType::String, ArgCount::None)
                .parse(&args, 2)?;

            if parsed_args.show_help() {
                println!("{}", include_str!("../docs/commands/train-bpe.txt"));
                return Ok(());
            }

            let case_sensitive = parsed_args.get_flag(0).unwrap() == "--case-sensitive";
            let dataset = parsed_args.arg_flags.get("--dataset").unwrap().to_string();
            let tokenizer_data = parsed_args.arg_flags.get("--tokenizer-data").unwrap().to_string();
            let reserve_tokens = parsed_args.arg_flags.get("--reserve-tokens").unwrap().parse().unwrap();
            let vocab_size = parsed_args.arg_flags.get("--vocab-size").unwrap().parse::<usize>().unwrap();
            let epoch = (vocab_size / 10).max(5) + 1;

            let mut config = BpeConfig::default();
            config.vocab_size = vocab_size;
            config.char_vocab_size = Some(vocab_size / 2);
            config.case_sensitive = case_sensitive;
            let char_count = count_chars(&dataset, config.unit)?;
            let mut tokenizer = TokenizerInner::from_char_count(&char_count, &config);

            let max_corpus_size = 1 << 24;

            for i in 0..epoch {
                let corpus = if is_dir(&dataset) {
                    get_corpus(&dataset, max_corpus_size)?
                } else {
                    read_bytes(&dataset)?
                };
                tokenizer.train(&corpus, 10);

                if i % 4 == 0 {
                    tokenizer.trim_tail(&dataset, &config, None)?;
                }

                tokenizer.compact();
                println!("Vocab-size: {} tokens", tokenizer.len());
            }

            tokenizer.trim_tail(&dataset, &config, None)?;
            tokenizer.compact();

            if reserve_tokens > 0 {
                tokenizer.reserve_tokens(reserve_tokens);
            }

            println!("Saving the tokenizer...");
            let t_s = serde_json::to_string_pretty(&tokenizer).unwrap();
            write_string(
                &tokenizer_data,
                &t_s,
                WriteMode::CreateOrTruncate,
            )?;

            Ok(())
        },
        Some("count-tokens") => {
            let parsed_args = ArgParser::new()
                // It can be a model file or a tokenizer file. The program will
                // detect the file type.
                .arg_flag_with_default("--model", "tokenizer.json", ArgType::String)  // path
                .arg_flag_with_default("--dataset", "dataset.txt", ArgType::String)  // path
                .args(ArgType::String, ArgCount::None)
                .parse(&args, 2)?;

            if parsed_args.show_help() {
                println!("{}", include_str!("../docs/commands/count-tokens.txt"));
                return Ok(());
            }

            let model = parsed_args.arg_flags.get("--model").unwrap().to_string();
            let dataset = parsed_args.arg_flags.get("--dataset").unwrap().to_string();

            let tokenizer = match read_string(&model) {
                Ok(s) => serde_json::from_str::<TokenizerInner>(&s)?,
                Err(_) => {
                    let bytes = read_bytes(&model)?;
                    let model: Model = bincode::deserialize(&bytes)?;
                    model.tokenizer
                },
            };

            let counted = count_tokens(&tokenizer, &dataset)?;
            let mut counted = counted.into_iter().collect::<Vec<_>>();
            counted.sort_by_key(|(_, count)| u64::MAX - *count);

            let counted = counted.into_iter().enumerate().map(
                |(i, (id, count))| vec![
                    (String::from("rank"), Value::from(i)),
                    (String::from("id"), Value::from(id)),
                    (String::from("token"), Value::from(tokenizer.decode_id(id))),
                    (String::from("count"), Value::from(count)),
                ].into_iter().collect::<Map<_, _>>()
            ).collect::<Vec<_>>();
            println!("{}", serde_json::to_string_pretty(&counted)?);
            Ok(())
        },
        Some("loss") => {
            let parsed_args = ArgParser::new()
                .arg_flag_with_default("--model", "model.dat", ArgType::String)  // path
                .arg_flag_with_default("--limit", "10", ArgType::uinteger())
                .args(ArgType::String, ArgCount::None)
                .parse(&args, 2)?;

            if parsed_args.show_help() {
                println!("{}", include_str!("../docs/commands/loss.txt"));
                return Ok(());
            }

            let model_path = parsed_args.arg_flags.get("--model").unwrap().to_string();
            let bytes = read_bytes(&model_path)?;
            let model: Model = bincode::deserialize(&bytes)?;
            let limit = parsed_args.arg_flags.get("--limit").unwrap().parse::<usize>().unwrap();
            let mut logs = model.logs.into_iter().filter(
                |log| matches!(log.kind, LogKind::TrainStep { .. })
            ).collect::<Vec<_>>();

            if logs.len() > limit {
                logs = logs[(logs.len() - limit)..].to_vec();
            }

            for log in logs {
                let LogKind::TrainStep { avg_loss, .. } = log.kind else { unreachable!() };
                println!("{avg_loss}");
            }

            Ok(())
        },
        Some("info") => {
            let parsed_args = ArgParser::new()
                .arg_flag_with_default("--model", "model.dat", ArgType::String)  // path
                .args(ArgType::String, ArgCount::None)
                .parse(&args, 2)?;

            if parsed_args.show_help() {
                println!("{}", include_str!("../docs/commands/info.txt"));
                return Ok(());
            }

            let model_path = parsed_args.arg_flags.get("--model").unwrap().to_string();
            let bytes = read_bytes(&model_path)?;
            let model: Model = bincode::deserialize(&bytes)?;
            let Hyperparameters {
                num_tokens,
                vocab_size,
                embedding_degree,
                num_layers,
                num_heads,
                head_size,
            } = model.hyperparameters;
            let mut rng = rand::thread_rng();
            let tokenizer = Tokenizer::from_inner(model.tokenizer.clone());

            let gpt = GPT::new(
                &mut rng,
                graph,
                is_gpu.then(|| batch_size), // Pre-allocate batches only when using GPUs
                vocab_size,
                embedding_degree,
                num_tokens,
                num_layers,
                num_heads,
                head_size,
                0.0,  // dropout
                vec![],  // logs
                model.pos_enc,
            )?;

            let mut info = ModelInfo {
                hyperparameters: model.hyperparameters,
                num_params: gpt.num_params(),
                logs: model.logs.clone(),
                tokens: vec![],
            };

            let embeddings = model.training_state.tensors.get("token_embedding").unwrap();
            let embedding_shape = embeddings.shape();
            assert_eq!(embedding_shape, [vocab_size, embedding_degree]);

            let blob = embeddings.blob();

            for token_id in 0..vocab_size {
                let mut heads = vec![];
                let token_embedding = &blob[(token_id * embedding_degree)..(token_id * embedding_degree + embedding_degree)];

                for i in 0..num_heads {
                    let curr_head = &token_embedding[(i * head_size)..(i * head_size + head_size)];
                    let rendered = curr_head.iter().map(|n| f2s(*n)).collect::<Vec<_>>().join("");
                    heads.push(rendered);
                }

                info.tokens.push(TokenInfo {
                    index: token_id,
                    string: tokenizer.untokenize(&[token_id]),
                    heads,
                });
            }

            println!("{}", serde_json::to_string_pretty(&info).unwrap());
            Ok(())
        },
        Some("compare") => {
            let parsed_args = ArgParser::new()
                .optional_flag(&["--reverse"])
                .arg_flag_with_default("--limit", "10", ArgType::integer_between(Some(1), None))
                .args(ArgType::String, ArgCount::Exact(2))  // path
                .parse(&args, 2)?;

            if parsed_args.show_help() {
                println!("{}", include_str!("../docs/commands/compare.txt"));
                return Ok(());
            }

            let model_paths = parsed_args.get_args_exact(2)?;
            let reverse = parsed_args.get_flag(0).is_some();
            let limit = parsed_args.arg_flags.get("--limit").unwrap().parse::<usize>().unwrap();
            let model1_path = model_paths[0].to_string();
            let model2_path = model_paths[1].to_string();
            let bytes = read_bytes(&model1_path)?;
            let model1: Model = bincode::deserialize(&bytes)?;
            let bytes = read_bytes(&model2_path)?;
            let model2: Model = bincode::deserialize(&bytes)?;
            let tokenizer = Tokenizer::from_inner(model1.tokenizer.clone());

            let mut same_until = model1.logs.len().min(model2.logs.len());
            let mut same_train_step = 0;
            let mut model1_train_step = 0;
            let mut model2_train_step = 0;

            for i in 0..(model1.logs.len().min(model2.logs.len())) {
                if model1.logs[i].id != model2.logs[i].id {
                    if i == 0 {
                        panic!("Cannot compare 2 different models!");
                    }

                    same_until = i - 1;
                    break;
                }

                if let LogKind::TrainStep { .. } = &model1.logs[i].kind {
                    same_train_step += 1;
                }
            }

            for i in same_until..model1.logs.len() {
                if let LogKind::TrainStep { .. } = &model1.logs[i].kind {
                    model1_train_step += 1;
                }
            }

            for i in same_until..model2.logs.len() {
                if let LogKind::TrainStep { .. } = &model2.logs[i].kind {
                    model2_train_step += 1;
                }
            }

            match (model1_train_step, model2_train_step) {
                (0, 0) => {
                    println!("{model1_path} and {model2_path} have gone through the same training session. There's nothing to compare.");
                    println!("The parent has been trained for {same_train_step} steps.");
                },
                (0, n) => {
                    println!("{model1_path} is parent of {model2_path}.");
                    println!("{model2_path} is {n}-steps further trained version of {model1_path}");
                },
                (m, 0) => {
                    println!("{model2_path} is parent of {model1_path}.");
                    println!("{model1_path} is {m}-steps further trained version of {model2_path}");
                },
                (m, n) => {
                    println!("{model1_path} and {model2_path} have the same parent.");
                    println!("{model1_path} is {m}-steps further trained version of the parent.");
                    println!("{model2_path} is {n}-steps further trained version of the parent.");
                },
            }

            if model1.hyperparameters != model2.hyperparameters {
                panic!("TODO: comparing 2 models with different hyperparameters.");
            }

            let mut cosine_by_key = vec![];

            for key in model1.training_state.tensors.keys() {
                if key == "token_embedding" {
                    continue;
                }

                let s = cosine(
                    model1.training_state.tensors.get(key).unwrap().blob(),
                    model2.training_state.tensors.get(key).unwrap().blob(),
                );
                cosine_by_key.push((key.to_string(), s));
            }

            cosine_by_key.sort_by_key(|(key, _)| key.to_string());
            cosine_by_key.sort_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            if reverse {
                cosine_by_key = cosine_by_key.into_iter().rev().collect();
            }

            for (key, cosine) in &cosine_by_key[..limit.min(cosine_by_key.len())] {
                println!("key: {key}, cosine: {cosine:.6}");
            }

            let embeddings1 = model1.training_state.tensors.get("token_embedding").unwrap();
            let embeddings2 = model2.training_state.tensors.get("token_embedding").unwrap();
            let vocab_size = model1.hyperparameters.vocab_size;
            let embedding_degree = model1.hyperparameters.embedding_degree;
            let head_size = model1.hyperparameters.head_size;
            let num_heads = model1.hyperparameters.num_heads;

            let blob1 = embeddings1.blob();
            let blob2 = embeddings2.blob();
            let mut cosine_by_token = vec![];

            for token_id in 0..vocab_size {
                let token_embedding1 = &blob1[(token_id * embedding_degree)..(token_id * embedding_degree + embedding_degree)];
                let token_embedding2 = &blob2[(token_id * embedding_degree)..(token_id * embedding_degree + embedding_degree)];

                for i in 0..num_heads {
                    let curr_head1 = &token_embedding1[(i * head_size)..(i * head_size + head_size)];
                    let curr_head2 = &token_embedding2[(i * head_size)..(i * head_size + head_size)];
                    cosine_by_token.push((token_id, tokenizer.untokenize(&[token_id]), i, cosine(&curr_head1, &curr_head2)));
                }
            }

            cosine_by_token.sort_by_key(|(_, token, _, _)| token.to_string());
            cosine_by_token.sort_by(|(_, _, _, a), (_, _, _, b)| a.partial_cmp(b).unwrap());

            if reverse {
                cosine_by_token = cosine_by_token.into_iter().rev().collect();
            }

            for (token_id, token, head_i, cosine) in &cosine_by_token[..limit.min(cosine_by_key.len())] {
                println!("token_id: {token_id}, token: {token:?}, head: {head_i}, cosine: {cosine:.6}");
            }

            Ok(())
        },
        Some("cluster-tokens") => {
            let parsed_args = ArgParser::new()
                .arg_flag_with_default("--model", "model.dat", ArgType::String)  // path
                .args(ArgType::String, ArgCount::None)
                .parse(&args, 2)?;

            if parsed_args.show_help() {
                println!("{}", include_str!("../docs/commands/cluster-tokens.txt"));
                return Ok(());
            }

            let model_path = parsed_args.arg_flags.get("--model").unwrap().to_string();
            let bytes = read_bytes(&model_path)?;
            let model: Model = bincode::deserialize(&bytes).unwrap();
            let tokenizer = Tokenizer::from_inner(model.tokenizer.clone());
            let Hyperparameters {
                vocab_size,
                embedding_degree,
                num_heads,
                head_size,
                ..
            } = model.hyperparameters;

            let mut by_head = vec![Vec::with_capacity(vocab_size); num_heads];
            let embeddings = model.training_state.tensors.get("token_embedding").unwrap();
            let blob = embeddings.blob();

            for token_id in 0..vocab_size {
                let token_embedding = &blob[(token_id * embedding_degree)..(token_id * embedding_degree + embedding_degree)];

                for i in 0..num_heads {
                    let curr_head = &token_embedding[(i * head_size)..(i * head_size + head_size)];
                    by_head[i].push(curr_head.to_vec());
                }
            }

            for i in 0..num_heads {
                let mut cosines = Vec::with_capacity(vocab_size * vocab_size / 2);

                for j in 0..vocab_size {
                    for k in (j + 1)..vocab_size {
                        cosines.push((j, k, cosine(&by_head[i][j], &by_head[i][k])));
                    }
                }

                // rev sort
                cosines.sort_by(|(_, _, a), (_, _, b)| b.partial_cmp(a).unwrap());

                println!("---- Head {i} ----");

                for (token_id1, token_id2, cosine) in cosines[..10].iter() {
                    println!(
                        "{:?} <-> {:?}: {cosine}",
                        tokenizer.untokenize(&[*token_id1]),
                        tokenizer.untokenize(&[*token_id2]),
                    );
                }
            }

            Ok(())
        },
        Some("insert-layer") => {
            let parsed_args = ArgParser::new()
                .arg_flag_with_default("--input", "model.dat", ArgType::String)  // path
                .arg_flag_with_default("--output", "inserted.dat", ArgType::String)  // path
                .arg_flag_with_default("--insert-at", "1", ArgType::uinteger())
                .args(ArgType::String, ArgCount::None)
                .parse(&args, 2)?;

            if parsed_args.show_help() {
                println!("{}", include_str!("../docs/commands/insert-layer.txt"));
                return Ok(());
            }

            let input_path = parsed_args.arg_flags.get("--input").unwrap().to_string();
            let output_path = parsed_args.arg_flags.get("--output").unwrap().to_string();
            let insert_at = parsed_args.arg_flags.get("--insert-at").unwrap().parse::<usize>().unwrap();

            let bytes = read_bytes(&input_path)?;
            let mut model: Model = bincode::deserialize(&bytes).unwrap();
            let mut rng = rand::thread_rng();

            model.hyperparameters.num_layers += 1;
            let Hyperparameters {
                num_tokens,
                vocab_size,
                embedding_degree,
                num_layers,
                num_heads,
                head_size,
            } = model.hyperparameters;

            // It has to be trained again from scratch.
            // But it would converge much faster... I hope!
            model.training_state.optimizer = Default::default();

            let mut new_tensors = HashMap::new();

            for key in [
                "token_embedding",
                "head_norm_bias",
                "head_norm_coeff",
                "head_map_bias",
                "head_map_weights",
            ] {
                new_tensors.insert(
                    key.to_string(),
                    model.training_state.tensors.remove(key).unwrap(),
                );
            }

            for i in 0..(num_layers - 1) {
                let new_i = if i < insert_at { i } else { i + 1 };

                for (new_key, old_key) in [
                    (format!("norm_{new_i}_bias"), format!("norm_{i}_bias")),
                    (format!("norm_{new_i}_coeff"), format!("norm_{i}_coeff")),
                    (format!("proj_{new_i}_bias"), format!("proj_{i}_bias")),
                    (format!("proj_{new_i}_weights"), format!("proj_{i}_weights")),
                    (format!("atten_norm_{new_i}_bias"), format!("atten_norm_{i}_bias")),
                    (format!("atten_norm_{new_i}_coeff"), format!("atten_norm_{i}_coeff")),
                    (format!("feedforward1_{new_i}_bias"), format!("feedforward1_{i}_bias")),
                    (format!("feedforward1_{new_i}_weights"), format!("feedforward1_{i}_weights")),
                    (format!("feedforward2_{new_i}_bias"), format!("feedforward2_{i}_bias")),
                    (format!("feedforward2_{new_i}_weights"), format!("feedforward2_{i}_weights")),
                ] {
                    new_tensors.insert(
                        new_key,
                        model.training_state.tensors.remove(&old_key).unwrap(),
                    );
                }

                for head in 0..num_heads {
                    for (new_key, old_key) in [
                        (format!("head_{new_i}_{head}_k"), format!("head_{i}_{head}_k")),
                        (format!("head_{new_i}_{head}_q"), format!("head_{i}_{head}_q")),
                        (format!("head_{new_i}_{head}_v"), format!("head_{i}_{head}_v")),
                    ] {
                        new_tensors.insert(
                            new_key,
                            model.training_state.tensors.remove(&old_key).unwrap(),
                        );
                    }
                }
            }

            assert_eq!(model.training_state.tensors.len(), 0);
            model.logs.push(Log::insert_layer(insert_at));
            model.logs.push(Log::reset_optimizer());
            model.training_state.tensors = new_tensors;

            // we're NOT setting the parameters of the new layer
            // instead, we'll instantiate an empty GPT and write the new training
            // state to the empty GPT.
            //
            // 2025-07-14: It seems like the new layer is messing up the entire model.
            //             How about cloning an existing layer?
            let bytes = bincode::serialize(&model)?;
            write_bytes(
                &output_path,
                &bytes,
                WriteMode::Atomic,
            )?;

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
                0.0,  // dropout
                vec![],  // logs
                model.pos_enc,
            )?;
            gpt.sync()?;
            gpt.set_training_state(model.training_state.clone(), false)?;

            let training_state = gpt.get_training_state()?;
            let model = Model {
                training_state,
                ..model.clone()
            };
            let bytes = bincode::serialize(&model)?;
            write_bytes(
                &output_path,
                &bytes,
                WriteMode::Atomic,
            )?;

            Ok(())
        },
        _ => todo!(),
    }
}

fn get_corpus(dir: &str, size_limit: u64) -> Result<Vec<u8>, Error> {
    let mut buffer = vec![];
    let mut files = read_dir(dir, false)?;
    files.shuffle(&mut rand::thread_rng());

    for f in files.iter() {
        let s = file_size(f)?;

        // delimiter between files
        buffer.append(&mut vec![b'\n']);

        if buffer.len() as u64 + s > size_limit {
            let l = size_limit - buffer.len() as u64;

            // avoid appending a too small chunk
            if l < 256 {
                break;
            }

            let start = rand::random::<u64>() % (s - l);
            buffer.append(&mut read_bytes_offset(f, start, start + l)?);
            break;
        }

        else {
            buffer.append(&mut read_bytes(f)?);
        }
    }

    Ok(buffer)
}

fn cosine(t1: &[f32], t2: &[f32]) -> f64 {
    assert_eq!(t1.len(), t2.len());

    let mut inner_product = 0.0f64;
    let mut t1_sqr_sum = 0.0f64;
    let mut t2_sqr_sum = 0.0f64;

    for i in 0..t1.len() {
        inner_product += t1[i] as f64 * t2[i] as f64;
        t1_sqr_sum += t1[i] as f64 * t1[i] as f64;
        t2_sqr_sum += t2[i] as f64 * t2[i] as f64;
    }

    inner_product / t1_sqr_sum.sqrt() / t2_sqr_sum.sqrt()
}
