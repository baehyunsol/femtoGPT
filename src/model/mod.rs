use crate::gpt::TrainingState;
use serde::{Deserialize, Serialize};

mod info;
mod log;

pub use info::{ModelInfo, TokenInfo, f2s, s2f};
pub use log::{Log, LogKind};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Model {
    // "byte" | "bpe"
    pub tokenizer: String,

    // some tokenizers require an extra data
    // I want it to be `serde_json::Value`,
    // but it seems like `bincode` does not
    // support the type.
    pub tokenizer_data: String,
    pub hyperparameters: Hyperparameters,
    pub training_state: TrainingState,
    pub logs: Vec<Log>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Serialize, PartialEq)]
pub struct Hyperparameters {
    // It's like a max-context-size
    pub num_tokens: usize,
    pub vocab_size: usize,
    pub embedding_degree: usize,
    pub num_layers: usize,

    // head_size * num_heads == embedding_degree
    pub num_heads: usize,
    pub head_size: usize,
}
