// I have created another file because I don't want to edit `src/gpt.rs`.
// I'm not gonna edit original files except `lib.rs`, `main.rs` and `tokenizer/*`
// so that it's easier to apply diffs from another forks.

use crate::gpt::TrainingState;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Model {
    // for now, the only available one is "byte"
    pub tokenizer: String,
    pub hyper_parameters: HyperParameters,
    pub training_state: TrainingState,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct HyperParameters {
    // It's like a max-context-size
    pub num_tokens: usize,
    pub embedding_degree: usize,
    pub num_layers: usize,
    pub num_heads: usize,
}
