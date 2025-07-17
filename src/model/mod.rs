use crate::error::Error;
use crate::gpt::TrainingState;
use crate::tokenizer::TokenizerInner;
use serde::{Deserialize, Serialize};
use std::str::FromStr;

mod info;
mod log;

pub use info::{ModelInfo, TokenInfo, f2s, s2f};
pub use log::{Log, LogKind};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Model {
    pub tokenizer: TokenizerInner,
    pub pos_enc: PosEnc,
    pub hyperparameters: Hyperparameters,
    pub training_state: TrainingState,
    pub logs: Vec<Log>,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, Serialize, PartialEq)]
pub enum PosEnc {
    None,

    // sinusoidal positional encoding, proposed by "Attention Is All You Need"
    Absolute,

    // TODO
    // It concats sinusoidal positional encoding to an embedding vector.
    // Now that the vector has `embedding_degree * 2` dimension, it mat-muls the
    // vector with `[embedding_degree * 2, embedding_degree]` matrix to make it
    // compatible with the other layers.
    // AbsoluteCat,

    // TODO
    // Rotary,
}

impl FromStr for PosEnc {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Error> {
        match s {
            "none" => Ok(Self::None),
            "absolute" => Ok(Self::Absolute),
            // "absolute-cat" => Ok(Self::AbsoluteCat),
            // "rotary" => Ok(Self::Rotary),
            _ => todo!(),
        }
    }
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
