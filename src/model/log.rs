use super::Hyperparameters;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Log {
    pub kind: LogKind,

    // I fork models often, and I want to track history of each model.
    // So I give a random u128 number to each log, in order to uniquely
    // identify each log.
    // I'm using `String` instead of `u128` because I'm worried `serde`
    // might not be able to handle `u128` properly.
    pub id: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum LogKind {
    Init {
        hyperparameters: Hyperparameters,
    },
    TrainSession {
        dropout: f32,
        file: String,
        file_hash: String,
    },
    TrainStep {
        avg_loss: f32,
        elapsed: u64,  // millis
        is_gpu: bool,
    },
    InsertLayer {
        index: usize,
    },
    ResetOptimizer,
}

impl Log {
    pub fn init(hyperparameters: Hyperparameters) -> Self {
        Log {
            kind: LogKind::Init { hyperparameters },
            id: format!("{:032x}", rand::random::<u128>()),
        }
    }

    pub fn train_session(dropout: f32, file: &str, file_content: &str) -> Self {
        Log {
            kind: LogKind::TrainSession {
                dropout,
                file: file.to_string(),
                file_hash: format!("{:020x}", hash(file_content.as_bytes())),
            },
            id: format!("{:032x}", rand::random::<u128>()),
        }
    }

    pub fn train_step(avg_loss: f32, elapsed: u64, is_gpu: bool) -> Self {
        Log {
            kind: LogKind::TrainStep { avg_loss, elapsed, is_gpu },
            id: format!("{:032x}", rand::random::<u128>()),
        }
    }

    pub fn insert_layer(index: usize) -> Self {
        Log {
            kind: LogKind::InsertLayer { index },
            id: format!("{:032x}", rand::random::<u128>()),
        }
    }

    pub fn reset_optimizer() -> Self {
        Log {
            kind: LogKind::ResetOptimizer,
            id: format!("{:032x}", rand::random::<u128>()),
        }
    }
}

// I wrote it for fun. Who cares.
fn hash(s: &[u8]) -> u128 {
    let mut r = 0xffff_ffff_ffff_ffff_ffff;

    for (i, b) in s.iter().enumerate() {
        let mut k = *b as u128;
        k |= ((i as u128) & 0xffff) << 16;
        k |= ((r >> 32) & 0xffff) << 32;
        k = 2 * k * k + k + 1;
        r += k;
    }

    r & 0xffff_ffff_ffff_ffff_ffff
}
