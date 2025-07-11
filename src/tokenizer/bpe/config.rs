use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub enum Unit {
    Char,
    Byte,
}

#[derive(Deserialize, Serialize)]
pub struct BpeConfig {
    pub unit: Unit,
    pub unk_token: String,

    pub vocab_size: usize,

    // It affects the initial dictionary build from `CharCount`.
    // It'd replace uncommon characters with unk_token.
    // It affects both `Unit::Char` and `Unit::Byte`.
    pub char_vocab_size: Option<usize>,
}

impl Default for BpeConfig {
    fn default() -> Self {
        BpeConfig {
            unit: Unit::Char,
            unk_token: String::from("<unk>"),
            vocab_size: 768,
            char_vocab_size: Some(512),
        }
    }
}
