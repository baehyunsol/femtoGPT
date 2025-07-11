use super::Tokenizer;
use crate::error::Error;

mod char_count;
mod config;
mod tokenizer;

use tokenizer::StateMachine;

pub use char_count::{CharCount, count_chars};
pub use config::BpeConfig;
pub use tokenizer::Tokenizer as BpeTokenizerInner;

pub struct BpeTokenizer {
    inner: BpeTokenizerInner,
    state_machine: StateMachine,
}

impl BpeTokenizer {
    pub fn new() -> Self {
        let inner = BpeTokenizerInner::dummy();
        let state_machine = inner.build_state_machine();

        BpeTokenizer {
            inner,
            state_machine,
        }
    }
}

impl Tokenizer for BpeTokenizer {
    fn name(&self) -> String {
        String::from("bpe")
    }

    fn vocab_size(&self) -> usize {
        self.inner.tokens.len()
    }

    fn tokenize(&self, string: &str) -> Vec<usize> {
        self.inner.encode_with_state_machine(string.as_bytes(), &self.state_machine)
    }

    fn untokenize(&self, tokens: &[usize]) -> String {
        self.inner.decode(tokens)
    }

    fn has_to_load(&self) -> bool {
        true
    }

    fn load_from_json(&mut self, data: &str) -> Result<(), Error> {
        let inner = serde_json::from_str::<BpeTokenizerInner>(data)?;
        let state_machine = inner.build_state_machine();

        self.inner = inner;
        self.state_machine = state_machine;
        Ok(())
    }

    fn dump_json(&self) -> Result<String, Error> {
        Ok(serde_json::to_string(&self.inner)?)
    }
}
