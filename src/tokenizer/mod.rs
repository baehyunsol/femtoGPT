mod char_count;
mod config;
mod encode;
mod inner;

use encode::StateMachine;

pub use char_count::{CharCount, count_chars};
pub use config::BpeConfig;
pub use inner::TokenizerInner;

pub type TokenId = usize;

pub struct Tokenizer {
    pub inner: TokenizerInner,
    state_machine: StateMachine,
}

impl Tokenizer {
    pub fn ascii() -> Self {
        Self::from_inner(TokenizerInner::ascii())
    }

    pub fn vocab_size(&self) -> usize {
        self.inner.tokens.len()
    }

    pub fn tokenize(&self, string: &str) -> Vec<usize> {
        self.inner.encode_with_state_machine(string.as_bytes(), &self.state_machine)
    }

    pub fn untokenize(&self, tokens: &[usize]) -> String {
        self.inner.decode(tokens)
    }

    pub fn from_inner(inner: TokenizerInner) -> Self {
        let state_machine = inner.build_state_machine();

        Tokenizer {
            inner,
            state_machine,
        }
    }
}
