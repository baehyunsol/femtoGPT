mod count;
mod config;
mod encode;
mod inner;
mod reserve;

use encode::StateMachine;

pub use count::{CharCount, count_chars, count_tokens};
pub use config::{BpeConfig, Unit};
pub use inner::TokenizerInner;
pub use reserve::{
    generate_reserved_token,
    is_reserved_token,
};

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

    pub fn from_tokens(tokens: Vec<String>) -> Self {
        let inner = TokenizerInner::from_tokens(tokens);
        Self::from_inner(inner)
    }

    pub fn reserve_tokens(&mut self, count: usize) {
        self.inner.reserve_tokens(count);
        self.state_machine = self.inner.build_state_machine();
    }
}
