mod bytes;
pub use bytes::*;

mod simple;
pub use simple::*;

mod sentencepiece;
pub use sentencepiece::*;

pub trait TokenizerImpl {
    fn vocab_size(&self) -> usize;
    fn tokenize(&self, string: &str) -> Vec<usize>;
    fn untokenize(&self, tokens: &[usize]) -> String;
}

pub type Tokenizer = SimpleTokenizer;
