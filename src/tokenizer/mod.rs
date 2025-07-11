mod byte;
pub use byte::*;

mod simple;
pub use simple::*;

mod sentencepiece;
pub use sentencepiece::*;

pub trait Tokenizer {
    fn name(&self) -> String;
    fn vocab_size(&self) -> usize;
    fn tokenize(&self, string: &str) -> Vec<usize>;
    fn untokenize(&self, tokens: &[usize]) -> String;
}
