use crate::error::Error;
use serde_json::Value;

mod bpe;
pub use bpe::*;

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
    fn has_to_load(&self) -> bool {
        false
    }
    fn load_from_json(&mut self, _data: &str) -> Result<(), Error> {
        Ok(())
    }
    fn dump_json(&self) -> Result<String, Error> {
        Ok(String::new())
    }
}
