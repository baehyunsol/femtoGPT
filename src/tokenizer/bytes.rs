use super::TokenizerImpl;

pub struct ByteTokenizer {}

impl ByteTokenizer {
    pub fn new(_dataset: &str) -> Self {
        Self {}
    }
}

impl TokenizerImpl for ByteTokenizer {
    fn vocab_size(&self) -> usize {
        256
    }
    fn tokenize(&self, string: &str) -> Vec<usize> {
        string.bytes().map(|b| b as usize).collect()
    }
    fn untokenize(&self, tokens: &[usize]) -> String {
        String::from_utf8_lossy(&tokens.iter().map(|b| *b as u8).collect::<Vec<u8>>()).to_string()
    }
}
