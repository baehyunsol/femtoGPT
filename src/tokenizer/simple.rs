use super::Tokenizer;

// byte tokenizer
pub struct SimpleTokenizer {}

impl SimpleTokenizer {
    pub fn new(_dataset: &str) -> Self {
        Self {}
    }
}

impl Tokenizer for SimpleTokenizer {
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
