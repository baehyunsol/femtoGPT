use super::Tokenizer;

pub struct ByteTokenizer;

impl Tokenizer for ByteTokenizer {
    fn name(&self) -> String {
        String::from("byte")
    }
    fn vocab_size(&self) -> usize {
        256
    }
    fn tokenize(&self, string: &str) -> Vec<usize> {
        string
            .as_bytes()
            .iter()
            .map(|b| *b as usize)
            .collect()
    }
    fn untokenize(&self, tokens: &[usize]) -> String {
        String::from_utf8_lossy(
            &tokens
                .iter()
                .map(|b| *b as u8)
                .collect::<Vec<u8>>())
        .to_string()
    }
}
