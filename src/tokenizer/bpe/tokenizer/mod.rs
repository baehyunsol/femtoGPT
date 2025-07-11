use super::{BpeConfig, CharCount};
use super::config::Unit;
use crate::error::Error;
use ragit_fs::{
    WriteMode,
    is_dir,
    read_bytes,
    read_dir,
    write_string,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

mod encode;

pub use encode::StateMachine;
pub type TokenId = usize;

#[derive(Deserialize, Serialize)]
pub struct Tokenizer {
    unit: Unit,
    // `unk` is contained in `tokens`
    pub unk: TokenId,
    pub tokens: HashMap<TokenId, Vec<u8>>,
}

impl Tokenizer {
    /// It's unusable. You have to call `.load(value)` to initialize it.
    pub fn dummy() -> Self {
        Tokenizer {
            unit: Unit::Char,
            unk: 0,
            tokens: HashMap::new(),
        }
    }

    /// You have to call this function if you want to use it with femtoGPT.
    pub fn compact(&mut self) {
        let mut unk = None;
        let mut new_tokens = HashMap::new();

        for (new_token_id, (token_id, token)) in self.tokens.iter().enumerate() {
            new_tokens.insert(new_token_id, token.to_vec());

            if *token_id == self.unk {
                unk = Some(new_token_id);
            }
        }

        self.unk = unk.unwrap();
        self.tokens = new_tokens;
    }

    pub fn from_char_count(char_count: &CharCount, config: &BpeConfig) -> Self {
        let mut counts = HashMap::new();
        let unk_bytes = config.unk_token.as_bytes();

        if let Some(byte_count) = &char_count.bytes {
            for (b, c) in byte_count.iter() {
                counts.insert(vec![*b], *c);
            }
        }

        if let Some(char_count) = &char_count.chars {
            for (ch, c) in char_count.iter() {
                counts.insert(ch.to_string().as_bytes().to_vec(), *c);
            }
        }

        if !counts.contains_key(unk_bytes) {
            counts.insert(unk_bytes.to_vec(), 0);
        }

        if let Some(limit) = config.char_dictionary_size {
            if counts.len() > limit {
                let mut sortable = counts.iter().map(|(b, c)| (b.clone(), *c)).collect::<Vec<_>>();
                sortable.sort_by_key(|(_, count)| u64::MAX - *count);
                sortable = sortable[..limit].to_vec();
                let last_b = &sortable[limit - 1].0;
                counts = sortable.iter().map(|(b, c)| (b.clone(), *c)).collect();

                if !counts.contains_key(unk_bytes) {
                    counts.remove(last_b);
                    counts.insert(unk_bytes.to_vec(), 0);
                }
            }
        }

        let mut tokens = HashMap::with_capacity(counts.len());
        let mut unk_id = None;

        for (token_id, bytes) in counts.into_keys().enumerate() {
            let token_id = token_id as TokenId;

            if bytes == unk_bytes {
                unk_id = Some(token_id);
            }

            tokens.insert(token_id, bytes);
        }

        Tokenizer {
            unit: config.unit,
            unk: unk_id.unwrap(),
            tokens,
        }
    }

    pub fn decode(&self, token_ids: &[TokenId]) -> String {
        let mut result = vec![];

        for token_id in token_ids.iter() {
            let mut token = self.tokens.get(token_id).map(
                |t| t.to_vec()
            ).unwrap_or_else(
                || self.tokens.get(&self.unk).unwrap().to_vec()
            );
            result.append(&mut token);
        }

        String::from_utf8_lossy(&result).to_string()
    }

    pub fn decode_id(&self, id: TokenId) -> Option<String> {
        self.tokens.get(&id).map(|v| String::from_utf8_lossy(v).to_string())
    }

    /// If it returns before running `num_epoch`, it returns false
    pub fn train(&mut self, corpus: &[u8], num_epoch: usize) -> bool {
        let mut token_ids = self.encode(corpus);

        for _ in 0..num_epoch {
            let mut pair_count = HashMap::new();

            for pair in token_ids.windows(2) {
                let pair = (pair[0], pair[1]);

                if pair.0 == self.unk || pair.1 == self.unk {
                    continue;
                }

                match pair_count.get_mut(&pair) {
                    Some(n) => { *n += 1; },
                    None => { pair_count.insert(pair, 1); },
                }
            }

            let top = pair_count.iter().max_by_key(|(_, c)| *c).unwrap();

            if *top.1 > 2 {
                let new_token_id = self.create_new_token(top.0.0, top.0.1);
                println!(
                    "New token: ({:?}: {}) + ({:?}: {}) = ({:?}: {}), appears {} times",
                    self.decode_id(top.0.0).unwrap(),
                    top.0.0,
                    self.decode_id(top.0.1).unwrap(),
                    top.0.1,
                    self.decode_id(new_token_id).unwrap(),
                    new_token_id,
                    top.1,
                );

                // replace old tokens in `token_ids` with the new token
                let mut i = 0;
                let mut new_token_ids = Vec::with_capacity(token_ids.len());

                loop {
                    match (token_ids.get(i), token_ids.get(i + 1)) {
                        (Some(t1), Some(t2)) => if *t1 == top.0.0 && *t2 == top.0.1 {
                            new_token_ids.push(new_token_id);
                            i += 2;
                        } else if *t2 == top.0.0 {
                            new_token_ids.push(*t1);
                            i += 1;
                        } else {
                            new_token_ids.push(*t1);
                            new_token_ids.push(*t2);
                            i += 2;
                        },
                        (Some(t1), None) => {
                            new_token_ids.push(*t1);
                            break;
                        },
                        (None, None) => {
                            break;
                        },
                        _ => unreachable!(),
                    }
                }

                token_ids = new_token_ids;
            }

            else {
                println!("Cannot create anymore token. Please try with another batch.");
                return false;
            }
        }

        true
    }

    pub fn trim_tail(
        &mut self,
        dataset: &str,
        config: &BpeConfig,
        dump_result_to: Option<String>,
    ) -> Result<(), Error> {
        if self.tokens.len() > config.dictionary_size {
            let state_machine = self.build_state_machine();
            let mut counts = self.tokens.keys().map(|token_id| (*token_id, 0)).collect::<HashMap<TokenId, u64>>();
            let files = if is_dir(dataset) {
                read_dir(dataset, false)?
            } else {
                vec![dataset.to_string()]
            };

            for file in files.iter() {
                let token_ids = self.encode_with_state_machine(&read_bytes(file)?, &state_machine);

                for token_id in token_ids.iter() {
                    match counts.get_mut(token_id) {
                        Some(n) => { *n += 1; },
                        None => { counts.insert(*token_id, 1); },
                    }
                }
            }

            let mut counts = counts.into_iter().collect::<Vec<_>>();
            counts.sort_by_key(|(_, c)| u64::MAX - *c);

            if let Some(path) = dump_result_to {
                let result = counts.iter().map(
                    |(token_id, count)| {
                        let v = self.tokens.get(token_id).unwrap();

                        TrimTailResult {
                            id: *token_id,
                            bytes: v.to_vec(),
                            string: String::from_utf8_lossy(v).to_string(),
                            appearance: *count,
                        }
                    }
                ).collect::<Vec<_>>();

                write_string(
                    &path,
                    &serde_json::to_string_pretty(&result)?,
                    WriteMode::CreateOrTruncate,
                )?;
            }

            // We should not remove `self.unk` from `self.tokens`. But at this point, we're not sure whether `self.unk` is included in `counts[dict_size..]` or `counts[..dict_size]`.
            let extra = counts[config.dictionary_size - 1];
            let mut removed_tokens = Vec::with_capacity(counts.len() - config.dictionary_size);

            for (token, _) in counts[config.dictionary_size..].iter() {
                if *token == self.unk {
                    removed_tokens.push((extra.0, self.decode_id(extra.0).unwrap()));
                    self.tokens.remove(&extra.0);
                }

                else {
                    removed_tokens.push((*token, self.decode_id(*token).unwrap()));
                    self.tokens.remove(token);
                }
            }

            println!(
                "Removed {} tokens: {:?}",
                removed_tokens.len(),
                removed_tokens,
            );
        }

        Ok(())
    }

    fn create_new_token(&mut self, id1: TokenId, id2: TokenId) -> TokenId {
        let mut curr_token_ids: Vec<_> = self.tokens.keys().collect();
        curr_token_ids.sort();
        let mut new_token_id = None;

        for i in 0..(curr_token_ids.len() - 1) {
            if curr_token_ids[i] + 1 != *curr_token_ids[i + 1] {
                new_token_id = Some(curr_token_ids[i] + 1);
            }
        }

        let new_token_id = new_token_id.unwrap_or(curr_token_ids[curr_token_ids.len() - 1] + 1);
        self.tokens.insert(
            new_token_id,
            vec![
                self.tokens.get(&id1).unwrap().to_vec(),
                self.tokens.get(&id2).unwrap().to_vec(),
            ].concat(),
        );
        new_token_id
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }
}

#[derive(Deserialize, Serialize)]
pub struct TrimTailResult {
    id: TokenId,
    bytes: Vec<u8>,
    string: String,
    appearance: u64,
}
