use super::{
    TokenId,
    TokenizerInner,
    Unit,
};
use crate::error::Error;
use ragit_fs::{
    is_dir,
    read_bytes,
    read_dir,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::collections::hash_map::Entry;

#[derive(Deserialize, Serialize)]
pub struct CharCount {
    pub chars: Option<HashMap<char, u64>>,
    pub bytes: Option<HashMap<u8, u64>>,
}

pub fn count_chars(
    dataset: &str,  // dir or file
    unit: Unit,
) -> Result<CharCount, Error> {
    let mut chars = HashMap::new();
    let mut bytes = HashMap::new();
    let files = if is_dir(dataset) {
        read_dir(dataset, false)?
    } else {
        vec![dataset.to_string()]
    };

    for file in files {
        println!("counting `{file}`...");
        let content = read_bytes(&file)?;

        if unit == Unit::Char {
            let s = String::from_utf8_lossy(&content);

            for c in s.chars() {
                match chars.entry(c) {
                    Entry::Occupied(mut e) => { e.insert(*e.get() + 1); },
                    Entry::Vacant(e) => { e.insert(1); },
                }
            }
        }

        else {
            for b in content.iter() {
                match bytes.entry(*b) {
                    Entry::Occupied(mut e) => { e.insert(*e.get() + 1); },
                    Entry::Vacant(e) => { e.insert(1); },
                }
            }
        }
    }

    Ok(CharCount {
        chars: if unit == Unit::Char { Some(chars) } else { None },
        bytes: if unit == Unit::Char { None } else { Some(bytes) },
    })
}

pub fn count_tokens(
    tokenizer: &TokenizerInner,
    dataset: &str,
) -> Result<HashMap<TokenId, u64>, Error> {
    // The result must contain all the tokens!
    let mut counts = tokenizer.tokens.keys().map(
        |key| (*key, 0)
    ).collect::<HashMap<_, _>>();
    let state_machine = tokenizer.build_state_machine();

    let files = if is_dir(dataset) {
        read_dir(dataset, false)?
    } else {
        vec![dataset.to_string()]
    };

    for file in files.iter() {
        let token_ids = tokenizer.encode_with_state_machine(&read_bytes(file)?, &state_machine);

        for token_id in token_ids.iter() {
            match counts.entry(*token_id) {
                Entry::Occupied(mut e) => { e.insert(*e.get() + 1); },
                Entry::Vacant(e) => { e.insert(1); },
            }
        }
    }

    Ok(counts)
}
