use super::config::{BpeConfig, Unit};
use crate::error::Error;
use ragit_fs::{
    is_dir,
    read_bytes,
    read_dir,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Deserialize, Serialize)]
pub struct CharCount {
    pub chars: Option<HashMap<char, u64>>,
    pub bytes: Option<HashMap<u8, u64>>,
}

pub fn count_chars(
    dataset: &str,  // dir or file
    config: &BpeConfig,
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

        if config.unit == Unit::Char {
            let s = String::from_utf8_lossy(&content);

            for c in s.chars() {
                match chars.get_mut(&c) {
                    Some(n) => { *n += 1; },
                    None => { chars.insert(c, 1); },
                }
            }
        }

        else {
            for b in content.iter() {
                match bytes.get_mut(b) {
                    Some(n) => { *n += 1; },
                    None => { bytes.insert(*b, 1); },
                }
            }
        }
    }

    Ok(CharCount {
        chars: if config.unit == Unit::Char { Some(chars) } else { None },
        bytes: if config.unit == Unit::Char { None } else { Some(bytes) },
    })
}
