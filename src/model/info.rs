use super::{Hyperparameters, Log};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelInfo {
    pub hyperparameters: Hyperparameters,
    pub num_params: usize,
    pub logs: Vec<Log>,
    pub tokens: Vec<TokenInfo>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TokenInfo {
    pub index: usize,
    pub string: String,
    pub heads: Vec<String>,
}

// Yeah, I know it's inefficient, but it works!
// I'll optimize it later.
pub fn f2s(f: f32) -> String {
    let sign = f >= 0.0;
    let mut f = f.abs();
    let mut exp: i32 = 0;

    while f < 1.0 {
        f *= 2.0;
        exp -= 1;

        if exp < -15 {
            return String::from("  ");
        }
    }

    while f > 2.0 {
        f /= 2.0;
        exp += 1;
    }

    let mantissa = ((f - 1.0) * 135.0).round() as i32;
    let mut n = (exp + 15) * 135 + mantissa;

    if sign {
        n += 4050;
    }

    let b1 = (n / 90) as u8;
    let b2 = (n % 90) as u8;
    return String::from_utf8(vec![b1 + 32, b2 + 32]).unwrap();
}

pub fn s2f(s: &[u8]) -> f32 {
    // 90 * 90 = 8100 = 2 * 30 * 135
    let mut n = (s[0] as u32 - 32) * 90 + s[1] as u32 - 32;
    let mut sign = false;

    if n >= 4050 {
        sign = true;
        n -= 4050;
    }

    let exp = (n / 135) as i32 - 15;
    let mantissa = n % 135;

    (if sign { 1.0 } else { -1.0 }) * (2.0f32).powi(exp) * (135 + mantissa) as f32 / 135.0
}
