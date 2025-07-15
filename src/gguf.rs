use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct GGUF {
    kv: HashMap<String, Value>,
    tensors: Vec<GGUFTensor>,
}

// NOTE: https://medium.com/@hsinhungw/gpt-2-detailed-model-architecture-6b1aad33d16b
//       It seems like femtoGPT and gpt-2 are not compatible...
// I'm working with https://huggingface.co/igorbkz/gpt2-Q8_0-GGUF
pub fn import_gguf(gguf: &[u8]) -> GGUF {
    if &gguf[..4] != b"GGUF" {
        panic!();
    }

    if gguf[4] != 3 {
        panic!("Unsupported gguf version: {}", gguf[4]);
    }

    let num_tensors = u64::from_le_bytes(gguf[8..16].to_vec().try_into().unwrap()) as usize;
    let num_meta = u64::from_le_bytes(gguf[16..24].to_vec().try_into().unwrap());

    let mut cursor = 24;
    let mut kv = HashMap::new();

    for _ in 0..num_meta {
        let key_len = u64::from_le_bytes(gguf[cursor..(cursor + 8)].to_vec().try_into().unwrap());
        cursor += 8;

        let key = String::from_utf8(gguf[cursor..(cursor + key_len as usize)].to_vec()).unwrap();
        cursor += key_len as usize;

        let value_type = u32::from_le_bytes(gguf[cursor..(cursor + 4)].to_vec().try_into().unwrap());
        cursor += 4;

        let value = match value_type {
            // u32
            4 => {
                let value = u32::from_le_bytes(gguf[cursor..(cursor + 4)].to_vec().try_into().unwrap());
                cursor += 4;
                serde_json::to_value(value).unwrap()
            },
            // u32
            5 => {
                let value = i32::from_le_bytes(gguf[cursor..(cursor + 4)].to_vec().try_into().unwrap());
                cursor += 4;
                serde_json::to_value(value).unwrap()
            },
            // f32
            6 => {
                let value = u32::from_le_bytes(gguf[cursor..(cursor + 4)].to_vec().try_into().unwrap());
                let value = f32::from_bits(value);
                cursor += 4;
                serde_json::to_value(value).unwrap()
            },
            // String
            8 => {
                let value_len = u64::from_le_bytes(gguf[cursor..(cursor + 8)].to_vec().try_into().unwrap()) as usize;
                cursor += 8;

                let value = String::from_utf8(gguf[cursor..(cursor + value_len)].to_vec()).unwrap();
                cursor += value_len;
                serde_json::to_value(value).unwrap()
            },
            // array
            9 => {
                let value_type = u32::from_le_bytes(gguf[cursor..(cursor + 4)].to_vec().try_into().unwrap());
                cursor += 4;

                match value_type {
                    // i32
                    5 => {
                        let array_len = u64::from_le_bytes(gguf[cursor..(cursor + 8)].to_vec().try_into().unwrap()) as usize;
                        cursor += 8;
                        let mut result = Vec::with_capacity(array_len);

                        for _ in 0..array_len {
                            let element = i32::from_le_bytes(gguf[cursor..(cursor + 4)].to_vec().try_into().unwrap());
                            cursor += 4;
                            result.push(element);
                        }

                        serde_json::to_value(result).unwrap()
                    },
                    // String
                    8 => {
                        let array_len = u64::from_le_bytes(gguf[cursor..(cursor + 8)].to_vec().try_into().unwrap()) as usize;
                        cursor += 8;
                        let mut result = Vec::with_capacity(array_len);

                        for _ in 0..array_len {
                            let element_len = u64::from_le_bytes(gguf[cursor..(cursor + 8)].to_vec().try_into().unwrap()) as usize;
                            cursor += 8;

                            let element = String::from_utf8(gguf[cursor..(cursor + element_len)].to_vec()).unwrap();
                            cursor += element_len;
                            result.push(element);
                        }

                        serde_json::to_value(result).unwrap()
                    },
                    _ => panic!("TODO: array with value_type {value_type}"),
                }
            },
            _ => panic!("TODO: value_type {value_type}"),
        };

        kv.insert(key, value);
    }

    let mut tensors = Vec::with_capacity(num_tensors);

    for _ in 0..num_tensors {
        let name_len = u64::from_le_bytes(gguf[cursor..(cursor + 8)].to_vec().try_into().unwrap()) as usize;
        cursor += 8;
        let name = String::from_utf8(gguf[cursor..(cursor + name_len)].to_vec()).unwrap();
        cursor += name_len;

        let n_dimensions = u32::from_le_bytes(gguf[cursor..(cursor + 4)].to_vec().try_into().unwrap()) as usize;
        cursor += 4;

        let mut dimensions = Vec::with_capacity(n_dimensions);

        for _ in 0..n_dimensions {
            dimensions.push(u64::from_le_bytes(gguf[cursor..(cursor + 8)].to_vec().try_into().unwrap()) as usize);
            cursor += 8;
        }

        let tensor_type = u32::from_le_bytes(gguf[cursor..(cursor + 4)].to_vec().try_into().unwrap());
        let tensor_type = TensorType::from(tensor_type);
        cursor += 4;

        let offset = u64::from_le_bytes(gguf[cursor..(cursor + 8)].to_vec().try_into().unwrap()) as usize;
        cursor += 8;

        tensors.push(GGUFTensor {
            name,
            dimensions,
            tensor_type,
            offset,
            tensor: vec![],
        });
    }

    for tensor in tensors.iter_mut() {
        let tensor_len = tensor.dimensions.iter().product::<usize>();
        let tensor_len_bytes = tensor_len * tensor.tensor_type.bit_width() / 8 + 1;
        tensor.tensor = tensor.tensor_type.dequantize(tensor_len, &gguf[tensor.offset..(tensor.offset + tensor_len_bytes)]);
    }

    GGUF {
        kv,
        tensors,
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
struct GGUFTensor {
    name: String,
    dimensions: Vec<usize>,
    tensor_type: TensorType,
    offset: usize,
    tensor: Vec<f32>,
}

//  0: f32
//  1: f16
//  2: q4_0
//  3: q4_1
//  4: q4_2
//  5: q4_3
//  6: q5_0
//  7: q5_1
//  8: q8_0
//  9: q8_1
// 10: q2_k
#[derive(Clone, Debug, Deserialize, Serialize)]
enum TensorType {
    F32,
    F16,
    Q8_0,
}

impl TensorType {
    pub fn bit_width(&self) -> usize {
        match self {
            TensorType::F32 => 32,
            TensorType::F16 => 16,
            TensorType::Q8_0 => 8,
        }
    }

    pub fn dequantize(&self, mut len: usize, bytes: &[u8]) -> Vec<f32> {
        let mut result = Vec::with_capacity(len);
        let mut cursor = 0;

        match self {
            TensorType::F32 => {
                for _ in 0..len {
                    let n = u32::from_le_bytes(bytes[cursor..(cursor + 4)].to_vec().try_into().unwrap());
                    let n = f32::from_bits(n);
                    cursor += 4;
                    result.push(n);
                }
            },
            TensorType::F16 => {
                for _ in 0..len {
                    let n = u16::from_le_bytes(bytes[cursor..(cursor + 2)].to_vec().try_into().unwrap());
                    let n = f16_from_bits(n);
                    cursor += 2;
                    result.push(n);
                }
            },
            // https://github.com/ggml-org/llama.cpp/blob/bdca38376f7e8dd928defe01ce6a16218a64b040/ggml/src/ggml-quants.c#L187
            TensorType::Q8_0 => {
                let d = u16::from_le_bytes(bytes[cursor..(cursor + 2)].to_vec().try_into().unwrap());
                let d = f16_from_bits(d);
                cursor += 2;

                for _ in 0..32 {
                    let x = i8::from_le_bytes([bytes[cursor]]);
                    let x = x as f32 * d;
                    cursor += 1;
                    result.push(x);
                    len -= 1;

                    if len == 0 {
                        return result;
                    }
                }
            },
            _ => panic!("TODO: {self:?}::dequantize()"),
        }

        result
    }
}

impl From<u32> for TensorType {
    fn from(n: u32) -> TensorType {
        match n {
            0 => TensorType::F32,
            1 => TensorType::F16,
            8 => TensorType::Q8_0,
            _ => panic!("TODO: tensor type {n}"),
        }
    }
}

// https://docs.rs/decompose-float/latest/decompose_float/index.html
// I don't want to use nightly rust for this.
fn f16_from_bits(n: u16) -> f32 {
    let is_neg = n >= (1 << 15);
    let mut exp = ((n >> 10) & 31) as i32 - 15;
    let mut mantissa = n & 1023;

    if exp == -15 {
        exp += 1;

        if mantissa == 0 {
            if is_neg {
                return -0.0;
            }

            else {
                return 0.0;
            }
        }

        let to_shift = 10 - mantissa.ilog2();
        mantissa <<= to_shift;
        exp -= to_shift as i32;
        mantissa -= 1 << 10;
    }

    else if exp == 16 {
        if mantissa == 0 {
            if is_neg {
                return f32::NEG_INFINITY;
            }

            else {
                return f32::INFINITY;
            }
        }

        else {
            return f32::NAN;
        }
    }

    mantissa += 1024;
    (if is_neg { -1.0 } else { 1.0 }) * 2.0f32.powi(exp) * mantissa as f32 / 1024.0
}
