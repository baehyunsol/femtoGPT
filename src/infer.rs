/// It implements GPT inference.
/// I'm creating a new function instead of modifying the one in `gpt.rs` because
///
/// 1. I want to do a lot of experiments with this, so I have to modify a lot.
///    But I don't want to rewrite `gpt.rs` because I respect its original author
///    and I want to keep his code as a reference implementation.
/// 2. I want to study how GPT works. The best way to study it is to implement one
///    from scratch.

use crate::model::{Model, PosEnc};
use crate::tensor::TensorOps;
use std::collections::HashMap;
use std::collections::hash_map::Entry;

pub type TokenId = usize;

pub struct Cache {
    // HashMap<(layer, token_seq), output>
    pub(crate) layer_output: HashMap<(usize, usize), Vec<f32>>,

    // HashMap<(layer, token_seq, head), (k, v)>
    pub(crate) kv: HashMap<(usize, usize, usize), (Vec<f32>, Vec<f32>)>,
}

impl Cache {
    pub fn new() -> Self {
        Cache {
            layer_output: HashMap::new(),
            kv: HashMap::new(),
        }
    }
}

pub struct Inference {
    // sorted by prob
    pub logit: Vec<(TokenId, f32)>,
    pub cache: Cache,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
enum KQV {
    K, Q, V,
}

impl Model {
    pub fn infer_step(
        &self,
        prompt: &[TokenId],
        mut cache: Cache,
    ) -> Inference {
        let tensors = &self.training_state.tensors.iter().map(|(k, v)| (k.to_string(), v.blob().to_vec())).collect::<HashMap<_, _>>();
        let embedding_degree = self.hyperparameters.embedding_degree;
        let num_layers = self.hyperparameters.num_layers;
        let num_heads = self.hyperparameters.num_heads;
        let head_size = self.hyperparameters.head_size;
        let vocab_size = self.hyperparameters.vocab_size;
        let head_size_sqrt_inv = 1.0 / (head_size as f32).sqrt();

        let token_embeddings = tensors.get("token_embedding").unwrap();
        let mut curr_input = prompt.iter().map(
            |token_id| token_embeddings[(*token_id * embedding_degree)..(*token_id * embedding_degree + embedding_degree)].to_vec()
        ).collect::<Vec<Vec<f32>>>();

        match self.pos_enc {
            PosEnc::Absolute => {
                for row in 0..curr_input.len() {
                    for col in 0..embedding_degree {
                        let k = row as f32;
                        let i = (col / 2) as f32;
                        let factor = 10000f32.powf(2f32 * i / embedding_degree as f32);

                        let pos = if col % 2 == 0 {
                            (k / factor).sin()
                        } else {
                            (k / factor).cos()
                        };
                        curr_input[row][col] += pos;
                    }
                }
            },
            PosEnc::None => {},
        }

        for layer in 0..num_layers {
            let norm_coeff = tensors.get(&format!("norm_{layer}_coeff")).unwrap();
            let norm_bias = tensors.get(&format!("norm_{layer}_bias")).unwrap();
            let proj_weights = tensors.get(&format!("proj_{layer}_weights")).unwrap();
            let proj_bias = tensors.get(&format!("proj_{layer}_bias")).unwrap();
            let atten_norm_coeff = tensors.get(&format!("atten_norm_{layer}_coeff")).unwrap();
            let atten_norm_bias = tensors.get(&format!("atten_norm_{layer}_bias")).unwrap();
            let feedforward1_weights = tensors.get(&format!("feedforward1_{layer}_weights")).unwrap();
            let feedforward1_bias = tensors.get(&format!("feedforward1_{layer}_bias")).unwrap();
            let feedforward2_weights = tensors.get(&format!("feedforward2_{layer}_weights")).unwrap();
            let feedforward2_bias = tensors.get(&format!("feedforward2_{layer}_bias")).unwrap();
            let mut kqv = HashMap::new();
            let mut normalized_inputs = vec![];

            // step 1. It iterates the tokens and calculates k, q and v matrices (or load them from cache).
            for (token_seq, input) in curr_input.iter().enumerate() {
                let normalized_input = layer_normalization(
                    input,
                    &norm_coeff,
                    &norm_bias,
                );

                for head in 0..num_heads {
                    let (k, q, v) = match cache.kv.entry((layer, token_seq, head)) {
                        Entry::Occupied(e) => {
                            let (k, v) = e.get().clone();

                            // we don't need query because we already have the layer output
                            (k, vec![], v)
                        },
                        Entry::Vacant(e) => {
                            let k_params = tensors.get(&format!("head_{layer}_{head}_k")).unwrap();
                            let k = mat_mul(&normalized_input, k_params, embedding_degree, head_size);

                            let q_params = tensors.get(&format!("head_{layer}_{head}_q")).unwrap();
                            let q = mat_mul(&normalized_input, q_params, embedding_degree, head_size);

                            let v_params = tensors.get(&format!("head_{layer}_{head}_v")).unwrap();
                            let v = mat_mul(&normalized_input, v_params, embedding_degree, head_size);

                            e.insert((k.clone(), v.clone()));
                            (k, q, v)
                        },
                    };

                    kqv.insert((token_seq, head, KQV::K), k);
                    kqv.insert((token_seq, head, KQV::Q), q);
                    kqv.insert((token_seq, head, KQV::V), v);
                }

                normalized_inputs.push(normalized_input);
            }

            // step 2. Now that it has all the k, q and v matrices, it calculates attention vectors.
            for token_seq in 0..curr_input.len() {
                let layer_output = match cache.layer_output.entry((layer, token_seq)) {
                    Entry::Occupied(e) => e.get().clone(),
                    Entry::Vacant(e) => {
                        let mut attens = vec![];

                        for head in 0..num_heads {
                            let curr_q = kqv.get(&(token_seq, head, KQV::Q)).unwrap();
                            let mut softmax_kq = Vec::with_capacity(token_seq + 1);

                            // It only looks at previous tokens (and itself) and calculates dot product of k * q.
                            // In this code, every tensors are flattend into an 1-D vector. There's no need to transpose.
                            for prev_token_seq in 0..(token_seq + 1) {
                                let prev_k = kqv.get(&(prev_token_seq, head, KQV::K)).unwrap();
                                let mut kq = curr_q.iter().zip(prev_k.iter()).map(|(q, k)| q * k).sum::<f32>();
                                kq *= head_size_sqrt_inv;
                                softmax_kq.push(kq);
                            }

                            let softmax_kq = softmax(&softmax_kq);

                            // weighted sum
                            let atten = softmax_kq.iter().enumerate().map(
                                |(i, s)| mat_coeff(kqv.get(&(i, head, KQV::V)).unwrap(), *s)
                            ).fold(vec![0.0; head_size], |a, b| mat_add(&a, &b));
                            attens.push(atten);
                        }

                        let atten_cat = mat_cat(&attens, head_size);
                        let atten_cat_proj = mat_mul(&atten_cat, proj_weights, embedding_degree, embedding_degree);
                        let atten_cat_proj = mat_add(&atten_cat_proj, proj_bias);
                        let residual = mat_add(&normalized_inputs[token_seq], &atten_cat_proj);

                        let normalized_residual = layer_normalization(
                            &residual,
                            &atten_norm_coeff,
                            &atten_norm_bias,
                        );

                        let feedforward1 = mat_mul(
                            &normalized_residual,
                            &feedforward1_weights,
                            embedding_degree,
                            4 * embedding_degree,
                        );
                        let feedforward1 = mat_add(&feedforward1, feedforward1_bias);
                        let feedforward1 = mat_gelu(&feedforward1);

                        let feedforward2 = mat_mul(
                            &feedforward1,
                            &feedforward2_weights,
                            4 * embedding_degree,
                            embedding_degree,
                        );
                        let feedforward2 = mat_add(&feedforward2, feedforward2_bias);
                        let layer_output = mat_add(&feedforward2, &normalized_residual);
                        e.insert(layer_output.clone());
                        layer_output
                    },
                };

                curr_input[token_seq] = layer_output;
            }
        }

        let head_norm_coeff = tensors.get("head_norm_coeff").unwrap();
        let head_norm_bias = tensors.get("head_norm_bias").unwrap();
        let norm_out = layer_normalization(
            curr_input.last().unwrap(),
            &head_norm_coeff,
            &head_norm_bias,
        );

        let head_map_weights = tensors.get("head_map_weights").unwrap();
        let head_map_bias = tensors.get("head_map_bias").unwrap();
        let one_hot_result = mat_mul(
            &norm_out,
            head_map_weights,
            embedding_degree,
            vocab_size,
        );
        let one_hot_result = mat_add(
            &one_hot_result,
            head_map_bias,
        );
        let one_hot_result = softmax(&one_hot_result);

        let mut one_hot_result_with_token_id = one_hot_result.iter().enumerate().map(
            |(i, p)| (i, *p)
        ).collect::<Vec<_>>();
        // rev sort
        one_hot_result_with_token_id.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());

        Inference {
            logit: one_hot_result_with_token_id,
            cache,
        }
    }
}

const EPSILON: f32 = 1e-5;

fn layer_normalization(
    layer: &[f32],
    coeff: &[f32],
    bias: &[f32],
) -> Vec<f32> {
    assert_eq!(layer.len(), coeff.len());
    assert_eq!(layer.len(), bias.len());

    let avg = layer.iter().sum::<f32>() / layer.len() as f32;
    let var = (layer.iter().map(|f| (f - avg).powi(2)).sum::<f32>() / layer.len() as f32 + EPSILON).sqrt();
    let var_inv = 1.0 / var;

    layer.iter().enumerate().map(
        |(i, n)| ((n - avg) * var_inv) * coeff[i] + bias[i]
    ).collect()
}

fn mat_add(m1: &[f32], m2: &[f32]) -> Vec<f32> {
    assert_eq!(m1.len(), m2.len());

    m1.iter().zip(m2.iter()).map(|(a, b)| *a + *b).collect()
}

fn mat_coeff(matrix: &[f32], coeff: f32) -> Vec<f32> {
    matrix.iter().map(|m| *m * coeff).collect()
}

// In this code, all the matrix multiplication has this shape: `[1, N] * [N, M] = [1, M]`
fn mat_mul(
    vector: &[f32],
    matrix: &[f32],
    input_length: usize,
    output_length: usize,
) -> Vec<f32> {
    assert_eq!(vector.len(), input_length);
    assert_eq!(matrix.len(), input_length * output_length);

    (0..output_length).map(
        |o| (0..input_length).map(
            |i| vector[i] * matrix[i * output_length + o]
        ).sum::<f32>()
    ).collect()
}

fn mat_cat(
    matrices: &[Vec<f32>],
    group: usize,
) -> Vec<f32> {
    let length = matrices[0].len();
    let mut offset = 0;
    let mut result = vec![];

    assert_eq!(length % group, 0);
    assert!(matrices.iter().all(|m| m.len() == length));

    while offset < length {
        for m in matrices.iter() {
            result.extend(&m[offset..(offset + group)]);
        }

        offset += group;
    }

    result
}

const SQRT_2_OVER_PI: f32 = 0.7978845608;
const GELU_CONST: f32 = 0.044715;

fn mat_gelu(matrix: &[f32]) -> Vec<f32> {
    matrix.iter().map(|x| 0.5 * x * ((SQRT_2_OVER_PI * (x + GELU_CONST * x.powi(3))).tanh() + 1.)).collect()
}

fn softmax(ns: &[f32]) -> Vec<f32> {
    let max = ns.iter().fold(f32::NEG_INFINITY, |a, b| f32::max(a, *b));
    let es = ns.iter().map(|n| (n - max).exp()).collect::<Vec<_>>();
    let sum = es.iter().sum::<f32>();
    es.iter().map(|e| e / sum).collect()
}
