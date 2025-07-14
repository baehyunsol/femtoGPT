# Inner structure of a model

```rs
struct TrainingState {
    tensors: HashMap<String, Tensor<f32>>,
    optimizer: OptimizerState,
}

struct OptimizerState {
    step: usize,
    state: HashMap<String, Tensor<f32>>,
}

struct Model {
    // "byte" | "bpe"
    tokenizer: String,

    // some tokenizers require an extra data
    // I want it to be `serde_json::Value`,
    // but it seems like `bincode` does not
    // support the type.
    tokenizer_data: String,
    hyperparameters: Hyperparameters,
    training_state: TrainingState,
}

struct Hyperparameters {
    // It's like a max-context-size
    num_tokens: usize,
    embedding_degree: usize,
    num_layers: usize,
    num_heads: usize,
}
```

- `TrainingState::tensors`
  - `token_embedding`
    - dim: `vocab_size * embedding_degree`
  - `norm_<X>_bias`, `norm_<X>_coeff`
    - normalize input before applying multi-head attention
    - X: layer
    - dim: `embedding_degree`
  - `head_<X>_<Y>_k`, `head_<X>_<Y>_q`, `head_<X>_<Y>_v`
    - X: layer, Y: head
    - dim: `embedding_degree * head_size` (head_size is embedding_degree / num_heads)
  - `proj_<X>_bias`, `proj_<X>_weights`
    - concat head results and project into embedding_degree
    - X: layer
    - dim: `embedding_degree * embedding_degree`
  - `atten_norm_<X>_bias`, `atten_norm_<X>_coeff`
    - add attention results to input and then normalize
    - X: layer
    - dim: `embedding_degree`
  - `feedforward1_<X>_bias`, `feedforward1_<X>_weights`, `feedforward2_<X>_bias`, `feedforward2_<X>_weights`
    - X: layer
    - dim (ff1 weights): `embedding_degree * (embedding_degree * 4)`
    - dim (ff1 bias): `embedding_degree * 4`
    - dim (ff2 weights): `(embedding_degree * 4) * embedding_degree`
    - dim (ff2 bias): `embedding_degree`
  - `head_norm_bias`, `head_norm_coeff`
    - normalize the output after the last layer
    - dim: `embedding_degree`
  - `head_map_bias`, `head_map_weights`
    - map from embedding_degree to vocab_size through a linear layer
    - dim (weights): `embedding_degree * vocab_size`
    - dim (bias): `vocab_size`
- `OptimizerState::state`
  - `<key>_m`, `<key>_v`
    - key: a key in `TrainingState::tensors`
