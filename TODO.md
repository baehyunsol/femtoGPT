# 5. Add a new head to an existing model

# 4. Learnable Absolute Position Encoding

Instead of a sinusoidal positional vector, it adds a learnable vector to the input embeddings. The vector is chosen according to the position.

# 3. Rotary Position Encoding

- https://medium.com/@ngiengkianyew/understanding-rotary-positional-encoding-40635a4d078e
- https://github.com/lucidrains/rotary-embedding-torch/blob/main/rotary_embedding_torch/rotary_embedding_torch.py
- https://arxiv.org/abs/2405.10436v1

Removing the current PE (sinusoidal) leads to better results. I need further experiments on this!

# 2. Add a new layer to an existing model

(Complete)

# 1. Add a new token to an existing model

1. Easy approach
  - Initialize the tokenizer with `<|reserve-token-001|>`, `<|reserve-token-002|>`, ... 
  - Wish that such tokens never appears in the dataset.
  - Replace a reserve token with one that I want. There's nothing to do with tensors.
  - Further train the model.
2. Difficult approach
  - Add a token to the dictionary.
  - Extend `token_embedding` tensor. Initialize the new vector with random input.
  - Extend `head_map_bias` and `head_map_weights` tensors.
  - Further train the model.
