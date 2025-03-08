# Swapping Dataset

I modified `SimpleTokenizer`. It uses a fixed tokenization, so that I can swap dataset while training a model.

# TODO

1. Check if this implementation makes sense.
  - [X] Loss converges as training continues.
  - [ ] A model with smaller loss generates more sane output.
  - [ ] A larger model is smarter.
2. Find usecase for femtoGPT
  - Training a model with large context-size is impossible on my machine. 256 is limit.
  - Training a 6.4M model with context-size 256 consumes 18GiB RAM. (using CPU training)
