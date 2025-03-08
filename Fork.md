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

# Lessons learnt

`context_size` parameter works differently than I thought. When training, it splits the training data into chunks, where each chunk has `context_size` tokens. For each chunk, it's trained to predict its next token. Start and end of each chunk are selected randomly. That means each dataset has to be big enough...

1. Make sure that each file is big enough (much larger than the context size).
2. First start training with context size 32, then make it 64, then 128, then 256.
