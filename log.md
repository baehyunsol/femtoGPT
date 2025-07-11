# 1. Training a very simple model 1 (success)

- tokenizer: simple (the one in the original repository, now deprecated)
- embedding degree: 64, num layers: 4, num heads 4 (232K params)
- dropout: 0, base_lr: 0.001, min_lr: 0.00001, warmup_steps: 100, decay_steps: 50000
- step: 685, loss: 1.4672, elapsed: 2m 16s (apple M3 pro)
- data: `python3 dummy_data/simple_dummy.py > dataset.txt`

It can produce a plausible output!!!

# 2. Training a very simple model 2 (success)

- tokenizer: byte
- embedding degree: 64, num layers: 4, num heads 4 (232K params)
- dropout: 0, base_lr: 0.001, min_lr: 0.00001, warmup_steps: 100, decay_steps: 50000
- step: 629, loss: 1.4760, elapsed: 2m 9s (apple M3 pro)
- data: `python3 dummy_data/simple_dummy.py > dataset.txt`

It can produce a plausible output!!!

# 3. Training a very simple model 3 (fail)

- tokenizer: byte
- embedding degree: 64, num layers: 6, num heads 4 (331K params)
- dropout: 0, base_lr: 0.001, min_lr: 0.00001, warmup_steps: 100, decay_steps: 50000
- step: 874, loss: 1.9082, elapsed: 4m 24s (apple M3 pro)
- data: `python3 dummy_data/simple_dummy.py > dataset.txt`

Its output is not as good as #1 and #2.

I tried training it again from scratch, with the same settings.

- step: 650, loss: 1.9255, elapsed: 3m 11s

It's still not working.

Why not? The only difference is that the model has more layers than before. Why is adding layers degrade the model performance?

# 4. Training a Rust coder 1 (fail)

- tokenizer: byte
- embedding degree: 64, num layers: 6, num heads 4 (331K params)
- dropout: 0, base_lr: 0.001, min_lr: 0.00001, warmup_steps: 100, decay_steps: 50000
- step: 370, loss: 2.6159, elapsed: 1m 45s (apple M3 pro)
- data: `cat ~/Documents/Rust/ragit/src/main.rs ~/Documents/Rust/ragit/src/index.rs ~/Documents/Rust/ragit/src/chunk.rs ~/Documents/Rust/ragit/src/index/commands/build.rs ~/Documents/Rust/ragit/src/uid.rs > dataset.txt`

It doesn't work at all! I guess I need a bigger model

# 5. Training a Rust coder 2 (fail)

- tokenizer: byte
- embedding degree: 256, num layers: 8, num heads 8 (6.4M params)
- dropout: 0, base_lr: 0.001, min_lr: 0.00001, warmup_steps: 100, decay_steps: 50000
- step: 274, loss: 2.7485, elapsed: 12m 41s (apple M3 pro)
- data: `cat ~/Documents/Rust/ragit/src/main.rs ~/Documents/Rust/ragit/src/index.rs ~/Documents/Rust/ragit/src/chunk.rs ~/Documents/Rust/ragit/src/index/commands/build.rs ~/Documents/Rust/ragit/src/uid.rs > dataset.txt`

It's using 5~6 GiB of RAM, when `num_tokens` is 64.

It takes too long to converge. I'll continue on this later.

# 6. Training a Rust coder 3 (fail)

- tokenizer: byte
- embedding degree: 64, num layers: 4, num heads 4 (232K params)
- dropout: 0, base_lr: 0.001, min_lr: 0.00001, warmup_steps: 100, decay_steps: 50000
- step: 508, loss: 2.3020, elapsed: 1m 43s (apple M3 pro)
- data: `cat ~/Documents/Rust/ragit/src/main.rs ~/Documents/Rust/ragit/src/index.rs ~/Documents/Rust/ragit/src/chunk.rs ~/Documents/Rust/ragit/src/index/commands/build.rs ~/Documents/Rust/ragit/src/uid.rs > dataset.txt`

It doesn't work.

Looking at its log_probs: `[(' ', 88.8%), ('}', 0.9%), ('l', 0.8%), ('i', 0.7%), ...]`. Indeed it has learnt something!

# 7. Training an addition model 1 (fail)

- tokenizer: byte
- embedding degree: 64, num layers: 4, num heads 4 (232K params)
- dropout: 0, base_lr: 0.001, min_lr: 0.00001, warmup_steps: 100, decay_steps: 50000
- step: 716, loss: 1.3845, elapsed: 2m 24s (apple M3 pro)
- data: `python3 dummy_data/addition_dummy.py > dataset.txt`

when prompted `13 + 29 = `,

```
13 + 29 = 92;
91 = 121;
2 + 1;
11 + 121 + 1;
9;
9 + 9;
6 + 1;
7 = 90 10 = 7;
60 1 = 1;
10 1 = 13;
6 = 60;
6 =
```

It isn't that bad for this small model. Let's try again with a bigger one.

# 8. Training an addition model 2 (fail)

- tokenizer: byte
- embedding degree: 64, num layers: 6, num heads 4 (331K params)
- dropout: 0, base_lr: 0.001, min_lr: 0.00001, warmup_steps: 100, decay_steps: 50000
- step: 996, loss: 1.3336, elapsed: ?m ??s (apple M3 pro)
  - I forgot to record how long it took.
- data: `python3 dummy_data/addition_dummy.py > dataset.txt`

It's using 0.8 ~ 1 GiB of RAM, when `num_tokens` is 64.

when prompted `13 + 29 = `,

```
13 + 29 = 161 = 161 + 170 + 61 = 139 = 931 = 112 = 131;
7;
6;
7;
31 + 16;
9 = 17;+ 139 + 67 + 63 = 93 1 1 9 1
```

It isn't that different from #7.

# 9. Training a very simple model 4 (success)

- tokenizer: byte
- embedding degree: 80, num layers: 6, num heads 4 (506K params)
- dropout: 0, base_lr: 0.001, min_lr: 0.00001, warmup_steps: 100, decay_steps: 50000
- step: 1856, loss: 1.4313, elapsed: 9m 52s (apple M3 pro)
- data: `python3 dummy_data/simple_dummy.py > dataset.txt`

I just figured out why I have failed #3. It's because larger models take longer to train, but I trained them too shortly.

# 10. Training a very simple model 5 (success)

- tokenizer: byte
- embedding degree: 32, num layers: 4, num heads 2 (67K params)
- dropout: 0, base_lr: 0.001, min_lr: 0.00001, warmup_steps: 100, decay_steps: 50000
- step: 528, loss: 1.5303, elapsed: 0m 50s (Intel Core Ultra 7 155H)
- data: `python3 dummy_data/simple_dummy.py > dataset.txt`

I wanted to see if even smaller models can be trained on this data. And it works...!!

# 11. Training an English model 1 (fail)

- tokenizer: byte
- embedding degree: 128, num layers: 6, num heads: 8 (1.2M params)
- dropout: 0, base_lr: 0.001, min_lr: 0.00001, warmup_steps: 100, decay_steps: 50000
- step: 400, loss: 3.2494, elapsed: ?m ??s (Intel Core Ultra 7 155H)
  - I forgot to record how long it took, but each step took roughly 800 ~ 900 ms.
- data: the original dataset in femtoGPT (https://github.com/keyvank/femtoGPT/blob/d00da389998bcef3ba483bb7a0614f4e874e1d06/dataset.txt)

Generated text: `earhh o th  eh  et ee reaat tr  retaoohtth  r  hsorta eaoth ort ree e oa  eaearhthoe  taaa  a e t`

I only had a few minutes and it's not enough to train this size of model. I have to try this again when I have more time.

# 12. Training an English model 2 (fail)

- tokenizer: bpe (50 epochs, 566 tokens)
- embedding degree: 160, num layers: 8, num heads: 8 (2.6M params)
- dropout: 0, base_lr: 0.001, min_lr: 0.00001, warmup_steps: 100, decay_steps: 50000
- step: 338, loss: 5.8933, elapsed: ?m ??s (Intel Core Ultra 7 155H)
  - I forgot to record how long it took, but each step took roughly 1400 ~ 1500 ms.
- data: the original dataset in femtoGPT (https://github.com/keyvank/femtoGPT/blob/d00da389998bcef3ba483bb7a0614f4e874e1d06/dataset.txt)

It's using 3~4 GiB of RAM, when `num_tokens` is 80.

I just wanted to test the new bpe tokenizer. I guess I have to increase the vocab size!
