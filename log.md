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

# 13. Training a very large model (fail)

- tokenizer: byte
  - It doesn't matter, though
- embedding degree: 640, num layers: 8, num heads: 16 (39.7M parameters)
- dropout: 0, base_lr: 0.001, min_lr: 0.00001, warmup_steps: 100, decay_steps: 50000
- step: n/a, loss: n/a, elapsed: n/a (Intel Core Ultra 7 155H)
- data: doesn't matter

I wanted to see how much memory does it take to train this large model.

The first step was fine, it consumed about 10 GiB of RAM, when `num_tokens` is 80. The first step took 18 seconds. But at the second step, it started consuming 90% of RAM (my machine has 32GB), so I just killed it.

# 14. Training a very large model (fail)

- tokenizer: bpe (1054 tokens)
- embedding degree: 768, num layers: 8, num heads: 16 (58.3M parameters)
- dropout: 0.1, base_lr: 0.001, min_lr: 0.00001, warmup_steps: 100, decay_steps: 50000
- step: 689, loss: 6.4152, elapsed: 9h 52m (AWS EC2 m5.4xlarge: 16vCPU, 64GB RAM)
  - Each step takes roughly a minute...
- data: see below

It's using 40~50 GiB of RAM, when `num_tokens` is 80.

It would take at least a few hundred hours to train this. I guess I have to try smaller models first.

How to run:

```sh
git -C .. clone https://github.com/baehyunsol/ragit
git -C ../ragit checkout e12153573
git -C .. clone https://github.com/gleam-lang/gleam
git -C ../gleam checkout bed57729d
git -C .. clone https://github.com/seanmonstar/warp
git -C ../warp checkout 1cbf029b1
git -C .. clone https://github.com/getzola/zola
git -C ../zola checkout c1b105051

cat ../ragit/src/**/*.rs ../gleam/compiler-core/src/**/*.rs ../warp/src/**/*.rs ../zola/src/**/*.rs src/**/*.rs > dataset.txt

cargo run --release -- train-bpe --epoch=100 --vocab-size=1024
cargo run --release -- init --tokenizer=bpe --num-tokens=80 --embedding-degree=768 --num-layers=8 --num-heads=16
nohup cargo run --release -- train --dropout=0.1 &
```

# 15. Training an addition model 3 (fail)

- tokenizer: byte
- embedding degree: 160, num layers: 6, num heads: 8 (1.9M parameters)
- dropout: 0, base_lr: 0.001, min_lr: 0.00001, warmup_steps: 100, decay_steps: 50000
- step: 802, loss: 2.4628, elapsed: 15m 20s (Intel Core Ultra 7 155H)
- data: `python3 dummy_data/addition_dummy.py > dataset.txt`

I don't understand. I just want it to generate lines that *look like* additions, even though the calculations are wrong. Below is the output that it generates.

```
1 + 1 =       ;  = 1=;11+= =   =   1 ;;1    1  11=1 ; 1 =  1 1= =; = 1= ; =;  1;1 ; ;      ==1= ;1==       =
```

# 16. Training an addition model 4 (fail)

- tokenizer: byte
- embedding degree: 160, num layers: 6, num heads: 8 (1.9M parameters)
- dropout: 0.1, base_lr: 0.001, min_lr: 0.00001, warmup_steps: 100, decay_steps: 50000
- step: 360, loss: 2.4748, elapsed: 6m 56s (Intel Core Ultra 7 155H)
  - NOTE: #15 reached loss 2.4 at around 200 step, and stuck there for 600 steps
- data: `python3 dummy_data/addition_dummy.py > dataset.txt`

I was curious whether setting dropout to 0.1 might change something. It didn't.

# 17. Training an addition model 5 (fail)

- tokenizer: byte
- embedding degree: 80, num layers 4, num heads: 4 (351K parameters)
- dropout: 0, base_lr: 0.001, min_lr: 0.00001, warmup_steps: 100, decay_steps: 50000
- step: 676, loss: 1.4216, elapsed: 3m 4s (Intel Core Ultra 7 155H)
  - NOTE: It took about 300 steps to reach loss 2.0.
- data: `python3 dummy_data/addition_dummy.py > dataset.txt`

Below is the response. I won't mark it "success" until it dumps `r"\d+\s?\+\d+\s?\=\d+\;\n"`.

```
10;
1;
1 + 15;
9 + 14 + 12 + 65;
1 + = 5615 = 54;
9;
6 + = 9;
1;
5 = 61;
10 + 1;
5;
9 + = 11140 = 9;
```

It converged much faster than #15 and #16, though. Maybe #15 would converge if I had more patience.

# 18. Training a Rust coder 4 (fail)

- tokenizer: bpe (1054 tokens)
- embedding: 80, num layers: 4, num heads: 4 (480K parameters)
- dropout: 0, base_lr: 0.001, min_lr: 0.00001, warmup_steps: 100, decay_steps: 50000
- step: 1018, loss: 4.7967, elapsed: ?m ??s (Intel Core Ultra 7 155H)
  - Each step took roughly 300 ms
- data: that of #14

It might converge someday. I'll try a bit larger model with the same dataset before I go to bed tonight.

# 19. Training a Rust coder 5 (fail)

- tokenizer: bpe (1054 tokens)
- embedding degree: 192, num layers: 8, num heads: 8 (3.9M parameters)
- dropout: 0.1, base_lr: 0.001, min_lr: 0.00001, warmup_steps: 100, decay_steps: 50000
- step: 7684, loss: 6.4670, elapsed: 8 ~ 9 hours (AWS EC2 m5.4xlarge: 16vCPU, 64GB RAM)
  - Each step takes roughly 3 ~ 4 seconds...
- data: that of #14

It's using 3~5 GiB of RAM, when `num_tokens` is 80.
