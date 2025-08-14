NOTE: All the trainings are done on CPU.

# 1. Training a very simple model 1 (success)

- tokenizer: simple (the one in the original repository, now deprecated), case sensitive
- positional encoding: absolute
- embedding degree: 64, num layers: 4, num heads 4 (232K params)
- dropout: 0, base_lr: 0.001, min_lr: 0.00001, warmup_steps: 100, decay_steps: 50000
- step: 685, loss: 1.4672, elapsed: 2m 16s (apple M3 pro)
- data: `python3 dummy_data/simple_dummy.py > dataset.txt`

The output looks like the model understood the pattern.

# 2. Training a very simple model 2 (success)

- tokenizer: byte (deprecated), case sensitive
- positional encoding: absolute
- embedding degree: 64, num layers: 4, num heads 4 (232K params)
- dropout: 0, base_lr: 0.001, min_lr: 0.00001, warmup_steps: 100, decay_steps: 50000
- step: 629, loss: 1.4760, elapsed: 2m 9s (apple M3 pro)
- data: `python3 dummy_data/simple_dummy.py > dataset.txt`

The output looks like the model understood the pattern.

# 3. Training a very simple model 3 (fail)

- tokenizer: byte (deprecated), case sensitive
- positional encoding: absolute
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

- tokenizer: byte (deprecated), case sensitive
- positional encoding: absolute
- embedding degree: 64, num layers: 6, num heads 4 (331K params)
- dropout: 0, base_lr: 0.001, min_lr: 0.00001, warmup_steps: 100, decay_steps: 50000
- step: 370, loss: 2.6159, elapsed: 1m 45s (apple M3 pro)
- data: `cat ~/Documents/Rust/ragit/src/main.rs ~/Documents/Rust/ragit/src/index.rs ~/Documents/Rust/ragit/src/chunk.rs ~/Documents/Rust/ragit/src/index/commands/build.rs ~/Documents/Rust/ragit/src/uid.rs > dataset.txt`

It doesn't work at all! I guess I need a bigger model

# 5. Training a Rust coder 2 (fail)

- tokenizer: byte (deprecated), case sensitive
- positional encoding: absolute
- embedding degree: 256, num layers: 8, num heads 8 (6.4M params)
- dropout: 0, base_lr: 0.001, min_lr: 0.00001, warmup_steps: 100, decay_steps: 50000
- step: 274, loss: 2.7485, elapsed: 12m 41s (apple M3 pro)
- data: `cat ~/Documents/Rust/ragit/src/main.rs ~/Documents/Rust/ragit/src/index.rs ~/Documents/Rust/ragit/src/chunk.rs ~/Documents/Rust/ragit/src/index/commands/build.rs ~/Documents/Rust/ragit/src/uid.rs > dataset.txt`

It's using 5~6 GiB of RAM, when `num_tokens` is 64.

It takes too long to converge. I'll continue on this later.

# 6. Training a Rust coder 3 (fail)

- tokenizer: byte (deprecated), case sensitive
- positional encoding: absolute
- embedding degree: 64, num layers: 4, num heads 4 (232K params)
- dropout: 0, base_lr: 0.001, min_lr: 0.00001, warmup_steps: 100, decay_steps: 50000
- step: 508, loss: 2.3020, elapsed: 1m 43s (apple M3 pro)
- data: `cat ~/Documents/Rust/ragit/src/main.rs ~/Documents/Rust/ragit/src/index.rs ~/Documents/Rust/ragit/src/chunk.rs ~/Documents/Rust/ragit/src/index/commands/build.rs ~/Documents/Rust/ragit/src/uid.rs > dataset.txt`

It doesn't work.

Looking at its log_probs: `[(' ', 88.8%), ('}', 0.9%), ('l', 0.8%), ('i', 0.7%), ...]`. Indeed it has learnt something!

# 7. Training an addition model 1 (fail)

- tokenizer: byte (deprecated), case sensitive
- positional encoding: absolute
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

- tokenizer: byte (deprecated), case sensitive
- positional encoding: absolute
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

- tokenizer: byte (deprecated), case sensitive
- positional encoding: absolute
- embedding degree: 80, num layers: 6, num heads 4 (506K params)
- dropout: 0, base_lr: 0.001, min_lr: 0.00001, warmup_steps: 100, decay_steps: 50000
- step: 1856, loss: 1.4313, elapsed: 9m 52s (apple M3 pro)
- data: `python3 dummy_data/simple_dummy.py > dataset.txt`

I just figured out why I have failed #3. It's because larger models take longer to train, but I trained them too shortly.

# 10. Training a very simple model 5 (success)

- tokenizer: byte (deprecated), case sensitive
- positional encoding: absolute
- embedding degree: 32, num layers: 4, num heads 2 (67K params)
- dropout: 0, base_lr: 0.001, min_lr: 0.00001, warmup_steps: 100, decay_steps: 50000
- step: 528, loss: 1.5303, elapsed: 0m 50s (Intel Core Ultra 7 155H)
- data: `python3 dummy_data/simple_dummy.py > dataset.txt`

I wanted to see if even smaller models can be trained on this data. And it works...!!

# 11. Training an English model 1 (fail)

- tokenizer: byte (deprecated), case sensitive
- positional encoding: absolute
- embedding degree: 128, num layers: 6, num heads: 8 (1.2M params)
- dropout: 0, base_lr: 0.001, min_lr: 0.00001, warmup_steps: 100, decay_steps: 50000
- step: 400, loss: 3.2494, elapsed: ?m ??s (Intel Core Ultra 7 155H)
  - I forgot to record how long it took, but each step took roughly 800 ~ 900 ms.
- data: the original dataset in femtoGPT (https://github.com/keyvank/femtoGPT/blob/d00da389998bcef3ba483bb7a0614f4e874e1d06/dataset.txt)

Generated text: `earhh o th  eh  et ee reaat tr  retaoohtth  r  hsorta eaoth ort ree e oa  eaearhthoe  taaa  a e t`

I only had a few minutes and it's not enough to train this size of model. I have to try this again when I have more time.

# 12. Training an English model 2 (fail)

- tokenizer: bpe (50 epochs, 566 tokens), case sensitive
- positional encoding: absolute
- embedding degree: 160, num layers: 8, num heads: 8 (2.6M params)
- dropout: 0, base_lr: 0.001, min_lr: 0.00001, warmup_steps: 100, decay_steps: 50000
- step: 338, loss: 5.8933, elapsed: ?m ??s (Intel Core Ultra 7 155H)
  - I forgot to record how long it took, but each step took roughly 1400 ~ 1500 ms.
- data: the original dataset in femtoGPT (https://github.com/keyvank/femtoGPT/blob/d00da389998bcef3ba483bb7a0614f4e874e1d06/dataset.txt)

It's using 3~4 GiB of RAM, when `num_tokens` is 80.

I just wanted to test the new bpe tokenizer. I guess I have to increase the vocab size!

# 13. Training a very large model (fail)

- tokenizer: byte (deprecated), case sensitive
  - It doesn't matter, though
- positional encoding: absolute
- embedding degree: 640, num layers: 8, num heads: 16 (39.7M parameters)
- dropout: 0, base_lr: 0.001, min_lr: 0.00001, warmup_steps: 100, decay_steps: 50000
- step: n/a, loss: n/a, elapsed: n/a (Intel Core Ultra 7 155H)
- data: doesn't matter

I wanted to see how much memory does it take to train this large model.

The first step was fine, it consumed about 10 GiB of RAM, when `num_tokens` is 80. The first step took 18 seconds. But at the second step, it started consuming 90% of RAM (my machine has 32GB), so I just killed it.

# 14. Training a very large model (fail)

- tokenizer: bpe (1054 tokens), case sensitive
- positional encoding: absolute
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

- tokenizer: byte (deprecated), case sensitive
- positional encoding: absolute
- embedding degree: 160, num layers: 6, num heads: 8 (1.9M parameters)
- dropout: 0, base_lr: 0.001, min_lr: 0.00001, warmup_steps: 100, decay_steps: 50000
- step: 802, loss: 2.4628, elapsed: 15m 20s (Intel Core Ultra 7 155H)
- data: `python3 dummy_data/addition_dummy.py > dataset.txt`

I don't understand. I just want it to generate lines that *look like* additions, even though the calculations are wrong. Below is the output that it generates.

```
1 + 1 =       ;  = 1=;11+= =   =   1 ;;1    1  11=1 ; 1 =  1 1= =; = 1= ; =;  1;1 ; ;      ==1= ;1==       =
```

# 16. Training an addition model 4 (fail)

- tokenizer: byte (deprecated), case sensitive
- positional encoding: absolute
- embedding degree: 160, num layers: 6, num heads: 8 (1.9M parameters)
- dropout: 0.1, base_lr: 0.001, min_lr: 0.00001, warmup_steps: 100, decay_steps: 50000
- step: 360, loss: 2.4748, elapsed: 6m 56s (Intel Core Ultra 7 155H)
  - NOTE: #15 reached loss 2.4 at around 200 step, and stuck there for 600 steps
- data: `python3 dummy_data/addition_dummy.py > dataset.txt`

I was curious whether setting dropout to 0.1 might change something. It didn't.

# 17. Training an addition model 5 (fail)

- tokenizer: byte (deprecated), case sensitive
- positional encoding: absolute
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

- tokenizer: bpe (1054 tokens), case sensitive
- positional encoding: absolute
- embedding: 80, num layers: 4, num heads: 4 (480K parameters)
- dropout: 0, base_lr: 0.001, min_lr: 0.00001, warmup_steps: 100, decay_steps: 50000
- step: 1018, loss: 4.7967, elapsed: ?m ??s (Intel Core Ultra 7 155H)
  - Each step took roughly 300 ms
- data: that of #14

It might converge someday. I'll try a bit larger model with the same dataset before I go to bed tonight.

# 19. Training a Rust coder 5 (fail)

- tokenizer: bpe (1054 tokens), case sensitive
- positional encoding: absolute
- embedding degree: 192, num layers: 8, num heads: 8 (3.9M parameters)
- dropout: 0.1, base_lr: 0.001, min_lr: 0.00001, warmup_steps: 100, decay_steps: 50000
- step: 7684, loss: 6.4670, elapsed: 8 ~ 9 hours (AWS EC2 m5.4xlarge: 16vCPU, 64GB RAM)
  - Each step takes roughly 3 ~ 4 seconds...
- data: that of #14

It's using 3~5 GiB of RAM, when `num_tokens` is 80.

# 20. Training a simple model 1

- tokenizer: byte (deprecated), case sensitive
- positional encoding: absolute
- embedding degree: 80, num layers: 4, num heads 4 (351K params)
- dropout: 0, base_lr: 0.001, min_lr: 0.00001, warmup_steps: 100, decay_steps: 50000
- step: 1447, loss: 1.6670, elapsed: ?m ??s (Intel Core Ultra 7 155H)
- data: `python3 dummy_data/simple_dummy2.py > dataset.txt`

Evaluation

| action | correct  | incorrect  | correct rate   |
|--------|----------|------------|----------------|
| 0      | 10       |  0         | 100%           |
| 1      |  0       | 11         |   0%           |
| 2      |  0       |  9         |   0%           |
| 3      |  0       |  7         |   0%           |
| 4      |  0       | 10         |   0%           |
| 5      |  0       |  8         |   0%           |
| 6      |  0       | 17         |   0%           |
| 7      |  0       | 10         |   0%           |
| 8      |  0       |  6         |   0%           |
| 9      | 12       |  0         | 100%           |

It doesn't understand what each action does, but it generates lines that resemble dataset.

I'll mark it "success" if all the action's correct rates are greater than 70%

# 21. Training a simple model 2 (fail)

- tokenizer: byte (deprecated), case sensitive
- positional encoding: absolute
- embedding degree: 96, num layers: 6, num heads 4 (718K params)
- dropout: 0.1, base_lr: 0.001, min_lr: 0.00001, warmup_steps: 100, decay_steps: 50000
- step: 1807, loss: 1.7154, elapsed: 12m 31s (apple M3 pro)
  - It reached loss 1.7 at around 1150 step.
- data: `python3 dummy_data/simple_dummy2.py > dataset.txt`

Evaluation

| action | correct  | incorrect  | correct rate   |
|--------|----------|------------|----------------|
| 0      |  7       |  0         | 100%           |
| 1      |  0       |  7         |   0%           |
| 2      |  0       | 15         |   0%           |
| 3      |  0       | 12         |   0%           |
| 4      |  0       | 10         |   0%           |
| 5      |  0       |  6         |   0%           |
| 6      |  0       | 11         |   0%           |
| 7      |  0       | 11         |   0%           |
| 8      |  0       | 10         |   0%           |
| 9      | 11       |  0         | 100%           |

# 22. Training a simple model 3 (fail)

- tokenizer: byte (deprecated), case sensitive
- positional encoding: absolute
- embedding degree: 144, num layers: 6, num heads 6 (1.5M params)
- dropout: 0.1, base_lr: 0.001, min_lr: 0.00001, warmup_steps: 100, decay_steps: 50000
- step: 1143, loss: 2.7072, elapsed: ??m ??s (apple M3 pro)
  - step 1 ~ step 28: loss 5.5
  - step 29 ~ step 160: loss 5.5 -> loss 2.7
  - step 160 ~ 1143: loss 2.7
- data: `python3 dummy_data/simple_dummy2.py > dataset.txt`

It seems like it's stuck at a wrong local optima. It's not smart enough to run the evaluation. It can't even generate proper lines!

# 23. Training a simple model 4 (fail)

- tokenizer: byte (deprecated), case sensitive
- positional encoding: absolute
- embedding degree: 144, num layers: 6, num heads 6 (1.5M params)
- dropout: 0.1, base_lr: 0.001, min_lr: 0.00001, warmup_steps: 100, decay_steps: 50000
- step: 4229, loss: 1.6876, elapsed: ??m ??s (apple M3 pro)
  - step 1 ~ step 28: loss 5.5
  - step 29 ~ step 140: loss 5.5 -> loss 2.7
  - step 141 ~ step 290: loss 2.7
  - step 291 ~ step 475: loss 2.7 -> loss 2.2
  - step 476 ~ step 857: loss 2.2
  - step 858 ~ step 1523: loss 2.2 -> loss 1.9
  - step 1524 ~ step 1884: loss 1.9 -> loss 1.8
  - step 1885 ~ step 4229: loss 1.8 -> loss 1.7
- data: `python3 dummy_data/simple_dummy2.py > dataset.txt`

#22 and #23 have exactly the same settings, but #23 is much better than #22, regarding the training losses.

Maybe I was wrong at #22. It seemed like #23 also fell into a local optima, but it got better very slowly.

I forgot to add the evaluation result, but it was worse than #21.

# 24. Training a simple model 5 (fail)

- tokenizer: byte (deprecated), case sensitive
- positional encoding: absolute
- embedding degree: 96, num layers: 8, num heads 4 (942K params)
- dropout: 0.1, base_lr: 0.001, min_lr: 0.00001, warmup_steps: 100, decay_steps: 50000
- step: 3756, loss: 1.6839, elapsed: ??m ??s (apple M3 pro)
  - step 1 ~ step 32: loss 5.5
  - step 33 ~ step 160: loss 5.5 -> loss 2.7
  - step 161 ~ step 446: loss 2.7
  - step 447 ~ step 1760: loss 2.7 -> loss 1.7
  - step 1761 ~ step 3292: loss 1.7
  - step 3292: loss 2.1 (It spiked suddenly)
  - step 3293 ~ step 3503: loss 2.1 -> loss 1.7
  - step 3504 ~ step 3756: loss 1.7
- data: `python3 dummy_data/simple_dummy2.py > dataset.txt`

Evaluation

| action | correct  | incorrect  | correct rate   |
|--------|----------|------------|----------------|
| 0      |  9       |  0         | 100.0%         |
| 1      |  1       | 12         |   7.7%         |
| 2      |  0       | 12         |   0.0%         |
| 3      |  0       | 10         |   0.0%         |
| 4      |  0       |  7         |   0.0%         |
| 5      |  0       |  9         |   0.0%         |
| 6      |  0       |  9         |   0.0%         |
| 7      |  0       | 13         |   0.0%         |
| 8      |  0       | 12         |   0.0%         |
| 9      |  6       |  0         | 100.0%         |

This is the first-ever model to understand an action other than 0 or 9.

# 25. Training a simple model 6 (fail)

- tokenizer: byte (deprecated), case sensitive
- positional encoding: absolute
- embedding degree: 144, num layers: 8, num heads 6 (2M params)
- dropout: 0.1, base_lr: 0.001, min_lr: 0.00001, warmup_steps: 100, decay_steps: 50000
- step 1234, loss: 2.6963, elapsed: ??m ??s (apple M3 pro)
  - step 1 ~ step 27: loss 5.5
  - step 28 ~ step 262: loss 5.5 -> loss 2.7
  - step 263 ~ 1234: loss 2.7
- data: `python3 dummy_data/simple_dummy2.py > dataset.txt`

# 26. Training a simple model 7 (fail)

- tokenizer: byte (deprecated), case sensitive
- positional encoding: absolute
- embedding degree: 144, num layers: 8, num heads 6 (2M params)
- dropout: 0.1, base_lr: 0.001, min_lr: 0.00001, warmup_steps: 100, decay_steps: 50000
- step 933, loss: 2.7182, elapsed: ??m ??s (apple M3 pro)
  - step 1 ~ step 26: loss 5.5
  - step 27 ~ step 201: loss 5.5 -> loss 2.7
  - step 202 ~ step 933: loss 2.7
- data: `python3 dummy_data/simple_dummy2.py > dataset.txt`

We have lost Euler.

# 27. Training a simple model 8 (fail)

- tokenizer: byte (deprecated), case sensitive
- positional encoding: absolute
- embedding degree: 64, num layers: 4, num heads 4 (232K params)
- dropout: 0.1, base_lr: 0.001, min_lr: 0.00001, warmup_steps: 100, decay_steps: 50000
- step: 1665, loss: 1.6631, elapsed: ??m ??s (apple M3 pro)
- data: `python3 dummy_data/simple_dummy2.py > dataset.txt`

I have used `train.py` to search for the best initial state.

Evaluation

| action | correct  | incorrect  | correct rate   |
|--------|----------|------------|----------------|
| 0      |  9       |  0         | 100.0%         |
| 1      |  0       |  6         |   0.0%         |
| 2      |  1       | 10         |   9.1%         |
| 3      |  0       | 12         |   0.0%         |
| 4      |  0       |  9         |   0.0%         |
| 5      |  0       |  7         |   0.0%         |
| 6      |  0       | 15         |   0.0%         |
| 7      |  0       | 12         |   0.0%         |
| 8      |  0       | 12         |   0.0%         |
| 9      |  7       |  0         | 100.0%         |

It's much smaller than #20 ~ #26, but does better than them. The lesson is that I have to wait longer until the model converges.

# 28. Training a Rust coder 6 (failed)

- tokenizer: bpe (630 tokens), case sensitive
- positional encoding: absolute
- embedding degree: 192, num layers: 8, num heads: 8 (3.7M parameters)
- dropout: 0.1, base_lr: 0.001, min_lr: 0.00001, warmup_steps: 100, decay_steps: 50000
- step: 9440 loss: 5.9775, elapsed: 11 ~ 12h (AWS EC2 m5.4xlarge: 16vCPU, 64GB RAM)
  - step 1 ~ step 39: loss 6.4
  - step 40 ~ step 90: loss 6.4 -> loss 6.0
  - step 91 ~ step 9440: loss 6.0
- data: see below

```sh
cat src/**/*.rs > dataset.txt

git -C .. clone https://github.com/baehyunsol/ragit
git -C ../ragit checkout e12153573
cat ../ragit/src/**/*.rs >> dataset.txt

git -C .. clone https://github.com/gleam-lang/gleam
git -C ../gleam checkout bed57729d
cat ../gleam/compiler-core/src/**/*.rs >> dataset.txt

git -C .. clone https://github.com/seanmonstar/warp
git -C ../warp checkout 1cbf029b1
cat ../warp/src/**/*.rs >> dataset.txt

git -C .. clone https://github.com/getzola/zola
git -C ../zola checkout c1b105051
cat ../zola/src/**/*.rs >> dataset.txt

git -C .. clone https://github.com/jafioti/luminal
git -C ../luminal checkout bd33460c9
cat ../luminal/src/**/*.rs >> dataset.txt

git -C .. clone https://github.com/rust-num/num-bigint
git -C ../num-bigint checkout 575cea47d
cat ../num-bigint/src/**/*.rs >> dataset.txt

git -C .. clone https://github.com/dimforge/nalgebra
git -C ../nalgebra checkout fa7afd5e4
cat ../nalgebra/src/**/*.rs >> dataset.txt

git -C .. clone https://github.com/bevyengine/bevy
git -C ../bevy checkout 2bddbdfd7
cat ../bevy/crates/bevy_core_pipeline/src/**/*.rs >> dataset.txt

git -C .. clone https://github.com/rust-lang/rust
git -C ../rust checkout 7f2065a4b
cat ../rust/compiler/rustc_mir_transform/src/**/*.rs >> dataset.txt

ls -l dataset.txt

cargo run --release -- train-bpe --epoch=60 --vocab-size=600
cargo run --release -- init --tokenizer=bpe --num-tokens=80 --embedding-degree=192 --num-layers=8 --num-heads=8
nohup cargo run --release -- train --dropout=0.1 &
```

At first, I thought it's a failure. Its loss was stuck at 6.0.

I ran `compare` command on 2 checkpoints (step 1140 and step 9420), and it seems like something is happening.

- `cargo run --release -- compare chk-001-step-1140.dat chk-002-step-9420.dat --limit=50`

```
chk-001-step-1140.dat is parent of chk-002-step-9420.dat.
chk-002-step-9420.dat is 8280-steps further trained version of chk-001-step-1140.dat
key: norm_0_bias, cosine: -0.186219
key: proj_0_bias, cosine: -0.149568
key: atten_norm_0_bias, cosine: 0.032173
key: feedforward2_0_bias, cosine: 0.203454
key: atten_norm_5_bias, cosine: 0.270509
key: norm_1_bias, cosine: 0.309950
key: feedforward2_5_bias, cosine: 0.374995
key: proj_2_bias, cosine: 0.389131
key: atten_norm_1_bias, cosine: 0.398604
key: norm_2_bias, cosine: 0.402201
key: feedforward1_0_bias, cosine: 0.431982
key: proj_1_bias, cosine: 0.448739
key: feedforward2_1_bias, cosine: 0.460932
key: head_norm_coeff, cosine: 0.482581
key: atten_norm_2_bias, cosine: 0.491970
key: atten_norm_3_bias, cosine: 0.686438
key: feedforward2_3_bias, cosine: 0.699300
key: proj_3_bias, cosine: 0.702777
key: feedforward1_1_bias, cosine: 0.725262
key: norm_3_bias, cosine: 0.783604
key: atten_norm_7_coeff, cosine: 0.787582
key: proj_6_bias, cosine: 0.787752
key: feedforward1_3_bias, cosine: 0.787844
key: feedforward2_2_bias, cosine: 0.788781
key: proj_5_bias, cosine: 0.792540
key: norm_5_bias, cosine: 0.799880
key: feedforward1_2_bias, cosine: 0.815701
key: norm_4_bias, cosine: 0.817524
key: norm_6_bias, cosine: 0.820367
key: proj_4_bias, cosine: 0.836463
key: atten_norm_4_bias, cosine: 0.843805
key: feedforward2_7_weights, cosine: 0.858503
key: proj_7_weights, cosine: 0.863350
key: feedforward2_4_bias, cosine: 0.868288
key: feedforward2_7_bias, cosine: 0.885023
key: proj_7_bias, cosine: 0.890920
key: norm_7_bias, cosine: 0.891247
key: atten_norm_7_bias, cosine: 0.896418
key: head_norm_bias, cosine: 0.900783
key: atten_norm_6_bias, cosine: 0.904012
key: feedforward1_5_bias, cosine: 0.910950
key: head_7_7_v, cosine: 0.921637
key: feedforward2_6_bias, cosine: 0.924358
key: norm_6_coeff, cosine: 0.925809
key: feedforward1_7_bias, cosine: 0.928416
key: feedforward1_7_weights, cosine: 0.929824
key: head_7_2_v, cosine: 0.930775
key: head_7_1_v, cosine: 0.933730
key: head_7_4_v, cosine: 0.943772
key: head_7_0_v, cosine: 0.946504
token_id: 439, token: "tex", head: 4, cosine: 0.449657
token_id: 153, token: "er", head: 4, cosine: 0.470878
token_id: 176, token: "<'a", head: 4, cosine: 0.494777
token_id: 294, token: "nder", head: 4, cosine: 0.514955
token_id: 26, token: "::", head: 4, cosine: 0.522974
token_id: 288, token: "Expr", head: 4, cosine: 0.531286
token_id: 68, token: "} ", head: 4, cosine: 0.534327
token_id: 315, token: "ex", head: 4, cosine: 0.544650
token_id: 31, token: ": S", head: 4, cosine: 0.564348
token_id: 177, token: " ", head: 3, cosine: 0.564892
token_id: 177, token: " ", head: 4, cosine: 0.569459
token_id: 272, token: "from", head: 4, cosine: 0.573965
token_id: 598, token: "_", head: 4, cosine: 0.599973
token_id: 393, token: "with", head: 4, cosine: 0.607397
token_id: 584, token: "sh", head: 4, cosine: 0.629185
token_id: 122, token: "ens", head: 4, cosine: 0.633083
token_id: 328, token: "st", head: 4, cosine: 0.637121
token_id: 414, token: "pub ", head: 4, cosine: 0.647366
token_id: 598, token: "_", head: 3, cosine: 0.647582
token_id: 274, token: "t ", head: 4, cosine: 0.649600
token_id: 564, token: "s", head: 4, cosine: 0.649679
token_id: 233, token: "\n    /// ", head: 4, cosine: 0.662096
token_id: 87, token: "of", head: 4, cosine: 0.667237
token_id: 21, token: "s: ", head: 4, cosine: 0.667291
token_id: 55, token: "line", head: 4, cosine: 0.669367
token_id: 473, token: "s ", head: 4, cosine: 0.670664
token_id: 138, token: "2, ", head: 4, cosine: 0.676291
token_id: 433, token: "e", head: 4, cosine: 0.680534
token_id: 470, token: "format", head: 4, cosine: 0.691581
token_id: 382, token: "ir", head: 4, cosine: 0.692219
token_id: 527, token: "//", head: 4, cosine: 0.693792
token_id: 503, token: "I", head: 4, cosine: 0.709106
token_id: 121, token: "/// ", head: 4, cosine: 0.711867
token_id: 391, token: "const", head: 4, cosine: 0.714640
token_id: 485, token: "ces", head: 4, cosine: 0.720311
token_id: 83, token: "Storage", head: 4, cosine: 0.722642
token_id: 147, token: "k", head: 4, cosine: 0.722670
token_id: 392, token: "<D", head: 4, cosine: 0.724198
token_id: 412, token: "al", head: 4, cosine: 0.725515
token_id: 82, token: "self.", head: 4, cosine: 0.726636
token_id: 594, token: "B", head: 4, cosine: 0.727489
token_id: 29, token: "ur", head: 4, cosine: 0.731126
token_id: 329, token: "ation", head: 4, cosine: 0.732588
token_id: 96, token: "Self::", head: 4, cosine: 0.733360
token_id: 558, token: "t", head: 4, cosine: 0.735875
token_id: 209, token: "///\n    /// ", head: 4, cosine: 0.738057
token_id: 407, token: "for", head: 4, cosine: 0.740767
token_id: 388, token: "t::", head: 4, cosine: 0.748152
token_id: 42, token: "() ", head: 4, cosine: 0.751697
token_id: 43, token: "&mut ", head: 4, cosine: 0.753123
```

- `cargo run --release -- compare chk-001-step-1140.dat chk-002-step-9420.dat --limit=50 --reverse`

```
chk-001-step-1140.dat is parent of chk-002-step-9420.dat.
chk-002-step-9420.dat is 8280-steps further trained version of chk-001-step-1140.dat
key: head_2_6_q, cosine: 1.000000
key: head_5_4_q, cosine: 1.000000
key: head_2_3_q, cosine: 1.000000
key: head_2_1_q, cosine: 1.000000
key: head_5_4_k, cosine: 1.000000
key: head_1_7_q, cosine: 1.000000
key: head_2_4_k, cosine: 1.000000
key: head_2_3_k, cosine: 1.000000
key: head_2_2_k, cosine: 1.000000
key: head_2_0_q, cosine: 1.000000
key: head_2_1_k, cosine: 1.000000
key: head_2_0_k, cosine: 1.000000
key: head_2_7_q, cosine: 1.000000
key: head_2_7_k, cosine: 1.000000
key: head_2_2_q, cosine: 1.000000
key: head_1_3_k, cosine: 1.000000
key: head_1_7_k, cosine: 1.000000
key: head_2_6_k, cosine: 1.000000
key: head_1_1_q, cosine: 1.000000
key: head_5_7_q, cosine: 1.000000
key: head_1_3_q, cosine: 1.000000
key: head_2_5_k, cosine: 1.000000
key: head_2_4_q, cosine: 1.000000
key: head_1_1_k, cosine: 1.000000
key: head_5_5_q, cosine: 1.000000
key: head_2_5_q, cosine: 1.000000
key: head_5_7_k, cosine: 1.000000
key: head_5_0_k, cosine: 1.000000
key: head_5_0_q, cosine: 1.000000
key: head_1_5_k, cosine: 1.000000
key: head_5_5_k, cosine: 1.000000
key: head_1_5_q, cosine: 1.000000
key: head_5_1_k, cosine: 1.000000
key: head_1_4_k, cosine: 1.000000
key: head_1_6_k, cosine: 1.000000
key: head_1_2_q, cosine: 1.000000
key: head_1_0_q, cosine: 1.000000
key: head_1_2_k, cosine: 1.000000
key: head_1_0_k, cosine: 1.000000
key: head_1_6_q, cosine: 1.000000
key: head_5_1_q, cosine: 1.000000
key: head_1_4_q, cosine: 1.000000
key: head_5_6_k, cosine: 1.000000
key: head_4_5_k, cosine: 1.000000
key: head_5_6_q, cosine: 1.000000
key: head_5_2_q, cosine: 1.000000
key: head_5_3_k, cosine: 1.000000
key: head_4_0_q, cosine: 1.000000
key: head_6_7_k, cosine: 1.000000
key: head_5_3_q, cosine: 1.000000
token_id: 266, token: "colum", head: 1, cosine: 1.000000
token_id: 266, token: "colum", head: 7, cosine: 1.000000
token_id: 266, token: "colum", head: 6, cosine: 1.000000
token_id: 266, token: "colum", head: 0, cosine: 1.000000
token_id: 266, token: "colum", head: 4, cosine: 1.000000
token_id: 266, token: "colum", head: 5, cosine: 1.000000
token_id: 266, token: "colum", head: 3, cosine: 1.000000
token_id: 266, token: "colum", head: 2, cosine: 1.000000
token_id: 518, token: "alField", head: 5, cosine: 1.000000
token_id: 518, token: "alField", head: 1, cosine: 1.000000
token_id: 518, token: "alField", head: 7, cosine: 1.000000
token_id: 518, token: "alField", head: 2, cosine: 0.999999
token_id: 518, token: "alField", head: 6, cosine: 0.999999
token_id: 518, token: "alField", head: 0, cosine: 0.999999
token_id: 489, token: "tor", head: 7, cosine: 0.999997
token_id: 417, token: ";", head: 1, cosine: 0.999996
token_id: 262, token: "9", head: 2, cosine: 0.999995
token_id: 262, token: "9", head: 7, cosine: 0.999995
token_id: 600, token: " to ", head: 5, cosine: 0.999994
token_id: 262, token: "9", head: 0, cosine: 0.999994
token_id: 463, token: "ag", head: 0, cosine: 0.999994
token_id: 362, token: "q", head: 5, cosine: 0.999994
token_id: 262, token: "9", head: 6, cosine: 0.999994
token_id: 482, token: "32", head: 5, cosine: 0.999994
token_id: 58, token: "fault", head: 5, cosine: 0.999994
token_id: 252, token: "rom", head: 6, cosine: 0.999993
token_id: 417, token: ";", head: 2, cosine: 0.999993
token_id: 54, token: "{\n            ", head: 5, cosine: 0.999992
token_id: 92, token: " {\n    ", head: 7, cosine: 0.999992
token_id: 600, token: " to ", head: 0, cosine: 0.999992
token_id: 482, token: "32", head: 0, cosine: 0.999991
token_id: 252, token: "rom", head: 5, cosine: 0.999991
token_id: 262, token: "9", head: 5, cosine: 0.999991
token_id: 252, token: "rom", head: 2, cosine: 0.999991
token_id: 54, token: "{\n            ", head: 7, cosine: 0.999991
token_id: 362, token: "q", head: 7, cosine: 0.999991
token_id: 428, token: "bas", head: 0, cosine: 0.999991
token_id: 538, token: " }", head: 0, cosine: 0.999990
token_id: 91, token: "::from_", head: 0, cosine: 0.999990
token_id: 538, token: " }", head: 2, cosine: 0.999990
token_id: 324, token: "fn ", head: 7, cosine: 0.999990
token_id: 417, token: ";", head: 0, cosine: 0.999989
token_id: 362, token: "q", head: 6, cosine: 0.999989
token_id: 463, token: "ag", head: 7, cosine: 0.999989
token_id: 252, token: "rom", head: 1, cosine: 0.999989
token_id: 541, token: "par", head: 7, cosine: 0.999989
token_id: 417, token: ";", head: 5, cosine: 0.999989
token_id: 376, token: ".clone()", head: 0, cosine: 0.999989
token_id: 120, token: ");\n        ", head: 5, cosine: 0.999989
token_id: 91, token: "::from_", head: 2, cosine: 0.999989
```

Maybe... something is happening!

1. qkv matrix of heads haven't changed much. The biggest change is 0.9216 of head_7_7_v, which isn't that big. Top 50 smallest changes are all qkv matrices.
2. Top 50 biggest changes in token embeddings are all happening at head-4. Maybe the other heads have converged and it's struggling to finetune head-4?

I have to give it another try!

```sh
cp chk-002-step-9420.dat model.dat

cargo run --release -- train --model model.dat --steps 3081 --dropout 0.1
cp model.dat chk-003-step-12500.dat

cargo run --release -- train --model model.dat --steps 2501 --dropout 0.1
cp model.dat chk-004-step-15000.dat

cargo run --release -- train --model model.dat --steps 2501 --dropout 0.1
cp model.dat chk-005-step-17500.dat

cargo run --release -- train --model model.dat --steps 2501 --dropout 0.1
cp model.dat chk-006-step-20000.dat

cargo run --release -- train --model model.dat --steps 2501 --dropout 0.1
cp model.dat chk-007-step-22500.dat

cargo run --release -- train --model model.dat --steps 2501 --dropout 0.1
cp model.dat chk-008-step-25000.dat
```

# 29. Comparing positional encodings 1

Below tuples are `(embedding_degree, num layers, num heads, positional encoding, steps)`

All the cases use the ascii tokenizer, and `python3 dummy_data/simple_sequence.py > dataset.txt` dataset.

You can see the entire training process below.

1. (144, 6, 6, none, 500)
  - loss: 0.7596
  - output: `a^^c@@g##e$$e%%e^^a@@g##b$$c%%e^^d@@c##f$$c%%e^^h@@c##c$$c%%e^^e@@d##c$$f%%d^^d@@a##e$$a%%b^^b@@b##e$`
2. (144, 7, 6, none, 700)
  - inserted a layer to #1 (layer-3), and trained 200 more steps
  - loss: 0.7472
  - output: `a@@e##f$$e%%e^^f@@e##g$$h%%d^^c@@e##c$$d%%c^^c@@e##d$$d%%e^^d@@e##f$$h%%e^^e@@c##c$$c%%d^^g@@f##g$$d%`
3. (144, 6, 6, none, 500)
  - loss: 0.7716
  - output: `a%%b^^g@@h##f$$e%%e^^b@@c##f$$c%%a^^b@@e##d$$c%%h^^a@@c##a$$c%%a^^f@@h##f$$f%%h^^e@@e##f$$f%%e^^a@@g#`
4. (144, 7, 6, none, 700)
  - inserted a layer to #3 (layer-3), and trained 200 more steps
  - loss: 0.7554
  - output: `a%%h^^h@@a##a$$f%%g^^f@@h##h$$g%%a^^b@@h##d$$f%%h^^e@@e##f$$b%%a^^f@@e##f$$g%%f^^f@@h##e$$g%%a^^f@@e#`
5. (144, 6, 6, none, 500)
  - loss: 0.7581
  - output: `a%%a^^g@@d##d$$f%%a^^g@@h##c$$g%%d^^f@@f##f$$g%%b^^f@@g##b$$g%%e^^d@@c##f$$d%%b^^e@@a##d$$f%%b^^g@@f#`
6. (144, 7, 6, none, 700)
  - inserted a layer to #5 (layer-3), and trained 200 more steps
  - loss: 0.7657
  - output: `a@@d##h$$a%%h^^b@@d##h$$a%%b^^c@@h##d$$c%%h^^c@@b##a$$a%%a^^b@@c##c$$d%%c^^d@@a##h$$c%%b^^c@@c##d$$b%`
7. (144, 6, 6, absolute, 500)
  - loss: 1.4594
  - output: `a@$@f%%%f^^^b%#%f^^$@##%a^^$@%#a$@h##g^$@b##%d^^$@e##f^$@@@@###%a^$$e#%c^$@@@e##%f^$$b#%a^^$@@@%##c$$`
8. (144, 7, 6, absolute, 700)
  - inserted a layer to #7 (layer-3), and trained 200 more steps
  - loss: 1.0524
  - output: `a%#c$$f%%a^^d@@c##g$$d%%a^^g@@e##b$$e%%h^^d@@g##f$$b%%a^^c@@d##g$$e%%a^^g@@e##f$$f%%b^^a@@b##%b^^g@@b`
9. (144, 6, 6, absolute, 500)
  - loss: 2.4040
  - output: `a^#@$$^#$#$$^##$$#$$@##$^^^@^$^$@$@#$@^#$^@$^@$^@$$###@^##$$@^#@$$@$#$$##@#@@$#@@$^#^@$#$@#@@$@#^@$#^`
10. (144, 7, 6, absolute, 700)
  - inserted a layer to #9 (layer-3), and trained 200 more steps
  - loss: 2.4025
  - output: `a%^@%#@%@^@#@^@@^^^^^@#%^^^^^%%#%^^@^%#^%#%%^^@@%#@@^#^##%@@#@##@^^##@^@#@@@#%##%#%@#^%@%@#%^^@@^^^^#`
11. (144, 6, 6, absolute, 500)
  - loss: 1.6935
  - output: `a^^#^##^^^^^^a@@$$g%%%%#^#^^^d@@@$@a%%#^##^a@@@d##%^#^c@$@g^^%^^^d@$$dg$$g%%%%##%#a@$@$d%###^#h@@$$g%`
12. (144, 7, 6, absolute, 700)
  - inserted a layer to #11 (layer-3), and trained 200 more steps
  - loss: 1.4461
  - output: `a#^^a$$h%%b^b@@h%%b^b@@c%#^a@@c%%b^^^^^d@@@a#%^^^d@@$a%%%#^^d@@$b#%%%#b@@$b%%%#^^b@@$c%%%%b@$$b%%%^h@`

Disabling positional-encoding made the models better! Models without positional encoding all learnt the patterns of the special characters (`@#$%^`), but failed to learn the alphabets.

The best absolute-pe model is worse than the worst none-pe model.

```sh
cargo build --release
cp target/release/femto-gpt gpt
echo "---------"
echo "init exp1"
./gpt init --model exp1.dat --tokenizer ascii --positional-encoding none --embedding-degree 144 --num-layers 6 --num-heads 6 1> mute
./gpt train --model exp1.dat --dropout 0.1 --steps 500 1> mute
echo loss
./gpt loss --model exp1.dat --limit 1
./gpt infer --model exp1.dat "a"

echo "exp1-ext"
./gpt insert-layer --input exp1.dat --output exp1-ext.dat --insert-at 3
./gpt train --model exp1-ext.dat --dropout 0.1 --steps 200 1> mute
echo loss
./gpt loss --model exp1-ext.dat --limit 1
./gpt infer --model exp1-ext.dat "a"

echo "---------"
echo "init exp2"
./gpt init --model exp2.dat --tokenizer ascii --positional-encoding none --embedding-degree 144 --num-layers 6 --num-heads 6 1> mute
./gpt train --model exp2.dat --dropout 0.1 --steps 500 1> mute
echo loss
./gpt loss --model exp2.dat --limit 1
./gpt infer --model exp2.dat "a"

echo "exp2-ext"
./gpt insert-layer --input exp2.dat --output exp2-ext.dat --insert-at 3
./gpt train --model exp2-ext.dat --dropout 0.1 --steps 200 1> mute
echo loss
./gpt loss --model exp2-ext.dat --limit 1
./gpt infer --model exp2-ext.dat "a"

echo "---------"
echo "init exp3"
./gpt init --model exp3.dat --tokenizer ascii --positional-encoding none --embedding-degree 144 --num-layers 6 --num-heads 6 1> mute
./gpt train --model exp3.dat --dropout 0.1 --steps 500 1> mute
echo loss
./gpt loss --model exp3.dat --limit 1
./gpt infer --model exp3.dat "a"

echo "exp3-ext"
./gpt insert-layer --input exp3.dat --output exp3-ext.dat --insert-at 3
./gpt train --model exp3-ext.dat --dropout 0.1 --steps 200 1> mute
echo loss
./gpt loss --model exp3-ext.dat --limit 1
./gpt infer --model exp3-ext.dat "a"

echo "---------"
echo "init exp4"
./gpt init --model exp4.dat --tokenizer ascii --positional-encoding absolute --embedding-degree 144 --num-layers 6 --num-heads 6 1> mute
./gpt train --model exp4.dat --dropout 0.1 --steps 500 1> mute
echo loss
./gpt loss --model exp4.dat --limit 1
./gpt infer --model exp4.dat "a"

echo "exp4-ext"
./gpt insert-layer --input exp4.dat --output exp4-ext.dat --insert-at 3
./gpt train --model exp4-ext.dat --dropout 0.1 --steps 200 1> mute
echo loss
./gpt loss --model exp4-ext.dat --limit 1
./gpt infer --model exp4-ext.dat "a"

echo "---------"
echo "init exp5"
./gpt init --model exp5.dat --tokenizer ascii --positional-encoding absolute --embedding-degree 144 --num-layers 6 --num-heads 6 1> mute
./gpt train --model exp5.dat --dropout 0.1 --steps 500 1> mute
echo loss
./gpt loss --model exp5.dat --limit 1
./gpt infer --model exp5.dat "a"

echo "exp5-ext"
./gpt insert-layer --input exp5.dat --output exp5-ext.dat --insert-at 3
./gpt train --model exp5-ext.dat --dropout 0.1 --steps 200 1> mute
echo loss
./gpt loss --model exp5-ext.dat --limit 1
./gpt infer --model exp5-ext.dat "a"

echo "---------"
echo "init exp6"
./gpt init --model exp6.dat --tokenizer ascii --positional-encoding absolute --embedding-degree 144 --num-layers 6 --num-heads 6 1> mute
./gpt train --model exp6.dat --dropout 0.1 --steps 500 1> mute
echo loss
./gpt loss --model exp6.dat --limit 1
./gpt infer --model exp6.dat "a"

echo "exp6-ext"
./gpt insert-layer --input exp6.dat --output exp6-ext.dat --insert-at 3
./gpt train --model exp6-ext.dat --dropout 0.1 --steps 200 1> mute
echo loss
./gpt loss --model exp6-ext.dat --limit 1
./gpt infer --model exp6-ext.dat "a"
```

# 30. Comparing positional encoding 2

It's like #29, but with `python3 dummy_data/simple_dummy2.py > dataset.txt`.

I prompted them with `"acbafbe3"`, whose answer is `"gih;\n"`

1. (144, 6, 6, none, 500)
  - loss: 1.8017
  - output: `acbafbe3gkh;\n`
2. (144, 7, 6, none, 700)
  - inserted a layer to #1 (layer-3), and trained 200 more steps
  - loss: 1.7795
  - output: `acbafbe3iih;\n`
3. (144, 6, 6, none, 500)
  - loss: 1.7824
  - output: `acbafbe3kkk;\n`
4. (144, 7, 6, none, 700)
  - inserted a layer to #3 (layer-3), and trained 200 more steps
  - loss: 1.7534
  - output: `acbafbe3hkh;\n`
5. (144, 6, 6, none, 500)
  - loss: 1.7750
  - output: `acbafbe3gh;\n`
6. (144, 7, 6, none, 700)
  - inserted a layer to #5 (layer-3), and trained 200 more steps
  - loss: 1.7538
  - output: `acbafbe3khi;\n`
7. (144, 6, 6, absolute, 500)
  - loss: 2.7156
  - output: `acbafbe3aecea...` (followed be a sequence of meaningless alphabets)
8. (144, 7, 6, absolute, 700)
  - inserted a layer to #7 (layer-3), and trained 200 more steps
  - loss: 2.7092
  - output: `acbafbe3cfafb...` (followed by a sequence of meaningless alphabets)
9. (144, 6, 6, absolute, 500)
  - loss: 2.2187
  - output: `acbafbe3;\n`
10. (144, 7, 6, absolute, 700)
  - inserted a layer to #9 (layer-3), and trained 200 more steps
  - loss: 2.2058
  - output: `acbafbe3\n`
11. (144, 6, 6, absolute, 500)
  - loss: 2.7286
  - output: `acbafbe3fdfff...` (followed by a sequence of meaningless alphabets)
12. (144, 7, 6, absolute, 700)
  - inserted a layer to #11 (layer-3), and trained 200 more steps
  - loss: 2.7123
  - output: `acbafbe3affcf...` (followed by a sequence of meaningless alphabets)

Removing PE is always better...

Also, inserting a layer and training 200 more steps always lowers the loss.

# 31. Training an English model 1 (fail)

- tokenizer: bpe (1024 tokens, 64 reserved tokens), case sensitive
- positional encoding: none
- embedding degree: 256, num layers: 8, num heads 8 (???M params)
- dropout: 0.1, base_lr: 0.001, min_lr: 0.00001, warmup_steps: 100, decay_steps: 50000
- step: ????, loss: ?.????, elapsed: ??m ??s (AWS EC2 m5.4xlarge: 16vCPU, 64GB RAM)
- data: downloaded some files from [huggingface](https://huggingface.co/datasets/wikimedia/wikipedia/tree/main/20231101.en) and postprocessed it
  - I picked 4255 documents which contains keyword `"rogramming language"` from the dataset and concat them with delimiter `"<|endofdocument|>"`

```sh
cargo run --release -- train-bpe --reserve-tokens 64 --vocab-size 1024
cargo run --release -- count-tokens > count-tokens.json

# check the tokens here!

# step 1: initialize 4 models and train each for 1000 steps

cargo run --release -- init --model exp1.dat --tokenizer bpe --positional-encoding none --num-tokens 80 --embedding-degree 256 --num-layers 8 --num-heads 8
cargo run --release -- train --model exp1.dat --dropout 0.1 --steps 1001
cp exp1.dat exp1-chk001.dat

cargo run --release -- init --model exp2.dat --tokenizer bpe --positional-encoding none --num-tokens 80 --embedding-degree 256 --num-layers 8 --num-heads 8
cargo run --release -- train --model exp2.dat --dropout 0.1 --steps 1001
cp exp2.dat exp2-chk001.dat

cargo run --release -- init --model exp3.dat --tokenizer bpe --positional-encoding none --num-tokens 80 --embedding-degree 256 --num-layers 8 --num-heads 8
cargo run --release -- train --model exp3.dat --dropout 0.1 --steps 1001
cp exp3.dat exp3-chk001.dat

cargo run --release -- init --model exp4.dat --tokenizer bpe --positional-encoding none --num-tokens 80 --embedding-degree 256 --num-layers 8 --num-heads 8
cargo run --release -- train --model exp4.dat --dropout 0.1 --steps 1001
cp exp4.dat exp4-chk001.dat

# step 2: train each for 5000 more steps

cargo run --release -- train --model exp1.dat --dropout 0.1 --steps 5001
cp exp1.dat exp1-chk002.dat

cargo run --release -- train --model exp2.dat --dropout 0.1 --steps 5001
cp exp2.dat exp2-chk002.dat

cargo run --release -- train --model exp3.dat --dropout 0.1 --steps 5001
cp exp3.dat exp3-chk002.dat

cargo run --release -- train --model exp4.dat --dropout 0.1 --steps 5001
cp exp4.dat exp4-chk002.dat

# step 3: insert a layer to each model and train 3000 steps

cargo run --release -- insert-layer --input exp1.dat --output exp1-ext1.dat --insert-at 6
cargo run --release -- train --model exp1-ext1.dat --dropout 0.1 --steps 3001
cp exp1-ext1.dat exp1-ext1-chk001.dat

cargo run --release -- insert-layer --input exp1.dat --output exp1-ext2.dat --insert-at 6
cargo run --release -- train --model exp1-ext2.dat --dropout 0.1 --steps 3001
cp exp1-ext2.dat exp1-ext2-chk001.dat

cargo run --release -- insert-layer --input exp2.dat --output exp2-ext1.dat --insert-at 6
cargo run --release -- train --model exp2-ext1.dat --dropout 0.1 --steps 3001
cp exp2-ext1.dat exp2-ext1-chk001.dat

cargo run --release -- insert-layer --input exp2.dat --output exp2-ext2.dat --insert-at 6
cargo run --release -- train --model exp2-ext2.dat --dropout 0.1 --steps 3001
cp exp2-ext2.dat exp2-ext2-chk001.dat

cargo run --release -- insert-layer --input exp3.dat --output exp3-ext1.dat --insert-at 6
cargo run --release -- train --model exp3-ext1.dat --dropout 0.1 --steps 3001
cp exp3-ext1.dat exp3-ext1-chk001.dat

cargo run --release -- insert-layer --input exp3.dat --output exp3-ext2.dat --insert-at 6
cargo run --release -- train --model exp3-ext2.dat --dropout 0.1 --steps 3001
cp exp3-ext2.dat exp3-ext2-chk001.dat

cargo run --release -- insert-layer --input exp4.dat --output exp4-ext1.dat --insert-at 6
cargo run --release -- train --model exp4-ext1.dat --dropout 0.1 --steps 3001
cp exp4-ext1.dat exp4-ext1-chk001.dat

cargo run --release -- insert-layer --input exp4.dat --output exp4-ext2.dat --insert-at 6
cargo run --release -- train --model exp4-ext2.dat --dropout 0.1 --steps 3001
cp exp4-ext2.dat exp4-ext2-chk001.dat

# step 4: further train models from step 3

cargo run --release -- train --model exp1-ext1.dat --dropout 0.1 --steps 3001
cp exp1-ext1.dat exp1-ext1-chk002.dat

cargo run --release -- train --model exp1-ext2.dat --dropout 0.1 --steps 3001
cp exp1-ext2.dat exp1-ext2-chk002.dat

cargo run --release -- train --model exp2-ext1.dat --dropout 0.1 --steps 3001
cp exp2-ext1.dat exp2-ext1-chk002.dat

cargo run --release -- train --model exp2-ext2.dat --dropout 0.1 --steps 3001
cp exp2-ext2.dat exp2-ext2-chk002.dat

cargo run --release -- train --model exp3-ext1.dat --dropout 0.1 --steps 3001
cp exp3-ext1.dat exp3-ext1-chk002.dat

cargo run --release -- train --model exp3-ext2.dat --dropout 0.1 --steps 3001
cp exp3-ext2.dat exp3-ext2-chk002.dat

cargo run --release -- train --model exp4-ext1.dat --dropout 0.1 --steps 3001
cp exp4-ext1.dat exp4-ext1-chk002.dat

cargo run --release -- train --model exp4-ext2.dat --dropout 0.1 --steps 3001
cp exp4-ext2.dat exp4-ext2-chk002.dat
```

All the 24 checkpoints have very similar loss (around 6.4). It seems like nothing's getting better even though I trained them for 12000 steps.

comparison between exp1-chk001 and exp1-chk002

```
exp1-chk001.dat is parent of exp1-chk002.dat.
exp1-chk002.dat is 5001-steps further trained version of exp1-chk001.dat
key: head_norm_coeff, cosine: 0.749735
key: atten_norm_7_coeff, cosine: 0.906037
key: feedforward2_7_weights, cosine: 0.908405
key: feedforward2_7_bias, cosine: 0.933107
key: head_norm_bias, cosine: 0.939939
key: feedforward1_7_weights, cosine: 0.952833
key: proj_7_weights, cosine: 0.956272
key: head_map_weights, cosine: 0.957193
key: atten_norm_7_bias, cosine: 0.962747
key: feedforward1_7_bias, cosine: 0.979152
key: norm_7_bias, cosine: 0.982568
key: head_map_bias, cosine: 0.987208
key: proj_7_bias, cosine: 0.988728
key: head_7_5_v, cosine: 0.990407
key: head_7_7_v, cosine: 0.991554
token_id: 756, token: "qu", head: 5, cosine: 1.000000
token_id: 1057, token: "engine", head: 0, cosine: 1.000000
token_id: 756, token: "qu", head: 0, cosine: 1.000000
token_id: 401, token: "projec", head: 0, cosine: 1.000000
token_id: 517, token: "mar", head: 0, cosine: 1.000000
token_id: 77, token: "are", head: 5, cosine: 1.000000
token_id: 255, token: "se", head: 5, cosine: 1.000000
token_id: 793, token: "compil", head: 5, cosine: 1.000000
token_id: 595, token: "provi", head: 0, cosine: 1.000000
token_id: 560, token: "bra", head: 0, cosine: 1.000000
token_id: 491, token: "g", head: 5, cosine: 1.000000
token_id: 793, token: "compil", head: 0, cosine: 1.000000
token_id: 900, token: "lar", head: 0, cosine: 1.000000
token_id: 448, token: "inte", head: 5, cosine: 1.000000
token_id: 565, token: "ver", head: 5, cosine: 1.000000
```

This is very, very strange. I mean, it's not what I've expected. For 5000 steps, token embeddings have not changed at all (99.9999%). `head_norm` and layer-7 changed, but it's much smaller than what I've seen in #28. There are 2 possible explanations.

1. The model is almost complete. When `head_norm` and layer-7 converge, the model will become smart and generate valid outputs.
2. The size of the model (6.8M parameters) is too small to understand English. It'll converge, but will never generate something useful.

comparison between exp1-ext1-chk001 and exp1-ext1-chk002

```
exp1-ext1-chk001.dat is parent of exp1-ext1-chk002.dat.
exp1-ext1-chk002.dat is 3001-steps further trained version of exp1-ext1-chk001.dat
key: atten_norm_8_coeff, cosine: 0.932131
key: proj_8_bias, cosine: 0.933034
key: atten_norm_8_bias, cosine: 0.956440
key: head_norm_coeff, cosine: 0.959888
key: feedforward2_8_bias, cosine: 0.975647
key: feedforward2_8_weights, cosine: 0.976924
key: norm_6_bias, cosine: 0.979071
key: feedforward1_8_bias, cosine: 0.979866
key: proj_6_bias, cosine: 0.984398
key: feedforward2_7_bias, cosine: 0.985090
key: atten_norm_7_bias, cosine: 0.986334
key: head_norm_bias, cosine: 0.987542
key: proj_7_bias, cosine: 0.988367
key: norm_8_bias, cosine: 0.989575
key: feedforward1_8_weights, cosine: 0.989715
token_id: 36, token: " ", head: 0, cosine: 1.000000
token_id: 597, token: "-", head: 0, cosine: 1.000000
token_id: 36, token: " ", head: 5, cosine: 1.000000
token_id: 10, token: "s", head: 0, cosine: 1.000000
token_id: 776, token: "vi", head: 0, cosine: 1.000000
token_id: 458, token: "a", head: 0, cosine: 1.000000
token_id: 178, token: "s ", head: 0, cosine: 1.000000
token_id: 36, token: " ", head: 3, cosine: 1.000000
token_id: 404, token: "ut", head: 0, cosine: 1.000000
token_id: 36, token: " ", head: 1, cosine: 1.000000
token_id: 323, token: "um", head: 0, cosine: 1.000000
token_id: 229, token: "V", head: 0, cosine: 1.000000
token_id: 576, token: "al", head: 0, cosine: 1.000000
token_id: 590, token: "m", head: 0, cosine: 1.000000
token_id: 899, token: "P", head: 0, cosine: 1.000000
```

well...

1. It hasn't learnt that much for 3000 steps. Again, there's no differences in word embeddings.
2. The inserted layer is 6, but the changes are mostly in layer 8 (the last layer).

# 32. Training an English model 2

- tokenizer: bpe (256 tokens, 64 preserved)
  - I wanted (256 tokens + 32 preserved), but I accidentally added preserved tokens twice.
- positional encoding: none
- embedding degree: 144, num layers: 6, num heads 6 (1.5M params)
- dropout: 0.0, base_lr: 0.001, min_lr: 0.00001, warmup_steps: 100, decay_steps: 50000
- step: ????, loss: ?.????, elapsed: ??m ??s (apple M3 pro)
  - step 1 ~ step 45: loss 5.7
  - step 46 ~ step 440: loss 5.7 -> loss 3.8
  - step 441 ~ step 580: loss 3.8
- data: `dataset.txt` from keyvank's repository (the shakespeare)

Something's happening..!!

So I stopped the training at step 580, inserted a layer at 4, and continued training.

- step 1 ~ step 62: loss 5.2 -> loss 3.8
- step 63 ~ step 192: loss 3.8 -> loss 3.7
- step 193 ~ step 280: loss 3.7
- step 281 ~ step 432: loss 3.7 -> loss 3.6

Good! Inserting a layer improves the model slightly. How about adding another?

I inserted a layer at 4, again, and continued training.

- step 1 ~ step 92: loss 7.58 -> loss 3.7
- step 93 ~ step 200: loss 3.7 -> loss 3.6
- step 201 ~ step 920: loss 3.6 -> loss 3.5

While training, I made some checkpoints and compared them. I found that,

1. Unlike #31, token embeddings were consistently changing (cosine 0.7 ~ 0.9).
2. `head_k` and `head_q` of the newly inserted layer are the most changed layers.
3. `head_v` of the newly inserted layer didn't change that much. Why?

---

The result isn't that bad. I prompted `BRUTUS:` 3 times and below are the generated outputs. I didn't cherry-pick the output.

```
BRUTUS: I ROKE:
What I spriends that you come of thy the creminking.

DULABETHA:
Thouly, thought to my howersay, prownies
That I be so bost morly.

KING RI
ThAMARDWARD IV:
Mardo not be
```

```
BRUTUS: CISI:
But say, I was thou well'll comest you.

CORILOLLANIO:
What you, well, whendost ond air,
I have sirtrangs and my sondake the sodone bence,
That partimpt be dave the brinces with like me of th
```

```
BRUTUS: VINCENTIO:
Ory, I was the granger!
This ave thee a mopright will you
That a is gotheet of such theet ber, my heaks aveir,
And bean is is the prower so thin thy hean osties,
Which me in this
```

It is much better than previous large models (#28, #31), but isn't good enough to mark it "success". Maybe I should further-train this model in later experiments!

But why? Why was #31 failure and #32 successful? Some differences are

1. #31 has much larger dataset (1MiB vs 35MiB)
2. #31 has much larger vocab size (1024 vs 256)
3. #31's model is larger.
  - #31 started with (embedding degree 256, num layers 8, num heads 8) and inserted a layer later.
  - #32 started with (embedding degree 144, num layers 6, num heads 6) and inserted 2 layers later.
4. #31's dropout is 0.1, and #32's is 0.
  - I didn't mean to test dropout, but I forgot to set the dropout of #32.

Why... maybe because a larger model requires much more steps to train. But #31's loss started at 6.4 and stayed there for 12000 steps! #32's loss started at 5.7 and began decreasing in less than 100 steps.

# 33. A lesson

I was doing a small experiment with `dummy_data/addition_dummy.py`. The model was initialized with (embedding degree 144, num layers 6, num heads 6). It converged to loss 1.8 very quickly, so I inserted a layer and continued training. I did this a few times. When a layer was inserted, the loss went up to 2.5 ~ 3.5 and quickly went down to 1.3 ~ 1.5. But there was one time, when a layer was inserted, the loss went up to 5.5 and never went below 2.5. Inserting a layer messed up a model!

The lesson is that, inserting a layer might mess up a model, so we always have to create a checkpoint before inserting a layer and checkout the checkpoint if the insertion goes wrong.

# 34. Training a Rust coder 7 (failed)

- tokenizer: bpe (1280 tokens + 64 reserved tokens), case sensitive
- positional encoding: none
- embedding degree: 324, num layers: 9, num heads 9 (12.2M params)
- num tokens: 128
- dropout: 0.1, base_lr: 0.001, min_lr: 0.00001, warmup_steps: 100, decay_steps: 50000
- step: 2697, loss: around 6.6, elapsed: ??m ??s (apple M3 pro)
  - each step takes 6 ~ 7 seconds
- data: see `exp34.nu`

I'm training 2 models in parallel. Both models' loss started at 7.2.

Both reached 6.6 quickly (at around 100 steps), and got stuck at 6.6 (currently both at step 2697).

# 35. Training a Rust coder 8 (failed)

- tokenizer: bpe (512 tokens + 64 reserved tokens), case sensitive
- positional encoding: none
- embedding degree: 96, num layers: 6, num heads 4 (780K params)
- num tokens: 128
- dropout: 0.1, base_lr: 0.001, min_lr: 0.00001, warmup_steps: 100, decay_steps: 50000
- steps: 730, loss: 4.4256, elapsed: 9m 20s (apple M3 pro)
  - each step takes 600 ~ 700 ms
- data: same as #34

The model's loss started at 6.35, reached 4.4 quickly (at around 540 steps), and got stuck at 4.4 (currently at step 730).

#34 and #35 are both bad, but #35 is bettern than #34. #34 has learnt something for 540 steps, while #35 learnt only for 100 steps. Again, it's proven that making a model bigger doesn't always make the model smarter.

# 36. Training a Rust coder 9 (failed)

- tokenizer: bpe (512 tokens + 64 reserved tokens), case sensitive
  - exactly same as #35
- positional encoding: none
- embedding degree: 144, num layers: 6, num heads 6 (1.6M params)
- num tokens: 128
- dropout: 0.1, base_lr: 0.001, min_lr: 0.00001, warmup_steps: 100, decay_steps: 50000
- steps: 780, loss: 4.1666, elapsed: 19m 47s (apple M3 pro)
  - each step takes 1200 ~ 1400 ms
- data: same as #34

The model's loss started at 6.35, reached 4.2 quickly (at around 520 steps), and got stuck at 4.2 (currently at step 780).

Maybe... #34's failure was due to the tokenizer being too big, not due to the number of layers and heads.

# 37. Training a Rust coder 10 (failed)

- tokenizer: bpe (1280 tokens + 64 reserved tokens), case sensitive
- positional encoding: none
- embedding degree: 96, num layers: 6, num heads 4 (928K params)
- num tokens: 128
- dropout: 0.1, base_lr: 0.001, min_lr: 0.00001, warmup_steps: 100, decay_steps: 50000
- steps: 1020, loss: 4.6270, elapsed: 20m 24s (apple M3 pro)
  - each step takes 600 ~ 700 ms
- data: same as #34

The model's loss started at 7.2, reached 4.7 quickly (at around 860 steps), and got stuck at 4.7 (currently at step 1020).

My assumption at #36 is wrong. #35 and #37 are the same except that #37 has a bigger tokenizer. According to my assumption, #37 must be much worse than #35 and #34 (because #34 has the same tokenizer but bigger heads and layers). But #37 is better than #34 and almost as good as #35.

My next guess is that #34 is just too big and the optimizer kept throwing meaningless gradients... But why? There are 70B models in the wild and #34 is only 12M. I have to start a training with a small model, then incrementally add layers and heads to the model. The problem is that I have `insert-layer` api, but I don't have `insert-head` api.

# 38. Training a Rust coder 11

- tokenizer: bpe (600 tokens + 40 reserved tokens), case sensitive
- positional encoding: none
- embedding degree: 162, num layers: 6, num heads 6 (2.1M params)
  - initially embedding degree: 162, num layers: 4, num heads 6 (1.4M params)
  - then became embedding degree: 162, num layers: 5, num heads 6 (1.7M params)
- num tokens: 128
- dropout: 0.1, base_lr: 0.001, min_lr: 0.00001, warmup_steps: 100, decay_steps: 50000
- steps: ????, loss: ?.????, elapsed: ??m ??s (apple M3 pro)
- data: same as #34

According to #37, I have to implement `insert-head` before doing #38. But I'm too lazy to do that. Instead, I began the training with many-head less-layer model. I'll add layers to the models.

I'm training 2 models at the same time. They both started at loss 6.46. I'm currently at step 186 and the losses are 5.31. So far so good!

I'm at step 248 and the losses are 4.75... great!
I'm at step 279 and the losses are 4.63... nice!
I'm at step 341 and the losses are 4.32... good!

I think it's a good timing to insert layers. You can find the whole process at `exp38.nu`.

- model1-ext1
  - loss 6.11 -> 4.65 (for the first 31 steps)
  - loss 4.65 -> 4.44 (for the next 31 steps)
  - loss 4.44 -> 4.25 (for the next 31 steps)
- model1-ext2
  - loss 5.96 -> 4.66 (for the first 31 steps)
  - loss 4.66 -> 4.37 (for the next 31 steps)
  - loss 4.37 -> 4.26 (for the next 31 steps)
- model2-ext1
  - loss 7.17 -> 5.00 (for the first 31 steps)
  - loss 5.00 -> 4.55 (for the next 31 steps)
  - loss 4.55 -> 4.33 (for the next 31 steps)
- model2-ext2
  - loss 6.43 -> 4.80 (for the first 31 steps)
  - loss 4.80 -> 4.41 (for the next 31 steps)
  - loss 4.41 -> 4.29 (for the next 31 steps)

So far so good.

I have waited a few more hours (1500 more steps), and the losses are now 3.4 ~ 3.6.

Before I went to bed, I thought there would be deviations between the models. I thought some models would have significantly smaller losses than the others. But they all have similar losses. Let me insert a layer and train a few more steps.

- model1-ext1-ext1
  - loss 7.87 -> 4.77 (for the first 31 steps)
  - loss 4.77 -> 3.76 (for the next 31 steps)
  - loss 3.76 -> 3.74 (for the next 31 steps)
  - loss 3.74 -> 3.68 (for the next 31 steps)
- model1-ext1-ext2
  - loss 7.06 -> 4.51 (for the first 31 steps)
  - loss 4.51 -> 3.58 (for the next 31 steps)
  - loss 3.58 -> 3.67 (for the next 31 steps)
  - loss 3.67 -> 3.70 (for the next 31 steps)
- model1-ext2-ext1
  - loss 7.09 -> 4.72 (for the first 31 steps)
  - loss 4.72 -> 3.82 (for the next 31 steps)
  - loss 3.82 -> 3.76 (for the next 31 steps)
  - loss 3.76 -> 3.65 (for the next 31 steps)
- model1-ext2-ext2
  - loss 8.17 -> 4.90 (for the first 31 steps)
  - loss 4.90 -> 3.87 (for the next 31 steps)
  - loss 3.87 -> 3.88 (for the next 31 steps)
  - loss 3.88 -> 3.67 (for the next 31 steps)
- model2-ext1-ext1
  - loss 8.40 -> 5.10 (for the first 31 steps)
  - loss 5.10 -> 4.14 (for the next 31 steps)
  - loss 4.14 -> 3.62 (for the next 31 steps)
  - loss 3.62 -> 3.61 (for the next 31 steps)
- model2-ext1-ext2
  - loss 8.14 -> 5.05 (for the first 31 steps)
  - loss 5.05 -> 3.93 (for the next 31 steps)
  - loss 3.93 -> 3.81 (for the next 31 steps)
  - loss 3.81 -> 3.71 (for the next 31 steps)
- model2-ext2-ext1
  - loss 8.42 -> 4.94 (for the first 31 steps)
  - loss 4.94 -> 3.87 (for the next 31 steps)
  - loss 3.87 -> 3.71 (for the next 31 steps)
  - loss 3.71 -> 3.67 (for the next 31 steps)
- model2-ext2-ext2
  - loss 6.49 -> 4.37 (for the first 31 steps)
  - loss 4.37 -> 3.73 (for the next 31 steps)
  - loss 3.73 -> 3.66 (for the next 31 steps)
  - loss 3.66 -> 3.59 (for the next 31 steps)

So far so good.

I have trained 248 more steps (total 372 steps per model). The losses are still 3.4 ~ 3.6, so I was wondering if they're learning something. I want to compare the models. Let's do some inferences.

1. prompt: "pub(crate) fn add_numbers("

model1-ext1-ext1

```rs
pub(crate) fn add_numbers(&mut self) {
        self.tcx.sess.skip(errors::Muture::Scope) {
            let err = if tcx.sess.file_oper().emit() {
                return false;
                return Some(insert();
                self.ren_diagnostic.emp!(
                self.dyn() {
                    }
                return self.requir(sulatch.clone(path, s),
                if name.path.span, name);
                }
                }
            }
            }
```

model1-ext1-ext2

```rs
pub(crate) fn add_numbers() {
        let mut supplicit_scope = &'_psp_pty_uns);
        let mut fig = &mut self,
        lobefi: fule: uile,
        "spplicability"bix"debug"fund".is_d");
        let mut spplit_ability = sc"debug"debug"_unwrap_target_optability"debug".un
```

model1-ext2-ext1

```rs
pub(crate) fn add_numbers() {
        self.madd();
        let sposs = &[12].block.0].make(_expr(subaseq, byma, &[b, basmspty, bbinder, subbinder, ");
    backenvarch, r(ax),
        scope, sma.casic_build.get_h_or.
```

model1-ext2-ext2

```rs
pub(crate) fn add_numbers().
    }
}

/// Searly: In printrange ard for `. The `
    /// to the protherwarten norit is not supilder pointer with a f that that poncan be used bit is `.
    pub fn print_inference_infer(
    #[inline]
    pub fn dcx: u32,
    /// Anapshould to
```

model2-ext1-ext1

```rs
pub(crate) fn add_numbers(),
        fic_und::fix, 0,
        indextrace: usize: Set,
        f: Floc: Function,
    ) -> T {
        debug!(self.curce::from_susizeq!(fxHashother),
        has_unsafe {
        }", 0)),
        name(),
            Ascrimport_stric!("maillabelp 3"),
        ff
```

model2-ext1-ext2

```rs
pub(crate) fn add_numbers(
        &mut self,
        asts: u32,
        fix: T: If: u32,
        T: u32,
        s,
        f32,
        bi64,
        data: i32,
        ct: u32,
        x: u32,
        rs: i32,
        fmonx: u32,
        _u32,
        mt::Si64,
        v,
        bi64,
        c_u32,
        v: u32,
        _wnsafe: 32
```

model2-ext2-ext1

```rs
pub(crate) fn add_numbers() where_endable GenericArgumerichecauses for `
    for `"uns` with "malready sted uns"debug")"f",
    "riting listed for we don""" => "rinst -mas" 1", " => ""und" ", "),
            _u86",
            "rh64"" => "
```

model2-ext2-ext2

```rs
pub(crate) fn add_numbers().into();
    let (self, c, self_ty, c) = self.tcx.type_ty();
    let ty = &[1];
    let ty = ffcx.type_ty();
    let bcx.type_fxt_ty = [(self.tcx = Typ_try_ptransm_valize_type_fx.ty.ase_drop_transcrip(f.is_unwrap_ore_type_type_ty
```

2. prompt: "/// This is a"

model1-ext1-ext1

`````rs
/// This is ailer.
            // I as only one does not a ````, ```
    /// ``` `.
    /// `````
    /// ```````.
    ///
    /// `````````````````.
    ///
    /// ``` ``````.
    ///
    /// `
    /// ``]` and ````.
    ///
    /// ```]` or `.
    ///
    /// ``
`````

model1-ext1-ext2

```rs
/// This is a
    /// ity bit withubdiagnostics for sulturing licure.
    pub(self) -> Option<Buile {
        #[super_parset_unwrap()]
    pub fn from_path(self, g: u32) -> bool {
        #[diagnostic)]
    //rustcfn_rustccc_target_print(suggc_mappenggct]
    #[sug!(
```

model1-ext2-ext1

```rs
/// This is aile only a baske the faill only to fulead mutilds a st mutains in it will be ding that export to the st spanichicht only backing it is the poss to alariated to the set that any cappe livistinge the sing the comp
```

model1-ext2-ext2

```rs
/// This is a
    /// in the ```````````````
    /// ```````````````````````
    /// `````
    /// ``````````````````/////////////////////////////////
```

model2-ext1-ext1

```rs
/// This is ails and hould not execking by for like are not ways in only result to init't with only othere in pro and prevalso we it not ne, the losures the bit we that mothere like it we it livisit and we coversions only doest ne
```

model2-ext1-ext2

```rs
/// This is awill (x, f ax Ablockswr gcts. more `&fi An`-s``-by`.
    ///
    /// Ataddy`-gc`"`.
    ///
    /// ``- or `
    /// #### and `-in not prefore safe worknowrrequostatic`can's
```

model2-ext2-ext1

```rs
/// This is aill conly init prevalary only willt are mits balignore the functions and the al d
        // d.
    /// backetime bit will as a aluillh as this function anonly recurrentailderive warr.
    /// unking in the mard be in as uary s
```

model2-ext2-ext2

```rs
/// This is aits in as the fmacros inke).
    ///
    /// ################![$line ------------------------------------------------------------------------------------------------------------------------------
```

3. prompt: "use crate::"

model1-ext1-ext1

```rs
use crate::relages::s;
use crate::traits::traits::{
        for def_id.def_id();

    cx().span_impl_crate_type_impl_def_id();
    #[lage_trait_ref(def_id]
    tcx item_impl]
    let traits = self.tcx;
        let ty = trait_items = tcx.is_unsion_trait_reg.is_cluggest_item(def_id);
    }

    /// Oution implif_trait item in vari
```

model1-ext1-ext2

```rs
use crate::mmir::mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm
```

model1-ext2-ext1

```rs
use crate::{printrinits, binput));

    let mut v = "eq";
    let mut gccc.srmailting.capplaybrmaintiplevarch(ardsccc.lec_undys")]
    subalreadytest_stction_valiatedsatompsatompty.d();
    // This
```

model1-ext2-ext2

```rs
use crate::ree::Deredicate::Errowed, Sy::Sighierce::Info::PlaceEq, ErrowKind::StrrowIndexVec::new(Layout)) => {}),
    };


use crate::Index::Datalange;

use crate::Mutput<'tcx, Bast_ty;
















use crate
```

model2-ext1-ext1

```rs
use crate::{Ent::From::SeSubdiagnostic(sym::rustcal::Node::errors::{{self, err};
use rustc_errors::{self, errors::{self, err, Labelement};
use rustc_span::errors::{self, attr, TyCtxt, Span, err, Striddle::span, Labelper::error, err};
use rustc_span, attrs::{
    {
    Foldle::{
        span:
```

model2-ext1-ext2

```rs
use crate::implicite;
use crate::fmt::malization::mm::{Clobalign::Inttractt::traits::Enc::Met::spec::FlayoutAss::Ass::ScureSub(InferOblanit::FltLayoutAssAnMut { Cx, Scation: Ty::Inter::Branyncas
```

model2-ext2-ext1

```rs
use crate::ffeatures::{{}
                };

    use crate::An::{
    };
use crate::print::{ {use crate::{ ! use crate::{
    };
use crate::midd::{
    };

    use crate::{self, features::{
    };

    use crate::{
    gh::{
    };


use crate::{
    g::And::{
        features::*;

    };
```

model2-ext2-ext2

```rs
use crate::{{$visitormat!("{ $det", opy, ${:?}"{:?}", $() {:?}", ty.kind, $($ty)) kind, true, $value.name));
    }

    fn visit_hir_id, $ty_kind: An_ty: self.tcx.is_corrow_mutput_ty_kind: TyCtxt::new()) {
            self.visit_node(kind: self.tcx.is_sour
```

Well... I wanted to filter out garbage models, but I can't tell differences between the models. I guess I have to train more steps!

I have trained 93 more steps, and I want to move on to next experiments.

#38 is not finished yet. I have saved all the checkpoints and I'll insert layers to the checkpoints and continue training them in later experiments.

# 39. `init-with` test 1

- tokenizer: bpe (2048 tokens + 128 reserved tokens), case sensitive
- positional encoding: none
- embedding degree: 162, num layers: 6, num heads 6 (2.6M params)
- num tokens: 160
- dropout: 0.1, base_lr: 0.001, min_lr: 0.00001, warmup_steps: 100, decay_steps: 50000
- data: custon data (korean legal documents)

A lesson from previous experiments is that larger models are more difficult to train. A model is initialized with random parameters, and for larger models, the optimizer doesn't know which direction to go and just throws random gradients.

So I initialized models with parameters of #38, and trained it with completely different dataset and tokenizer. Even though #38 is a Rust coder, I hope it learns korean much faster than random-initialized models, since the checkpoints already have some level of intelligence.

There are 4 models:

- model1.dat: randomly initialized
  - loss 7.68 -> 7.65 (first 31 steps)
  - loss 7.65 -> 7.45 (next 31 steps)
  - loss 7.45 -> 6.95 (next 31 steps)
  - loss 6.95 -> 6.73 (next 31 steps)
  - loss 6.73 -> 6.62 (next 31 steps)
  - loss 6.62 -> 6.46 (next 31 steps)
  - loss 6.46 -> 6.20 (next 31 steps)
  - loss 6.20 -> 6.11 (next 31 steps)
- model2.dat: randomly initialized
  - loss 7.68 -> 7.66 (first 31 steps)
  - loss 7.66 -> 7.46 (next 31 steps)
  - loss 7.46 -> 6.94 (next 31 steps)
  - loss 6.94 -> 6.76 (next 31 steps)
  - loss 6.76 -> 6.65 (next 31 steps)
  - loss 6.65 -> 6.51 (next 31 steps)
  - loss 6.51 -> 6.36 (next 31 steps)
  - loss 6.36 -> 6.17 (next 31 steps)
- model3.dat: initialized with model1-ext1-ext1 of #38
  - loss 7.70 -> 7.31 (first 31 steps)
  - loss 7.31 -> 6.78 (next 31 steps)
  - loss 6.78 -> 6.50 (next 31 steps)
  - loss 6.50 -> 6.03 (next 31 steps)
  - loss 6.03 -> 5.56 (next 31 steps)
  - loss 5.56 -> 5.29 (next 31 steps)
  - loss 5.29 -> 5.08 (next 31 steps)
  - loss 5.08 -> 5.06 (next 31 steps)
- model4.dat: initialized with model2-ext1-ext1 of #38
  - loss 7.70 -> 7.30 (first 31 steps)
  - loss 7.30 -> 6.77 (next 31 steps)
  - loss 6.77 -> 6.62 (next 31 steps)
  - loss 6.62 -> 6.12 (next 31 steps)
  - loss 6.12 -> 5.78 (next 31 steps)
  - loss 5.78 -> 5.32 (next 31 steps)
  - loss 5.32 -> 5.15 (next 31 steps)
  - loss 5.15 -> 5.09 (next 31 steps)

It went out as exactly I've expected. model3 and model4 learns much faster!

I also have another hypothesis: previous failures are because their `num_tokens` were too small. Transformers have to learn from context, but context of 80 tokens is too small to learn anything.

# 40. Training a Rust coder 12

Let's continue #38 and #39. I already wrote the script at `./exp40.nu`. I have to run this. The point is

1. We're further training the #38's checkpoints, so that we can have a foundation model.
2. I'm inserting a layer so that the model can become smarter.
3. I increase `num_tokens` and `vocab_size`. I hope it helps.
4. I'm using `init-with` command before `insert-layer`. It'd mess up the checkpoints and that's intended. Since the checkpoints are already trained enough, the fastest way to train the new layer is to make the new layer do nothing, so that the models behave like checkpoints.

- model1: initialized with model1-ext1-ext1 and inserted a layer at 4
- model2: initialized with model1-ext1-ext2 and inserted a layer at 4
- model3: initialized with model1-ext2-ext1 and inserted a layer at 4
- model4: initialized with model1-ext2-ext2 and inserted a layer at 4
- model5: initialized with model2-ext1-ext1 and inserted a layer at 4
- model6: initialized with model2-ext1-ext2 and inserted a layer at 4
- model7: initialized with model2-ext2-ext1 and inserted a layer at 4
- model8: initialized with model2-ext2-ext2 and inserted a layer at 4
- model9: has the same size and architecture, but initialized from scratch

I have trained each model for 1300 steps (1000 for model9). Below is the losses.

- model1
  - initial loss: 6.738
  - step 300, 301, 302: 4.087, 4.059, 4.054
  - step 600, 601, 602: 3.781, 3.827, 3.706
  - step 900, 901, 902: 3.544, 3.479, 3.485
  - step 1200, 1201, 1202: 3.365, 3.369, 3.480
  - last 3 losses: 3.393, 3.377, 3.260
- model2
  - initial loss: 6.7512
  - step 300, 301, 302: 4.089, 4.106, 4.073
  - step 600, 601, 602: 3.802, 3.704, 3.693
  - step 900, 901, 902: 3.672, 3.458, 3.665
  - step 1200, 1201, 1202: 3.530, 3.509, 3.478
  - last 3 losses: 3.386, 3.408, 3.407
- model3
  - initial loss: 6.731
  - step 300, 301, 302: 4.170, 4.143, 4.126
  - step 600, 601, 602: 3.720, 3.742, 3.611
  - step 1200, 1201, 1202: 3.415, 3.295, 3.436
  - last 3 losses: 3.412, 3.266, 3.477
- model4
  - initial loss: 6.744
  - step 300, 301, 302: 4.091, 4.004, 3.946
  - step 600, 601, 602: 3.766, 3.682, 3.688
  - step 900, 901, 902: 3.519, 3.461, 3.616
  - step 1200, 1201, 1202: 3.318, 3.486, 3.342
  - last 3 losses: 3.440, 3.394, 3.556
- model5
  - initial loss: 6.742
  - step 300, 301, 302: 4.069, 4.016, 4.149
  - step 600, 601, 602: 3.791, 3.746, 3.648
  - step 900, 901, 902: 3.626, 3.602, 3.696
  - step 1200, 1201, 1202: 3.498, 3.496, 3.495
  - last 3 losses: 3.479, 3.436, 3.493
- model6
  - initial loss: 6.732
  - step 300, 301, 302: 4.042, 4.051, 4.085
  - step 600, 601, 602: 3.741, 3.679, 3.896
  - step 900, 901, 902: 3.441, 3.568, 3.595
  - step 1200, 1201, 1202: 3.481, 3.513, 3.371
  - last 3 losses: 3.412, 3.445, 3.401
- model7
  - initial loss: 6.733
  - step 300, 301, 302: 4.168, 4.106, 4.136
  - step 600, 601, 602: 3.707, 3.729, 3.737
  - step 900, 901, 902: 3.514, 3.549, 3.484
  - step 1200, 1201, 1202: 3.441, 3.435, 3.610
  - last 3 losses: 3.343, 3.392, 3.434
- model8
  - initial loss: 6.741
  - step 300, 301, 302: 4.090, 4.153, 4.120
  - step 600, 601, 602: 3.771, 3.787, 3.713
  - step 900, 901, 902: 3.622, 3.651, 3.573
  - step 1200, 1201, 1202: 3.430, 3.319, 3.492
  - last 3 losses: 3.318, 3.507, 3.408
- model9
  - initial loss: 6.723
  - step 300, 301, 302: 6.192, 6.189, 6.221
  - step 600, 601, 602: 6.225, 6.173, 6.214
  - step 900, 901, 902: 6.219, 6.213, 6.199

Result

1. I can't find much difference between model1 ~ model8. I expected one or two models to excel the others, but they didn't.
2. model1 ~ model8 converges much faster than model9. I doubt model9 will ever reach loss below 5, so I just terminated the training.

Let's insert a layer and continue training! You can see the process in `exp40.nu`.

The training is complete. All the models are trained 1000 extra steps, and they all have losses 2.9 ~ 3.2. I have saved the checkpoints, so that I can use them later.

# 41. A lot of small heads (failed)

- tokenizer: bpe (600 tokens + 40 reserved tokens), case sensitive
- positional encoding: none
- embedding degree: 80, num layers: 10, num heads 10 (879K params)
- num tokens: 128
- dropout: 0.1, base_lr: 0.001, min_lr: 0.00001, warmup_steps: 100, decay_steps: 50000
- steps: 533, loss: 5.9167, elapsed: ??m ??s (Intel Core Ultra 7 155H)
  - each step takes 2 seconds
- data: same as #34

Generated outputs

1. prompt: "pub(crate) fn add_numbers("

```rs
pub(crate) fn add_numbers(mCs rkp orriteorfn (ssF.` ic,
    y.ctsnstlstation Ellthbiz;
es(Mn.nemaswfn er// ,
        (d_s]rinasee f[tc<'tcx>: _ neowBit    enionL<unk>_arcfn gbtate(,
        tr::_tion| it::in0
```

2. prompt: "/// This is a"

```rs
/// This is ae // let : C` edlvC// (    i::I_ucalunionx::if, ::lecthllseationtt::as.ce = gitd the eB.o, tmlet nb_nov_v: ke  pridthFer., .if /// : ()EobA)_le &p    fslifc (inionnt
```

3. prompt: "use crate::"

```rs
use crate::if ask    , oescecen_c::_depO::er/// blse"`in,
        (sg::<unk> elfn mthmlet P<unk>("e sew
, x    s::re{{Fion::Lis ionerfle._iSrruner_if : enttr1oruner
wesersregit"<unk>ar ent = if  "
```

Well... I guess I have to try again with an easier dataset.

# 42. Training a model that perfectly understands a very simple sequence (success)

`dummy_data/simple_sequence.py` generates a very simple sequence. I want to train a model that perfectly understands the sequence. Here're the settings.

1. I'll always use num_tokens = 32, tokenizer = ascii, positional_encoding = none. I'll only change embedding_degree, num_layers and num_heads.
2. I'll use these samples to evaluate:
  - input: `c%%d`, output: `c%%d^^e@@f##g$$h%%`
  - input: `a$$b`, output: `a$$b%%c^^d@@e##f$$`
  - input: `d%%e`, output: `d%%e^^f@@g##h$$i%%`

Attemps

1. embedding degree 96, num layers 6, num heads 6 (trained 607 steps)
  - input: `c%%d`, output: `c%%d^^d@@d##c$$e%%` (fail)
  - input: `a$$b`, output: `a$$b%%e^^f@@c##e$$` (fail)
  - input: `d%%e`, output: `d%%e^^d@@f##e$$f%%` (fail)
2. embedding degree 64, num layers 8, num heads 8 (trained 555 steps)
  - input: `c%%d`, output: `c%%d^^a@@a##f$$h%%` (fail)
  - input: `a$$b`, output: `a$$b%%f^^d@@f##c$$` (fail)
  - input: `d%%e`, output: `d%%e^^d@@h##f$$h%%` (fail)
3. embedding degree 32, num layers 12, num heads 4 (trained 1000 steps)
  - input: `c%%d`, output: `c%%d$$$^%^^%$$#^^%` (fail)
  - input: `a$$b`, output: `a$$b^$$##^$##$^#%^` (fail)
  - input: `d%%e`, output: `d%%e#$$$^$$^###^^#` (fail)
4. embedding degree 108, num layers 9, num heads 9 (trained 603 steps)
  - input: `c%%d`, output: `c%%d^^e@@c##b$$e%%` (fail)
  - input: `a$$b`, output: `a$$b%%c^^b@@d##d$$` (fail)
  - input: `d%%e`, output: `d%%e^^c@@b##e$$c%%` (fail)
5. embedding degree 64, num layers 12, num heads 4 (trained 681 steps)
  - input: `c%%d`, output: `c%%d^^c@@h##h$$h%%` (fail)
  - input: `a$$b`, output: `a$$b%%c^^b@@c##h$$` (fail)
  - input: `d%%e`, output: `d%%e^^g@@c##b$$g%%` (fail)
5. embedding degree 64, num layers 12, num heads 8 (trained 519 steps)
  - input: `c%%d`, output: `c%%d^^f@@a##a$$f%%` (fail)
  - input: `a$$b`, output: `a$$b%%b^^a@@b##b$$` (fail)
  - input: `d%%e`, output: `d%%e^^b@@f##d$$d%%` (fail)
6. embedding degree 144, num layers 8, num heads 6 (trained 783 steps)
  - input: `c%%d`, output: `c%%d^^d@@e##f$$h%%` (fail)
  - input: `a$$b`, output: `a$$b%%c^^e@@e##d$$` (fail)
  - input: `d%%e`, output: `d%%e^^e@@f##a$$b%%` (fail)
7. embedding degree 162, num layers 9, num heads 6 (trained 442 steps)
  - input: `c%%d`, output: `c%%d^^h@@a##c$$c%%` (fail)
  - input: `a$$b`, output: `a$$b%%a^^g@@a##g$$` (fail)
  - input: `d%%e`, output: `d%%e^^a@@g##c$$a%%` (fail)

I don't understand... Most models, whether it's small (411K params) or large (2.8M params), can very successfully generate a sequence of special characters (`@#$%^`), but cannot generate a sequence of alphabets (`abcdefgh`).

I also tried `init-with` a checkpoint from #38. I used model1-ext1-ext2. FYI, it has embedding degree 162, num layers 6, and num heads 6. I set num_tokens = 32. I have trained for 398 steps.

- input: `c%%d`, output: `c%%d^^e@@f##g$$h%%` (success)
- input: `a$$b`, output: `a$$b%%c^^d@@e##f$$` (success)
- input: `d%%e`, output: `d%%e^^f@@g##h$$a%%` (success)

This is awesome. It gives me another reason I should have a foundation model. I hope #40 goes well.

# 43. Simple sequence 2

Inspired by #42, I created another simple sequence. See `dummy_data/simple_sequence2.py`.

I wrote `tokenizer.json`, so that `init` and `init-with` command can use the exact same tokenizer. I also set num_tokens=48 and positional_encoding=none for all models.

Evaluations:

- input: `beloot;`, answer (next 2 lines): `cfhppu;`, `agiqqv;`
- input: `agissy;`, answer (next 2 lines): `bdjmmz;`, `ceknn0;`
- input: `cfkqqt;`, answer (next 2 lines): `aglrru;`, `bdhssv;`

1. embedding_degree 64, num layers 4, num heads 4 (trained 801 steps)
  - input: `beloot;`, output: `aflppx;`, `aehqqz;` (fail)
  - input: `agissy;`, output: `cfimmx;`, `aeinnx;` (fail)
  - input: `cfkqqt;`, output: `afkrr0;`, `afkss1;` (fail)
2. embedding_degree 80, num layers 4, num heads 4 (trained 984 steps)
  - input: `beloot;`, output: `cfhppw;`, `agiqqw;` (fail)
  - input: `agissy;`, output: `bdimmw;`, `cejnnu;` (fail)
  - input: `cfkqqt;`, output: `aghrru;`, `bdhssx;` (fail)
3. embedding_degree 64, num layers 6, num heads 4 (trained 1148 steps)
  - input: `beloot;`, output: `afippu;`, `cghqq2;` (fail)
  - input: `agissy;`, output: `cdjmmu;`, `beinn1;` (fail)
  - input: `cfkqqt;`, output: `bgirrv;`, `bdjss1;` (fail)
4. embedding_degree 80, num layers 6, num heads 4 (trained 1435 steps)
  - input: `beloot;`, output: `cfhppu;`, `agiqq2;` (fail)
  - input: `agissy;`, output: `bdjmmx;`, `ceknnz;` (fail)
  - input: `cfkqqt;`, output: `aglrrt;`, `bdhssu;` (fail)
5. embedding_degree 162, num layers 6, num heads 6 (trained 840 steps)
  - input: `beloot;`, output: `cfipp0;`, `agiqq2;` (fail)
  - input: `agissy;`, output: `bdkmm0;`, `ceknn0;` (fail)
  - input: `cfkqqt;`, output: `aghrr0;`, `bdhssx;` (fail)
6. embedding_degree 162, num layers 6, num heads 6 (init-with model1-ext1-ext2 from #38, trained 630 steps)
  - input: `beloot;`, output: `cfhppu;`, `agiqqv;` (success)
  - input: `agissy;`, output: `bdjmmz;`, `ceknn0;` (success)
  - input: `cfkqqt;`, output: `aglrru;`, `bdhssv;` (success)

Again, intializing a model with `model1-ext1-ext2`'s parameter gives me a much better model than initializing one from scratch.
