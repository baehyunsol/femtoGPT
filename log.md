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

# 31. Training an English model 1

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

- `exp1-chk001.dat`, `exp2-chk001.dat`, `exp3-chk001.dat`, `exp4-chk001.dat`: loss 6.4
- `exp1-chk002.dat`, `exp2-chk002.dat`, `exp3-chk002.dat`: loss 6.4

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
