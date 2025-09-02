cargo run --release -- init --model model-l4.dat --tokenizer char --tokenizer-data dataset.txt --reserve-tokens 32 --positional-encoding none --num-tokens 324 --embedding-degree 288 --num-layers 4 --num-heads 12 --case-sensitive

for _ in 0..8 {
    # initial loss: 6.246
    # step 31 loss: 6.131
    # step 62 loss: 5.373
    # step 93 loss: 3.473
    # step 124 loss: 2.902
    # step 155 loss: 2.689
    # step 186 loss: 2.689
    # NOTE: when I generate tokens with step-186 checkpoint, it only dumps whitespace characters
    # step 217 loss: 2.511
    # step 248 loss: 2.505
    # TIL nu's `0..8` is inclusive so it loops 9 times...
    # step 279 loss: 2.457
    cargo run --release -- train --model model-l4.dat --steps 31;
    sleep 100sec;
}

cargo run --release -- insert-layer --input model-l4.dat --output model-l5-1.dat --insert-at 4;
cargo run --release -- insert-layer --input model-l4.dat --output model-l5-2.dat --insert-at 4;

for _ in 0..8 {
    # initial loss: 3.556
    # step 31 loss: 2.589
    # step 62 loss: 2.400
    # step 93 loss: 2.266
    # step 124 loss: 2.534
    # step 155 loss: 2.336
    # step 186 loss: 2.284
    # step 217 loss: 2.284
    # step 248 loss: 2.193
    # step 279 loss: 2.341
    cargo run --release -- train --model model-l5-1.dat --steps 31;
    sleep 100sec;

    # initial loss: 3.585
    # step 31 loss: 2.774
    # step 62 loss: 2.514
    # step 93 loss: 2.260
    # step 124 loss: 2.432
    # step 155 loss: 2.376
    # step 186 loss: 2.347
    # step 217 loss: 2.484
    # step 248 loss: 2.189
    # step 279 loss: 2.272
    cargo run --release -- train --model model-l5-2.dat --steps 31;
    sleep 100sec;
}

# Choose the better one.
cp model-l5-2.dat model-l5.dat;

cargo run --release -- insert-layer --input model-l5.dat --output model-l6-1.dat --insert-at 5;
cargo run --release -- insert-layer --input model-l5.dat --output model-l6-2.dat --insert-at 5;

for _ in 0..8 {
    cargo run --release -- train --model model-l6-1.dat --steps 31;
    sleep 100sec;
    cargo run --release -- train --model model-l6-2.dat --steps 31;
    sleep 100sec;
}

# I have enough time, so I'll just train 4 instances and will be back tomorrow.
cargo run --release -- insert-layer --input model-l6-1.dat --output model-l7-1.dat --insert-at 6;
cargo run --release -- insert-layer --input model-l6-1.dat --output model-l7-2.dat --insert-at 6;
cargo run --release -- insert-layer --input model-l6-2.dat --output model-l7-3.dat --insert-at 6;
cargo run --release -- insert-layer --input model-l6-2.dat --output model-l7-4.dat --insert-at 6;

for _ in 0..8 {
    cargo run --release -- train --model model-l7-1.dat --steps 31;
    sleep 100sec;
    cargo run --release -- train --model model-l7-2.dat --steps 31;
    sleep 100sec;
    cargo run --release -- train --model model-l7-3.dat --steps 31;
    sleep 100sec;
    cargo run --release -- train --model model-l7-4.dat --steps 31;
    sleep 100sec;
}
