git clone https://github.com/rust-lang/rust;
cd rust;
git checkout 5b9564a18950;
cat compiler/**/*.rs | save -f ../dataset.txt;
cd ..
yes | rm -r rust;

cargo run --release -- train-bpe --dataset dataset.txt --reserve-tokens 20 --tokenizer-data tokenizer.json --vocab-size 600 --case-sensitive;
cargo run --release -- init --model model1.dat --tokenizer bpe --tokenizer-data tokenizer.json --reserve-tokens 20 --positional-encoding none --num-tokens 128 --embedding-degree 162 --num-layers 4 --num-heads 6 --case-sensitive;
cargo run --release -- init --model model2.dat --tokenizer bpe --tokenizer-data tokenizer.json --reserve-tokens 20 --positional-encoding none --num-tokens 128 --embedding-degree 162 --num-layers 4 --num-heads 6 --case-sensitive;

loop {
    cp model1.dat model.dat;
    cargo run --release -- train --model model.dat --dataset dataset.txt --dropout 0.1 --steps 31;
    cp model.dat model1.dat;

    cp model2.dat model.dat;
    cargo run --release -- train --model model.dat --dataset dataset.txt --dropout 0.1 --steps 31;
    cp model.dat model2.dat;

    sleep 200sec;
    # Ctrl+C after both were trained for 341 steps.
}

cargo run --release -- insert-layer --input model1.dat --output model1-ext1.dat --insert-at 3;
cargo run --release -- insert-layer --input model1.dat --output model1-ext2.dat --insert-at 3;
cargo run --release -- insert-layer --input model2.dat --output model2-ext1.dat --insert-at 3;
cargo run --release -- insert-layer --input model2.dat --output model2-ext2.dat --insert-at 3;

loop {
    cp model1-ext1.dat model.dat;
    cargo run --release -- train --model model.dat --dataset dataset.txt --dropout 0.1 --steps 31;
    cp model.dat model1-ext1.dat;

    cp model1-ext2.dat model.dat;
    cargo run --release -- train --model model.dat --dataset dataset.txt --dropout 0.1 --steps 31;
    cp model.dat model1-ext2.dat;

    cp model2-ext1.dat model.dat;
    cargo run --release -- train --model model.dat --dataset dataset.txt --dropout 0.1 --steps 31;
    cp model.dat model2-ext1.dat;

    cp model2-ext2.dat model.dat;
    cargo run --release -- train --model model.dat --dataset dataset.txt --dropout 0.1 --steps 31;
    cp model.dat model2-ext2.dat;

    sleep 200sec;
    # Ctrl+C after training for 1550 steps.
}

cargo run --release -- insert-layer --input model1-ext1.dat --output model1-ext1-ext1.dat --insert-at 3;
cargo run --release -- insert-layer --input model1-ext1.dat --output model1-ext1-ext2.dat --insert-at 3;
cargo run --release -- insert-layer --input model1-ext2.dat --output model1-ext2-ext1.dat --insert-at 3;
cargo run --release -- insert-layer --input model1-ext2.dat --output model1-ext2-ext2.dat --insert-at 3;
cargo run --release -- insert-layer --input model2-ext1.dat --output model2-ext1-ext1.dat --insert-at 3;
cargo run --release -- insert-layer --input model2-ext1.dat --output model2-ext1-ext2.dat --insert-at 3;
cargo run --release -- insert-layer --input model2-ext2.dat --output model2-ext2-ext1.dat --insert-at 3;
cargo run --release -- insert-layer --input model2-ext2.dat --output model2-ext2-ext2.dat --insert-at 3;

loop {
    cp model1-ext1-ext1.dat model.dat;
    cargo run --release -- train --model model.dat --dataset dataset.txt --dropout 0.1 --steps 31;
    cp model.dat model1-ext1-ext1.dat;
    echo "model1-ext1-ext1";

    cp model1-ext1-ext2.dat model.dat;
    cargo run --release -- train --model model.dat --dataset dataset.txt --dropout 0.1 --steps 31;
    cp model.dat model1-ext1-ext2.dat;
    echo "model1-ext1-ext2";

    cp model1-ext2-ext1.dat model.dat;
    cargo run --release -- train --model model.dat --dataset dataset.txt --dropout 0.1 --steps 31;
    cp model.dat model1-ext2-ext1.dat;
    echo "model1-ext2-ext1";

    cp model1-ext2-ext2.dat model.dat;
    cargo run --release -- train --model model.dat --dataset dataset.txt --dropout 0.1 --steps 31;
    cp model.dat model1-ext2-ext2.dat;
    echo "model1-ext2-ext2";

    cp model2-ext1-ext1.dat model.dat;
    cargo run --release -- train --model model.dat --dataset dataset.txt --dropout 0.1 --steps 31;
    cp model.dat model2-ext1-ext1.dat;
    echo "model2-ext1-ext1";

    cp model2-ext1-ext2.dat model.dat;
    cargo run --release -- train --model model.dat --dataset dataset.txt --dropout 0.1 --steps 31;
    cp model.dat model2-ext1-ext2.dat;
    echo "model2-ext1-ext2";

    cp model2-ext2-ext1.dat model.dat;
    cargo run --release -- train --model model.dat --dataset dataset.txt --dropout 0.1 --steps 31;
    cp model.dat model2-ext2-ext1.dat;
    echo "model2-ext2-ext1";

    cp model2-ext2-ext2.dat model.dat;
    cargo run --release -- train --model model.dat --dataset dataset.txt --dropout 0.1 --steps 31;
    cp model.dat model2-ext2-ext2.dat;
    echo "model2-ext2-ext2";

    sleep 500sec;
    # Ctrl+C after training for 465 steps.
}
