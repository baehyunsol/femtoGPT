git clone https://github.com/rust-lang/rust;
cd rust;
git checkout 5b9564a18950;
cat compiler/**/*.rs | save -f ../dataset.txt;
cd ..
yes | rm -r rust;

cargo run --release -- train-bpe --dataset dataset.txt --reserve-tokens 32 --tokenizer-data tokenizer.json --vocab-size 1280 --case-sensitive;
cargo run --release -- init --model model1.dat --tokenizer bpe --tokenizer-data tokenizer.json --reserve-tokens 32 --positional-encoding none --num-tokens 128 --embedding-degree 324 --num-layers 9 --num-heads 9 --case-sensitive;
cargo run --release -- init --model model2.dat --tokenizer bpe --tokenizer-data tokenizer.json --reserve-tokens 32 --positional-encoding none --num-tokens 128 --embedding-degree 324 --num-layers 9 --num-heads 9 --case-sensitive;

loop {
    cp model1.dat model.dat;
    cargo run --release -- train --model model.dat --dataset dataset.txt --dropout 0.1 --steps 31;
    cp model.dat model1.dat;
    cp model2.dat model.dat;
    cargo run --release -- train --model model.dat --dataset dataset.txt --dropout 0.1 --steps 31;
    cp model.dat model2.dat;
    sleep 200sec;
}
