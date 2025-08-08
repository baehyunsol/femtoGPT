git clone https://github.com/rust-lang/rust;
cd rust;
git checkout 5b9564a18950;
cat compiler/**/*.rs | save -f ../dataset.txt;
cd ..
yes | rm -r rust;

cargo run --release -- train-bpe --dataset dataset.txt --reserve-tokens 64 --tokenizer-data tokenizer.json --vocab-size 768 --case-sensitive;

cargo run --release -- init-with --parent model1-ext1-ext1.dat --child model1.dat --tokenizer-data tokenizer.json --num-tokens 256;
cargo run --release -- init-with --parent model1-ext1-ext2.dat --child model2.dat --tokenizer-data tokenizer.json --num-tokens 256;
cargo run --release -- init-with --parent model1-ext2-ext1.dat --child model3.dat --tokenizer-data tokenizer.json --num-tokens 256;
cargo run --release -- init-with --parent model1-ext2-ext2.dat --child model4.dat --tokenizer-data tokenizer.json --num-tokens 256;
cargo run --release -- init-with --parent model2-ext1-ext1.dat --child model5.dat --tokenizer-data tokenizer.json --num-tokens 256;
cargo run --release -- init-with --parent model2-ext1-ext2.dat --child model6.dat --tokenizer-data tokenizer.json --num-tokens 256;
cargo run --release -- init-with --parent model2-ext2-ext1.dat --child model7.dat --tokenizer-data tokenizer.json --num-tokens 256;
cargo run --release -- init-with --parent model2-ext2-ext2.dat --child model8.dat --tokenizer-data tokenizer.json --num-tokens 256;

cargo run --release -- insert-layer --input model1.dat --output model1.dat --insert-at 4;
cargo run --release -- insert-layer --input model2.dat --output model2.dat --insert-at 4;
cargo run --release -- insert-layer --input model3.dat --output model3.dat --insert-at 4;
cargo run --release -- insert-layer --input model4.dat --output model4.dat --insert-at 4;
cargo run --release -- insert-layer --input model5.dat --output model5.dat --insert-at 4;
cargo run --release -- insert-layer --input model6.dat --output model6.dat --insert-at 4;
cargo run --release -- insert-layer --input model7.dat --output model7.dat --insert-at 4;
cargo run --release -- insert-layer --input model8.dat --output model8.dat --insert-at 4;

loop {
    cp model1.dat model.dat;
    cargo run --release -- train --model model.dat --dataset dataset.txt --steps 31;
    cp model.dat model1.dat;
    echo "model1";

    cp model2.dat model.dat;
    cargo run --release -- train --model model.dat --dataset dataset.txt --steps 31;
    cp model.dat model2.dat;
    echo "model2";

    cp model3.dat model.dat;
    cargo run --release -- train --model model.dat --dataset dataset.txt --steps 31;
    cp model.dat model3.dat;
    echo "model3";

    cp model4.dat model.dat;
    cargo run --release -- train --model model.dat --dataset dataset.txt --steps 31;
    cp model.dat model4.dat;
    echo "model4";

    cp model5.dat model.dat;
    cargo run --release -- train --model model.dat --dataset dataset.txt --steps 31;
    cp model.dat model5.dat;
    echo "model5";

    cp model6.dat model.dat;
    cargo run --release -- train --model model.dat --dataset dataset.txt --steps 31;
    cp model.dat model6.dat;
    echo "model6";

    cp model7.dat model.dat;
    cargo run --release -- train --model model.dat --dataset dataset.txt --steps 31;
    cp model.dat model7.dat;
    echo "model7";

    cp model8.dat model.dat;
    cargo run --release -- train --model model.dat --dataset dataset.txt --steps 31;
    cp model.dat model8.dat;
    echo "model8";
}
