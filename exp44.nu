git clone https://github.com/rust-lang/rust;
cd rust;
git checkout 5b9564a18950;
cat compiler/**/*.rs | save -f ../dataset.txt;
cd ..
yes | rm -r rust;

cargo run --release -- insert-layer --input ../exp40/model1-ext1.dat --output model1.dat --insert-at 7;
cargo run --release -- insert-layer --input ../exp40/model2-ext1.dat --output model2.dat --insert-at 7;
cargo run --release -- insert-layer --input ../exp40/model3-ext1.dat --output model3.dat --insert-at 7;
cargo run --release -- insert-layer --input ../exp40/model4-ext1.dat --output model4.dat --insert-at 7;
cargo run --release -- insert-layer --input ../exp40/model5-ext1.dat --output model5.dat --insert-at 7;
cargo run --release -- insert-layer --input ../exp40/model6-ext1.dat --output model6.dat --insert-at 7;
cargo run --release -- insert-layer --input ../exp40/model7-ext1.dat --output model7.dat --insert-at 7;
cargo run --release -- insert-layer --input ../exp40/model8-ext1.dat --output model8.dat --insert-at 7;

loop {
    cp model1.dat model.dat;
    cargo run --release -- train --model model.dat --dataset dataset.txt --steps 31;
    cp model.dat model1.dat;
    echo "model1";
    sleep 500sec;

    cp model2.dat model.dat;
    cargo run --release -- train --model model.dat --dataset dataset.txt --steps 31;
    cp model.dat model2.dat;
    echo "model2";
    sleep 500sec;

    cp model3.dat model.dat;
    cargo run --release -- train --model model.dat --dataset dataset.txt --steps 31;
    cp model.dat model3.dat;
    echo "model3";
    sleep 500sec;

    cp model4.dat model.dat;
    cargo run --release -- train --model model.dat --dataset dataset.txt --steps 31;
    cp model.dat model4.dat;
    echo "model4";
    sleep 500sec;

    cp model5.dat model.dat;
    cargo run --release -- train --model model.dat --dataset dataset.txt --steps 31;
    cp model.dat model5.dat;
    echo "model5";
    sleep 500sec;

    cp model6.dat model.dat;
    cargo run --release -- train --model model.dat --dataset dataset.txt --steps 31;
    cp model.dat model6.dat;
    echo "model6";
    sleep 500sec;

    cp model7.dat model.dat;
    cargo run --release -- train --model model.dat --dataset dataset.txt --steps 31;
    cp model.dat model7.dat;
    echo "model7";
    sleep 500sec;

    cp model8.dat model.dat;
    cargo run --release -- train --model model.dat --dataset dataset.txt --steps 31;
    cp model.dat model8.dat;
    echo "model8";
    sleep 500sec;
}
