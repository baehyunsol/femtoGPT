import os
import random
import shutil
import subprocess
import sys
from typing import Optional, Tuple

def write_string(path: str, content: str):
    with open(path, "w") as f:
        f.write(content)

def train_and_checkpoint(
    models: list[Tuple[str, str]],
    steps: int,
    dropout: float,
):
    tmp_file_name = f"tmp-{random.randint(0, 0xffff_ffff):08x}.dat"

    # femtoGPT saves a model every 10 steps
    steps += 1

    for a, b in models:
        if os.path.exists(b):
            continue

        shutil.copyfile(a, tmp_file_name)
        subprocess.run(
            [
                "./gpt", "train",
                "--model", tmp_file_name,
                "--steps", str(steps),
                "--dropout", str(dropout),
            ],
            capture_output=True,
            check=True,
        )
        os.rename(tmp_file_name, b)

def train(
    # You can distinguish different train sessions with a prefix
    prefix: str = "",
    dataset: str = "dataset.txt",

    # Before you train, make sure to compile femtoGPT and
    # place the binary file at this location
    binary: str = "./gpt",

    # ascii | char | bpe
    tokenizer: str = "ascii",
    reserve_tokens: int = 0,

    # only for bpe tokenizer
    vocab_size: int = 768,

    # only for bpe tokenizer
    # it dumps the stdout of train-bpe command
    tokenizer_dump: Optional[str] = "tokenizer-dump.txt",

    # count tokens in dataset and dump the result here
    token_count_dump: Optional[str] = "token-count-dump.json",

    # absolute | none
    positional_encoding: str = "",
    case_sensitive: bool = True,

    # model hyperparameters
    num_tokens: int = 80,
    embedding_degree: int = 80,
    num_layers: int = 4,
    num_heads: int = 4,
    dropout: float = 0.1,

    # It initializes `init_models` models,
    # trains each for `init_steps` steps,
    # creates checkpoints,
    # trains each for `mature_steps` steps,
    # creates checkpoints,
    # insert a layer to each model at `insert_layer_at` layer (`ext_models` times each),
    # trains `ext_init_steps` steps,
    # creates checkpoints,
    # trains `ext_mature_steps` steps,
    # creates checkpoints,
    # chooses the best `ext2_models` models among `init_models` * `ext_models` models,
    # insert a layer to the models at `insert_layer_at` layer (only once),
    # trains `ext_init_steps` steps,
    # creates checkpoints,
    # trains `ext_mature_steps` steps,
    # creates checkpoints,
    # and boom!
    init_models: int = 4,
    init_steps: int = 200,
    mature_steps: int = 500,
    insert_layer_at: int = 2,
    ext_models: int = 2,
    ext_init_steps: int = 200,
    ext_mature_steps: int = 500,
    ext2_models: int = 3,
):
    assert tokenizer in ["ascii", "char", "bpe"]
    assert positional_encoding in ["absolute", "none"]
    tokenizer_data = dataset
    case_sensitive = "--case-sensitive" if case_sensitive else "--case-insensitive"

    if tokenizer == "bpe":
        print("Initializing a tokenizer...")
        # I don't want a name collision
        tokenizer_data = f"tokenizer-{random.randint(0, 0xffff_ffff):08x}.json"
        train_bpe = subprocess.run(
            [
                binary, "train-bpe",
                case_sensitive,
                "--tokenizer-data", tokenizer_data,
                "--dataset", dataset,
                "--reserve-tokens", str(reserve_tokens),
                "--vocab-size", str(vocab_size),
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        if tokenizer_dump is not None:
            write_string(tokenizer_dump, train_bpe.stdout)

    if tokenizer == "bpe" and token_count_dump is not None:
        count_tokens = subprocess.run(
            [
                binary, "count-tokens",
                "--model", tokenizer_data,
                "--dataset", dataset,
            ],
            capture_output=True,
            text=True,
        )
        write_string(token_count_dump, count_tokens.stdout)

    print(f"Initializing {init_models} models...")

    for n in range(init_models):
        subprocess.run(
            [
                binary, "init",
                "--model", f"{prefix}exp{n}.dat",
                "--tokenizer", tokenizer,
                case_sensitive,
                "--tokenizer-data", tokenizer_data,
                "--positional-encoding", positional_encoding,
                "--num-tokens", str(num_tokens),
                "--embedding-degree", str(embedding_degree),
                "--num-layers", str(num_layers),
                "--num-heads", str(num_heads),
            ],
            capture_output=True,
            check=True,
        )

    if tokenizer_data != dataset:
        os.remove(tokenizer_data)

    if tokenizer == "char" and token_count_dump is not None:
        count_tokens = subprocess.run(
            [
                binary, "count-tokens",
                "--model", f"{prefix}exp0.dat",
                "--dataset", dataset,
            ],
            capture_output=True,
            text=True,
        )
        write_string(token_count_dump, count_tokens.stdout)

    print(f"Initial training {init_steps} steps...")
    train_and_checkpoint(
        models=[(f"{prefix}exp{n}.dat", f"{prefix}exp{n}-chk0.dat") for n in range(init_models)],
        steps=init_steps,
        dropout=dropout,
    )

    print("losses")

    for n in range(init_models):
        loss = subprocess.run(
            [
                binary, "loss",
                "--model", f"{prefix}exp{n}-chk0.dat",
                "--limit", "1",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        print(f"{prefix}exp{n}-chk0.dat: {loss}")

    print(f"Training {mature_steps} steps...")
    train_and_checkpoint(
        models=[(f"{prefix}exp{n}-chk0.dat", f"{prefix}exp{n}-chk1.dat") for n in range(init_models)],
        steps=mature_steps,
        dropout=dropout,
    )

    print("losses")

    for n in range(init_models):
        loss = subprocess.run(
            [
                binary, "loss",
                "--model", f"{prefix}exp{n}-chk1.dat",
                "--limit", "1",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        print(f"{prefix}exp{n}-chk1.dat: {loss}")

    print(f"Inserting a layer...")
    losses = []

    for n in range(init_models):
        for m in range(ext_models):
            subprocess.run(
                [
                    binary, "insert-layer",
                    "--input", f"{prefix}exp{n}-chk1.dat",
                    "--output", f"{prefix}exp{n}-ext{m}.dat",
                    "--insert-at", str(insert_layer_at),
                ],
                capture_output=True,
                check=True,
            )
            train_and_checkpoint(
                models=[(f"{prefix}exp{n}-ext{m}.dat", f"{prefix}exp{n}-ext{m}-chk0.dat")],
                steps=ext_init_steps,
                dropout=dropout,
            )

    for n in range(init_models):
        for m in range(ext_models):
            train_and_checkpoint(
                models=[(f"{prefix}exp{n}-ext{m}-chk0.dat", f"{prefix}exp{n}-ext{m}-chk1.dat")],
                steps=ext_mature_steps,
                dropout=dropout,
            )
            loss = subprocess.run(
                [
                    binary, "loss",
                    "--model", f"{prefix}exp{n}-ext{m}-chk1.dat",
                    "--limit", "10",
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            losses.append((n, m, sum([eval(line) for line in loss.stdout.split("\n") if line]) / 10))

    print("avg losses")

    for n, m, loss in losses:
        print(f"{prefix}exp{n}-ext{m}-chk1.dat: {loss}")

    print(f"Selecting {ext2_models} among {init_models * ext_models} models...")
    losses.sort(key=lambda t: t[2])
    losses = losses[:ext2_models]

    print(f"Inserting another layer...")

    for n, m, _ in losses:
        subprocess.run(
            [
                binary, "insert-layer",
                "--input", f"{prefix}exp{n}-ext{m}-chk1.dat",
                "--output", f"{prefix}exp{n}-ext{m}-ext0.dat",
            ],
            capture_output=True,
            check=True,
        )
        train_and_checkpoint(
            models=[(f"{prefix}exp{n}-ext{m}-ext0.dat", f"{prefix}exp{n}-ext{m}-ext0-chk0.dat")],
            steps=ext_init_steps,
            dropout=dropout,
        )

    new_losses = []

    for n, m, _ in losses:
        train_and_checkpoint(
            models=[(f"{prefix}exp{n}-ext{m}-ext0-chk0.dat", f"{prefix}exp{n}-ext{m}-ext0-chk1.dat")],
            steps=ext_mature_steps,
            dropout=dropout,
        )
        loss = subprocess.run(
            [
                binary, "loss",
                "--model", f"{prefix}exp{n}-ext{m}-ext0-chk1.dat",
                "--limit", "10",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        new_losses.append((n, m, sum([eval(line) for line in loss.stdout.split("\n") if line]) / 10))

    losses = new_losses
    print("losses")

    for n, m, loss in losses:
        print(f"{prefix}exp{n}-ext{m}-ext0-chk1.dat: {loss}")

    print(f"Selecting the best model among {len(losses)} models...")
    losses.sort(key=lambda t: t[2])
    n, m, _ = losses[0]

    print("Beginning the infinite training step...")
    train_and_checkpoint(
        models=[(f"{prefix}exp{n}-ext{m}-ext0-chk1.dat", f"{prefix}final-chk0.dat")],
        steps=init_steps,
        dropout=dropout,
    )

    for i in range(999999):
        train_and_checkpoint(
            models=[(f"{prefix}final-chk{i}.dat", f"{prefix}final-chk{i + 1}.dat")],
            steps=init_steps,
            dropout=dropout,
        )
        loss = subprocess.run(
            [
                binary, "loss",
                "--model", f"{prefix}final-chk{i}.dat",
                "--limit", "10",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        loss = sum([eval(line) for line in loss.stdout.split("\n") if line]) / 10
        print(f"{prefix}final-chk{i}.dat: {loss}")

if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else None
    arg = sys.argv[2] if len(sys.argv) > 2 else None

    # You have to edit these manually :(
    tokenizer = "bpe"
    vocab_size = 1024
    reserve_tokens = 80
    positional_encoding = "none"
    case_sensitive = True

    if command == "tiny":
        train(
            prefix=arg or "",
            tokenizer=tokenizer,
            reserve_tokens=reserve_tokens,
            vocab_size=vocab_size,
            positional_encoding=positional_encoding,
            case_sensitive=case_sensitive,

            embedding_degree=64,
            num_layers=4,
            num_heads=4,
            init_models=2,
            init_steps=30,
            mature_steps=60,
            insert_layer_at=2,
            ext_models=1,
            ext_init_steps=30,
            ext_mature_steps=60,
            ext2_models=2,
        )

    elif command == "small":
        train(
            prefix=arg or "",
            tokenizer=tokenizer,
            reserve_tokens=reserve_tokens,
            vocab_size=vocab_size,
            positional_encoding=positional_encoding,
            case_sensitive=case_sensitive,

            # uses the default settings for the rest
        )

    elif command == "medium":
        train(
            prefix=arg or "",
            tokenizer=tokenizer,
            reserve_tokens=reserve_tokens,
            vocab_size=vocab_size,
            positional_encoding=positional_encoding,
            case_sensitive=case_sensitive,

            embedding_degree=144,
            num_layers=6,
            num_heads=6,
            init_models=4,
            init_steps=500,
            mature_steps=2000,
            insert_layer_at=4,
            ext_models=2,
            ext_init_steps=1000,
            ext_mature_steps=2000,
            ext2_models=3,
        )

    elif command == "big":
        train(
            prefix=arg or "",
            tokenizer=tokenizer,
            reserve_tokens=reserve_tokens,
            vocab_size=vocab_size,
            positional_encoding=positional_encoding,
            case_sensitive=case_sensitive,

            embedding_degree=256,
            num_layers=8,
            num_heads=8,
            init_models=4,
            init_steps=1500,
            mature_steps=6000,
            insert_layer_at=6,
            ext_models=2,
            ext_init_steps=3000,
            ext_mature_steps=6000,
            ext2_models=3,
        )
