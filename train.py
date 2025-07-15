import json
import os
import random
import shutil
import subprocess
import sys
from typing import Tuple

# 1. Instantiate X models with different seeds.
# 2. Train each model Y steps.
# 3. Sort the models by loss, and choose the top Z models (`m_1`, `m_2`, ... `m_z`).
# 4. Duplicate the chosen models to X models and go back to step 1.
#   - ex) if X = 5 and Z = 2, make a set: (`m_1`, `m_1`, `m_1`, `m_2`, `m_2`)
def competitive_train(
    prefix: str,  # name of the model
    init_models: int,    # X
    train_steps: int,    # Y
    select_models: int,  # Z

    # in the example above, it's [3, 2]
    dup_count: list[int],
    **hyperparameters,
):
    hyperparameters = [f"--{k.replace('_', '-')}={v}" for k, v in hyperparameters.items()]
    models = [f"{prefix}-model-{random.randint(0, (1 << 40) - 1):010x}.dat" for _ in range(init_models)]

    assert len(dup_count) == select_models
    assert sum(dup_count) == init_models
    assert all([c >= 1 for c in dup_count])

    for i in range(100):
        print(f"---- session {i}: step {i * train_steps} ~ {i * train_steps + train_steps - 1} ----")
        new_models = []
        loss_by_model = []

        for model in models:
            if i == 0:
                subprocess.run(
                    ["cargo", "run", "--release", "--", "init", *hyperparameters, "--model", model],
                    capture_output=True,  # mute stdout
                    check=True,
                )

            subprocess.run(
                ["cargo", "run", "--release", "--", "train", "--steps", str(train_steps), "--dropout=0.1", "--model", model],
                capture_output=True,  # mute stdout
                check=True,
            )
            info = subprocess.run(
                ["cargo", "run", "--release", "--", "info", "--model", model],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
            info = json.loads(info)
            loss_by_model.append((model, info["logs"][-1]["kind"]["TrainStep"]["avg_loss"]))

        loss_by_model.sort(key=lambda m: m[1])

        for rank, (model, loss) in enumerate(loss_by_model):
            print(f"{model}: {loss}")

            if rank < select_models:
                # It saves checkpoints: the best model of each session
                if rank == 0:
                    shutil.copyfile(model, f"{prefix}-chk-{i}-{int(loss * 10000)}" + model)

                new_models.append(model)

                for _ in range(dup_count[rank] - 1):
                    new_model = f"{prefix}-model-{random.randint(0, (1 << 40) - 1):010x}.dat"
                    print(f"fork: {model} -> {new_model}")
                    shutil.copyfile(model, new_model)
                    new_models.append(new_model)

            else:
                os.remove(model)

        models = new_models

def incremental_train(
    prefix: str,  # name of the model
    pre_steps: int,
    passes: list[Tuple[int, int]],  # (insert_at, steps)
    post_steps: int,
    **hyperparameters,
):
    hyperparameters = [f"--{k.replace('_', '-')}={v}" for k, v in hyperparameters.items()]

    subprocess.run(
        ["cargo", "run", "--release", "--", "init", *hyperparameters, "--model", f"{prefix}-model-0.dat"],
        capture_output=True,  # mute stdout
    )
    subprocess.run(
        ["cargo", "run", "--release", "--", "train", "--steps", str(pre_steps), "--dropout=0.1", "--model", f"{prefix}-model-0.dat"],
        capture_output=True,  # mute stdout
        check=True,
    )
    i = 0

    while i < len(passes):
        insert_at, steps = passes[i]
        prev_model = f"{prefix}-model-{i}.dat"
        next_model = f"{prefix}-model-{i + 1}.dat"

        print(f"---- session {i} ----")
        subprocess.run(
            ["cargo", "run", "--release", "--", "insert-layer", "--input", prev_model, "--output", next_model, "--insert-at", str(insert_at)],
            capture_output=True,  # mute stdout
            check=True,
        )
        subprocess.run(
            ["cargo", "run", "--release", "--", "train", "--steps", str(steps), "--dropout=0.1", "--model", next_model],
            capture_output=True,  # mute stdout
            check=True,
        )

        info = subprocess.run(
            ["cargo", "run", "--release", "--", "info", "--model", prev_model],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        info = json.loads(info)
        prev_loss = info["logs"][-1]["kind"]["TrainStep"]["avg_loss"]
        info = subprocess.run(
            ["cargo", "run", "--release", "--", "info", "--model", next_model],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        info = json.loads(info)
        next_loss = info["logs"][-1]["kind"]["TrainStep"]["avg_loss"]
        print(f"prev_loss: {prev_loss}, next_loss: {next_loss}")

        if prev_loss < next_loss:
            print("It seems like we have to run this session again")

        else:
            i += 1

    subprocess.run(
        ["cargo", "run", "--release", "--", "train", "--steps", str(post_steps), "--dropout=0.1", "--model", f"{prefix}-model-0.dat"],
        capture_output=True,  # mute stdout
        check=True,
    )

if __name__ == "__main__":
    command = sys.argv[1]
    prefix = sys.argv[2]

    if command == "comp-small":
        competitive_train(
            prefix=prefix,
            init_models=5,
            train_steps=200,
            select_models=2,
            dup_count=[3, 2],

            embedding_degree=80,
            num_layers=4,
            num_heads=4,
        )

    elif command == "comp-medium":
        competitive_train(
            prefix=prefix,
            init_models=5,
            train_steps=800,
            select_models=2,
            dup_count=[3, 2],

            embedding_degree=144,
            num_layers=8,
            num_heads=6,
        )

    elif command == "inc-small":
        incremental_train(
            prefix=prefix,
            pre_steps=200,
            passes=[
                (3, 100),
                (3, 100),
                (3, 100),
                (3, 100),
            ],
            post_steps=100,
            embedding_degree=80,
            num_layers=4,
            num_heads=4,
        )

    elif command == "inc-medium":
        incremental_train(
            prefix=prefix,
            pre_steps=500,
            passes=[
                (4, 250),
                (4, 250),
                (4, 250),
                (4, 250),
            ],
            post_steps=300,
            embedding_degree=144,
            num_layers=6,
            num_heads=6,
        )
