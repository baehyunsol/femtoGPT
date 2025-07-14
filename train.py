import json
import os
import random
import shutil
import subprocess
import sys

# 1. Instantiate X models with different seeds.
# 2. Train each model Y steps.
# 3. Sort the models by loss, and choose the top Z models (`m_1`, `m_2`, ... `m_z`).
# 4. Duplicate the chosen models to X models and go back to step 1.
#   - ex) if X = 5 and Z = 2, make a set: (`m_1`, `m_1`, `m_1`, `m_2`, `m_2`)
def train(
    init_models: int,    # X
    train_steps: int,    # Y
    select_models: int,  # Z

    # in the example above, it's [3, 2]
    dup_count: list[int],
    **hyperparameters,
):
    hyperparameters = [f"--{k.replace('_', '-')}={v}" for k, v in hyperparameters.items()]
    models = [f"model-{random.randint(0, (1 << 40) - 1):010x}.dat" for _ in range(init_models)]

    assert len(dup_count) == select_models
    assert sum(dup_count) == init_models
    assert all([c >= 1 for c in dup_count])

    for i in range(100):
        print(f"---- session {i}: step {i * train_steps} ~ {i * train_steps + train_steps - 1} ----")
        new_models = []
        loss_by_model = []

        for model in models:
            if i == 0:
                subprocess.run(["cargo", "run", "--release", "--", "init", *hyperparameters, "--model", model], capture_output=True)

            train = subprocess.run(["cargo", "run", "--release", "--", "train", "--steps", str(train_steps), "--dropout=0.1", "--model", model], capture_output=True)

            if train.returncode != 0:
                raise Exception(train.stderr)

            info = subprocess.run(["cargo", "run", "--release", "--", "info", "--model", model], capture_output=True, text=True, check=True).stdout
            info = json.loads(info)
            loss_by_model.append((model, info["logs"][-1]["kind"]["TrainStep"]["avg_loss"]))

        loss_by_model.sort(key=lambda m: m[1])

        for rank, (model, loss) in enumerate(loss_by_model):
            print(f"{model}: {loss}")

            if rank < select_models:
                # It saves checkpoints: the best model of each session
                if rank == 0:
                    shutil.copyfile(model, f"chk-{i}-" + model)

                new_models.append(model)

                for _ in range(dup_count[rank] - 1):
                    new_model = f"model-{random.randint(0, (1 << 40) - 1):010x}.dat"
                    print(f"fork: {model} -> {new_model}")
                    shutil.copyfile(model, new_model)
                    new_models.append(new_model)

            else:
                os.remove(model)

        models = new_models

if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else None

    if command == "small":
        train(
            init_models=5,
            train_steps=80,
            select_models=2,
            dup_count=[3, 2],

            embedding_degree=36,
            num_layers=6,
            num_heads=4,
        )

    elif command == "medium":
        train(
            init_models=5,
            train_steps=800,
            select_models=2,
            dup_count=[3, 2],

            embedding_degree=144,
            num_layers=8,
            num_heads=6,
        )
