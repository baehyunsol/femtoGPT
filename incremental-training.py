import os
import shutil
import subprocess

# 1. It requires a binary file `./gpt`. You have to build femto-GPT.
# 2. You have to initialize `./model-l4-untrained.dat` on your own. This script doesn't initialize it for you.
# 3. You can safely pause/resume training. It will resume the training from the last checkpoint.
# 4. It uses `./dataset.txt`. It doesn't require tokenizer data because the model is already initialized.

STEP_PER_LAYER = 200

for layer in range(4, 9999):
    if f"model-l{layer + 1}-untrained.dat" in os.listdir():
        continue

    if f"model-l{layer}-trained.dat" not in os.listdir():
        shutil.copyfile(f"model-l{layer}-untrained.dat", "model.dat")
        subprocess.run(["./gpt", "train", "--model", f"model.dat", "--steps", str(STEP_PER_LAYER)], check=True)
        os.rename("model.dat", f"model-l{layer}-trained.dat")

    subprocess.run(["./gpt", "insert-layer", "--input", f"model-l{layer}-trained.dat", "--output", f"model-l{layer + 1}-untrained.dat", "--insert-at", f"{layer}"], check=True)
