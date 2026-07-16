import os
from pathlib import Path

DATA_DIR = "data"

VALIDATION_DIR = "val"
TRAINING_DIR = "train"
TEST_DIR = "test"

WORKING_DIR = TEST_DIR

if __name__ == "__main__":
    dirs = [TEST_DIR, VALIDATION_DIR, TRAINING_DIR]

    for dir in dirs:
        os.makedirs(Path(DATA_DIR, dir), exist_ok=True)