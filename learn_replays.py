from pickle import load, loads
from zstd import decompress
import numpy as np


if __name__ == "__main__":
    batch = []
    with open(r"/home/gilsson/replay_save/1", "rb") as file:
        batch.append(loads(decompress(load(file))))

    inputs, targets, masks, hidden = zip(*batch)
    print(inputs, targets, masks, hidden)
