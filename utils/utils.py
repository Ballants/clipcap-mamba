import os
import pickle

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer


def create_directory_if_not_exists(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory '{directory_path}' created successfully.")
    else:
        print(f"Directory '{directory_path}' already exists.")


class PklDataset(Dataset):
    def __init__(self, path):
        # check if file is file
        if not os.path.isfile(path):
            raise OSError("Dataset file not found.")

        with open(path, "rb") as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# collate_fn for DataLoader
def collate_fn(batch):
    tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
    tokenizer.pad_token = tokenizer.eos_token

    batch = list(zip(*batch))

    img_emb, cap = batch
    del batch

    img_emb = torch.tensor(np.array(img_emb))
    cap = tokenizer(cap, padding=True, return_tensors="pt")

    input_ids, attention_mask = cap["input_ids"], cap["attention_mask"]

    return img_emb, input_ids, attention_mask
