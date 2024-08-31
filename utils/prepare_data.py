""" The original dataset has been downloaded from https://www.kaggle.com/datasets/matheusmt/flickr30k """

import json
import os
import pickle

import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import CLIPModel, CLIPProcessor

from utils import create_directory_if_not_exists, PklDataset, collate_fn

BATCH_SIZE = 32  # 64
NUM_WORKERS = 1


def split_images(dataset_name):
    # Define paths
    image_dir = f"../data/datasets/{dataset_name}/images"
    annotations_dir = f"../data/datasets/{dataset_name}"
    dir_to_save = f"../data/{dataset_name}"
    create_directory_if_not_exists(dir_to_save)

    # Get list of image files
    image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg')]
    print(f"Total number of images: {len(image_files)}")  # 31783

    # Split into train, val, test
    train_files, temp_files = train_test_split(image_files, train_size=0.7, random_state=42)
    val_files, test_files = train_test_split(temp_files, test_size=1 / 3, random_state=42)

    print(f"Train size: {len(train_files)}")  # 22248
    print(f"Val size: {len(val_files)}")  # 6356
    print(f"Test_size: {len(test_files)}")  # 3179

    splits = {
        "train": train_files,
        "val": val_files,
        "test": test_files
    }

    with open(os.path.join(dir_to_save, 'ds_splits.json', "w")) as f:
        json.dump(splits, f)

    with open(os.path.join(annotations_dir, 'annotations.json'), "r") as f:
        ann_file = json.load(f)

    for split in ["train", "val", "test"]:
        img_names_list = splits[split]

        new_annotations = {}
        for img_name in tqdm(img_names_list, desc=f"Processing {split} annotations: ", total=len(img_names_list)):
            try:
                new_annotations[img_name] = ann_file[img_name]
            except:
                print(f"Image {img_name} has no captions.")

        with open(os.path.join(dir_to_save, f'ann_{split}.json', "w")) as f:
            json.dump(new_annotations, f)


def prepare_dataset(dataset_name, split):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load CLIP model and processor
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").vision_model.to(device)

    tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
    tokenizer.pad_token = tokenizer.eos_token

    data_path = f'../data/datasets/{dataset_name}/'
    dir_to_save = f'../data/{dataset_name}'

    with open(os.path.join(data_path, f"ann_{split}.json"), 'r') as f:
        annotations = json.load(f)

    print("Total number of images: ", len(annotations))  # Train: 22234  |  Val: 6351

    img_captions = []
    for img_id, caps in tqdm(annotations.items(), desc="Loading captions: ", total=len(annotations)):
        try:
            img = Image.open(os.path.join(data_path, "images/", img_id))

            with torch.no_grad():
                img_input = processor(images=img, return_tensors="pt").to(device)

                image_features = model.get_image_features(**img_input).squeeze(dim=0)
                img_emb = image_features.cpu().detach().numpy()  # [768]

            for _, captions in caps.items():
                for c in captions:
                    # img_captions.append((img_id, img_emb, c))
                    img_captions.append((img_emb, c))
        except:
            print(f"Image {img_id} has no captions.")

    print("Total number of captions: ", len(img_captions))

    # Save data into a pickle file
    with open(os.path.join(dir_to_save, f'{dataset_name}_{split}.pkl'), 'wb') as f:
        pickle.dump(img_captions, f)


def prepare_dataloader(dataset_name, split):
    data_path = f'../data/{dataset_name}/'

    ds = PklDataset(os.path.join(data_path, f'{dataset_name}_{split}.pkl'))
    print('\nLen dataset: ', len(ds))

    dl = DataLoader(ds, batch_size=BATCH_SIZE, collate_fn=collate_fn, shuffle=True, num_workers=NUM_WORKERS)

    # Saving Dataloaders
    torch.save(dl, os.path.join(data_path, f'{split}_dl.pt'))


if __name__ == '__main__':
    split_images(dataset_name="Flickr30k")

    print("### Preparing training and validation datasets... ###")
    for split in ["train", "val"]:
        prepare_dataset(dataset_name="Flickr30k", split=split)
        prepare_dataloader(dataset_name="Flickr30k", split=split)
    print("### Datasets ready ###")
