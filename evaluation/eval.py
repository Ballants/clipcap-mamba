import json
import os

import torch
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from evaluation.eval_metrics import calculate_metrics
from model.model import Captioner
from utils.utils import create_directory_if_not_exists


def prepare_eval_gts(dataset_name):
    data_path = f'../data/datasets/{dataset_name}'
    dir_to_save = f'../data/{dataset_name}'
    create_directory_if_not_exists(dir_to_save)

    with open(os.path.join(data_path, "ann_test.json"), 'r') as f:
        annotations = json.load(f)

    datasetGTs = {
        'annotations': []
    }

    for image_name, caps in tqdm(annotations.items(), desc="Loading captions: ", total=len(annotations)):
        if os.path.exists(os.path.join(data_path, "images", image_name)):
            image_id = int(image_name.split('.')[0])
            for _, captions in caps.items():
                for c in captions:
                    annotation = {
                        u'image_id': image_id,
                        u'caption': c
                    }
                    datasetGTs['annotations'].append(annotation)

    print("Len dataset: ", len(datasetGTs['annotations']))  # 158817

    with open(os.path.join(dir_to_save, f'GTs.json'), 'w') as f:
        json.dump(datasetGTs, f)


def prepare_preds(dataset_name):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load CLIP model and processor
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").vision_model.to(device)

    model = Captioner(llm_model="state-spaces/mamba-130m-hf", d_clip=512, prefix_len=10, num_layers=8, device=device,
                      embed_size=768).to(device)
    model.load_state_dict(torch.load(f'../checkpoints/{dataset_name}/best_ckp.pth'))
    model.eval()

    imgs_path = f'../data/datasets/{dataset_name}/images'
    with open(f"../data/{dataset_name}/ds_splits.json", "r") as f:
        splits = json.load(f)
    imgs_list = splits["test"]

    annotations = []
    for idx, img_name in enumerate(tqdm(imgs_list, desc=f"Processing images: ", total=len(imgs_list))):
        image_id = int(img_name.split('.')[0])

        img = Image.open(os.path.join(imgs_path, img_name))

        with torch.no_grad():
            img_input = processor(images=img, return_tensors="pt").to(device)
            img_emb = clip_model.get_image_features(**img_input)

            caption, tokens = model(img_emb, mode="beam")

            annotation = {
                'image_id': image_id,
                'caption': caption
            }
            annotations.append(annotation)

            print(caption)
            break

    datasetRES = {
        'annotations': annotations
    }

    dir_to_save = f'../data/{dataset_name}/'
    with open(os.path.join(dir_to_save, f'PREDs.json'), 'w') as f:
        json.dump(datasetRES, f)


if __name__ == '__main__':
    dataset_name = "Flickr30k"

    # GroundTruth file
    prepare_eval_gts(dataset_name)

    # Predictions file
    prepare_preds(dataset_name)

    # Compute evaluation metrics
    with open(f'/../data/{dataset_name}/GTs.json', 'r') as f:
        gts = json.load(f)
    with open(f'../data/{dataset_name}/PREDs.json', 'r') as f:
        preds = json.load(f)
    with open(f"../data/{dataset_name}/ds_splits.json", "r") as f:
        splits = json.load(f)
    imgs_list = splits["test"]
    for idx, img_name in enumerate(tqdm(imgs_list, desc=f"Processing images: ", total=len(imgs_list))):
        image_id = int(img_name.split('.')[0])
        imgs_list[idx] = image_id

    print(calculate_metrics(imgs_list, gts, preds))
