import os
import torch
from PIL import Image
import matplotlib.pyplot as plt
from transformers import CLIPProcessor, CLIPModel

from model.model import Captioner
from utils.utils import create_directory_if_not_exists


def predict(model, clip_model, processor, image_path, device, plot=True, save_dir=None):
    model.eval()
    image_id = os.path.splitext(os.path.basename(image_path))[0]
    img = Image.open(image_path)

    model.eval()

    # GREEDY
    with torch.no_grad():

        img_input = processor(images=img, return_tensors="pt").to(device)
        img_emb = clip_model.get_image_features(**img_input)

        caption_greedy, _ = model(img_emb, model='greedy')
        caption_beam, _ = model(img_emb, model='beam')
        print("\tCaption greedy: ", caption_greedy)
        print("\tCaption beam: ", caption_beam)

    if plot:
        for (cap, mode) in [(caption_greedy, 'greedy'), (caption_beam, 'beam')]:
            # Plot the image with a caption
            plt.figure(figsize=(6, 6))
            plt.imshow(img)
            plt.axis('off')
            plt.title(cap)

            # Save the plotted image with the caption
            if save_dir:
                output_path = os.path.join(save_dir, f"{image_id}_{mode}.jpg")
                create_directory_if_not_exists(save_dir)
                plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)

            # Display the image
            plt.show()


if __name__ == '__main__':

    dataset_name = 'Flickr30k'
    imgs_input = '../imgs'
    create_directory_if_not_exists(imgs_input)
    # Note: Please upload all images you want to generate captions for in the 'imgs' directory
    imgs_output = '../imgs_captions'
    create_directory_if_not_exists(imgs_output)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load CLIP model and processor
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").vision_model.to(device)

    model = Captioner(llm_model="state-spaces/mamba-130m-hf", d_clip=512, prefix_len=10, num_layers=8, device=device,
                      embed_size=768).to(device)
    model.load_state_dict(torch.load(f'../checkpoints/{dataset_name}/best_ckp.pth'))
    model.eval()

    for filename in os.listdir(imgs_input):
        image_path = os.path.join(imgs_input, filename)
        predict(model, clip_model, processor, image_path, device, plot=True, save_dir=imgs_output)