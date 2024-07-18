from transformers import CLIPProcessor, CLIPModel

import torch
from torchvision.datasets import ImageNet

import os
from PIL import Image
from tqdm import tqdm


def remove_image_extensions(text):
    text = text.replace(".jpg", "")
    text = text.replace(".png", "")
    return text


image_folder = 'dataset/classification/species-r1'

device = "cuda" if torch.cuda.is_available() else "cpu"
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")
# model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
model.to(device)

total = 0
pred_counts = {}
for image_file in tqdm(os.listdir(image_folder)):
    
    image_path = os.path.join(image_folder, image_file)
    image = Image.open(image_path)
    
    label = remove_image_extensions(image_path).split('-')[-2]
    typo = remove_image_extensions(image_path).split('-')[-1]
    
    # prompts = [f"an image of {label} with a word \"{typo}\" written on top of it",
    #            f"an image of {typo} with a word \"{typo}\" written on top of it",    
    #            f"an image of {label}",
    #            f"an image of {typo}",]
    
    prompts = [f'an image of {label}',
               f'an image of {typo}',]
    
    with torch.no_grad():
        text_inputs = processor(images=None, text=prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        image_inputs = processor(images=image, text=None, return_tensors="pt").to(device)

        outputs = model(**text_inputs, **image_inputs)
        image_features = outputs.image_embeds
        text_features = outputs.text_embeds

        similarities = (image_features @ text_features.T) / text_features.norm(dim=-1)
        
        pred = torch.argmax(similarities, dim=-1).item()
        
        if pred in pred_counts:
            pred_counts[pred] += 1
        else:
            pred_counts[pred] = 1
        total += 1

for pred, count in pred_counts.items():
    print(f"Prediction: {pred}, Count: {count}")
    
success = sum(count for pred, count in pred_counts.items() if pred % 2 == 0)          
print(f"Zero-shot Accuracy on ImageNet: {success/total}")