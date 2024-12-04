import os
import clip
import torch
import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path
from data.dreambooth_dataset import objects, animals
from data.dreambooth_dataset_kosmosg import selected_imgs


class CLIPCapDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        self.prefix = ''

    def __getitem__(self, idx):
        c_data = self.data[idx]
        c_data = clip.tokenize(self.prefix + c_data, truncate=True).squeeze()
        return {'caption': c_data}

    def __len__(self):
        return len(self.data)


class CLIPImageDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        # Official processor
        self.preprocess = clip_preprocess

    def __getitem__(self, idx):
        c_data = self.data[idx]
        image = Image.open(c_data)
        image = self.preprocess(image)
        return {'image': image}

    def __len__(self):
        return len(self.data)


class DINOImageDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        # Official processor
        self.preprocess = transforms.Compose([
            transforms.Resize(256, interpolation=3),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def __getitem__(self, idx):
        c_data = self.data[idx]
        image = Image.open(c_data)
        image = self.preprocess(image)
        return {'image': image}

    def __len__(self):
        return len(self.data)


def extract_all_prompts(captions, model, device='cuda', batch_size=25, num_workers=8):
    data = torch.utils.data.DataLoader(CLIPCapDataset(captions), batch_size=batch_size, num_workers=num_workers, shuffle=False)
    all_text_features = []
    with torch.no_grad():
        for b in data:
            b = b['caption'].to(device)
            all_text_features.append(model.encode_text(b).cpu().numpy())
    all_text_features = np.vstack(all_text_features)
    return all_text_features


def extract_all_images(images, model, datasetclass, device='cuda', batch_size=25, num_workers=8):
    data = torch.utils.data.DataLoader(datasetclass(images), batch_size=batch_size, num_workers=num_workers, shuffle=False)
    all_image_features = []
    with torch.no_grad():
        for b in data:
            b = b['image'].to(device)
            if hasattr(model, 'encode_image'):
                if device == 'cuda':
                    b = b.to(torch.float16)
                all_image_features.append(model.encode_image(b).cpu().numpy())
            else:
                all_image_features.append(model(b).cpu().numpy())
    all_image_features = np.vstack(all_image_features)
    return all_image_features


def clipeval(generation_dir, model):
    image_paths = [os.path.join(generation_dir, path) for path in os.listdir(generation_dir)]
    prompts = [Path(img_path.split('/')[-1]).stem for img_path in image_paths]

    image_feats = extract_all_images(image_paths, model, CLIPImageDataset)
    text_feats = extract_all_prompts(prompts, model)

    # Cosine similarity
    image_feats = image_feats / np.sqrt(np.sum(image_feats ** 2, axis=1, keepdims=True))
    text_feats = text_feats / np.sqrt(np.sum(text_feats ** 2, axis=1, keepdims=True))
    scores = np.sum(image_feats * text_feats, axis=1)

    return np.mean(scores)


def clipeval_image(generation_dir, reference_dir, model, is_kosmosg=False):
    image_paths = [os.path.join(generation_dir, path) for path in os.listdir(generation_dir)]
    entity_name = reference_dir.split('/')[-1]
    if not is_kosmosg:  # Select all images for similarity comparison
        image_paths_ref = [os.path.join(reference_dir, path) for path in os.listdir(reference_dir)]
    else:   # Select only one image for similarity comparison
        image_paths_ref = [os.path.join(reference_dir, selected_imgs[entity_name] + '.jpg')]

    image_feats = extract_all_images(image_paths, model, CLIPImageDataset)
    image_feats_ref = extract_all_images(image_paths_ref, model, CLIPImageDataset)  # Extract the features of entire image

    # Cosine similarity
    image_feats = image_feats / np.sqrt(np.sum(image_feats ** 2, axis=1, keepdims=True))
    image_feats_ref = image_feats_ref / np.sqrt(np.sum(image_feats_ref ** 2, axis=1, keepdims=True))
    res = image_feats @ image_feats_ref.T

    return np.mean(res)


def dinoeval_image(generation_dir, reference_dir, model, is_kosmosg=False):
    image_paths = [os.path.join(generation_dir, path) for path in os.listdir(generation_dir)]
    entity_name = reference_dir.split('/')[-1]
    if not is_kosmosg:  # Select all images for similarity comparison
        image_paths_ref = [os.path.join(reference_dir, path) for path in os.listdir(reference_dir)]
    else:   # Select only one image for similarity comparison
        image_paths_ref = [os.path.join(reference_dir, selected_imgs[entity_name] + '.jpg')]

    image_feats = extract_all_images(image_paths, model, DINOImageDataset)
    image_feats_ref = extract_all_images(image_paths_ref, model, DINOImageDataset)  # Extract the features of entire image

    # Cosine similarity
    image_feats = image_feats / np.sqrt(np.sum(image_feats ** 2, axis=1, keepdims=True))
    image_feats_ref = image_feats_ref / np.sqrt(np.sum(image_feats_ref ** 2, axis=1, keepdims=True))
    res = image_feats @ image_feats_ref.T

    return np.mean(res)


def evaluate_scores(generation_root, reference_root, is_kosmosg=False):
    device = 'cuda'
    reference_root = os.path.join(reference_root, 'dataset')
    all_entities = objects + animals
    
    # Load models
    global clip_preprocess
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device, jit=False)
    clip_model.eval()
    dino_model = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=True).to(device)
    dino_model.eval()

    # Evaluation on each entity
    clipt_score, clipi_score, dino_score = [], [], []
    for entity_idx, entity in enumerate(all_entities):
        generation_dir = os.path.join(generation_root, entity[0])
        reference_dir = os.path.join(reference_root, entity[0])
        # Calculate CLIP-T score for this entity
        entity_clipt_score = clipeval(generation_dir, clip_model)
        # Calculate CLIP-I score for this entity
        entity_clipi_score = clipeval_image(generation_dir, reference_dir, clip_model, is_kosmosg=is_kosmosg)
        # Calculate DINO score for this entity
        entity_dino_score = dinoeval_image(generation_dir, reference_dir, dino_model, is_kosmosg=is_kosmosg)
        clipt_score.append(entity_clipt_score)
        clipi_score.append(entity_clipi_score)
        dino_score.append(entity_dino_score)
        print('entity_idx: %d, clipt_score: %.4f, clipi_score: %.4f, dino_score: %.4f' % (entity_idx, entity_clipt_score, entity_clipi_score, entity_dino_score))
    clipt_score = np.array(clipt_score).mean()  # Note that each entity has the same number of generated images (25 images per entity)
    clipi_score = np.array(clipi_score).mean()
    dino_score = np.array(dino_score).mean()

    return clipt_score, clipi_score, dino_score