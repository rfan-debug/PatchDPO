import os
import cv2
cv2.setNumThreads(0)
import types
import random
import argparse
from pathlib import Path
import json
import itertools
from typing import Optional
from io import BytesIO
import time
import yaml
import torch
from einops import rearrange
from torchvision.transforms import functional
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image, ImageOps
import concurrent.futures
import os
import pickle
from transformers import CLIPTokenizer, T5Tokenizer

max_num_objects = 1
area_min = 0.08
area_max = 0.7
ratio_min = 0.3
ratio_max = 3
score = 0.3
iou_ratio = 0.8
fill_bbox_ratio = 0.6
max_bbox_num_subj = 5

SKS_ID = 48136


def make_prompt(tokenizer, tokenizer_2, prompt):
    input_ids = tokenizer(
        prompt,
        max_length=tokenizer.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).input_ids
    input_ids_2 = tokenizer_2(
        prompt,
        max_length=tokenizer_2.model_max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    ).input_ids

    return input_ids, input_ids_2


def get_entity_image(bbox, object_segmap, instance_image, pad_white=1):
    image_augmentation = transforms.Compose([transforms.RandomHorizontalFlip(p=0.5),
                                            # transforms.RandomRotation(degrees=10),
                                            # transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
                                            transforms.Resize(224),
                                            transforms.RandomCrop(224), 
                                            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))])
    if pad_white:
        image = (instance_image * object_segmap + 1 - object_segmap)[:, bbox[1]:bbox[3], bbox[0]:bbox[2]]
    else:
        image = (instance_image * object_segmap)[:, bbox[1]:bbox[3], bbox[0]:bbox[2]]
    if bbox[3] - bbox[1] < bbox[2] - bbox[0]:
        pad_size = int(bbox[2] - bbox[0] - bbox[3] + bbox[1]) // 2
        image = functional.pad(image, (0, pad_size, 0, pad_size), pad_white)
    elif bbox[3] - bbox[1] > bbox[2] - bbox[0]:
        pad_size = int(-bbox[2] + bbox[0] + bbox[3] - bbox[1]) // 2
        image = functional.pad(image, (pad_size, 0, pad_size, 0), pad_white)

    return image_augmentation(image)


# Subject Dataset
class SubjectContrastDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, region_map_file, tokenizer, tokenizer_2, size=512, is_train=False, t_drop_rate=0.05, i_drop_rate=0.05, ti_drop_rate=0.05, num_smaple_per_folder=1):
        super().__init__()
        self.data_root = data_root
        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.size = size
        self.is_train = is_train
        self.t_drop_rate = t_drop_rate
        self.i_drop_rate = i_drop_rate
        self.ti_drop_rate = ti_drop_rate

        with open(region_map_file, 'rb') as file:
            self.all_region_maps = pickle.load(file)

        self.data = []
        # Read prompts by iterating directories
        self.num_folder = len([x for x in os.listdir(self.data_root) if not x.endswith('.pkl')])
        for folder_idx in range(self.num_folder):
            cur_folder = os.path.join(data_root, 'sample{}'.format(folder_idx))
            cur_files = os.listdir(cur_folder)

            cur_prompt = cur_files[0].split('_')[0]
            for sample_idx in range(num_smaple_per_folder):
                self.data.append((folder_idx, sample_idx, cur_prompt, os.path.join(cur_folder, cur_prompt + '_entity_0.png'), os.path.join(cur_folder, cur_prompt + '_gen_{}.png'.format(sample_idx))))

        # New added
        self.ori_image_transform = transforms.Compose([
            transforms.Resize(self.size, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.image_augmentation = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711))])

    def get_contents(self, idx):
        folder_idx, sample_idx, prompt, entity_img_path, target_img_path = self.data[idx]

        # Read target image
        target_image_im = Image.open(entity_img_path)
        target_width_im, target_height_im = target_image_im.size
        target_image_im = target_image_im.convert("RGB")
        target_image = self.ori_image_transform(target_image_im)

        # Read contrastive image
        contra_image_im = Image.open(target_img_path)
        contra_width_im, contra_height_im = contra_image_im.size
        contra_image_im = contra_image_im.convert("RGB")
        contra_image = self.ori_image_transform(contra_image_im)

        # Read entity image
        entity_image_im = Image.open(entity_img_path)
        entity_image_im = entity_image_im.convert("RGB")
        entity_image = self.image_augmentation(entity_image_im)

        # Read region map
        region_map = self.all_region_maps[folder_idx][sample_idx]
        target_rmap = torch.from_numpy(region_map['target_map'])
        contra_rmap = torch.from_numpy(region_map['gene_map'])

        # Get image size & crop_coords_top_left & target size
        original_size = torch.Tensor([[target_height_im, target_width_im], [contra_height_im, contra_width_im]])
        start_y, start_x = 0, 0
        crop_coords_top_left = torch.tensor([[start_y, start_x], [start_y, start_x]])
        target_size = torch.tensor([[self.size, self.size], [self.size, self.size]])

        # CFG training
        drop_image_embed = 0
        rand_num = random.random()
        if rand_num < self.i_drop_rate:
            drop_image_embed = 1
        elif rand_num < (self.i_drop_rate + self.t_drop_rate):
            prompt = ""
        elif rand_num < (self.i_drop_rate + self.t_drop_rate + self.ti_drop_rate):
            prompt = ""
            drop_image_embed = 1

        # Get input_ids * input_ids_2
        input_ids, input_ids_2 = make_prompt(self.tokenizer, self.tokenizer_2, prompt)

        content = {
            'prompt': prompt,
            'text_input_ids': input_ids,
            'text_input_ids_2': input_ids_2,
            'target_image': target_image,
            'contra_image': contra_image,
            'entity_image': entity_image,
            'target_rmap': target_rmap,
            'contra_rmap': contra_rmap,
            'original_size': original_size,
            'crop_coords_top_left': crop_coords_top_left,
            'target_size': target_size,
            'drop_image_embed': drop_image_embed,
        }

        return content

    def getitem_info(self, idx):
        # index = self.data[idx]
        result = self.get_contents(idx)
        return result

    def __getitem__(self, idx): 
        while True:
            result = self.getitem_info(idx)
            if result is not None:
                return result
            else:
                idx = random.randint(0, len(self.data) -1)
                # print("WARNING:use a random idx:{} ".format(idx))
                continue

    def __len__(self):
        return len(self.data)