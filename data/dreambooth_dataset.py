import os
import json
import torch
import random
import numpy as np
from PIL import Image
from torchvision import transforms


objects = [
    ('backpack', 'backpack'),
    ('backpack_dog', 'backpack'),
    ('bear_plushie', 'stuffed animal'),
    ('berry_bowl', 'bowl'),
    ('can', 'can'),
    ('candle', 'candle'),
    ('clock', 'clock'),
    ('colorful_sneaker', 'sneaker'),
    ('duck_toy', 'toy'),
    ('fancy_boot', 'boot'),
    ('grey_sloth_plushie', 'stuffed animal'),
    ('monster_toy', 'toy'),
    ('pink_sunglasses', 'glasses'),
    ('poop_emoji', 'toy'),
    ('rc_car', 'toy'),
    ('red_cartoon', 'cartoon'),
    ('robot_toy', 'toy'),
    ('shiny_sneaker', 'sneaker'),
    ('teapot', 'teapot'),
    ('vase', 'vase'),
    ('wolf_plushie', 'stuffed animal'),
]


animals = [
    ('cat', 'cat'),
    ('cat2', 'cat'),
    ('dog', 'dog'),
    ('dog2', 'dog'),
    ('dog3', 'dog'),
    ('dog5', 'dog'),
    ('dog6', 'dog'),
    ('dog7', 'dog'),
    ('dog8', 'dog'),
]


object_prompts = [
    'a {0} in the jungle',
    'a {0} in the snow',
    'a {0} on the beach',
    'a {0} on a cobblestone street',
    'a {0} on top of pink fabric',
    'a {0} on top of a wooden floor',
    'a {0} with a city in the background',
    'a {0} with a mountain in the background',
    'a {0} with a blue house in the background',
    'a {0} on top of a purple rug in a forest',
    'a {0} with a wheat field in the background',
    'a {0} with a tree and autumn leaves in the background',
    'a {0} with the Eiffel Tower in the background',
    'a {0} floating on top of water',
    'a {0} floating in an ocean of milk',
    'a {0} on top of green grass with sunflowers around it',
    'a {0} on top of a mirror',
    'a {0} on top of the sidewalk in a crowded street',
    'a {0} on top of a dirt road',
    'a {0} on top of a white rug',
    'a red {0}',
    'a purple {0}',
    'a shiny {0}',
    'a wet {0}',
    'a cube shaped {0}'
]


animal_prompts = [
    'a {0} in the jungle',
    'a {0} in the snow',
    'a {0} on the beach',
    'a {0} on a cobblestone street',
    'a {0} on top of pink fabric',
    'a {0} on top of a wooden floor',
    'a {0} with a city in the background',
    'a {0} with a mountain in the background',
    'a {0} with a blue house in the background',
    'a {0} on top of a purple rug in a forest',
    'a {0} wearing a red hat',
    'a {0} wearing a santa hat',
    'a {0} wearing a rainbow scarf',
    'a {0} wearing a black top hat and a monocle',
    'a {0} in a chef outfit',
    'a {0} in a firefighter outfit',
    'a {0} in a police outfit',
    'a {0} wearing pink glasses',
    'a {0} wearing a yellow shirt',
    'a {0} in a purple wizard outfit',
    'a red {0}',
    'a purple {0}',
    'a shiny {0}',
    'a wet {0}',
    'a cube shaped {0}'
]


def resize_padding_image(img, size=224):
    aspect_ratio = img.width / img.height
    if aspect_ratio > 1.0:
        new_width = size
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = size
        new_width = int(new_height * aspect_ratio)

    padding_left = (size - new_width) // 2
    padding_right = size - new_width - padding_left
    padding_top = (size - new_height) // 2
    padding_bottom = size - new_height - padding_top

    transform = transforms.Compose([
        transforms.Resize((new_height, new_width)),
        transforms.Pad((padding_left, padding_top, padding_right, padding_bottom), fill=255),
    ])
    transformed_img = transform(img)
    return transformed_img


def resize_image(img, size=224):
    transform = transforms.Compose([
        transforms.Resize((size, size)),
    ])
    transformed_img = transform(img)
    return transformed_img


class DreamboothDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, tokenizer, size=512):
        super().__init__()
        self.size = size
        self.tokenizer = tokenizer
        self.all_entities = {
            'objects': objects,
            'animals': animals,
        }

        self.img_dir = os.path.join(data_root, 'dataset')
        self.json_dir = os.path.join(data_root, 'json_data')
        self.data = []
        entity_idx = 0
        for key in self.all_entities:
            cur_entities = self.all_entities[key]
            for cur_entity in cur_entities:
                entity_dir_name, entity_name = cur_entity[0], cur_entity[1]
                entity_img_names = [x.split('.')[0] for x in os.listdir(os.path.join(data_root, 'json_data', entity_dir_name))]
                prompts = object_prompts if 'object' in key else animal_prompts
                for prompt in prompts:
                    prompt = prompt.replace('{0}', entity_name)
                    data_dict = {
                        'entity_idx': entity_idx,
                        'index': len(self.data),
                        'prompt': prompt,
                        'entity_dir_name': entity_dir_name,
                        'entity_img_names': entity_img_names,
                        'entity_name': entity_name,
                    }
                    self.data.append(data_dict)
                entity_idx += 1

    def get_result(self, sample, original_size=800):
        entity_idx = sample["entity_idx"]
        item = sample["index"]
        prompt = sample["prompt"]
        entity_dir_name = sample["entity_dir_name"]
        entity_img_names = sample["entity_img_names"]
        entity_name = sample["entity_name"]

        entity_img_paths = [os.path.join(self.img_dir, entity_dir_name, img_name + '.jpg') for img_name in entity_img_names]
        entity_json_paths = [os.path.join(self.json_dir, entity_dir_name, img_name + '.json') for img_name in entity_img_names]

        # Read images
        entity_imgs = [Image.open(img_path) for img_path in entity_img_paths]
        # Read jsons
        entity_datas = []
        for json_path in entity_json_paths:
            with open(json_path, 'r') as f:
                entity_datas.append(json.load(f))
        entity_bboxes = [np.array(data['mask'][1]['box']).astype(np.int32) for data in entity_datas]

        processed_entity_imgs = []
        for entity_bbox, entity_img in zip(entity_bboxes, entity_imgs):
            cur_size = entity_img.size[0]
            entity_bbox = (np.array(entity_bbox) / original_size * cur_size).astype(np.int32)   # Resize the bounding box
            entity_img = entity_img.crop((entity_bbox[0], entity_bbox[1], entity_bbox[2], entity_bbox[3]))
            entity_img = resize_padding_image(entity_img, size=224)
            processed_entity_imgs.append(entity_img)

        ret_val = {"entity_dir_name": entity_dir_name,
                   "entity_name": entity_name,
                   "entity_img": processed_entity_imgs,
                   "prompt": prompt,
                   "entity_idx": entity_idx}

        return ret_val

    def getitem_info(self, idx):
        result = self.data[idx]
        result = self.get_result(result)
        return result

    def __getitem__(self, idx): 
        while True:
            result = self.getitem_info(idx)
            if result is not None:
                return result
            else:
                idx = random.randint(0, len(self.data) -1)
                continue

    def __len__(self):
        return len(self.data)