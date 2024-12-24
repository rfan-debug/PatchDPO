import os

import json
import torch
import argparse
from diffusers import StableDiffusionXLPipeline
from ip_adapter import IPAdapterPlusXL
from transformers import CLIPTokenizer
from data.dreambooth_dataset import DreamboothDataset
from data.dreambooth_dataset_kosmosg import DreamboothDatasetKosmosg
from accelerate import Accelerator
from evaluation.evaluate_dreambooth_func import evaluate_scores


def collate_fn(data):
    entity_idxes = [example["entity_idx"] for example in data]
    prompts = [example["prompt"] for example in data]
    entity_dir_names = [example["entity_dir_name"] for example in data]
    entity_names = [example["entity_name"] for example in data]
    clip_images = []
    for example in data:
        clip_images.append(example["entity_img"])

    return {
        "entity_idxes": entity_idxes,
        "clip_images": clip_images,
        "prompts": prompts,
        "entity_dir_names": entity_dir_names,
        "entity_names": entity_names,
    }


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser("metric", add_help=False)
    parser.add_argument("--base_model_path", type=str)
    parser.add_argument("--image_encoder_path", type=str)
    parser.add_argument("--ip_ckpt", type=str)
    parser.add_argument("--data_root", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--scale", type=float, default=0.6)
    parser.add_argument("--is_kosmosg", type=str2bool, default=False)
    return parser.parse_args()


args = parse_args()
base_model_path = args.base_model_path
image_encoder_path = args.image_encoder_path
output_dir = args.output_dir
ip_ckpt = args.ip_ckpt

accelerator = Accelerator()
device = "mps"
resolution = 512
batch_size = 2
num_tokens = 16
max_num_objects = 1
num_samples = 1
scale = args.scale

# Load model
tokenizer = CLIPTokenizer.from_pretrained(base_model_path, subfolder="tokenizer")
tokenizer_2 = CLIPTokenizer.from_pretrained(base_model_path, subfolder="tokenizer_2")

# Load SDXL pipeline
pipe = StableDiffusionXLPipeline.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    add_watermarker=False,
)
pipe.enable_vae_slicing()
pipe.to(device)

# Load ip-adapter
ip_model = IPAdapterPlusXL(pipe, image_encoder_path, None, ip_ckpt, device, num_tokens=num_tokens)

# Load dataset
dataset_type = DreamboothDataset if not args.is_kosmosg else DreamboothDatasetKosmosg   # Here, choose the dataset type (original version or kosmosg version)
dataset = dataset_type(args.data_root, tokenizer=tokenizer, size=resolution)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
dataloader = accelerator.prepare(dataloader)

os.makedirs(output_dir, exist_ok=True)
print('Scale : {}'.format(scale))

for step, batch in enumerate(dataloader):
    clip_images = batch['clip_images']
    entity_idxes = batch['entity_idxes']
    prompts = batch['prompts']
    entity_dir_names = batch['entity_dir_names']
    entity_names = batch['entity_names']

    generated_images = ip_model.generate_multi(pil_image=clip_images, num_samples=num_samples, num_inference_steps=30, seed=42, prompt=prompts, scale=scale)
    
    for cur_idx in range(batch_size):
        cur_prompt = prompts[cur_idx]
        entity_idx = entity_idxes[cur_idx]
        entity_dir_name = entity_dir_names[cur_idx]
        entity_dir = os.path.join(output_dir, entity_dir_name)
        os.makedirs(entity_dir, exist_ok=True)
        
        if isinstance(clip_images[cur_idx], list):
            cur_entity_image = clip_images[cur_idx][0]
        else:
            cur_entity_image = clip_images[cur_idx]

        cur_generated_images = generated_images[cur_idx * num_samples: (cur_idx + 1) * num_samples]

        for idx, image in enumerate(cur_generated_images):
            image.save(os.path.join(entity_dir, '{}.png'.format(cur_prompt)))

accelerator.wait_for_everyone()
if accelerator.is_main_process:
    clipt_score, clipi_score, dino_score = evaluate_scores(output_dir, args.data_root, args.is_kosmosg)
    save_score_dict = {
        'clipt_score': str(clipt_score),
        'clipi_score': str(clipi_score),
        'dino_score': str(dino_score),
    }
    save_score_path = os.path.join(output_dir, 'all_score.json')
    with open(save_score_path, 'w') as file:
        json.dump(save_score_dict, file, indent=4)
    print('clipt_score: %.4f, clipi_score: %.4f, dino_score: %.4f' % (clipt_score, clipi_score, dino_score))