import os
import random
import argparse
from pathlib import Path
import json
import itertools
import time

import torch
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from PIL import Image
from transformers import CLIPImageProcessor
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from diffusers import AutoencoderKL, DDPMScheduler, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPTextModelWithProjection

from ip_adapter.resampler import Resampler
from ip_adapter.utils import is_torch2_available
if is_torch2_available():
    from ip_adapter.attention_processor import IPAttnProcessor2_0 as IPAttnProcessor, AttnProcessor2_0 as AttnProcessor
else:
    from ip_adapter.attention_processor import IPAttnProcessor, AttnProcessor

# New added
import copy
import logging
from data.subject_dataset import SubjectContrastDataset


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If you are using multi-GPU
    np.random.seed(seed)  # Numpy module
    random.seed(seed)  # Python random module
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def weighted_mse_loss(predictions, targets, region_weight=None):
    diff = predictions - targets
    squared_diff = diff ** 2

    if region_weight is not None:
        squared_diff = squared_diff * region_weight
    
    loss = torch.mean(squared_diff)
    return loss


def get_state_dict(old_state_dict):
    image_proj_sd = {}
    ip_sd = {}
    for k in old_state_dict:
        if k.startswith("unet"):
            pass
        elif k.startswith("image_proj_model"):
            image_proj_sd[k.replace("image_proj_model.", "")] = old_state_dict[k].cpu().half()
        elif k.startswith("adapter_modules"):
            ip_sd[k.replace("adapter_modules.", "")] = old_state_dict[k].cpu().half()
    state_dict = {"image_proj": image_proj_sd, "ip_adapter": ip_sd}
    return state_dict


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def recover_image(img_tensor, mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)):
    mean = torch.FloatTensor(mean).cuda() if img_tensor.device.type == 'cuda' else torch.FloatTensor(mean)
    std = torch.FloatTensor(std).cuda() if img_tensor.device.type == 'cuda' else torch.FloatTensor(std)
    mean = mean.view(-1, 1, 1)
    std = std.view(-1, 1, 1)
    img_tensor = img_tensor * std + mean
    img_tensor = torch.clamp(img_tensor, 0, 1)
    img_np = img_tensor.permute(1, 2, 0).mul(255).cpu().byte().numpy()
    img_pil = Image.fromarray(img_np, 'RGB')

    return img_pil


def collate_fn(data):
    prompt = [example["prompt"] for example in data]
    text_input_ids = torch.cat([example["text_input_ids"] for example in data], dim=0)
    text_input_ids_2 = torch.cat([example["text_input_ids_2"] for example in data], dim=0)
    target_image = torch.stack([example["target_image"] for example in data], dim=0)
    contra_image = torch.stack([example["contra_image"] for example in data], dim=0)
    entity_image = torch.stack([example["entity_image"] for example in data], dim=0)
    target_rmap = torch.stack([example["target_rmap"] for example in data], dim=0)
    contra_rmap = torch.stack([example["contra_rmap"] for example in data], dim=0)
    original_size = torch.stack([example["original_size"] for example in data], dim=0)
    crop_coords_top_left = torch.stack([example["crop_coords_top_left"] for example in data], dim=0)
    target_size = torch.stack([example["target_size"] for example in data], dim=0)
    drop_image_embed = [example["drop_image_embed"] for example in data]

    return {
        "prompt": prompt,
        "text_input_ids": text_input_ids,
        "text_input_ids_2": text_input_ids_2,
        "target_image": target_image,
        "contra_image": contra_image,
        "entity_image": entity_image,
        "target_rmap": target_rmap,
        "contra_rmap": contra_rmap,
        "original_size": original_size,
        "crop_coords_top_left": crop_coords_top_left,
        "target_size": target_size,
        "drop_image_embed": drop_image_embed,
    }


class IPAdapter(torch.nn.Module):
    """IP-Adapter"""
    def __init__(self, unet, image_proj_model, adapter_modules, ckpt_path=None):
        super().__init__()
        self.unet = unet
        self.image_proj_model = image_proj_model
        self.adapter_modules = adapter_modules

        if ckpt_path is not None:
            self.load_from_checkpoint(ckpt_path)

    def forward(self, noisy_latents, timesteps, encoder_hidden_states, unet_added_cond_kwargs, image_embeds):
        # ip_tokens = self.image_proj_model(image_embeds)
        ip_tokens = self.image_proj_model(image_embeds)

        encoder_hidden_states = torch.cat([encoder_hidden_states, ip_tokens], dim=1)
        # Predict the noise residual
        noise_pred = self.unet(noisy_latents, timesteps, encoder_hidden_states, added_cond_kwargs=unet_added_cond_kwargs).sample
        return noise_pred

    def load_from_checkpoint(self, ckpt_path: str):
        # Calculate original checksums
        orig_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        orig_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        state_dict = torch.load(ckpt_path, map_location="cpu")

        # Check if 'latents' exists in both the saved state_dict and the current model's state_dict
        strict_load_image_proj_model = True
        if "latents" in state_dict["image_proj"] and "latents" in self.image_proj_model.state_dict():
            # Check if the shapes are mismatched
            if state_dict["image_proj"]["latents"].shape != self.image_proj_model.state_dict()["latents"].shape:
                print(f"Shapes of 'image_proj.latents' in checkpoint {ckpt_path} and current model do not match.")
                print("Removing 'latents' from checkpoint and loading the rest of the weights.")
                del state_dict["image_proj"]["latents"]
                strict_load_image_proj_model = False

        # Load state dict for image_proj_model and adapter_modules
        self.image_proj_model.load_state_dict(state_dict["image_proj"], strict=strict_load_image_proj_model)
        self.adapter_modules.load_state_dict(state_dict["ip_adapter"], strict=True)

        # Calculate new checksums
        new_ip_proj_sum = torch.sum(torch.stack([torch.sum(p) for p in self.image_proj_model.parameters()]))
        new_adapter_sum = torch.sum(torch.stack([torch.sum(p) for p in self.adapter_modules.parameters()]))

        # Verify if the weights have changed
        assert orig_ip_proj_sum != new_ip_proj_sum, "Weights of image_proj_model did not change!"
        assert orig_adapter_sum != new_adapter_sum, "Weights of adapter_modules did not change!"

        print(f"Successfully loaded weights from checkpoint {ckpt_path}")
    

def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_ip_adapter_path",
        type=str,
        default=None,
        help="Path to pretrained ip adapter model. If not specified weights are initialized randomly.",
    )
    parser.add_argument(
        "--data_json_file",
        type=str,
        default=None,
        help="Training data",
    )
    parser.add_argument(
        "--data_root_path",
        type=str,
        default="",
        required=True,
        help="Training data root path",
    )
    parser.add_argument(
        "--image_encoder_path",
        type=str,
        default=None,
        required=True,
        help="Path to CLIP image encoder",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="sd-ip_adapter",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images"
        ),
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Learning rate to use.",
    )
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--num_train_epochs", type=int, default=100)
    parser.add_argument(
        "--train_batch_size", type=int, default=8, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--noise_offset", type=float, default=None, help="noise offset")
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument(
        "--stop_step",
        type=int,
        default=30000,
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=2000,
        help=(
            "Save a checkpoint of the training state every X updates"
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    # New added
    parser.add_argument("--patch_quality_file", type=str, default=None)
    parser.add_argument("--use_dpo_loss", type=str2bool, default=False)
    
    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args
    

def main():
    args = parse_args()
    logging_dir = Path(args.output_dir, args.logging_dir)

    # Set seed
    set_seed(42)  # dagger

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)

    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
        logger = logging.getLogger('my_logger')
        logger.setLevel(logging.DEBUG)
        file_handler = logging.FileHandler(os.path.join(args.output_dir, 'log.log'))
        file_handler.setLevel(logging.DEBUG)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
    else:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())

    logger.info(args)
    # Load scheduler, tokenizer and models.
    noise_scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    tokenizer_2 = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2")
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder_2")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="unet")
    image_encoder = CLIPVisionModelWithProjection.from_pretrained(args.image_encoder_path)
    # Freeze parameters of models to save more memory
    unet.requires_grad_(False)
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    image_encoder.requires_grad_(False)
    
    # Ip-adapter-plus
    num_tokens = 16
    image_proj_model = Resampler(
        dim=1280,
        depth=4,
        dim_head=64,
        heads=20,
        num_queries=num_tokens,
        embedding_dim=image_encoder.config.hidden_size,
        output_dim=unet.config.cross_attention_dim,
        ff_mult=4,
    )
    # Init adapter modules
    attn_procs = {}
    unet_sd = unet.state_dict()
    for name in unet.attn_processors.keys():
        cross_attention_dim = None if name.endswith("attn1.processor") else unet.config.cross_attention_dim
        if name.startswith("mid_block"):
            hidden_size = unet.config.block_out_channels[-1]
        elif name.startswith("up_blocks"):
            block_id = int(name[len("up_blocks.")])
            hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
        elif name.startswith("down_blocks"):
            block_id = int(name[len("down_blocks.")])
            hidden_size = unet.config.block_out_channels[block_id]
        if cross_attention_dim is None:
            attn_procs[name] = AttnProcessor()
        else:
            layer_name = name.split(".processor")[0]
            weights = {
                "to_k_ip.weight": unet_sd[layer_name + ".to_k.weight"],
                "to_v_ip.weight": unet_sd[layer_name + ".to_v.weight"],
            }
            attn_procs[name] = IPAttnProcessor(hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, num_tokens=num_tokens)
            attn_procs[name].load_state_dict(weights)
    unet.set_attn_processor(attn_procs)
    adapter_modules = torch.nn.ModuleList(unet.attn_processors.values())
    
    ip_adapter = IPAdapter(unet, image_proj_model, adapter_modules, args.pretrained_ip_adapter_path)
    
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16
    # unet.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device) # use fp32
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    text_encoder_2.to(accelerator.device, dtype=weight_dtype)
    image_encoder.to(accelerator.device, dtype=weight_dtype)
    
    # Optimizer
    params_to_opt = itertools.chain(ip_adapter.adapter_modules.parameters())
    # params_to_opt = itertools.chain(ip_adapter.image_proj_model.parameters(), ip_adapter.adapter_modules.parameters())
    optimizer = torch.optim.AdamW(params_to_opt, lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Dataloader
    train_dataset = SubjectContrastDataset(args.data_root_path, args.patch_quality_file, tokenizer=tokenizer, tokenizer_2=tokenizer_2, size=args.resolution, num_smaple_per_folder=1)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
    
    # Prepare everything with our `accelerator`.
    ip_adapter, optimizer, train_dataloader = accelerator.prepare(ip_adapter, optimizer, train_dataloader)

    global_step = 0
    for epoch in range(0, args.num_train_epochs):
        begin = time.perf_counter()
        for step, batch in enumerate(train_dataloader):
            load_data_time = time.perf_counter() - begin
            # New added
            cat_images = torch.cat([batch['target_image'], batch['contra_image']], dim=0)    # (2 * B, C, H, W), target + contra
            prompt = batch['prompt'] + batch['prompt']
            original_size = torch.cat(batch["original_size"].chunk(2, dim=1), dim=0).squeeze(dim=1).to(accelerator.device)
            crop_coords_top_left = torch.cat(batch["crop_coords_top_left"].chunk(2, dim=1), dim=0).squeeze(dim=1).to(accelerator.device)
            target_size = torch.cat(batch["target_size"].chunk(2, dim=1), dim=0).squeeze(dim=1).to(accelerator.device)

            with accelerator.accumulate(ip_adapter):
                # Convert images to latent space
                with torch.no_grad():
                    # vae of sdxl should use fp32
                    latents = vae.encode(cat_images.to(accelerator.device, dtype=torch.float32)).latent_dist.mean
                    latents = latents * vae.config.scaling_factor
                    latents = latents.to(accelerator.device, dtype=weight_dtype)

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                if args.noise_offset:
                    # https://www.crosslabs.org//blog/diffusion-with-offset-noise
                    noise += args.noise_offset * torch.randn((latents.shape[0], latents.shape[1], 1, 1)).to(accelerator.device, dtype=weight_dtype)

                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()
                # New added, make timesteps and noise same for pairs in DPO
                timesteps = timesteps.chunk(2)[0].repeat(2)
                noise = noise.chunk(2)[0].repeat(2, 1, 1, 1)

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
            
                clip_images = []
                for clip_image, drop_image_embed in zip(batch["entity_image"].squeeze(dim=1), batch["drop_image_embed"]):
                    if drop_image_embed == 1:
                        clip_images.append(torch.zeros_like(clip_image))
                    else:
                        clip_images.append(clip_image)
                clip_images = torch.stack(clip_images, dim=0)
                with torch.no_grad():
                    # New added
                    image_embeds = image_encoder(clip_images.to(accelerator.device, dtype=weight_dtype), output_hidden_states=True).hidden_states[-2]
                    # New added, repeat image embeds for DPO
                    image_embeds = image_embeds.repeat(2, 1, 1)

                with torch.no_grad():
                    encoder_output = text_encoder(batch['text_input_ids'].to(accelerator.device), output_hidden_states=True)
                    text_embeds = encoder_output.hidden_states[-2]
                    encoder_output_2 = text_encoder_2(batch['text_input_ids_2'].to(accelerator.device), output_hidden_states=True)
                    pooled_text_embeds = encoder_output_2[0]
                    text_embeds_2 = encoder_output_2.hidden_states[-2]
                    text_embeds = torch.concat([text_embeds, text_embeds_2], dim=-1) # Concat
                    # New added, repeat text embeds for DPO
                    text_embeds = text_embeds.repeat(2, 1, 1)
                    pooled_text_embeds = pooled_text_embeds.repeat(2, 1)
                        
                # add cond
                add_time_ids = [
                    batch["original_size"].to(accelerator.device),
                    batch["crop_coords_top_left"].to(accelerator.device),
                    batch["target_size"].to(accelerator.device),
                ]
                add_time_ids = torch.cat(add_time_ids, dim=1).to(accelerator.device, dtype=weight_dtype)
                unet_added_cond_kwargs = {"text_embeds": pooled_text_embeds, "time_ids": add_time_ids}
                
                noise_pred = ip_adapter(noisy_latents, timesteps, text_embeds, unet_added_cond_kwargs, image_embeds)

                if not args.use_dpo_loss:
                    loss = F.mse_loss(noise_pred.float(), noise.float(), reduction="mean")
                else:
                    target_rmap = batch["target_rmap"].unsqueeze(dim=1)   # (bsz, 1, fea_h, fea_w)
                    contra_rmap = batch["contra_rmap"].unsqueeze(dim=1)   # (bsz, 1, fea_h, fea_w)
                    region_weight = torch.cat([1 - target_rmap, contra_rmap], dim=0)  # (2 * bsz, 1, fea_h, fea_w)

                    loss = weighted_mse_loss(noise_pred.float(), noise.float(), region_weight)

                # Gather the losses across all processes for logging (if we use distributed training).
                avg_loss = accelerator.gather(loss.repeat(args.train_batch_size)).mean().item()
                
                # Backpropagate
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()

                if accelerator.is_main_process and step % 10 == 0:
                    logger.info("Epoch {}, step {}, data_time: {}, time: {}, step_loss: {}".format(
                        epoch, step, load_data_time, time.perf_counter() - begin, avg_loss))
            
            global_step += 1

            if global_step % args.save_steps == 0 or global_step == args.stop_step:
                save_dir = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, "model.bin")
                # accelerator.save_state(save_path)
                unwrapped_ip_adapter = accelerator.unwrap_model(ip_adapter, keep_fp32_wrapper=True)
                old_state_dict = unwrapped_ip_adapter.state_dict()
                state_dict = get_state_dict(old_state_dict)
                torch.save(state_dict, save_path)
            
            begin = time.perf_counter()
            if global_step >= args.stop_step:
                break
        if global_step >= args.stop_step:
            break
                
if __name__ == "__main__":
    main()