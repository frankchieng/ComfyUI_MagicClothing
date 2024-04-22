import argparse
import folder_paths

import torch
import os
import numpy as np
from PIL import Image
from diffusers import UniPCMultistepScheduler, AutoencoderKL, ControlNetModel

from einops import rearrange
from .garment_adapter.garment_diffusion import ClothAdapter
from .pipelines.OmsDiffusionControlNetPipeline import OmsDiffusionControlNetPipeline
from diffusers.pipelines import StableDiffusionControlNetPipeline
device = "cuda" if torch.cuda.is_available() else "cpu"
pipe_path = "SG161222/Realistic_Vision_V4.0_noVAE"

def find_safetensors_files(directory):
    safetensors_files = [
        filename
        for filename in os.listdir(directory)
        if filename.endswith('.safetensors') and os.path.isfile(os.path.join(directory, filename))
    ]
    return safetensors_files

class GarmentLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_path": (folder_paths.get_filename_list("checkpoints"),),
                "enable_cloth_guidance": ("BOOLEAN", {"default": True}),
            }
        }
    
    RETURN_TYPES= ("MODEL",)
    RETURN_NAMES=("garment_pipe",)
    FUNCTION="load_pipe"
    CATEGORY = "MagicClothing"

    def load_pipe(self, model_path,enable_cloth_guidance):
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(dtype=torch.float16)
        control_net_openpose = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_openpose", torch_dtype=torch.float16)
        if enable_cloth_guidance:
            pipe = OmsDiffusionControlNetPipeline.from_pretrained(pipe_path, vae=vae, torch_dtype=torch.float16, controlnet=control_net_openpose)
        else:
            pipe = StableDiffusionControlNetPipeline.from_pretrained(pipe_path, vae=vae, torch_dtype=torch.float16, controlnet=control_net_openpose)
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        print(model_path)
        full_net = ClothAdapter(pipe, folder_paths.get_full_path("checkpoints", model_path), device, enable_cloth_guidance)
        return (full_net,)

class GarmentGenerate:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "garment_pipe": ("MODEL",),
                "cloth_image": ("IMAGE",),
                "prompt": ("STRING", {"default": "a photography of a model"}),
                "cloth_mask_image": ("IMAGE", ),
            },
            "optional": {
                "num_samples": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                "n_prompt": ("STRING", {"default": "bare, monochrome, lowres, bad anatomy, worst quality, low quality"}),
                "seed": ("INT", {"default": 42}),
                "scale": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10.0, "step": 0.1}),
                "cloth_guidance_scale": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10.0, "step": 0.1}),
                "sample_steps": ("INT", {"default": 20, "min": 1, "max": 100, "step": 1}),
                "height": ("INT", {"default": 768, "min": 256, "max": 1024, "step": 1}),
                "width": ("INT", {"default": 576, "min": 192, "max": 768, "step": 1}),
                "pose_image": ("IMAGE", ),        
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("images", "cloth_mask_image")
    OUTPUT_NODE = True
    CATEGORY = "MagicClothing"
    FUNCTION = "garment_generation"
    
    def garment_generation(self, garment_pipe, cloth_image, prompt, num_samples, n_prompt, seed, scale, cloth_guidance_scale, sample_steps, height, width, cloth_mask_image, pose_image=None):
        a_prompt = ', best quality, high quality'
        numpy_image = torch.squeeze(cloth_image, 0)
        numpy_image = (numpy_image.numpy() * 255).astype(np.uint8)
        cloth_image = Image.fromarray(numpy_image)

        cloth_mask_image = torch.squeeze(cloth_mask_image, 0)
        cloth_mask_image=(cloth_mask_image.numpy() * 255).astype(np.uint8)
        cloth_mask_image=Image.fromarray(cloth_mask_image)

        print(pose_image.shape)
        pose_image = rearrange(pose_image, "b h w c -> b c h w")
                                                   
        images, cloth_mask_image = garment_pipe.generate(
            cloth_image,
            cloth_mask_image, 
            prompt, 
            a_prompt, 
            num_samples, 
            n_prompt, 
            seed, 
            scale, 
            cloth_guidance_scale, 
            sample_steps, 
            height, 
            width,
            image=pose_image
        )
                    
        images = np.array(images).astype(np.float32) / 255.0
        images = torch.from_numpy(images)
        cloth_mask_image = np.array(cloth_mask_image).astype(np.float32) / 255.0
        cloth_mask_image = torch.unsqueeze(torch.from_numpy(cloth_mask_image), 0)
        return (images, cloth_mask_image)