import argparse
import folder_paths
import torch
import os
import numpy as np
from PIL import Image
from diffusers import UniPCMultistepScheduler, AutoencoderKL
from diffusers.pipelines import StableDiffusionPipeline

from .garment_adapter.garment_diffusion import ClothAdapter
from .pipelines.OmsDiffusionPipeline import OmsDiffusionPipeline

device = "cuda" if torch.cuda.is_available() else "cpu"
pipe_path = "SG161222/Realistic_Vision_V4.0_noVAE"
faceid_version = ['FaceID', 'FaceIDPlus', 'FaceIDPlusV2']

folder_paths.folder_names_and_paths["ckpt"] = (
    [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints'),
    ],
    [".safetensors"]
)

checkpoints_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')
ipadapter_faceid_path = os.path.join(checkpoints_path, 'ipadapter_faceid')

def find_safetensors_files(directory):
    safetensors_files = [
        filename
        for filename in os.listdir(directory)
        if filename.endswith('.safetensors') and os.path.isfile(os.path.join(directory, filename))
    ]
    return safetensors_files

class GarmentGenerate:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "cloth_image": ("IMAGE",),
                "prompt": ("STRING", {"default": "a photography of a model"}),
                "model_path": (find_safetensors_files(checkpoints_path),),
                "enable_cloth_guidance": ("BOOLEAN", {"default": True}),
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
                "faceid_version": (faceid_version,),
                "face_image": ("IMAGE", ),
                "pose_image": ("IMAGE", ),                
                "cloth_mask_image": ("IMAGE", ),
            },
        }

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("images", "cloth_mask_image")
    OUTPUT_NODE = True
    CATEGORY = "MagicClothing"
    FUNCTION = "garment_generation"
    
    def garment_generation(self, cloth_image, prompt, model_path, enable_cloth_guidance, num_samples, n_prompt, seed, scale, cloth_guidance_scale, sample_steps, height, width, faceid_version, cloth_mask_image=None, face_image=None, pose_image=None):
        vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(dtype=torch.float16)
        a_prompt = 'best quality, high quality'
        numpy_image = torch.squeeze(cloth_image, 0)
        numpy_image = (numpy_image.numpy() * 255).astype(np.uint8)
        cloth_image = Image.fromarray(numpy_image)
        #ipadapter_faceid generation
        if face_image is not None and pose_image is None:
            if enable_cloth_guidance:
                pipe = OmsDiffusionPipeline.from_pretrained(pipe_path, vae=vae, torch_dtype=torch.float16)
            else:
                pipe = StableDiffusionPipeline.from_pretrained(pipe_path, vae=vae, torch_dtype=torch.float16)
            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

            face_image = torch.squeeze(face_image, 0)
            face_image = (face_image.numpy() * 255).astype(np.uint8)
            face_image = Image.fromarray(face_image)
            if faceid_version == "FaceID":
                ip_lora = folder_paths.get_full_path("loras", "ip-adapter-faceid_sd15_lora.safetensors")
                ip_ckpt = folder_paths.get_full_path("ipadapter", "ip-adapter-faceid_sd15.bin")
                pipe.load_lora_weights(ip_lora)
                pipe.fuse_lora()
                from .garment_adapter.garment_ipadapter_faceid import IPAdapterFaceID
                
                ip_model = IPAdapterFaceID(pipe, folder_paths.get_full_path("ckpt", model_path), ip_ckpt, device, enable_cloth_guidance)
                result = ip_model.generate(cloth_image, face_image, cloth_mask_image, prompt, a_prompt, n_prompt, num_samples, seed, scale, cloth_guidance_scale, sample_steps, height, width)
            else:
                if faceid_version == "FaceIDPlus":
                    ip_lora = folder_paths.get_full_path("loras", "ip-adapter-faceid-plus_sd15_lora.safetensors")
                    ip_ckpt = folder_paths.get_full_path("ipadapter", "ip-adapter-faceid-plus_sd15.bin")
                    v2 = False
                else:
                    ip_lora = folder_paths.get_full_path("loras", "ip-adapter-faceid-plusv2_sd15_lora.safetensors")
                    ip_ckpt = folder_paths.get_full_path("ipadapter", "ip-adapter-faceid-plusv2_sd15.bin")
                    v2 = True

                pipe.load_lora_weights(ip_lora)
                pipe.fuse_lora()
                image_encoder_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
                from .garment_adapter.garment_ipadapter_faceid import IPAdapterFaceIDPlus as IPAdapterFaceID

                ip_model = IPAdapterFaceID(pipe, folder_paths.get_full_path("ckpt", model_path), image_encoder_path, ip_ckpt, device, enable_cloth_guidance)
                result = ip_model.generate(cloth_image, face_image, cloth_mask_image, prompt, a_prompt, n_prompt, num_samples, seed, scale, cloth_guidance_scale, sample_steps, height, width, shortcut=v2)
                
            if result is None:
                raise NotImplementedError("face detection error,plz try another portrait!")
            else:
                images, cloth_mask_image = result
        
        #ipadapter_faceid with controlnet openpose generation
        elif face_image is not None and pose_image is not None:
            from .pipelines.OmsDiffusionControlNetPipeline import OmsDiffusionControlNetPipeline
            from diffusers.pipelines import StableDiffusionControlNetPipeline
            #from controlnet_aux import OpenposeDetector    
            from diffusers import ControlNetModel  
            #openpose_model = OpenposeDetector.from_pretrained("lllyasviel/ControlNet").to(device)
            control_net_openpose = ControlNetModel.from_pretrained("lllyasviel/control_v11p_sd15_openpose", torch_dtype=torch.float16)
            if enable_cloth_guidance:
                pipe = OmsDiffusionControlNetPipeline.from_pretrained(pipe_path, vae=vae, controlnet=control_net_openpose, torch_dtype=torch.float16)
            else:
                pipe = StableDiffusionControlNetPipeline.from_pretrained(pipe_path, vae=vae, controlnet=control_net_openpose, torch_dtype=torch.float16)
            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)     
            face_image = torch.squeeze(face_image, 0)
            face_image = (face_image.numpy() * 255).astype(np.uint8)
            face_image = Image.fromarray(face_image)
            
            pose_image = torch.squeeze(pose_image, 0)
            pose_image = (pose_image.numpy() * 255).astype(np.uint8)
            pose_image = Image.fromarray(pose_image)        
            
            if faceid_version == "FaceID":
                ip_lora = folder_paths.get_full_path("loras", "ip-adapter-faceid_sd15_lora.safetensors")
                ip_ckpt = folder_paths.get_full_path("ipadapter", "ip-adapter-faceid_sd15.bin")
                pipe.load_lora_weights(ip_lora)
                pipe.fuse_lora()
                from .garment_adapter.garment_ipadapter_faceid import IPAdapterFaceID
                
                ip_model = IPAdapterFaceID(pipe, folder_paths.get_full_path("ckpt", model_path), ip_ckpt, device, enable_cloth_guidance)
                result = ip_model.generate(cloth_image, face_image, cloth_mask_image, prompt, a_prompt, n_prompt, num_samples, seed, scale, cloth_guidance_scale, sample_steps, height, width, image=pose_image)
            else:
                if faceid_version == "FaceIDPlus":
                    ip_lora = folder_paths.get_full_path("loras", "ip-adapter-faceid-plus_sd15_lora.safetensors")
                    ip_ckpt = folder_paths.get_full_path("ipadapter", "ip-adapter-faceid-plus_sd15.bin")
                    v2 = False
                else:
                    ip_lora = folder_paths.get_full_path("loras", "ip-adapter-faceid-plusv2_sd15_lora.safetensors")
                    ip_ckpt = folder_paths.get_full_path("ipadapter", "ip-adapter-faceid-plusv2_sd15.bin")
                    v2 = True

                pipe.load_lora_weights(ip_lora)
                pipe.fuse_lora()
                image_encoder_path = "laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
                from .garment_adapter.garment_ipadapter_faceid import IPAdapterFaceIDPlus as IPAdapterFaceID

                ip_model = IPAdapterFaceID(pipe, folder_paths.get_full_path("ckpt", model_path), image_encoder_path, ip_ckpt, device, enable_cloth_guidance)
                result = ip_model.generate(cloth_image, face_image, cloth_mask_image, prompt, a_prompt, n_prompt, num_samples, seed, scale, cloth_guidance_scale, sample_steps, height, width, shortcut=v2, image=pose_image)
                
            if result is None:
                raise NotImplementedError("face detection error,plz try another portrait!")
            else:
                images, cloth_mask_image = result                                   
                   
        #only cloth reference image generation
        elif face_image is None and pose_image is None:
            if enable_cloth_guidance:
                pipe = OmsDiffusionPipeline.from_pretrained(pipe_path, vae=vae, torch_dtype=torch.float16)
            else:
                pipe = StableDiffusionPipeline.from_pretrained(pipe_path, vae=vae, torch_dtype=torch.float16)
            pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
            full_net = ClothAdapter(pipe, folder_paths.get_full_path("ckpt", model_path), device, enable_cloth_guidance)
            images, cloth_mask_image = full_net.generate(cloth_image, cloth_mask_image, prompt, a_prompt, num_samples, n_prompt, seed, scale, cloth_guidance_scale, sample_steps, height, width)
                    
        images = np.array(images).astype(np.float32) / 255.0
        images = torch.from_numpy(images)
        cloth_mask_image = np.array(cloth_mask_image).astype(np.float32) / 255.0
        cloth_mask_image = torch.unsqueeze(torch.from_numpy(cloth_mask_image), 0)
        return (images, cloth_mask_image)
        
    
