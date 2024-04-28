### Updates:
- ✅ [2024/04/17] only cloth image with prompt generation
- ✅ [2024/04/18] IPAdapter FaceID with human face detection and synthesize with cloth image generation
- ✅ [2024/04/18] IPAdapter FaceID with controlnet openpose and synthesize with cloth image generation
- ✅ [2024/04/19] lower-body and full-body models for preliminary experiment
- ✅ [2024/04/26] AnimateDiff and cloth inpaiting have been supported
  
U can contact me thr ![twitter_1](https://github.com/frankchieng/ComfyUI_Aniportrait/assets/130369523/27b4fcae-e50c-477d-86f4-dacf7fd052f4)[twitter](https://twitter.com/kurtqian) ![wechat_1](https://github.com/frankchieng/ComfyUI_Aniportrait/assets/130369523/b95cd0a2-4188-4eb3-b1de-5f6eeab71045) Weixin：GalaticKing

#### [the main workflow](https://github.com/frankchieng/ComfyUI_MagicClothing/blob/main/assets/magic_clothing_workflow.json)
![1713496499658](https://github.com/frankchieng/ComfyUI_MagicClothing/assets/130369523/59f380c8-faf9-4544-ae57-3aa36021652c)

#### [IPAdapater FaceID workflow](https://github.com/frankchieng/ComfyUI_MagicClothing/blob/main/assets/ipadapter_faceid_workflow.json)
![1713496598257](https://github.com/frankchieng/ComfyUI_MagicClothing/assets/130369523/05bd294f-fd9f-439e-bfbf-2da3541ebb79)

#### [IPAdapater FaceID chained with controlnet openpose workflow](https://github.com/frankchieng/ComfyUI_MagicClothing/blob/main/assets/ipadapter_faceid_openpose_workflow.json)
![IPadapter_faceid_openpose](https://github.com/frankchieng/ComfyUI_MagicClothing/assets/130369523/3fca5f7f-f9db-410a-bc33-9f69f6442ecf)

#### [lower-body full-body workflow](https://github.com/frankchieng/ComfyUI_MagicClothing/blob/main/assets/lower%26full_body_workflow.json)
![lower_body](https://github.com/frankchieng/ComfyUI_MagicClothing/assets/130369523/39a589fb-dea1-4985-84b3-d89bf46038b1)
![dress](https://github.com/frankchieng/ComfyUI_MagicClothing/assets/130369523/8204c985-5da8-4723-ba40-119da03b2df3)

#### [full-body workflow with IPadapterFaceid](https://github.com/frankchieng/ComfyUI_MagicClothing/blob/main/assets/fullbody_ipadaterfaceid_workflow.json)
![fullbody_ipadapter](https://github.com/frankchieng/ComfyUI_MagicClothing/assets/130369523/be180181-3690-4803-a52b-47e6ee1192ab)

#### [cloth inpainting workflow](https://github.com/frankchieng/ComfyUI_MagicClothing/blob/main/assets/cloth_inpainting_workflow.json)
![cloth_inpainting](https://github.com/frankchieng/ComfyUI_MagicClothing/assets/130369523/41afc874-bd55-4089-8895-027921a46b44)

#### [AnimateDiff workflow](https://github.com/frankchieng/ComfyUI_MagicClothing/blob/main/assets/magic_clothing_animatediff_workflow.json)
<div align="left">
    <img src="https://github.com/frankchieng/ComfyUI_MagicClothing/assets/130369523/680f55a0-d4b3-4e85-9c07-86a81e2e5fc9" width="15%">
    <img src="https://github.com/frankchieng/ComfyUI_MagicClothing/assets/130369523/2b5580bc-afe9-40e8-8a08-43a2629fbf2d" width="15%">
</div>

you should run under custom_nodes directory of ComfyUI
```shell
git clone https://github.com/frankchieng/ComfyUI_MagicClothing.git
```
then run 
```shell
pip install -r requirements.txt
```

> download the models of cloth_segm.pth , magic_clothing_768_vitonhd_joint.safetensors(upper-body model),OMS_1024_VTHD+DressCode_200000.safetensors(lower-body and full-body model) from 
 🤗[Huggingface](https://huggingface.co/ShineChen1024/MagicClothing) and place them at the checkpoints directory, If u wanna to run AnimateDiff you should place garment_extractor.safetensors and ip_layer.pth in checkpoints/stable_ckpt directory
#### you should try the combination of miscellaneous hyperparameters especially when you inference with the lower and full body model,just for experiment now
> install the [ComfyUI_IPAdapter_plus](https://github.com/cubiq/ComfyUI_IPAdapter_plus) custom node at first if you wanna to experience the ipadapterfaceid.Then download the IPAdapter FaceID models from [IP-Adapter-FaceID](https://huggingface.co/h94/IP-Adapter-FaceID) and place them as the following placement structure
> For cloth inpainting, i just installed the [Segment anything](https://github.com/storyicon/comfyui_segment_anything) node,you can utilize other SOTA model to seg out the cloth from background.
> #####  tips:If you wanna to run the controlnet openpose part,you have to install the [comfyui_controlnet_aux](https://github.com/Fannovel16/comfyui_controlnet_aux) custome code as well as download the body_pose_model.pth, facenet.pth and hand_pose_model.pth at [openpose models](https://huggingface.co/lllyasviel/Annotators) and place them in custom_nodes/comfyui_controlnet_aux/ckpts/lllyasviel/Annotators
```text
ComfyUI
|-- models
|   |-- ipadapter
|   |   |-- ip-adapter-faceid-plus_sd15.bin
|   |   |-- ip-adapter-faceid-plusv2_sd15.bin
|   |   |-- ip-adapter-faceid_sd15.bin
|   |-- loras
|   |   |-- ip-adapter-faceid-plus_sd15_lora.safetensors
|   |   |-- ip-adapter-faceid-plusv2_sd15_lora.safetensors
|   |   |-- ip-adapter-faceid_sd15_lora.safetensors
|-- custom_nodes
|   |-- ComfyUI_MagicClothing
|   |   |-- checkpoints
|   |   |   |-- cloth_segm.pth
|   |   |   |-- magic_clothing_768_vitonhd_joint.safetensors
|   |   |   |-- OMS_1024_VTHD+DressCode_200000.safetensors
|   |   |   |-- stable_ckpt
|   |   |   |   |-- garment_extractor.safetensors
|   |   |   |   |-- ip_layer.pth
```

