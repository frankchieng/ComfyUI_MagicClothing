### Updates:
- ✅ [2024/04/17] only cloth image with prompt generation
- ✅ [2024/04/18] IPAdapter FaceID with human face detection and synthesize with cloth image generation

#### [the main workflow](https://github.com/frankchieng/ComfyUI_MagicClothing/blob/main/magic_clothing_workflow.json)
![magic_clothing](https://github.com/frankchieng/ComfyUI_MagicClothing/assets/130369523/6a9e7d05-69c1-4c79-8769-90d876a0a031)

#### [IPAdapater FaceID workflow](https://github.com/frankchieng/ComfyUI_MagicClothing/blob/main/ipadapter_faceid_workflow.json)
![IPadapter_faceid](https://github.com/frankchieng/ComfyUI_MagicClothing/assets/130369523/b42a8510-4076-49ea-932e-e6c8aee344f2)

you should run under custom_nodes directory of ComfyUI
```shell
git clone https://github.com/frankchieng/ComfyUI_MagicClothing.git
```
then run 
```shell
pip install -r requirements.txt
```

> download the models of cloth_segm.pth and magic_clothing_768_vitonhd_joint.safetensors from 
 🤗[Huggingface](https://huggingface.co/ShineChen1024/MagicClothing) and place them at the checkpoints directory

> download the IPAdapter FaceID models from [IP-Adapter-FaceID](https://huggingface.co/h94/IP-Adapter-FaceID) and place them under the checkpoints/ipadapter_faceid directory

```text
./checkpoints/
|-- ipadapter_faceid
|   |-- ip-adapter-faceid-plus_sd15.bin
|   |-- ip-adapter-faceid-plus_sd15_lora.safetensors
|   |-- ip-adapter-faceid-plusv2_sd15.bin
|   |-- ip-adapter-faceid-plusv2_sd15_lora.safetensors
|   |-- ip-adapter-faceid_sd15.bin
|   |-- ip-adapter-faceid_sd15_lora.safetensors
|-- cloth_segm.pth
|-- magic_clothing_768_vitonhd_joint.safetensors
```
