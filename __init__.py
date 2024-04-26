from .nodes import GarmentGenerate, AnimatediffGenerate, ClothInpainting

NODE_CLASS_MAPPINGS = {
    "MagicClothing_Generate": GarmentGenerate,
    "MagicClothing_Animatediff": AnimatediffGenerate,
    "MagicClothing_Inpainting": ClothInpainting,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MagicClothing_Generate": "Human Garment Generation",
    "MagicClothing_Animatediff": "Human Garment AnimateDiff Generation",     
    "MagicClothing_Inpainting": "Human Garment Inpainting",
}
