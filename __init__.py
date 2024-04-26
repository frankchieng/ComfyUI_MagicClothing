from .nodes import GarmentGenerate, AnimatediffGenerate

NODE_CLASS_MAPPINGS = {
    "MagicClothing_Generate": GarmentGenerate,
    "MagicClothing_Animatediff": AnimatediffGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MagicClothing_Generate": "Human Garment Generation",
    "MagicClothing_Animatediff": "Human Garment AnimateDiff Generation",     
}
