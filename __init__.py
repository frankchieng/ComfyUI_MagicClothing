from .nodes import GarmentGenerate,GarmentLoader

NODE_CLASS_MAPPINGS = {
    "MagicClothingLoader":GarmentLoader,
    "MagicClothing_Uwear_Generate": GarmentGenerate,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MagicClothing_Uwear_Generate": "Human Garment Generation",
    "MagicClothingLoader":"GarmentLoader"
}