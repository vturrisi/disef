from src.data.caltech101 import Caltech101
from src.data.dtd import DTD
from src.data.eurosat import EuroSAT
from src.data.fgvc_aircraft import FGVCAircraft
from src.data.flowers102 import Flowers102
from src.data.food101 import Food101
from src.data.imagenet import ImageNet
from src.data.oxford_pets import OxfordPets
from src.data.stanford_cars import StanfordCars
from src.data.sun397 import SUN397

__all__ = [
    "Caltech101",
    "DTD",
    "EuroSAT",
    "FGVCAircraft",
    "Flowers102",
    "Food101",
    "ImageNet",
    "OxfordPets",
    "StanfordCars",
    "SUN397",
]

DATA = {
    "caltech101": Caltech101,
    "dtd": DTD,
    "eurosat": EuroSAT,
    "fgvc_aircraft": FGVCAircraft,
    "flowers102": Flowers102,
    "food101": Food101,
    "imagenet": ImageNet,
    "oxford_pets": OxfordPets,
    "stanford_cars": StanfordCars,
    "sun397": SUN397,
}
