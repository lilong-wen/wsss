from PIL import Image
import torchvision.transforms as T
from torchvision.transforms import Compose, Resize, CenterCrop, \
        ToTensor, Normalize, RandomResizedCrop, RandomRotation

def _convert_to_rgb(image):
    return image.convert('RGB')

def _transform(n_px: int, is_train: bool):
    normalize = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    if is_train:
        return Compose([
            # RandomRotation((-20, 20)),
            RandomResizedCrop(n_px, scale=(0.95, 1.0), interpolation=T.InterpolationMode.BICUBIC),
            _convert_to_rgb,
            ToTensor(),
            normalize,
        ])
    else:
        return Compose([
            Resize(n_px, interpolation=T.InterpolationMode.BICUBIC),
            CenterCrop(n_px),
            _convert_to_rgb,
            ToTensor(),
            normalize,
        ])