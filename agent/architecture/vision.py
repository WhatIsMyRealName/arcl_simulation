"""Lazy frontal image decoding; metadata stays JSON serializable."""
from pathlib import Path
import numpy as np
import torch
from torch import nn
from PIL import Image
from .actions import image_available


def load_image(world):
    if not image_available(world):
        return None
    path = Path(world['front_camera']['path'])
    try:
        with Image.open(path) as source:
            array = np.array(source.convert('RGB').resize((160, 120)), dtype=np.float32, copy=True)
    except (OSError, ValueError) as error:
        raise ValueError(f'Cannot decode frontal image {path}: {error}') from error
    return torch.from_numpy(array).permute(2, 0, 1).div_(255.)


class FrontCameraEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(nn.Conv2d(3, 16, 5, stride=2, padding=2), nn.ReLU(),
                                 nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),
                                 nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
                                 nn.AdaptiveAvgPool2d(1), nn.Flatten())

    def forward(self, world, device):
        pixels = load_image(world)
        if pixels is None:
            embedding = torch.zeros((1, 64), device=device)
        else:
            embedding = self.cnn(pixels.unsqueeze(0).to(device))
        age = (world.get('front_camera') or {}).get('age_s')
        age = min(1., max(0., float(age))) if age is not None else 1.
        return torch.cat((embedding, embedding.new_tensor([[float(pixels is not None), age]])), dim=-1)

    def forward_many(self, worlds, device):
        images = [load_image(world) for world in worlds]
        valid = [i for i, pixels in enumerate(images) if pixels is not None]
        embedding = torch.zeros((len(worlds), 64), device=device)
        if valid:
            values = self.cnn(torch.stack([images[i] for i in valid]).to(device))
            embedding = embedding.index_copy(0, torch.tensor(valid, device=device), values)
        metadata = []
        for world, pixels in zip(worlds, images):
            age = (world.get('front_camera') or {}).get('age_s')
            metadata.append([float(pixels is not None), min(1., max(0., float(age))) if age is not None else 1.])
        return torch.cat((embedding, embedding.new_tensor(metadata)), dim=-1)


def validate_image_reference(world):
    """Validate even stale referenced files when preparing/loading a dataset."""
    path = (world.get('front_camera') or {}).get('path')
    if path:
        try:
            with Image.open(path) as image:
                image.verify()
        except (OSError, ValueError) as error:
            raise ValueError(f'Cannot decode frontal image {path}: {error}') from error
