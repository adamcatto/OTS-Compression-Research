"""Small vision benchmarks: CIFAR classification and Cityscapes segmentation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ....core.experiments import ExperimentSpec
from ...api import TaskDomain, TrainingTask


class SmallConvNet(nn.Module):
    def __init__(self, num_classes: int = 10, width: int = 64) -> None:
        super().__init__()
        channels = (width, width * 2, width * 4)
        layers = []
        in_channels = 3
        for out_channels in channels:
            layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.GELU(),
                    nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.GELU(),
                    nn.MaxPool2d(2),
                ]
            )
            in_channels = out_channels
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(channels[-1], num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images).mean(dim=(-2, -1))
        return self.classifier(features)


class TinyVisionTransformer(nn.Module):
    def __init__(
        self,
        *,
        image_size: int = 32,
        patch_size: int = 4,
        num_classes: int = 10,
        dimension: int = 192,
        depth: int = 6,
        heads: int = 3,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if image_size % patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        patches = (image_size // patch_size) ** 2
        self.patch_embedding = nn.Conv2d(
            3, dimension, kernel_size=patch_size, stride=patch_size
        )
        self.class_token = nn.Parameter(torch.zeros(1, 1, dimension))
        self.position = nn.Parameter(torch.zeros(1, patches + 1, dimension))
        layer = nn.TransformerEncoderLayer(
            d_model=dimension,
            nhead=heads,
            dim_feedforward=round(dimension * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=depth, enable_nested_tensor=False
        )
        self.normalization = nn.LayerNorm(dimension)
        self.classifier = nn.Linear(dimension, num_classes)
        nn.init.normal_(self.class_token, std=0.02)
        nn.init.normal_(self.position, std=0.02)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        patches = self.patch_embedding(images).flatten(2).transpose(1, 2)
        token = self.class_token.expand(images.shape[0], -1, -1)
        features = torch.cat((token, patches), dim=1)
        features = self.encoder(features + self.position[:, : features.shape[1]])
        return self.classifier(self.normalization(features[:, 0]))


class SmallUNet(nn.Module):
    def __init__(self, num_classes: int = 19, base_channels: int = 32) -> None:
        super().__init__()
        self.down1 = self._block(3, base_channels)
        self.down2 = self._block(base_channels, base_channels * 2)
        self.down3 = self._block(base_channels * 2, base_channels * 4)
        self.bridge = self._block(base_channels * 4, base_channels * 8)
        self.up3 = nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 2, 2)
        self.decode3 = self._block(base_channels * 8, base_channels * 4)
        self.up2 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 2, 2)
        self.decode2 = self._block(base_channels * 4, base_channels * 2)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, 2, 2)
        self.decode1 = self._block(base_channels * 2, base_channels)
        self.classifier = nn.Conv2d(base_channels, num_classes, 1)

    @staticmethod
    def _block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        down1 = self.down1(images)
        down2 = self.down2(F.max_pool2d(down1, 2))
        down3 = self.down3(F.max_pool2d(down2, 2))
        bridge = self.bridge(F.max_pool2d(down3, 2))
        decode3 = self.decode3(torch.cat((self.up3(bridge), down3), dim=1))
        decode2 = self.decode2(torch.cat((self.up2(decode3), down2), dim=1))
        decode1 = self.decode1(torch.cat((self.up1(decode2), down1), dim=1))
        return self.classifier(decode1)


class _CityscapesView(Dataset):
    _TRAIN_IDS = (7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33)

    def __init__(self, root: Path, split: str, height: int, width: int) -> None:
        try:
            from torchvision.datasets import Cityscapes
        except ImportError as exc:
            raise RuntimeError("Cityscapes requires the torchvision package") from exc
        self.dataset = Cityscapes(
            root=str(root), split=split, mode="fine", target_type="semantic"
        )
        self.height = height
        self.width = width
        lookup = torch.full((256,), 255, dtype=torch.long)
        for train_id, raw_id in enumerate(self._TRAIN_IDS):
            lookup[raw_id] = train_id
        self.lookup = lookup

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms import functional as transform

        image, mask = self.dataset[index]
        size = [self.height, self.width]
        image = transform.resize(image, size, interpolation=InterpolationMode.BILINEAR)
        image = transform.pil_to_tensor(image).float().div_(255.0)
        image = transform.normalize(
            image,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        mask = transform.resize(mask, size, interpolation=InterpolationMode.NEAREST)
        mask = transform.pil_to_tensor(mask).squeeze(0).long()
        valid = mask < len(self.lookup)
        converted = torch.full_like(mask, 255)
        converted[valid] = self.lookup[mask[valid]]
        return image, converted


def _loader(
    dataset: Dataset,
    *,
    batch_size: int,
    shuffle: bool,
    workers: int,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
        drop_last=shuffle and len(dataset) >= batch_size,
    )


class VisionClassificationTask(TrainingTask):
    name = "vision.classification"
    domain = TaskDomain.VISION

    def build_model(self, spec: ExperimentSpec) -> nn.Module:
        config = dict(spec.model)
        architecture = str(config.pop("architecture"))
        if architecture == "small_convnet":
            return SmallConvNet(**config)
        if architecture == "tiny_vit":
            return TinyVisionTransformer(**config)
        raise ValueError(f"unsupported vision architecture: {architecture}")

    def build_dataloaders(
        self, spec: ExperimentSpec, datasets_dir: Path
    ) -> Tuple[DataLoader, Optional[DataLoader]]:
        dataset = dict(spec.dataset)
        name = str(dataset.get("name", "cifar10")).lower()
        training = spec.training
        batch_size = int(training.get("batch_size", 128))
        workers = int(training.get("num_workers", 0))
        if name == "cifar10":
            try:
                from torchvision import datasets, transforms
            except ImportError as exc:
                raise RuntimeError("CIFAR-10 requires the torchvision package") from exc
            root = Path(dataset.get("root", datasets_dir / "cifar10"))
            normalize = transforms.Normalize(
                (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
            )
            train = datasets.CIFAR10(
                root=str(root),
                train=True,
                download=bool(dataset.get("download", True)),
                transform=transforms.Compose(
                    [
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        normalize,
                    ]
                ),
            )
            validation = datasets.CIFAR10(
                root=str(root),
                train=False,
                download=bool(dataset.get("download", True)),
                transform=transforms.Compose([transforms.ToTensor(), normalize]),
            )
        else:
            raise ValueError(f"unsupported classification dataset: {name}")
        return (
            _loader(
                train,
                batch_size=batch_size,
                shuffle=True,
                workers=workers,
                seed=spec.seed,
            ),
            None
            if validation is None
            else _loader(
                validation,
                batch_size=batch_size,
                shuffle=False,
                workers=workers,
                seed=spec.seed,
            ),
        )

    def loss(
        self, model: nn.Module, batch: Any, spec: ExperimentSpec
    ) -> torch.Tensor:
        images, labels = batch
        return F.cross_entropy(model(images), labels)


class VisionSegmentationTask(TrainingTask):
    name = "vision.segmentation"
    domain = TaskDomain.VISION

    def build_model(self, spec: ExperimentSpec) -> nn.Module:
        config = dict(spec.model)
        architecture = str(config.pop("architecture"))
        if architecture != "small_unet":
            raise ValueError(f"unsupported segmentation architecture: {architecture}")
        # Image dimensions describe data, not U-Net parameters.
        config.pop("image_height", None)
        config.pop("image_width", None)
        return SmallUNet(**config)

    def build_dataloaders(
        self, spec: ExperimentSpec, datasets_dir: Path
    ) -> Tuple[DataLoader, Optional[DataLoader]]:
        dataset = dict(spec.dataset)
        name = str(dataset.get("name", "cityscapes")).lower()
        training = spec.training
        batch_size = int(training.get("batch_size", 4))
        workers = int(training.get("num_workers", 0))
        height = int(spec.model.get("image_height", 256))
        width = int(spec.model.get("image_width", 512))
        if name == "cityscapes":
            root = Path(dataset.get("root", datasets_dir / "cityscapes"))
            train = _CityscapesView(root, "train", height, width)
            validation = _CityscapesView(root, "val", height, width)
        else:
            raise ValueError(f"unsupported segmentation dataset: {name}")
        return (
            _loader(
                train,
                batch_size=batch_size,
                shuffle=True,
                workers=workers,
                seed=spec.seed,
            ),
            None
            if validation is None
            else _loader(
                validation,
                batch_size=batch_size,
                shuffle=False,
                workers=workers,
                seed=spec.seed,
            ),
        )

    def loss(
        self, model: nn.Module, batch: Any, spec: ExperimentSpec
    ) -> torch.Tensor:
        images, masks = batch
        return F.cross_entropy(model(images), masks, ignore_index=255)
