"""Speech Commands classification with a compact Conformer-style encoder."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ....core.experiments import ExperimentSpec
from ...api import TaskDomain, TrainingTask


SPEECH_COMMAND_LABELS = (
    "backward", "bed", "bird", "cat", "dog", "down", "eight", "five",
    "follow", "forward", "four", "go", "happy", "house", "learn", "left",
    "marvin", "nine", "no", "off", "on", "one", "right", "seven", "sheila",
    "six", "stop", "three", "tree", "two", "up", "visual", "wow", "yes", "zero",
)


class ConformerFeedForward(nn.Module):
    def __init__(self, dimension: int, hidden_dimension: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.LayerNorm(dimension),
            nn.Linear(dimension, hidden_dimension),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dimension, dimension),
            nn.Dropout(dropout),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class ConformerConvolution(nn.Module):
    def __init__(self, dimension: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("Conformer convolution kernel must be odd")
        self.normalization = nn.LayerNorm(dimension)
        self.pointwise_in = nn.Conv1d(dimension, dimension * 2, 1)
        self.depthwise = nn.Conv1d(
            dimension,
            dimension,
            kernel_size,
            padding=kernel_size // 2,
            groups=dimension,
        )
        self.batch_normalization = nn.BatchNorm1d(dimension)
        self.pointwise_out = nn.Conv1d(dimension, dimension, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.normalization(inputs).transpose(1, 2)
        hidden = F.glu(self.pointwise_in(hidden), dim=1)
        hidden = F.silu(self.batch_normalization(self.depthwise(hidden)))
        hidden = self.pointwise_out(hidden).transpose(1, 2)
        return self.dropout(hidden)


class ConformerBlock(nn.Module):
    def __init__(
        self,
        dimension: int,
        heads: int,
        feedforward_dimension: int,
        convolution_kernel: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.feedforward1 = ConformerFeedForward(
            dimension, feedforward_dimension, dropout
        )
        self.attention_norm = nn.LayerNorm(dimension)
        self.attention = nn.MultiheadAttention(
            dimension, heads, dropout=dropout, batch_first=True
        )
        self.attention_dropout = nn.Dropout(dropout)
        self.convolution = ConformerConvolution(
            dimension, convolution_kernel, dropout
        )
        self.feedforward2 = ConformerFeedForward(
            dimension, feedforward_dimension, dropout
        )
        self.output_norm = nn.LayerNorm(dimension)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = inputs + 0.5 * self.feedforward1(inputs)
        normalized = self.attention_norm(hidden)
        attended, _ = self.attention(
            normalized, normalized, normalized, need_weights=False
        )
        hidden = hidden + self.attention_dropout(attended)
        hidden = hidden + self.convolution(hidden)
        hidden = hidden + 0.5 * self.feedforward2(hidden)
        return self.output_norm(hidden)


class SmallConformerClassifier(nn.Module):
    def __init__(
        self,
        *,
        num_classes: int = len(SPEECH_COMMAND_LABELS),
        sample_rate: int = 16_000,
        n_fft: int = 400,
        hop_length: int = 160,
        dimension: int = 144,
        depth: int = 4,
        heads: int = 4,
        feedforward_dimension: int = 576,
        convolution_kernel: int = 15,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        del sample_rate  # Captured by the experiment spec; no learned effect here.
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.register_buffer("window", torch.hann_window(n_fft), persistent=False)
        self.input_projection = nn.Linear(n_fft // 2 + 1, dimension)
        self.subsample = nn.Conv1d(dimension, dimension, 3, stride=2, padding=1)
        self.blocks = nn.ModuleList(
            [
                ConformerBlock(
                    dimension,
                    heads,
                    feedforward_dimension,
                    convolution_kernel,
                    dropout,
                )
                for _ in range(depth)
            ]
        )
        self.normalization = nn.LayerNorm(dimension)
        self.classifier = nn.Linear(dimension, num_classes)

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        spectrum = torch.stft(
            waveforms,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=True,
        )
        features = torch.log1p(spectrum.abs()).transpose(1, 2)
        hidden = self.input_projection(features)
        hidden = self.subsample(hidden.transpose(1, 2)).transpose(1, 2)
        for block in self.blocks:
            hidden = block(hidden)
        hidden = self.normalization(hidden).mean(dim=1)
        return self.classifier(hidden)


class _SpeechCommandsView(Dataset):
    def __init__(
        self,
        root: Path,
        split: str,
        *,
        download: bool,
        samples: int,
        sample_rate: int,
    ) -> None:
        try:
            from torchaudio.datasets import SPEECHCOMMANDS
        except (ImportError, OSError) as exc:
            raise RuntimeError(
                "Speech Commands requires a torchaudio build matching PyTorch"
            ) from exc
        self.dataset = SPEECHCOMMANDS(
            str(root), download=download, subset=split
        )
        self.samples = samples
        self.sample_rate = sample_rate
        self.label_indices = {label: index for index, label in enumerate(SPEECH_COMMAND_LABELS)}

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        waveform, sample_rate, label, _, _ = self.dataset[index]
        if sample_rate != self.sample_rate:
            raise ValueError(
                f"expected sample rate {self.sample_rate}, received {sample_rate}"
            )
        waveform = waveform.mean(dim=0)
        waveform = F.pad(waveform[: self.samples], (0, max(0, self.samples - len(waveform))))
        return waveform, torch.tensor(self.label_indices[label], dtype=torch.long)


class SpeechCommandsTask(TrainingTask):
    name = "speech.commands"
    domain = TaskDomain.AUDIO

    def build_model(self, spec: ExperimentSpec) -> nn.Module:
        config = dict(spec.model)
        architecture = str(config.pop("architecture"))
        if architecture != "small_conformer":
            raise ValueError(f"unsupported speech architecture: {architecture}")
        return SmallConformerClassifier(**config)

    def build_dataloaders(
        self, spec: ExperimentSpec, datasets_dir: Path
    ) -> Tuple[DataLoader, Optional[DataLoader]]:
        dataset = dict(spec.dataset)
        name = str(dataset.get("name", "speech_commands")).lower()
        batch_size = int(spec.training.get("batch_size", 32))
        workers = int(spec.training.get("num_workers", 0))
        sample_rate = int(spec.model.get("sample_rate", 16_000))
        samples = int(dataset.get("samples", sample_rate))
        if name == "speech_commands":
            root = Path(dataset.get("root", datasets_dir / "speech_commands"))
            download = bool(dataset.get("download", True))
            train = _SpeechCommandsView(
                root,
                "training",
                download=download,
                samples=samples,
                sample_rate=sample_rate,
            )
            validation = _SpeechCommandsView(
                root,
                "validation",
                download=False,
                samples=samples,
                sample_rate=sample_rate,
            )
        else:
            raise ValueError(f"unsupported speech dataset: {name}")
        generator = torch.Generator().manual_seed(spec.seed)
        common = {
            "batch_size": batch_size,
            "num_workers": workers,
            "pin_memory": torch.cuda.is_available(),
            "generator": generator,
        }
        return (
            DataLoader(
                train,
                shuffle=True,
                drop_last=len(train) >= batch_size,
                **common,
            ),
            None
            if validation is None
            else DataLoader(validation, shuffle=False, **common),
        )

    def loss(
        self, model: nn.Module, batch: Any, spec: ExperimentSpec
    ) -> torch.Tensor:
        waveforms, labels = batch
        return F.cross_entropy(model(waveforms), labels)
