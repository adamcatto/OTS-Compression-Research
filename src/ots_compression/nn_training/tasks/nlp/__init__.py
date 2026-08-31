"""Small causal language-model training on WikiText."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from ....core.experiments import ExperimentSpec
from ...api import TaskDomain, TrainingTask


class ByteTokenizer:
    """Deterministic dependency-free tokenizer with an explicit document token."""

    vocab_size = 257
    document_token = 256

    def encode_documents(self, documents: Iterable[str]) -> torch.Tensor:
        tokens = []
        for document in documents:
            tokens.extend(document.encode("utf-8", errors="replace"))
            tokens.append(self.document_token)
        return torch.tensor(tokens, dtype=torch.long)


class TokenBlockDataset(Dataset):
    def __init__(self, tokens: torch.Tensor, sequence_length: int) -> None:
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        self.tokens = tokens
        self.sequence_length = sequence_length
        self.blocks = max(0, (len(tokens) - 1) // sequence_length)

    def __len__(self) -> int:
        return self.blocks

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = index * self.sequence_length
        values = self.tokens[start : start + self.sequence_length + 1]
        return values[:-1], values[1:]


class CausalTransformerLM(nn.Module):
    def __init__(
        self,
        *,
        vocab_size: int = ByteTokenizer.vocab_size,
        sequence_length: int = 256,
        dimension: int = 256,
        depth: int = 6,
        heads: int = 8,
        feedforward_dimension: int = 1024,
        dropout: float = 0.0,
        tie_embeddings: bool = True,
    ) -> None:
        super().__init__()
        self.sequence_length = sequence_length
        self.token_embedding = nn.Embedding(vocab_size, dimension)
        self.position_embedding = nn.Parameter(
            torch.zeros(1, sequence_length, dimension)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=dimension,
            nhead=heads,
            dim_feedforward=feedforward_dimension,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            layer, num_layers=depth, enable_nested_tensor=False
        )
        self.normalization = nn.LayerNorm(dimension)
        self.output = nn.Linear(dimension, vocab_size, bias=False)
        self.apply(self._initialize)
        if tie_embeddings:
            self.output.weight = self.token_embedding.weight
        nn.init.normal_(self.position_embedding, std=0.02)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        length = tokens.shape[1]
        if length > self.sequence_length:
            raise ValueError("input exceeds configured sequence length")
        hidden = self.token_embedding(tokens) + self.position_embedding[:, :length]
        causal_mask = torch.triu(
            torch.ones(length, length, device=tokens.device, dtype=torch.bool),
            diagonal=1,
        )
        hidden = self.transformer(hidden, mask=causal_mask, is_causal=True)
        return self.output(self.normalization(hidden))


def _wikitext_tokens(
    variant: str,
    split: str,
    max_tokens: Optional[int],
    cache_dir: Path,
) -> torch.Tensor:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "WikiText requires the Hugging Face 'datasets' package"
        ) from exc
    dataset = load_dataset(
        "Salesforce/wikitext", variant, split=split, cache_dir=str(cache_dir)
    )
    tokens = ByteTokenizer().encode_documents(record["text"] for record in dataset)
    if max_tokens is not None:
        tokens = tokens[:max_tokens]
    return tokens


def _loader(
    tokens: torch.Tensor,
    *,
    sequence_length: int,
    batch_size: int,
    workers: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    dataset = TokenBlockDataset(tokens, sequence_length)
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


class CausalLanguageModelingTask(TrainingTask):
    name = "nlp.causal_lm"
    domain = TaskDomain.NLP

    def build_model(self, spec: ExperimentSpec) -> nn.Module:
        config = dict(spec.model)
        architecture = str(config.pop("architecture"))
        if architecture != "causal_transformer":
            raise ValueError(f"unsupported language model architecture: {architecture}")
        return CausalTransformerLM(**config)

    def build_dataloaders(
        self, spec: ExperimentSpec, datasets_dir: Path
    ) -> Tuple[DataLoader, Optional[DataLoader]]:
        dataset = dict(spec.dataset)
        name = str(dataset.get("name", "wikitext")).lower()
        sequence_length = int(spec.model.get("sequence_length", 256))
        batch_size = int(spec.training.get("batch_size", 16))
        workers = int(spec.training.get("num_workers", 0))
        if name == "wikitext":
            variant = str(dataset.get("variant", "wikitext-2-raw-v1"))
            maximum = dataset.get("max_tokens")
            max_tokens = None if maximum is None else int(maximum)
            cache_dir = Path(dataset.get("root", datasets_dir / "huggingface"))
            train_tokens = _wikitext_tokens(
                variant, "train", max_tokens, cache_dir
            )
            validation_tokens = _wikitext_tokens(
                variant, "validation", max_tokens, cache_dir
            )
        else:
            raise ValueError(f"unsupported language dataset: {name}")
        return (
            _loader(
                train_tokens,
                sequence_length=sequence_length,
                batch_size=batch_size,
                workers=workers,
                shuffle=True,
                seed=spec.seed,
            ),
            None
            if validation_tokens is None
            else _loader(
                validation_tokens,
                sequence_length=sequence_length,
                batch_size=batch_size,
                workers=workers,
                shuffle=False,
                seed=spec.seed,
            ),
        )

    def loss(
        self, model: nn.Module, batch: Any, spec: ExperimentSpec
    ) -> torch.Tensor:
        inputs, targets = batch
        logits = model(inputs)
        return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
