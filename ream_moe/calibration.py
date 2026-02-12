from __future__ import annotations

from typing import Iterable, List

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedTokenizerBase

from .adapters.base import CalibrationBatch


class TextDataset(Dataset):
    """
    Simple in-memory text dataset for calibration.
    """

    def __init__(self, texts: List[str]):
        self.texts = texts

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> str:
        return self.texts[idx]


def build_calibration_batches(
    tokenizer: PreTrainedTokenizerBase,
    texts: Iterable[str],
    max_seq_len: int = 512,
    batch_size: int = 4,
) -> Iterable[CalibrationBatch]:
    """
    Yield `CalibrationBatch` objects from raw text, suitable for REAM.

    This utility is intentionally simple; for production use you might
    want to plug in HuggingFace `datasets` with streaming, c4, math and
    code mixtures, etc.
    """

    dataset = TextDataset(list(texts))

    def collate(batch_texts: List[str]) -> CalibrationBatch:
        enc = tokenizer(
            batch_texts,
            max_length=max_seq_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return CalibrationBatch(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
        )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)

    for batch in loader:
        # Move tensors to GPU lazily in the adapter; here we keep them on CPU.
        yield batch

