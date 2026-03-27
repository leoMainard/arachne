"""CamemBERT fine-tuning classifier (requires torch + transformers)."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from arachne.models.base import BaseTableClassifier
from arachne.data.preprocessing import LABEL_TO_ID, ID_TO_LABEL, LABELS


class TransformerClassifier(BaseTableClassifier):
    """Fine-tuned CamemBERT (or any HuggingFace sequence classification model)."""

    def __init__(self, model_config: dict, features_config: dict):
        self._model_config = model_config
        self._features_config = features_config
        self._model = None
        self._tokenizer = None
        self._device = None
        self._classes = LABELS.copy()

    def _get_device(self) -> "torch.device":
        import torch
        device_str = self._model_config.get("params", {}).get("device", "auto")
        if device_str == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device_str)

    def fit(
        self,
        texts_train: list[str],
        labels_train: list[str],
        texts_val: Optional[list[str]] = None,
        labels_val: Optional[list[str]] = None,
    ) -> None:
        try:
            import torch
            from torch.utils.data import Dataset, DataLoader
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            from torch.optim import AdamW
            from transformers import get_linear_schedule_with_warmup
        except ImportError as e:
            raise ImportError(
                "torch and transformers are required for transformer models. "
                "Install with: pip install arachne[transformers]"
            ) from e

        params = self._model_config.get("params", {})
        model_name = params.get("model_name", "camembert-base")
        num_labels = params.get("num_labels", 4)
        dropout = params.get("dropout", 0.1)
        max_length = self._model_config.get("max_length", 512)

        training_cfg = self._model_config.get("training", {})
        epochs = training_cfg.get("epochs", 5)
        batch_size = training_cfg.get("batch_size", 16)
        lr = training_cfg.get("learning_rate", 2e-5)
        warmup_ratio = training_cfg.get("warmup_ratio", 0.1)
        weight_decay = training_cfg.get("weight_decay", 0.01)

        self._device = self._get_device()

        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels,
            hidden_dropout_prob=dropout,
            attention_probs_dropout_prob=dropout,
        )
        self._model.to(self._device)

        class TableDataset(Dataset):
            def __init__(self, texts, labels, tokenizer, max_len):
                self.encodings = tokenizer(
                    texts,
                    truncation=True,
                    padding=True,
                    max_length=max_len,
                    return_tensors="pt",
                )
                self.labels = torch.tensor(
                    [LABEL_TO_ID.get(l, 3) for l in labels], dtype=torch.long
                )

            def __len__(self):
                return len(self.labels)

            def __getitem__(self, idx):
                return {
                    **{k: v[idx] for k, v in self.encodings.items()},
                    "labels": self.labels[idx],
                }

        train_dataset = TableDataset(texts_train, labels_train, self._tokenizer, max_length)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        optimizer = AdamW(self._model.parameters(), lr=lr, weight_decay=weight_decay)
        total_steps = len(train_loader) * epochs
        warmup_steps = int(total_steps * warmup_ratio)
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

        self._model.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for batch in train_loader:
                batch = {k: v.to(self._device) for k, v in batch.items()}
                optimizer.zero_grad()
                outputs = self._model(**batch)
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch + 1}/{epochs} — loss: {avg_loss:.4f}")

    def predict(self, texts: list[str]) -> list[str]:
        proba = self.predict_proba(texts)
        indices = np.argmax(proba, axis=1)
        return [ID_TO_LABEL[i] for i in indices]

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        import torch
        if self._model is None or self._tokenizer is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")

        self._model.eval()
        all_probs = []

        batch_size = 32
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            encodings = self._tokenizer(
                batch_texts,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt",
            )
            encodings = {k: v.to(self._device) for k, v in encodings.items()}
            with torch.no_grad():
                outputs = self._model(**encodings)
            probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()
            all_probs.append(probs)

        return np.vstack(all_probs)

    def get_classes(self) -> list[str]:
        return self._classes

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        if self._model is not None:
            self._model.save_pretrained(directory / "hf_model")
        if self._tokenizer is not None:
            self._tokenizer.save_pretrained(directory / "hf_model")

    @classmethod
    def load(cls, directory: Path) -> "TransformerClassifier":
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification

        hf_path = directory / "hf_model"
        instance = cls.__new__(cls)
        instance._classes = LABELS.copy()
        instance._model_config = {}
        instance._features_config = {}
        instance._tokenizer = AutoTokenizer.from_pretrained(str(hf_path))
        instance._model = AutoModelForSequenceClassification.from_pretrained(str(hf_path))
        instance._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        instance._model.to(instance._device)
        return instance
