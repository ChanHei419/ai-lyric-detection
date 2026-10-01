"""Fine-tune a Wav2Vec2 CTC model on paired (audio, transcript) data.

Expected CSV columns: ``audio`` (file path) and ``text``.

Usage:
    python scripts/fine_tune_wav2vec2.py --data data/train.csv --output-dir models/wav2vec2-lyrics
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="CSV file with audio,text columns")
    parser.add_argument("--base-model", default="facebook/wav2vec2-base-960h")
    parser.add_argument("--output-dir", default="models/wav2vec2-lyrics")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--sample-rate", type=int, default=16000)
    return parser


@dataclass
class CTCDataCollator:
    """Pad input values and labels into a training batch."""

    processor: object

    def __call__(self, features: list[dict]) -> dict:
        input_features = [{"input_values": f["input_values"]} for f in features]
        batch = self.processor.pad(
            input_features, padding=True, return_tensors="pt"
        )

        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(
            label_features, padding=True, return_tensors="pt"
        )
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )
        batch["labels"] = labels
        return batch


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    import numpy as np
    from datasets import Audio, load_dataset
    from transformers import (
        Trainer,
        TrainingArguments,
        Wav2Vec2ForCTC,
        Wav2Vec2Processor,
    )

    from lyric_detector.metrics import word_error_rate

    processor = Wav2Vec2Processor.from_pretrained(args.base_model)
    model = Wav2Vec2ForCTC.from_pretrained(
        args.base_model,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
    )

    dataset = load_dataset("csv", data_files=args.data)["train"]
    dataset = dataset.cast_column("audio", Audio(sampling_rate=args.sample_rate))

    def prepare(batch):
        audio = batch["audio"]
        batch["input_values"] = processor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_values[0]
        batch["labels"] = processor.tokenizer(batch["text"]).input_ids
        return batch

    dataset = dataset.map(prepare, remove_columns=dataset.column_names)

    def compute_metrics(prediction) -> dict:
        predicted_ids = np.argmax(prediction.predictions, axis=-1)
        label_ids = prediction.label_ids.copy()
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        predictions = processor.batch_decode(predicted_ids)
        references = processor.batch_decode(label_ids, group_tokens=False)
        scores = [
            word_error_rate(reference, prediction_text)
            for reference, prediction_text in zip(references, predictions)
        ]
        return {"wer": sum(scores) / len(scores) if scores else 0.0}

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_ratio=0.1,
        eval_strategy="no",
        save_strategy="epoch",
        logging_steps=10,
        group_by_length=True,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=CTCDataCollator(processor),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)
    print(f"Saved fine-tuned model to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
