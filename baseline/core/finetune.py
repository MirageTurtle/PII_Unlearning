"""Supervised fine-tuning with optional retain-data regularization."""

import transformers

from .dataset import DefaultDataset, ForgetRetainDataset
from .utils import load_model_and_tokenizer


class SFTGDRTrainer(transformers.Trainer):
    """Minimize prepared-text loss plus retain-text loss with equal weights."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Each forward pass already returns a mean loss, without token-count kwargs.
        self.model_accepts_loss_kwargs = False

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        prepared_inputs, retain_inputs, _ = inputs
        prepared_outputs = model(**prepared_inputs)
        retain_outputs = model(**retain_inputs)
        loss = prepared_outputs.loss + retain_outputs.loss
        return (loss, prepared_outputs) if return_outputs else loss


def finetune(
    model_dir: str,
    data_file: str,
    out_dir: str,
    epochs: int = 5,
    per_device_batch_size: int = 2,
    learning_rate: float = 1e-5,
    max_len: int = 4096,
    tokenizer_dir: str | None = None,
    retain_data_file: str | None = None,
):
    """Fine-tune a model on text and save the resulting checkpoint."""
    model, tokenizer = load_model_and_tokenizer(model_dir, tokenizer_dir=tokenizer_dir)

    if retain_data_file is None:
        dataset = DefaultDataset(data_file, tokenizer=tokenizer, max_len=max_len)
        trainer_class = transformers.Trainer
    else:
        dataset = ForgetRetainDataset(
            data_file,
            tokenizer=tokenizer,
            retain_file_path=retain_data_file,
            max_len=max_len,
        )
        trainer_class = SFTGDRTrainer

    training_args = transformers.TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=per_device_batch_size,
        learning_rate=learning_rate,
        num_train_epochs=epochs,
        optim="adamw_torch",
        lr_scheduler_type="cosine",
        bf16=True,
        report_to="none",  # Disable wandb
        remove_unused_columns=retain_data_file is None,
    )

    trainer = trainer_class(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        data_collator=dataset.get_collate_fn(),
    )

    model.config.use_cache = False  # silence the warnings.
    trainer.train()
    trainer.save_model(out_dir)
