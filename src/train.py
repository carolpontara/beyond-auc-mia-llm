from __future__ import annotations

import os
from typing import Dict, List

from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def build_text_dataset(texts: List[str]) -> Dataset:
    return Dataset.from_dict({"text": texts})


def prepare_lm_datasets(
    tokenizer,
    train_texts: List[str],
    eval_texts: List[str],
    max_length: int,
):
    train_ds = build_text_dataset(train_texts)
    eval_ds = build_text_dataset(eval_texts)

    def tokenize_function(batch: Dict[str, List[str]]) -> Dict[str, List[List[int]]]:
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    tokenized_train = train_ds.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
    )
    tokenized_eval = eval_ds.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
    )
    return tokenized_train, tokenized_eval


def load_tokenizer_and_model(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.config.pad_token_id = tokenizer.pad_token_id
    return tokenizer, model


def train_model(config: Dict, train_texts: List[str], eval_texts: List[str]):
    tokenizer, model = load_tokenizer_and_model(config["model_name"])

    tokenized_train, tokenized_eval = prepare_lm_datasets(
        tokenizer=tokenizer,
        train_texts=train_texts,
        eval_texts=eval_texts,
        max_length=config["max_length"],
    )

    output_dir = os.path.join(config["output_dir"], "model")
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=config["num_train_epochs"],
        learning_rate=config["learning_rate"],
        per_device_train_batch_size=config["train_batch_size"],
        per_device_eval_batch_size=config["eval_batch_size"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        warmup_ratio=config["warmup_ratio"],
        weight_decay=config["weight_decay"],
        logging_steps=config["logging_steps"],
        save_total_limit=config["save_total_limit"],
        fp16=config["fp16"],
        report_to=[],
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        seed=config["seed"],
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

    return tokenizer, trainer.model