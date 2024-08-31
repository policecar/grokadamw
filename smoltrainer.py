"""
smoltrainer.py and dummydataset.jsonl are copied from https://github.com/nisten/grokadamw/
with small modifications.
"""

import json
import logging
import os
import torch
import torch.nn as nn
import warnings

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from datasets import load_dataset, Dataset, concatenate_datasets
from grokadamw import GrokAdamW
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from torch.cuda.amp import autocast
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

MODEL_NAME = "nisten/Biggie-SmoLlm-0.15B-Base"
MAX_LENGTH = 2048
BATCH_SIZE = 12
LEARNING_RATE = 2e-4
MAX_STEPS = 3000
GRADIENT_ACCUMULATION_STEPS = 2
NUM_WARMUP_STEPS = 30
OUTPUT_DIR = "./longcustom_finetuned_results"
CUSTOM_DATASET_PATH = "dummydataset.jsonl"

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def load_custom_dataset(file_path):
    logger.info(f"🔍 Loading custom dataset from {file_path}")
    try:
        with open(file_path, "r") as f:
            data = [json.loads(line) for line in tqdm(f, desc="Loading JSON lines")]

        texts = []
        for item in tqdm(data, desc="Formatting conversations"):
            conversation = item["conversations"]
            formatted_text = ""
            for turn in conversation:
                if turn["from"] == "human":
                    formatted_text += f"Human: {turn['value']}\n\n"
                elif turn["from"] == "gpt":
                    formatted_text += f"Assistant: {turn['value']}\n\n"
            texts.append(formatted_text.strip())

        return Dataset.from_dict({"text": texts})
    except Exception as e:
        logger.error(f"❌ Failed to load custom dataset: {str(e)}")
        return None


def format_capybara_prompts(examples):
    texts = []
    for conversation in examples["conversation"]:
        formatted_text = ""
        for turn in conversation:
            if "input" in turn:
                formatted_text += f"Human: {turn['input']}\n\n"
            if "output" in turn:
                formatted_text += f"Assistant: {turn['output']}\n\n"
        texts.append(formatted_text.strip())
    return {"text": texts}


class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.grokking_signal = 0.0

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        )
        return (loss, outputs) if return_outputs else loss

    def training_step(self, model, inputs):
        model.train()
        inputs = self._prepare_inputs(inputs)

        with autocast(dtype=torch.bfloat16):
            loss = self.compute_loss(model, inputs)

        if isinstance(loss, tuple):
            loss, _ = loss  # Unpack the tuple if loss is a tuple

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        loss.backward()
        self.grokking_signal = loss.item()

        return loss.detach()


def grokking_signal_fn():
    return trainer.grokking_signal


def main():
    logger.info(f"🚀 Initializing {MODEL_NAME} finetuning with GrokAdamW")

    try:
        _ = AutoConfig.from_pretrained(MODEL_NAME)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, torch_dtype=torch.bfloat16, use_cache=False
        )
    except Exception as e:
        logger.error(f"❌ Failed to load model or tokenizer: {str(e)}")
        return

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    logger.info("📚 Loading datasets")
    custom_dataset = load_custom_dataset(CUSTOM_DATASET_PATH)

    if custom_dataset is None:
        logger.error("❌ Failed to load custom dataset. Aborting.")
        return

    try:
        capybara_dataset = load_dataset("LDJnr/Capybara", split="train")
        capybara_dataset = capybara_dataset.map(
            format_capybara_prompts,
            batched=True,
            remove_columns=capybara_dataset.column_names,
        )
    except Exception as e:
        logger.error(f"❌ Failed to load Capybara dataset: {str(e)}")
        capybara_dataset = Dataset.from_dict({"text": []})

    logger.info(f"📊 Custom dataset size: {len(custom_dataset)}")
    logger.info(f"📊 Capybara dataset size: {len(capybara_dataset)}")

    combined_dataset = concatenate_datasets([custom_dataset, capybara_dataset])
    combined_dataset = combined_dataset.shuffle(seed=42)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

    logger.info("🔢 Tokenizing dataset")
    tokenized_dataset = combined_dataset.map(
        tokenize_function, batched=True, remove_columns=combined_dataset.column_names
    )

    logger.info("🏋️ Setting up the training arguments")
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        weight_decay=0.01,
        # bf16=True,
        logging_steps=10,
        save_steps=300,
        save_total_limit=10,
        dataloader_num_workers=4,
        warmup_steps=NUM_WARMUP_STEPS,
        gradient_checkpointing=True,
        evaluation_strategy="steps",
        eval_steps=300,
        max_steps=MAX_STEPS,
        fp16=False,  # We're using bf16, so disable fp16
        optim="adamw_hf",  # Using the huggingface one allows us to use custom optimizers
        lr_scheduler_type="cosine",  # Cosine learning rate decay
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    optimizer = GrokAdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        alpha_init=0.98,
        lamb=2.0,
        gamma=0.1,
        grokking_signal_fns=[grokking_signal_fn],
        grokking_signal_decay_rate=0.1,
        gradient_clipping=1.0,
    )

    logger.info("🏃‍♂️ Initializing Trainer with GrokAdamW")
    global trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_dataset.select(range(min(1000, len(tokenized_dataset)))),
        data_collator=data_collator,
        tokenizer=tokenizer,
        optimizers=(optimizer, None),  # This line tells it to use GrokAdamW
    )

    logger.info("🔥 Starting the training with GrokAdamW")
    try:
        trainer.train()
    except Exception as e:
        logger.error(f"❌ Training failed: {str(e)}")
        return

    logger.info("💾 Saving the model")
    try:
        trainer.save_model(OUTPUT_DIR)
    except Exception as e:
        logger.error(f"❌ Failed to save model: {str(e)}")

    logger.info("🎉 Finetuning with GrokAdamW completed!")


if __name__ == "__main__":
    main()
