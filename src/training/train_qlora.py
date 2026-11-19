"""
QLoRA fine-tuning of the Code Executor model (Mistral-7B-Instruct v0.3) on one NVIDIA GPU.

Data: data/training/code_sft_{train,val}.jsonl from `python -m src.training.dataset_prep`
(prompt/completion pairs; the loss is computed on the completion, i.e. the code, only).

    python -m src.training.train_qlora                  # ~16 GB GPU is enough (T4 / A10G / L4)
    python -m src.training.train_qlora --epochs 1 --max-steps 5   # smoke test

Then import the adapter into Ollama and point the app at it: see src/training/README.md.
"""
import argparse
import os

import torch
from datasets import load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from config.settings import settings

TRAIN_FILE = "data/training/code_sft_train.jsonl"
VAL_FILE = "data/training/code_sft_val.jsonl"
OUTPUT_DIR = "./models/mistral-qlora-final"


def train_qlora(epochs: int = 3, max_steps: int = -1):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "QLoRA 4-bit training with bitsandbytes requires an NVIDIA GPU with CUDA. "
            "Please run this script on a GPU instance (e.g., Google Colab, RunPod, or AWS T4/A10G)."
        )
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. bf16 on Ampere or newer (A10G, L4, A100), fp16 on older cards (T4)
    use_bf16 = torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

    # 2. 4-bit NF4 weights with double quantisation: a 7B model fits in ~5 GB of GPU memory
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )

    # 3. Base model = the model Ollama serves as `mistral` (v0.3), so the adapter fits it
    model = AutoModelForCausalLM.from_pretrained(
        settings.BASE_MODEL_NAME, quantization_config=bnb_config, device_map="auto"
    )
    model.config.use_cache = False  # the KV cache is useless in training and clashes with checkpointing

    tokenizer = AutoTokenizer.from_pretrained(settings.BASE_MODEL_NAME)
    # Mistral has no pad token. Use <unk>, not </s>: if padding were </s>, the end-of-answer token
    # could be masked out with the padding and the model would not learn where the code block ends.
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = "right"

    # 4. Casts norms to fp32 and enables input gradients so LoRA can train on a 4-bit base
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # 5. LoRA adapters on every attention and MLP projection
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # 6. Conversational prompt/completion data: TRL applies the Mistral chat template itself
    #    (no hand-written <s>[INST] tags, which would add a second BOS token)
    dataset = load_dataset("json", data_files={"train": TRAIN_FILE, "validation": VAL_FILE})

    training_args = SFTConfig(
        output_dir=OUTPUT_DIR,
        max_length=2048,                 # 5-6 chunks of 512 characters + the rules fit with room to spare
        completion_only_loss=True,       # learn to write the code, not to repeat the 10-K context
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,   # effective batch of 8
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_steps=0.05,               # 5% of the steps (transformers 5: a float is a ratio)
        num_train_epochs=epochs,
        max_steps=max_steps,
        optim="paged_adamw_8bit",
        fp16=not use_bf16,
        bf16=use_bf16,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,     # keep the epoch with the lowest validation loss
        metric_for_best_model="eval_loss",
        save_total_limit=2,
        logging_steps=5,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.model.print_trainable_parameters()

    print("Starting QLoRA fine-tuning...")
    trainer.train()
    print(trainer.evaluate())

    # Only the adapter (~42M parameters) is saved, not the 7B base
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Training complete. Adapter saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=-1, help="stop early (smoke test)")
    args = parser.parse_args()
    train_qlora(args.epochs, args.max_steps)
