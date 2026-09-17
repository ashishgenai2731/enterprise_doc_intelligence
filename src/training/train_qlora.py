import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from config.settings import settings


def train_qlora():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "QLoRA 4-bit training with bitsandbytes requires an NVIDIA GPU with CUDA. "
            "Please run this script on a GPU instance (e.g., Google Colab, RunPod, or AWS T4/A10G)."
        )

    output_dir = "./models/mistral-qlora-final"
    os.makedirs(output_dir, exist_ok=True)

    # 1. Auto-detect optimal compute precision (bf16 for Ampere+, fp16 for older)
    use_bf16 = torch.cuda.is_bf16_supported()
    compute_dtype = torch.bfloat16 if use_bf16 else torch.float16

    # 2. Quantization setup for NF4 precision
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True
    )

    # 3. Base model & tokenizer loading
    model = AutoModelForCausalLM.from_pretrained(
        settings.BASE_MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto"
    )
    model.config.use_cache = False  # Prevents warning with gradient checkpointing

    tokenizer = AutoTokenizer.from_pretrained(settings.BASE_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Ensures clean batch alignment

    # 4. Prepare model layers for 4-bit quantization
    model = prepare_model_for_kbit_training(model)

    # 5. LoRA Adapters Configuration (Attention + MLP modules)
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # 6. Dataset preparation
    dataset = load_dataset("json", data_files="data/fin_qa_dataset.jsonl")

    # 7. SFT Trainer Configuration
    training_args = SFTConfig(
        output_dir=output_dir,
        dataset_text_field="text",
        max_seq_length=1024,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        num_train_epochs=3,
        optim="paged_adamw_8bit",
        fp16=not use_bf16,
        bf16=use_bf16,
        gradient_checkpointing=True,
        save_strategy="epoch"
    )

    # SFTTrainer cleanly wraps peft_config without double-wrapping
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        peft_config=peft_config,
        args=training_args
    )

    print("Starting QLoRA Fine-Tuning...")
    trainer.train()

    # Save final adapter weights and tokenizer
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"Training complete! Saved adapter weights to {output_dir}")


if __name__ == "__main__":
    train_qlora()