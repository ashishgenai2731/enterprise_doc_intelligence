import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from config.settings import settings


def train_qlora():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "QLoRA 4-bit training with bitsandbytes requires an NVIDIA GPU with CUDA. "
            "Please run this script on a GPU instance (e.g., Google Colab, RunPod, or AWS T4/A10G)."
        )

    os.makedirs("./models/mistral-qlora-adapter", exist_ok=True)

    # 1. Quantization setup for NF4 precision
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    # 2. Base model loading
    model = AutoModelForCausalLM.from_pretrained(
        settings.BASE_MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(settings.BASE_MODEL_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # 3. LoRA Adapters Injection
    model = prepare_model_for_kbit_training(model)
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)

    # 4. Dataset preparation
    dataset = load_dataset("json", data_files="data/fin_qa_dataset.jsonl")

    # 5. Trainer Configuration
    training_args = SFTConfig(
        output_dir="./models/mistral-qlora-adapter",
        dataset_text_field="text",
        max_seq_length=1024,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        logging_steps=10,
        num_train_epochs=3,
        optim="paged_adamw_8bit",
        fp16=True
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        peft_config=peft_config,
        args=training_args
    )

    print("Starting QLoRA Fine-Tuning...")
    trainer.train()
    trainer.model.save_pretrained("./models/mistral-qlora-final")
    tokenizer.save_pretrained("./models/mistral-qlora-final")
    print("Training complete! Saved adapter weights to ./models/mistral-qlora-final")


if __name__ == "__main__":
    train_qlora()