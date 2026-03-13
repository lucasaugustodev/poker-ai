"""
PokerBench Fine-Tuning Script
Fine-tunes Qwen 2.5 7B with QLoRA on PokerBench dataset
Optimized for RTX 5060 (8GB VRAM)
"""
import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig

# ── Paths ──────────────────────────────────────────────────────
os.environ["HF_HOME"] = "D:/poker-ai/hf_cache"
OUTPUT_DIR = "D:/poker-ai/poker-llm-lora"
MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"

# ── Load dataset ───────────────────────────────────────────────
print("Loading PokerBench dataset...")
dataset = load_dataset("RZ412/PokerBench", cache_dir="D:/poker-ai/hf_cache")

def format_prompt(example):
    text = (
        "<|im_start|>user\n"
        f"{example['instruction'].strip()}<|im_end|>\n"
        "<|im_start|>assistant\n"
        f"{example['output'].strip()}<|im_end|>"
    )
    return {"text": text}

train_dataset = dataset["train"].map(format_prompt, remove_columns=["instruction", "output"])
test_dataset = dataset["test"].map(format_prompt, remove_columns=["instruction", "output"])

print(f"Train: {len(train_dataset)} examples")
print(f"Test:  {len(test_dataset)} examples")

# ── QLoRA 4-bit config ────────────────────────────────────────
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# ── Load model ─────────────────────────────────────────────────
print(f"\nLoading {MODEL_NAME} in 4-bit...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir="D:/poker-ai/hf_cache")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    cache_dir="D:/poker-ai/hf_cache",
)
model = prepare_model_for_kbit_training(model)

# ── LoRA config ────────────────────────────────────────────────
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
trainable, total = model.get_nb_trainable_parameters()
print(f"\nTrainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

# ── SFT Config (all-in-one for TRL 0.29+) ─────────────────────
sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    max_steps=5000,                    # ~80k examples, enough for convergence
    per_device_train_batch_size=4,     # 3B model uses less VRAM, bigger batch
    gradient_accumulation_steps=2,     # effective batch = 8
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_steps=100,
    lr_scheduler_type="cosine",
    logging_steps=25,
    save_steps=500,
    save_total_limit=4,
    bf16=True,
    optim="paged_adamw_8bit",
    gradient_checkpointing=True,
    max_grad_norm=0.3,
    report_to="none",
    dataloader_num_workers=0,
    eval_strategy="no",                # skip eval to save time
    seed=42,
    max_length=512,
    dataset_text_field="text",
    packing=False,
)

# ── Trainer ────────────────────────────────────────────────────
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    processing_class=tokenizer,
)

# ── Train! ─────────────────────────────────────────────────────
print("\nStarting training...")
print(f"Effective batch: {sft_config.per_device_train_batch_size} x {sft_config.gradient_accumulation_steps} = {sft_config.per_device_train_batch_size * sft_config.gradient_accumulation_steps}")
print(f"Output: {OUTPUT_DIR}")

trainer.train()

# ── Save ───────────────────────────────────────────────────────
print("\nSaving LoRA adapter...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\nDone! LoRA adapter saved to {OUTPUT_DIR}")
print("Next step: merge + convert to GGUF for Ollama")
