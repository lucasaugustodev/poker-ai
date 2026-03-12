"""
PokerBench Fine-Tuning Script
Fine-tunes Llama 3.1 8B with QLoRA on PokerBench dataset
Optimized for RTX 5060 (8GB VRAM)
"""
import os
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# ── Paths ──────────────────────────────────────────────────────
os.environ["HF_HOME"] = "D:/poker-ai/hf_cache"
OUTPUT_DIR = "D:/poker-ai/poker-llm-lora"
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"

# ── Load dataset ───────────────────────────────────────────────
print("Loading PokerBench dataset...")
dataset = load_dataset("RZ412/PokerBench", cache_dir="D:/poker-ai/hf_cache")

# Format into chat-style prompt
def format_prompt(example):
    text = (
        "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"
        f"{example['instruction'].strip()}\n"
        "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{example['output'].strip()}"
        "<|eot_id|>"
    )
    return {"text": text}

train_dataset = dataset["train"].map(format_prompt, remove_columns=["instruction", "output"])
test_dataset = dataset["test"].map(format_prompt, remove_columns=["instruction", "output"])

print(f"Train: {len(train_dataset)} examples")
print(f"Test:  {len(test_dataset)} examples")
print(f"Sample:\n{train_dataset[0]['text'][:500]}...")

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
    torch_dtype=torch.bfloat16,
)
model = prepare_model_for_kbit_training(model)

# ── LoRA config ────────────────────────────────────────────────
lora_config = LoraConfig(
    r=16,                      # rank - higher = more capacity but more VRAM
    lora_alpha=32,             # scaling factor
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
trainable, total = model.get_nb_trainable_parameters()
print(f"\nTrainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

# ── Training args (optimized for 8GB VRAM) ─────────────────────
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1,               # 1 epoch over 563k examples is plenty
    per_device_train_batch_size=2,     # small batch for 8GB VRAM
    gradient_accumulation_steps=8,     # effective batch = 2*8 = 16
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    logging_steps=50,
    save_steps=2000,
    save_total_limit=3,
    bf16=True,                         # use bfloat16
    optim="paged_adamw_8bit",          # 8-bit optimizer saves VRAM
    gradient_checkpointing=True,       # trades compute for VRAM
    max_grad_norm=0.3,
    group_by_length=True,              # batch similar-length sequences
    report_to="none",                  # no wandb/tensorboard
    dataloader_num_workers=2,
    eval_strategy="steps",
    eval_steps=2000,
    seed=42,
)

# ── Trainer ────────────────────────────────────────────────────
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    processing_class=tokenizer,
    max_seq_length=1024,               # poker scenarios fit in ~600 tokens
    dataset_text_field="text",
    packing=True,                      # pack short examples together
)

# ── Train! ─────────────────────────────────────────────────────
print("\n🚀 Starting training...")
print(f"Epochs: {training_args.num_train_epochs}")
print(f"Batch size: {training_args.per_device_train_batch_size} x {training_args.gradient_accumulation_steps} = {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"Total steps: ~{len(train_dataset) // (training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps)}")
print(f"Output: {OUTPUT_DIR}")

trainer.train()

# ── Save ───────────────────────────────────────────────────────
print("\nSaving LoRA adapter...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print(f"\n✅ Done! LoRA adapter saved to {OUTPUT_DIR}")
print("Next step: merge + convert to GGUF for Ollama")
