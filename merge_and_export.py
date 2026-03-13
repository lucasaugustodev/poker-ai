"""
Merge LoRA adapter with base model and save as full model for GGUF conversion
"""
import sys, os
sys.stdout.reconfigure(encoding='utf-8')
os.environ["HF_HOME"] = "D:/poker-ai/hf_cache"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_PATH = "D:/poker-ai/poker-llm-lora/checkpoint-5000"
MERGED_PATH = "D:/poker-ai/poker-llm-merged"
CACHE_DIR = "D:/poker-ai/hf_cache"

# Step 1: Load base model in float16 (no quantization for merge)
print("Loading base model in float16...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="cpu",  # CPU to avoid OOM, we just need to save
    cache_dir=CACHE_DIR,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)

# Step 2: Load and merge LoRA
print(f"Loading LoRA adapter from {ADAPTER_PATH}...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)

print("Merging LoRA weights into base model...")
model = model.merge_and_unload()

# Step 3: Save merged model
print(f"Saving merged model to {MERGED_PATH}...")
model.save_pretrained(MERGED_PATH)
tokenizer.save_pretrained(MERGED_PATH)

print(f"Done! Merged model saved to {MERGED_PATH}")
print("Next: convert to GGUF format")
