"""
Test the PokerBench LoRA checkpoint (1000 steps)
Loads Qwen2.5-3B-Instruct + LoRA adapter and runs poker scenarios
"""
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset

os.environ["HF_HOME"] = "D:/poker-ai/hf_cache"

MODEL_NAME = "Qwen/Qwen2.5-3B-Instruct"
ADAPTER_PATH = "D:/poker-ai/poker-llm-lora/checkpoint-1000"
CACHE_DIR = "D:/poker-ai/hf_cache"

# ── Load model + adapter ──────────────────────────────────────
print(f"Loading {MODEL_NAME} in 4-bit...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    cache_dir=CACHE_DIR,
)

print(f"Loading LoRA adapter from {ADAPTER_PATH}...")
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

# ── Generate function ─────────────────────────────────────────
def ask_poker(question: str, max_tokens: int = 256) -> str:
    messages = [{"role": "user", "content": question}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.1,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response.strip()

# ── Test with PokerBench examples ─────────────────────────────
print("\n" + "="*60)
print("TESTING WITH POKERBENCH TEST SET")
print("="*60)

dataset = load_dataset("RZ412/PokerBench", cache_dir=CACHE_DIR)
test_data = dataset["test"]

# Test first 5 examples from test set
correct = 0
total = min(10, len(test_data))

for i in range(total):
    example = test_data[i]
    question = example["instruction"].strip()
    expected = example["output"].strip()

    print(f"\n{'─'*60}")
    print(f"Q{i+1}: {question[:200]}...")

    answer = ask_poker(question, max_tokens=128)

    print(f"Expected: {expected}")
    print(f"Model:    {answer[:200]}")

    # Simple match check
    if expected.lower() in answer.lower() or answer.lower().startswith(expected.lower()[:20]):
        correct += 1
        print("✓ MATCH")
    else:
        print("✗ MISMATCH")

print(f"\n{'='*60}")
print(f"Score: {correct}/{total} ({100*correct/total:.0f}%)")
print("="*60)

# ── Custom poker scenarios ────────────────────────────────────
print("\n\n" + "="*60)
print("CUSTOM POKER QUESTIONS")
print("="*60)

custom_questions = [
    "You are playing Texas Hold'em. You have Ah Kh. The flop is Qh Jh 2c. There are 2 players. The pot is $100. Your opponent bets $50. What should you do?",
    "You have pocket aces (As Ad) preflop in a 6-max game. You are in the cutoff position. What is the correct action?",
    "You have 7s 2d offsuit in the big blind. A tight player raises 3x from under the gun. What should you do?",
]

for i, q in enumerate(custom_questions):
    print(f"\n{'─'*60}")
    print(f"Q: {q}")
    answer = ask_poker(q)
    print(f"A: {answer}")
