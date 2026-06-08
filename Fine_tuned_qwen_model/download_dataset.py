from datasets import load_dataset

print("Connecting to Hugging Face and downloading DRTriton RL Data...")

# Download the training split
dataset = load_dataset("hkust-nlp/drkernel-rl-data", split="train")

# Save it locally in JSONL format
dataset.to_json("drkernel_pytorch_to_triton.jsonl")

print(f"Success! Saved {len(dataset)} code examples.")