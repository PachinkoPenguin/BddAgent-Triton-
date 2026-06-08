import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from transformers import TextStreamer
from unsloth.chat_templates import get_chat_template  # <-- Key import

print("1. Loading your Triton Specialist...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "qwen2.5-triton-specialist",
    max_seq_length = 2048,
    load_in_4bit = True,
)

FastLanguageModel.for_inference(model)

# 1.5 Remind the tokenizer how ChatML format works
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "chatml",
)

print("2. Downloading DRTriton validation data...")
dataset_val = load_dataset("hkust-nlp/drkernel-validation-data", split="validation")

example = dataset_val[0]
raw_data = example["prompt"]

# 2.5 Extract plain text from the list provided by the dataset
if isinstance(raw_data, list) and len(raw_data) > 0:
    user_text = raw_data[0]['content']
else:
    user_text = str(raw_data)

print("\n=== PYTORCH PROBLEM ===")
# Print only the first 300 characters to avoid flooding the terminal
print(user_text[:300] + "...\n[Text truncated for readability]")
print("===========================\n")

messages = [
    {"role": "system", "content": "You are an expert GPU programmer. Convert the following PyTorch code to optimal Triton code."},
    {"role": "user", "content": user_text}
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize = True,
    add_generation_prompt = True,
    return_tensors = "pt",
).to("cuda")

print("3. Generating Triton Kernel...\n")
streamer = TextStreamer(tokenizer, skip_prompt=True)

_ = model.generate(
    input_ids = inputs,
    streamer = streamer,
    max_new_tokens = 1024, 
    temperature = 0.3,        
    repetition_penalty = 1.15, 
    use_cache = True
)