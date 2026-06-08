import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth.chat_templates import get_chat_template

print("1. Loading Qwen 2.5-Coder-7B model in 4-bit...")
max_seq_length = 2048  # Maximum code length
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Qwen/Qwen2.5-Coder-7B",
    max_seq_length = max_seq_length,
    load_in_4bit = True,  # CRITICAL to fit in your RTX 5060 Ti
)

print("2. Configuring LoRA adapters (Muscle Memory)...")
model = FastLanguageModel.get_peft_model(
    model,
    r = 16,  # Adapter size (16 is a good balance)
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0,
    bias = "none",
    use_gradient_checkpointing = "unsloth",  # Saves a lot of VRAM
)

print("3. Preparing SFT dataset (Cold-Start)...")
# Using the correct dataset that DOES contain Triton responses
dataset = load_dataset("hkust-nlp/drkernel-coldstart-8k", split="train")

# Configure tokenizer to use Qwen's ChatML format
tokenizer = get_chat_template(
    tokenizer,
    chat_template = "chatml",
)

def format_dataset(example):
    # The 'coldstart' dataset already has a 'messages' column in OpenAI format
    # Here we convert it into a single text block for training
    formatted_text = tokenizer.apply_chat_template(
        example["messages"], 
        tokenize = False, 
        add_generation_prompt = False
    )
    return {"text": formatted_text}

dataset = dataset.map(format_dataset)

print("4. Starting training...")
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 5, 
    args = TrainingArguments(
        per_device_train_batch_size = 2,  # Process 2 examples at a time
        gradient_accumulation_steps = 4,  # Simulates a batch size of 8
        warmup_steps = 5,
        max_steps = 1000,  # Quick test --- 60 steps
        learning_rate = 2e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 10,
        optim = "adamw_8bit",  # 8-bit optimizer to save RAM
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

trainer_stats = trainer.train()

print("5. Saving your specialist model...")
model.save_pretrained("qwen2.5-triton-specialist")
tokenizer.save_pretrained("qwen2.5-triton-specialist")
print("Training completed successfully!")