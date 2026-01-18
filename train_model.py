# train_mac.py
import os
import torch
from unsloth import FastLanguageModel
from transformers import TrainingArguments
from trl import SFTTrainer
from datasets import load_dataset

# ========================================
# CONFIGURATION - ADJUST THESE VALUES
# ========================================
MODEL_NAME = "unsloth/llama-3-8b-bnb-4bit"  # Quantized base model
DATASET_NAME = "sft_output/langchain_ready_kz_history_sft_data.json"  # Example dataset (replace with your own)
OUTPUT_DIR = "./model_output"
MAX_SEQ_LENGTH = 2048
BATCH_SIZE = 4  # Reduce if you get OOM errors (MacBooks typically have 8-64GB unified RAM)
GRADIENT_ACCUMULATION_STEPS = 4
EPOCHS = 3
LEARNING_RATE = 2e-4
RANK = 64  # LoRA rank (higher = more trainable params)

# ========================================
# MAC-SPECIFIC OPTIMIZATIONS
# ========================================
# Force MPS usage for Apple Silicon
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Disable tokenizers parallelism (prevents deadlocks on Mac)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ========================================
# LOAD MODEL & TOKENIZER
# ========================================
print("Loading model...")
model, tokenizer = FastLanguageModel.from_pretrained(
    MODEL_NAME,
    max_seq_length=MAX_SEQ_LENGTH,
    dtype=None,  # Auto-detect (bfloat16 on M1/M2/M3)
    load_in_4bit=True,  # Critical for memory savings
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=RANK,
    target_modules=["q_proj", "k_proj", "v_correct", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing=True,  # Essential for long sequences
    random_state=3407,
)

# ========================================
# PREPARE DATASET
# ========================================
def formatting_prompts_func(examples):
    """Format dataset into instruction-response pairs"""
    instructions = examples["instruction"]
    outputs = examples["output"]
    texts = []
    for instruction, output in zip(instructions, outputs):
        text = f"### Instruction:\n{instruction}\n\n### Response:\n{output}{tokenizer.eos_token}"
        texts.append(text)
    return {"text": texts}

dataset = load_dataset(DATASET_NAME, split="train")
dataset = dataset.map(formatting_prompts_func, batched=True)

# ========================================
# TRAINING CONFIGURATION
# ========================================
training_args = TrainingArguments(
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    warmup_steps=5,
    num_train_epochs=EPOCHS,
    learning_rate=LEARNING_RATE,
    fp16=not torch.cuda.is_bf16_supported(),  # Use bfloat16 if available (M1+)
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=1,
    optim="adamw_8bit",  # 8-bit optimizer saves memory
    weight_decay=0.01,
    lr_scheduler_type="linear",
    seed=3407,
    output_dir=OUTPUT_DIR,
    report_to="none",  # Disable logging integrations
    save_strategy="epoch",
)

# ========================================
# START TRAINING
# ========================================
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    dataset_text_field="text",
    max_seq_length=MAX_SEQ_LENGTH,
    dataset_num_proc=1,  # Avoid multiprocessing issues on Mac
    packing=False,  # Disable packing for simplicity
    args=training_args,
)

print("Starting training...")
trainer.train()

# ========================================
# SAVE FINAL MODEL
# ========================================
print("Saving model...")
model.save_pretrained_merged("final_model", tokenizer, save_method="merged_16bit")
tokenizer.save_pretrained("final_model")

print("Training completed! Model saved to ./final_model")