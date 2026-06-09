# Qwen-2.5 Triton Specialist

This repository contains a specialized Large Language Model (LLM) fine-tuned to translate standard PyTorch code into highly optimized Triton kernels. It utilizes a LoRA adapter trained on the `Qwen/Qwen2.5-Coder-7B` base model.

## 📥 Download the Model Weights

Due to GitHub's file size limitations, the LoRA weights (`adapter_model.safetensors`) are hosted externally.

1. Download the model weights from [Google Drive](https://drive.google.com/file/d/1XNCUpSRuE2bbJh5G0K3Ks653CCXTCXHO/view?usp=sharing)
2. Place the downloaded `adapter_model.safetensors` file inside the `qwen2.5-triton-specialist/` folder:
   ```
   Fine_tuned_qwen_model/
   └── qwen2.5-triton-specialist/
       └── adapter_model.safetensors  <-- Place file here
   ```

## 📂 Repository Structure

* `qwen2.5-triton-specialist/`: Contains the fine-tuned LoRA weights (`adapter_model.safetensors`) and model configuration. **(Required for inference - see download instructions above)**.
* `validate_specialist.py`: The main inference script. It loads the base model, applies the specialist adapter, and tests it against the DRTriton validation dataset.
* `train_specialist.py`: The script used for the Supervised Fine-Tuning (SFT) process via Unsloth.
* `download_dataset.py`: Utility script to fetch the training data from Hugging Face.

## 🚀 Quick Start for Teammates (Inference)

To run the model and generate Triton kernels, follow these steps to set up your local Python environment and install the required dependencies.

### 1. Prerequisites

* **Python 3.12+**
* **NVIDIA GPU** with at least 8GB VRAM (16GB+ recommended for 4-bit loading), or **Apple Silicon Mac** (M1/M2/M3)
* **Hugging Face Token** (optional but recommended for faster downloads): Set `HF_TOKEN` in your environment.

### 2. Create and Activate a Virtual Environment

It is highly recommended to use a virtual environment to avoid dependency conflicts.

**Windows (PowerShell)**
```powershell
python -m venv triton_env
.\triton_env\Scripts\Activate.ps1
```

**Windows (CMD)**
```cmd
python -m venv triton_env
triton_env\Scripts\activate.bat
```

**WSL / macOS / Linux**
```bash
python3 -m venv triton_env
source triton_env/bin/activate
```

> Verify activation by running `which python` (Linux/Mac/WSL) or `where python` (Windows). To exit, type `deactivate`.

### 3. Install Dependencies

With the environment activated, install the required libraries based on your platform:

**Linux / WSL (NVIDIA GPU)**
```bash
pip install torch transformers datasets accelerate bitsandbytes peft trl
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
```

**macOS / Apple Silicon**
```bash
pip install torch transformers datasets accelerate peft
```

> **Note on Build Tools:** WSL/Linux users must install C++ build tools (`sudo apt install build-essential python3.12-dev`) because Unsloth requires them. Mac and native Windows users do not need to worry about this.

### 4. Run the Specialist

Execute the inference script. It will automatically download the base Qwen 2.5 model (if not already cached), apply the local LoRA weights, and output a generated Triton kernel.

```bash
python validate_specialist.py
```

## 🏋️ Training (Optional)

If you want to retrain or fine-tune the model further:

1. Download the dataset:
   ```bash
   python download_dataset.py
   ```
2. Start training:
   ```bash
   python train_specialist.py
   ```

> **Important:** Training requires **Linux or WSL** with an NVIDIA GPU. You must also install the C++ build tools before running the training script:
>
> ```bash
> sudo apt install build-essential python3.12-dev -y
> ```
>
> Training uses 4-bit quantization and LoRA adapters to fit within consumer GPU memory (e.g., RTX 5060 Ti).

## 📜 License & Acknowledgements

**Base Model:**
This model is a fine-tuned version of Qwen/Qwen2.5-Coder-7B. The base model and its weights are licensed under the Apache 2.0 License by Alibaba Cloud.

**Training Data:**
The fine-tuning process utilizes the hkust-nlp/drkernel-rldata dataset created by hkust-nlp. This dataset is distributed under the MIT License:

Copyright (c) hkust-nlp

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
