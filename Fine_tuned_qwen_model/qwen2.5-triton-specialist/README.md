---
base_model: unsloth/qwen2.5-coder-7b-bnb-4bit
library_name: peft
pipeline_tag: text-generation
tags:
- base_model:adapter:unsloth/qwen2.5-coder-7b-bnb-4bit
- lora
- sft
- transformers
- trl
- unsloth
- triton
- code-generation
- pytorch
---

# Qwen-2.5 Triton Specialist

This repository contains a LoRA adapter fine-tuned to translate standard PyTorch code into highly optimized Triton kernels. It is built on top of the `Qwen/Qwen2.5-Coder-7B` base model using 4-bit quantization.

## Model Details

### Model Description

- **Developed by:** Triple T
- **Model type:** LoRA adapter (PEFT)
- **Language(s) (NLP):** Code (Python, Triton)
- **License:** Apache 2.0
- **Finetuned from model:** Qwen/Qwen2.5-Coder-7B (via unsloth/qwen2.5-coder-7b-bnb-4bit)

### Model Sources

- **Repository:** [GitHub](https://github.com/PachinkoPenguin/BddAgent-Triton-)

## Uses

### Direct Use

This adapter is designed to be loaded with the base Qwen2.5-Coder-7B model to generate optimized Triton kernels from PyTorch code snippets. It is intended for developers and researchers working on GPU kernel optimization.

### Downstream Use

Can be integrated into automated code optimization pipelines, IDE assistants, or educational tools for learning Triton programming.

### Out-of-Scope Use

- Not intended for general-purpose code generation outside of Triton kernel optimization
- Should not be used for production-critical systems without thorough validation

## Bias, Risks, and Limitations

- Generated Triton kernels may require manual verification for correctness and performance
- The model is trained on a specific dataset and may not generalize to all PyTorch patterns
- As with all code generation models, outputs should be reviewed before deployment

### Recommendations

Always validate generated kernels against reference implementations. Test thoroughly on target hardware before production use.

## How to Get Started with the Model

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("unsloth/qwen2.5-coder-7b-bnb-4bit")
model = PeftModel.from_pretrained(base_model, "./qwen2.5-triton-specialist")
```

See the parent repository's README for full setup instructions.

## Training Details

### Training Data

The model was fine-tuned using the `hkust-nlp/drkernel-rl-data` dataset, which contains PyTorch-to-Triton code pairs.

### Training Procedure

#### Training Hyperparameters

- **Training regime:** 4-bit quantized LoRA (QLoRA)
- **Method:** Supervised Fine-Tuning (SFT)
- **Framework:** Unsloth + TRL

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

Evaluation is performed using the DRTriton validation dataset.

#### Metrics

Code correctness, syntactic validity of generated Triton kernels, and performance comparison against baseline implementations.

## Technical Specifications

### Model Architecture and Objective

- **Base Architecture:** Qwen2.5-Coder-7B
- **Adapter Type:** LoRA
- **Quantization:** 4-bit (bitsandbytes)

### Compute Infrastructure

#### Hardware

Training compatible with consumer NVIDIA GPUs (e.g., RTX 5060 Ti) thanks to 4-bit quantization.

#### Software

- Python 3.12+
- PyTorch
- Transformers
- PEFT
- Unsloth
- TRL

## Citation

If you use this model in your research, please cite the base model and training dataset:

**BibTeX:**
```bibtex
@misc{qwen2.5-coder,
  title={Qwen2.5-Coder: Technical Report},
  author={Qwen Team},
  year={2024}
}
```

## License

This adapter is licensed under the Apache 2.0 License, inherited from the base model Qwen/Qwen2.5-Coder-7B by Alibaba Cloud.

The training dataset `hkust-nlp/drkernel-rl-data` is licensed under the MIT License.

## Model Card Contact

For questions or issues, please open an issue in the parent repository.

### Framework versions

- PEFT 0.19.1
