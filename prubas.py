import json
from game.actionContext import ActionContext
from tools.tritonTools import get_gpu_specs, execute_pytorch_code, execute_triton_code, validate_outputs, save_to_dataset

ctx = ActionContext({"dataset_path": "test_dataset.jsonl"})

# 1. GPU Specs
print("=== GPU Specs ===")
specs = get_gpu_specs(ctx)
print(specs)

# 2. Ejecutar PyTorch
print("\n=== PyTorch Execution ===")
pytorch_code = """
import torch
a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
result = torch.matmul(a, b)
"""
pytorch_result = execute_pytorch_code(ctx, pytorch_code)
print(pytorch_result)

# 3. Ejecutar Triton (simulado con PyTorch para probar en local sin GPU)
print("\n=== Triton Execution (simulado) ===")
triton_code = """
import torch
def launch():
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
    return torch.matmul(a, b)
"""
triton_result = execute_triton_code(ctx, triton_code)
print(triton_result)

# 4. Validar outputs
print("\n=== Validation ===")
validation = validate_outputs(
    ctx,
    json.dumps(pytorch_result["result"]),
    json.dumps(triton_result["result"])
)
print(validation)

# 5. Guardar en dataset
print("\n=== Save to Dataset ===")
saved = save_to_dataset(
    ctx,
    pytorch_code=pytorch_code,
    triton_code=triton_code,
    gpu_specs=json.dumps(specs),
    pytorch_execution=json.dumps(pytorch_result),
    triton_execution=json.dumps(triton_result),
    validation=json.dumps(validation)
)
print(saved)