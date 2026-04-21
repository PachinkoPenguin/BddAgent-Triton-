from game.tools import register_tool
from game.actionContext import ActionContext
import subprocess

@register_tool(tags = ["triton", "hardware"])
def get_gpu_specs(action_context: ActionContext) -> dict:
    """
    Extract GPU hardware specifications relevant for Triton kernel generation.
    """

    try:
        import torch

        if not torch.cuda.is_available():
            return {"error": "No GPU available"}
        
        device = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device)

        return {
            "device_name": props.name,
            "total_memory_mb": props.total_memory // (1024 * 1024),
            "multiprocessor_count": props.multi_processor_count,
            "max_threads_per_block": props.max_threads_per_block,
            "warp_size": props.warp_size,
            "max_shared_memory_per_block_kb": props.shared_memory_per_block // 1024,
        }
    
    except Exception as e:
        return {"error": str(e)}

@register_tool(tags = ["triton", "generation"])
def generate_triton_kernel(action_context: ActionContext, pytorch_code: str, gpu_specs: str) -> str:
    """
    Generate a Triton kernel from PyTorch code using GPU hardware specifications.
    """

    from game.memory import Prompt
    generate_response = action_context.get("llm")
    if not generate_response:
        return "LLM not available in action context"
    
    prompt = f"""You are an expert in GPU programming and Triton kernels.
Convert this PyTorch code to an optimized Triton kernel using the provided GPU specifications.

PyTorch code:
{pytorch_code}

GPU Specifications:
{gpu_specs}

Rules:
- Use the GPU specifications specs to set block sizes and memory constraints.
- the kernel must produce mathematically identical results to the PyTorch code
- Include al necessary imports
- Include a launch function that calls the kernel
- Return ONLY the complete Python code, no explanations

Generate the Triton kernel now:
"""
    
    response = generate_response(Prompt(messages=[
        {"role": "system", "content": "You are an expert Triton GPU kernel developer. Return only working Python code."},
        {"role": "user", "content": prompt}
    ]))

    return response

@register_tool(tags=["triton", "execution"])
def execute_pytorch_code(action_context: ActionContext, pytorch_code: str) -> dict:
    """
    Execute PyTorch code and capture result and execution time.
    """
    import time
    import torch

    try:
        namespace = {}
        exec(pytorch_code, namespace)

        start = time.time()
        exec(pytorch_code, namespace)
        end = time.time()

        result_tensor = None
        for val in namespace.values():
            if isinstance(val, torch.Tensor):
                result_tensor = val

        return {
            "success": True,
            "execution_time_ms": round((end - start) * 1000, 4),
            "result": result_tensor.tolist() if result_tensor is not None else None,
            "result_shape": list(result_tensor.shape) if result_tensor is not None else None,
            "result_dtype": str(result_tensor.dtype) if result_tensor is not None else None,
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
    
@register_tool(tags=["triton", "execution"])
def execute_triton_code(action_context: ActionContext, triton_code: str) -> dict:
    """
    Execute Triton kernel code and capture result and execution time.
    """
    import time
    import torch

    try:
        namespace = {}
        exec(triton_code, namespace)

        launch_fn = None
        for name, val in namespace.items():
            if callable(val) and name.startswith("launch"):
                launch_fn = val
                break

        if launch_fn is None:
            return {
                "success": False,
                "error": "No launch function found. Function must start with 'launch'"
            }

        start = time.time()
        result = launch_fn()
        end = time.time()

        return {
            "success": True,
            "execution_time_ms": round((end - start) * 1000, 4),
            "result": result.tolist() if isinstance(result, torch.Tensor) else None,
            "result_shape": list(result.shape) if isinstance(result, torch.Tensor) else None,
            "result_dtype": str(result.dtype) if isinstance(result, torch.Tensor) else None,
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }
    
@register_tool(tags=["triton", "validation"])
def validate_outputs(action_context: ActionContext, pytorch_result: str, triton_result: str) -> dict:
    """
    Compare PyTorch and Triton outputs and record difference metrics.
    """
    import torch
    import json

    try:
        pytorch_tensor = torch.tensor(json.loads(pytorch_result))
        triton_tensor = torch.tensor(json.loads(triton_result))

        if pytorch_tensor.shape != triton_tensor.shape:
            return {
                "shape_match": False,
                "reason": f"Shape mismatch: PyTorch {list(pytorch_tensor.shape)} vs Triton {list(triton_tensor.shape)}",
                "pytorch_result": pytorch_tensor.tolist(),
                "triton_result": triton_tensor.tolist(),
            }

        diff = (pytorch_tensor.float() - triton_tensor.float()).abs()

        return {
            "shape_match": True,
            "pytorch_result": pytorch_tensor.tolist(),
            "triton_result": triton_tensor.tolist(),
            "max_difference": float(diff.max()),
            "mean_difference": float(diff.mean()),
            "min_difference": float(diff.min()),
        }
    except Exception as e:
        return {
            "shape_match": False,
            "reason": str(e)
        }

@register_tool(tags=["triton", "dataset"])
def save_to_dataset(action_context: ActionContext, 
                    pytorch_code: str,
                    triton_code: str,
                    gpu_specs: str,
                    pytorch_execution: str,
                    triton_execution: str,
                    validation: str) -> dict:
    """
    Save a PyTorch-Triton pair with execution results to a JSONL dataset file.
    """
    import json
    import os

    try:
        record = {
            "pytorch_code": pytorch_code,
            "triton_code": triton_code,
            "gpu_specs": json.loads(gpu_specs),
            "pytorch_execution": json.loads(pytorch_execution),
            "triton_execution": json.loads(triton_execution),
            "validation": json.loads(validation),
            "success": json.loads(triton_execution).get("success", False)
        }

        output_path = action_context.get("dataset_path", "dataset.jsonl")

        with open(output_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

        return {
            "saved": True,
            "path": output_path
        }
    except Exception as e:
        return {
            "saved": False,
            "error": str(e)
        }