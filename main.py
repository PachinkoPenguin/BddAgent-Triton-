import json
from typing import List, Dict
import os
import sys
import re
import traceback

from game.actionContext import create_action_context_with_registry
from game.actions import DecoratorActionRegistry
from game.agent import Agent, AgentRegistry
from game.environment import ActionContextEnvironment
from game.llms import create_simple_llm_function
from game.memory import Goal, Memory
from game.agentLanguage import AgentFunctionCallingActionLanguage

import tools.agentTools, tools.fileTools, tools.promptTools, tools.otherTools, tools.devEvalTools, tools.tritonTools

class PyTorchToTritonProcessor:
    def __init__(self, llm_function, input_path: str, output_path: str = "triton_kernels.jsonl"):
        self.llm_function = llm_function
        self.input_path = input_path
        self.output_path = output_path
        self.pytorch_code_samples = self.load_pytorch_samples()

    def load_pytorch_samples(self) -> List[Dict]:
        samples = []
        with open(self.input_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    sample = json.loads(line)
                    pytorch_code = sample['code']
                    newJSON = {
                        "pytorch_code": pytorch_code
                    }
                    samples.append(newJSON)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON line: {line}\nError: {str(e)}")
        print(samples)
        return samples
    
    def create_triton_task(self, pytorch_code: str) -> str:
        task = f"""
AVAILABLE AGENTS (use EXACTLY these names):
- "TritonDeveloper": generates Triton kernel code
- "TritonReviewer": reviews and validates Triton code

YOU MUST COMPLETE ALL THESE STEPS IN ORDER, DO NOT SKIP ANY:
1. Use get_gpu_specs tool to get GPU architecture information.
2. Call agent "TritonDeveloper" with the PyTorch code and GPU specs to generate a Triton kernel.
3. Call agent "TritonReviewer" with the generated kernel to validate it.
4. Use execute_pytorch_code tool to run the original PyTorch code and get results.
5. Use execute_triton_code tool to run the generated Triton kernel and get results.
6. Use validate_outputs tool to compare both results.
7. You MUST Use save_to_dataset tool to save everything.
8. Only after completing ALL steps above, call terminate.

PyTorch code:
{pytorch_code}

CRITICAL: You MUST call save_to_dataset before terminate. Do not skip steps.
"""
        return task

    def process(self):
        for sample in self.pytorch_code_samples:
            try:
                pytorch_code = sample['pytorch_code']
                
                sharedMemory = Memory()
                registry = AgentRegistry()

                mainAgent = create_project_manager_agent(self.llm_function)
                codingAgent = create_developer_agent(self.llm_function)
                codeReviewer = create_code_reviewer_agent(self.llm_function)

                registry.register_agent("TritonDeveloper", codingAgent.run)
                registry.register_agent("TritonReviewer", codeReviewer.run)

                task = self.create_triton_task(pytorch_code)

                action_context = {
                    "agent_registry": registry,
                    "target_language": "python",
                    "project_type": "triton_kernel",
                    "shared_memory": sharedMemory,
                    "llm": self.llm_function,
                    "dataset_path": self.output_path
                }

                result_memory = mainAgent.run(
                    user_input=task,
                    memory=sharedMemory,
                    action_context_props=action_context)
            except Exception as e:
                print(f"Error processing sample {sample}: {str(e)}")
                traceback.print_exc()
        return result_memory


def create_project_manager_agent(llm_function) -> Agent:
    goals = [
        Goal(1, "PyTorch Analysis", "Thoroughly analyze the provided PyTorch code and requirements to understand the functionality and performance characteristics"),
        Goal(2, "Agent Coordination", "Delegate specific coding tasks to the developer agent and coordinate the review process with the code reviewer agent"),
        Goal(3, "Project Oversight", "Ensure the overall project stays on track, meets requirements, and maintains high quality through effective coordination and feedback")
    ]

    action_registry = DecoratorActionRegistry(tags=["triton", "hardware", "execution", "dataset", "selective", "file_operations"])
    action_registry.register_terminate_tool()

    agent_language = AgentFunctionCallingActionLanguage()
    environment = ActionContextEnvironment()

    return Agent(
        goals = goals,
        agent_language = agent_language,
        action_registry = action_registry,
        generate_response = llm_function,
        environment = environment,
        agent_name = "Triton Project Manager Agent",
    )

def create_developer_agent(llm_function) -> Agent:

    goals = [
        Goal(1, "Generate Triton Code", "Write efficient Triton code based on the PyTorch implementation and requirements provided by the project manager"),
        Goal(2, "Implement Functionality", "Ensure the generated code correctly implements the required functionality and meets performance standards"),
        Goal(3, "Iterate Based on Feedback", "Refine and improve code based on feedback from the project manager and code reviewer")
    ]


    # Tools for development and coding
    action_registry = DecoratorActionRegistry(tags=["generation", "coding"])
    action_registry.register_terminate_tool()

    agent_language = AgentFunctionCallingActionLanguage()
    environment = ActionContextEnvironment()

    return Agent(
        goals=goals,
        agent_language=agent_language,
        action_registry=action_registry,
        generate_response=llm_function,
        environment=environment,
        agent_name="Triton Developer Agent",
        max_iterations=10
    )

def create_code_reviewer_agent(llm_function) -> Agent:

    goals = [
        Goal(1, "Code Quality Review",
             "Review code for quality, best practices, and potential issues"),
        Goal(2, "Performance Review",
             "Analyze code for performance optimization opportunities"),
        Goal(3, "Security Analysis",
             "Identify security vulnerabilities and suggest improvements"),
    ]

    # Tools for review and quality assurance
    action_registry = DecoratorActionRegistry(tags=["validation", "review"])
    action_registry.register_terminate_tool()

    agent_language = AgentFunctionCallingActionLanguage()
    environment = ActionContextEnvironment()

    return Agent(
        goals=goals,
        agent_language=agent_language,
        action_registry=action_registry,
        generate_response=llm_function,
        environment=environment,
        agent_name="Triton Code Reviewer",
        max_iterations=10
    )

def main():

    if not os.getenv("GEMINI_API_KEY"):
        raise ValueError("GEMINI_API_KEY environment variable not set")
    models = [
        "gemini/gemini-1.5-flash",
        "gemini/gemini-2.0-flash",
        "gemini/gemini-2.0-flash-lite",
        "gemini/gemini-2.5-flash",
        "gemini/gemini-2.5-pro",
        "azure/gpt-4.1-mini"
    ]
    llm_function = create_simple_llm_function(models[5])

    tritonProcessor = PyTorchToTritonProcessor(llm_function, input_path="C:/Users/cesar/OneDrive - Instituto Tecnologico y de Estudios Superiores de Monterrey/8vo Semestre/DiseñoAvanzado/BddAgent-Triton-/tritonCodeBlocks.jsonl", output_path="/triton_results")
    memory = tritonProcessor.process()
    print(memory.items[-1])

if __name__ == "__main__":
    main()