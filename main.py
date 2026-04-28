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


class DevEvalProcessor:
    def __init__(self,
                 lm_prompt_jsonl_path: str,
                 mode: str,
                 output_path: str = "unified_test_data.jsonl"):
        self.lm_prompt_jsonl_path = lm_prompt_jsonl_path
        self.mode = mode
        self.output_path = output_path
        os.makedirs(output_path, exist_ok=True)
        self.tests = self.load_tests()
        self.results = []

    def load_tests(self) -> List[Dict]:
        tests = []
        with open(self.lm_prompt_jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    test_case = json.loads(line)
                    input_code = test_case['input_code']
                    namespace = test_case['namespace']
                    newJSON = {
                        "namespace": namespace,
                        "input_code": input_code,
                    }   
                    if self.mode == 'local_file_completion':
                        newJSON['context_above'] = test_case['contexts_above']
                    elif self.mode == 'local_file_infiling':
                        newJSON['context_above'] = test_case['contexts_above']
                        newJSON['context_below'] = test_case['contexts_below']
                    tests.append(newJSON)

                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON line: {line}\nError: {str(e)}")
        return tests[:185]

    def generate_jsonl(self):
        out_file = os.path.join(self.output_path, self.mode + '_results4.jsonl')
        with open(out_file, 'w', encoding='utf-8') as f:
            for result in self.results:
                json_line = json.dumps(result)
                f.write(json_line + '\n')
        print(f"Generated JSONL file at {out_file}")

    def create_deveval_task(self, test, mode):
        namespace = test['namespace']
        requirements = test['input_code']
        
        context_str = ""
        context_instruction = ""
    
        if mode == 'without_context':
            context_instruction = "Generate code based only on requirements and common Python patterns."
        elif mode == 'local_file_completion':
            context_str = f"Context above: {test['context_above']}"
            context_instruction = "Use patterns, imports, and helper functions from the context above."
        elif mode == 'local_file_infiling':
            context_str = f"Context above: {test['context_above']}\nContext below: {test['context_below']}"
            context_instruction = "Use patterns from both context above and below. Ensure code fits between them."
        
        task = f"""
DEVEVAL COORDINATION: {namespace}

Requirements: {requirements}
{context_str}

STRATEGY: {context_instruction}

EXECUTE WORKFLOW:
1. Use analyze_deveval_requirements to understand the task
2. Generate the bdd_tests needed for validation
2. Use call_agent_with_reflection to call 'DevEvalCoder' with coding task andd the bdd_tests as validation criteria
3. Use call_agent_with_selected_context to call 'DevEvalReviewer' for validation based on the bdd_tests.
4. Extract final clean function body
5. validate_function_body(function_body=<step 3 result>, requirements="{requirements}")
   - If validation fails, regenerate from step 2 with fixes
6. terminate(message=<validated function body from step 3>)

Coordinate the team to produce working Python code.
CRITICAL: Final output must be ONLY function body, 4-space indented, no 'def' line.
"""
        return task

    def extract_final_code_from_memory(self, memory, namespace=None):
        """Extraer código con debugging."""

        def extract_function_body_from_complete(text, is_from_json=False):
            """Extracción robusta que maneja JSON anidados."""
            
            # Paso 1: Desenrollar JSON anidados múltiples veces
            original_text = text
            for _ in range(5):  # Máximo 5 niveles de anidamiento
                try:
                    data = json.loads(text)
                    if isinstance(data, dict):
                        # Buscar 'result' en cualquier nivel
                        if 'result' in data:
                            text = str(data['result'])
                        elif 'agent' in data and 'result' in data:
                            text = str(data['result'])
                    elif isinstance(data, str):
                        text = data
                    else:
                        break
                except json.JSONDecodeError:
                    break
            
            # Paso 2: Limpiar markdown
            text = re.sub(r'```python\s*\n(.*?)\n```', r'\1', text, flags=re.DOTALL)
            text = re.sub(r'```\s*\n(.*?)\n```', r'\1', text, flags=re.DOTALL)
            
            # Paso 3: Buscar función def explícitamente
            lines = text.split('\n')
            
            # Encontrar línea que contiene 'def '
            function_start = -1
            for i, line in enumerate(lines):
                if 'def ' in line and '(' in line and ':' in line:
                    function_start = i
                    break
            
            if function_start == -1:
                return None
            
            # Extraer desde la función encontrada
            function_lines = lines[function_start:]
            body_lines = []
            found_def = False
            main_indent = 0
            in_docstring = False
            docstring_char = None
            
            for line in function_lines:
                stripped = line.strip()
                current_indent = len(line) - len(line.lstrip())
                
                if not found_def and stripped.startswith('def '):
                    found_def = True
                    main_indent = current_indent
                    continue
                
                if found_def:
                    # 🔥 SALTAR DOCSTRINGS
                    if not in_docstring:
                        if stripped.startswith('"""') or stripped.startswith("'''"):
                            docstring_char = '"""' if stripped.startswith('"""') else "'''"
                            if stripped.count(docstring_char) >= 2:
                                # Docstring de una línea, saltar completamente
                                continue
                            else:
                                # Inicio de docstring multilínea
                                in_docstring = True
                                continue
                    else:
                        # Estamos dentro de docstring, buscar el final
                        if docstring_char and docstring_char in line:
                            in_docstring = False
                            docstring_char = None
                        continue
                    
                    # Código real de la función
                    if stripped:

                        #### Nueva mierda, ver si funciona o si es pura cola xd

                        if current_indent <= main_indent and stripped.startswith('def ') and current_indent == main_indent:
                            break  # Nueva función encontrada
                        
                        # Detectar texto explicativo final
                        if (current_indent <= main_indent and 
                            any(phrase in stripped.lower() for phrase in [
                                'this completes', 'task completed', 'implementation complete'
                            ])):
                            break
                        
                        # Normalizar indentación
                        if current_indent <= main_indent:
                            body_lines.append('    ' + stripped)
                        else:
                            relative = current_indent - main_indent
                            new_indent = 4 + (relative // 4) * 4
                            body_lines.append(' ' * new_indent + stripped)
                    else:
                        body_lines.append('')
            
            # Limpiar resultado
            while body_lines and not body_lines[-1].strip():
                body_lines.pop()
            
            result = '\n'.join(body_lines) if body_lines else None
            
            if result:
                result = re.sub(r'\s*🎉.*?Agent session completed.*$', '', result, flags=re.DOTALL)
                result = result.rstrip()
            
            return result

        
        print(f"\n🔍 DEBUGGING EXTRACTION for {namespace}:")
        
        for item in reversed(memory.items):
            content = str(item.get("content", ""))
            
            # Buscar en JSON result
            try:
                data = json.loads(content)
                if "result" in data:
                    result_text = str(data["result"])
                    if "def " in result_text:
                        print(f"📦 Found function in JSON result")
                        function_body = extract_function_body_from_complete(result_text, is_from_json=True)
                        if function_body and len(function_body.strip()) > 10:
                            print(f"✅ Extracted from JSON result")
                            return function_body
            except json.JSONDecodeError:
                pass
            
            # Buscar en texto plano
            if "def " in content and len(content) > 50:
                print(f"📝 Trying text content")
                function_body = extract_function_body_from_complete(content, is_from_json=False)
                if function_body and len(function_body.strip()) > 10:
                    print(f"✅ Extracted from text")
                    return function_body
        
        return "    pass  # No implementation found"

    def clean_extracted_code(self, raw_code):
        """Limpiar código extraído para DevEval."""
        
        lines = raw_code.split('\n')
        cleaned_lines = []
        skip_next = False
        
        for line in lines:
            # Saltar líneas de documentación y comentarios largos
            if skip_next:
                if '"""' in line or "'''" in line:
                    skip_next = False
                continue
                
            stripped = line.strip()
            
            # Saltar docstrings
            if '"""' in line or "'''" in line:
                if line.count('"""') == 1 or line.count("'''") == 1:
                    skip_next = True
                continue
            
            # Saltar signature de función
            if stripped.startswith('def ') and '(' in stripped and ':' in stripped:
                continue
                
            # Saltar líneas vacías o comentarios explicativos
            if not stripped or \
            stripped.startswith('# Note:') or \
            stripped.startswith('# This') or \
            any(skip_phrase in stripped.lower() for skip_phrase in ['here is', 'this completes']):
                continue
            
            # Limpiar indentación - asegurar 4 espacios mínimo
            if stripped:
                current_indent = len(line) - len(line.lstrip())
                if current_indent == 0:
                    # Agregar indentación base de 4 espacios
                    cleaned_lines.append("    " + stripped)
                else:
                    # Normalizar indentación a múltiplos de 4, mínimo 4
                    new_indent = max(4, ((current_indent + 3) // 4) * 4)
                    cleaned_lines.append(" " * new_indent + stripped)
        
        # Remover líneas vacías al final
        while cleaned_lines and not cleaned_lines[-1].strip():
            cleaned_lines.pop()
        
        result = '\n'.join(cleaned_lines)
        
        # Validar que tenemos código útil
        if not result.strip() or len(result.split('\n')) < 2:
            return "    pass"
        
        return result

    def process(self):
        llm_function = create_simple_llm_function("azure/gpt-4.1-mini")
        for test in self.tests:
            try:
                namespace = test['namespace']
                requirement = test['input_code']
                if 'context_above' in test:
                    context_above = test['context_above']
                if 'context_below' in test:
                    context_below = test['context_below']
                sharedMemory = Memory()
                registry = AgentRegistry()

                mainAgent = Agent(
                    goals=[
                        Goal(1, "DevEval Analysis","Use tools to analyze DevEval requirements thoroughly"),
                        Goal(2, "Agent Coordination", "Delegate to coding agent and coordinate review process"),
                        Goal(3, "Quality Assurance", "Ensure code meets DevEval stgandards through review tools")
                    ],
                    agent_language=AgentFunctionCallingActionLanguage(),
                    action_registry=DecoratorActionRegistry(tags=[ "selective", "deveval", "analysis"]),
                    generate_response=llm_function,
                    environment=ActionContextEnvironment(),
                    agent_name="DevEvalCoordinator",
                    max_iterations=12
                )
                codingAgent = Agent(
                    goals = [
                        Goal(1, "Complete Function Generation", "Generate complete Python functions with proper signatures"),
                        Goal(2, "Working Implementation", "Write actual working Python code that solves the given requirements"),
                    ],
                    agent_language=AgentFunctionCallingActionLanguage(),
                    action_registry=DecoratorActionRegistry(tags=[ "deveval"]),
                    generate_response=llm_function,
                    environment=ActionContextEnvironment(),
                    agent_name="DevEvalCoder",
                    max_iterations=8
                )
                codeReviewer = create_code_reviewer_agent(llm_function)

                registry.register_agent("DevEvalCoder", codingAgent.run)
                registry.register_agent("DevEvalReviewer", codeReviewer.run)
                mainAgent.action_registry.register_terminate_tool()
                codingAgent.action_registry.register_terminate_tool()
                codeReviewer.action_registry.register_terminate_tool()

                task = self.create_deveval_task(test, self.mode)

                action_context = {
                    "agent_registry": registry,
                    "target_language": "python",
                    "project_type": "deveval_function",
                    "namespace": namespace,
                    "shared_memory": sharedMemory
                }

                result_memory = mainAgent.run(
                    user_input=task,
                    memory=sharedMemory,
                    action_context_props=action_context
                )
                final_code = self.extract_final_code_from_memory(result_memory, namespace=namespace)

                print(f"\n🛠️ Cleaned Final Code for {namespace}:\n{final_code}\n")
                    # print(final_result)
                    # print(f"\nFinal Result for test {namespace}: ", final_result.get("content", "No content"))
                self.results.append({
                    "namespace": namespace,
                    "completion": final_code
                    })

            except Exception as e:
                print(f"Error processing test {test}: {str(e)}")
                traceback.print_exc()
        self.generate_jsonl()

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
        llm_function = create_simple_llm_function("azure/gpt-4.1-mini")
        for sample in self.pytorch_code_samples:
            try:
                pytorch_code = sample['pytorch_code']
                
                sharedMemory = Memory()
                registry = AgentRegistry()

                mainAgent = create_project_manager_agent(llm_function)
                codingAgent = create_developer_agent(llm_function)
                codeReviewer = create_code_reviewer_agent(llm_function)

                registry.register_agent("TritonDeveloper", codingAgent.run)
                registry.register_agent("TritonReviewer", codeReviewer.run)

                task = self.create_triton_task(pytorch_code)

                action_context = {
                    "agent_registry": registry,
                    "target_language": "python",
                    "project_type": "triton_kernel",
                    "shared_memory": sharedMemory,
                    "llm": llm_function,
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

def dump_memory_jsonl(memory, out_dir: str = "results", filename: str = "final_memory.jsonl"):
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, filename)
    with open(out_file, "w", encoding="utf-8") as fh:
        for item in getattr(memory, "items", []):
            # Use default=str to serialize non-JSON types (timestamps, etc.)
            fh.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")
    print(f"Saved memory JSONL to {out_file}")
    return out_file

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
    llm_function = create_simple_llm_function(models[3])

    project_manager = create_project_manager_agent(llm_function)
    developer = create_developer_agent(llm_function)
    code_reviewer = create_code_reviewer_agent(llm_function)

    registry = AgentRegistry()


    registry.register_agent("project_manager", project_manager)
    registry.register_agent("developer", developer)
    registry.register_agent("code_reviewer", code_reviewer)

    shared_memory = Memory()

    # action_context = create_action_context_with_registry(
    #     registry=registry,
    #     llm_function=llm_function,
    #     memory=shared_memory,
    #     target_language="python"
    # )
    mode = 'without_context'
    if len(sys.argv) >1:
        mode = sys.argv[1]

    # processor = DevEvalProcessor(lm_prompt_jsonl_path="C:/Users/cesar/7mo Semestre/DevEval/DevEval/Experiments/prompt/LM_prompt_elements.jsonl", mode='local_file_completion', output_path='results')
    # # processor = DevEvalProcessor(lm_prompt_jsonl_path="/home/piga/BddAgent/data/LM_prompt_elements.jsonl", mode=mode, output_path='results')
    # processor.process()

    # try:
    #     result_memory = project_manager.run(
    #         user_input=task,
    #         memory=shared_memory,
    #         action_context_props={
    #             "agent_registry": registry,
    #             "target_language": "Java",
    #             "project_type": "backend_api"
    #         })

    #     if result_memory.items:
    #         final_result = result_memory.items[-1]
    #         print("\nFinal Result: ", final_result.get("content", "No content"))
    #         # print("\nFull Memory:")
    #         # for item in result_memory.items:
    #         #     timestamp = item.get("timestamp", "N/A")
    #         #     role = item.get("role", "N/A")
    #         #     content = item.get("content", "N/A")
    #         #     print(f"[{timestamp}] ({role}): {content}\n")
    #     dump_memory_jsonl(result_memory, out_dir="results", filename="project_manager_memory.jsonl")

    # except Exception as e:
    #     print(f"Error during agent execution: {str(e)}")
    #     traceback.print_exc()

    tritonProcessor = PyTorchToTritonProcessor(llm_function, input_path="C:/Users/cesar/OneDrive - Instituto Tecnologico y de Estudios Superiores de Monterrey/8vo Semestre/DiseñoAvanzado/BddAgent-Triton-/tritonCodeBlocks.jsonl", output_path="/triton_results")
    memory = tritonProcessor.process()
    print(memory.items[-1])

if __name__ == "__main__":
    main()