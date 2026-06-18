from litellm import completion
from typing import Callable
import json
import os
import requests
from dotenv import load_dotenv
from dataclasses import dataclass

from game.memory import Prompt

load_dotenv()

# Configurar Azure para litellm
if os.getenv("AZURE4_OPENAI_KEY"):
    os.environ["AZURE_API_KEY"] = os.getenv("AZURE4_OPENAI_KEY")
    os.environ["AZURE_API_BASE"] = os.getenv("AZURE4_OPENAI_ENDPOINT")
    os.environ["AZURE_API_VERSION"] = os.getenv("AZURE4_OPENAI_API_VERSION", "2024-12-01-preview")


def create_grammar_constrained_llm_function(model_name: str, grammar_path: str) -> Callable:
    """LLM function that calls Ollama's /api/generate with a GBNF grammar.

    Uses Ollama's native generate endpoint (not the OpenAI-compatible one) so
    the grammar parameter reaches llama.cpp's constrained sampler directly.
    Only valid for ollama/* model names.
    """
    with open(grammar_path, "r") as f:
        grammar = f.read()

    ollama_model = model_name.replace("ollama/", "")
    ollama_base_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")

    def llm_function(prompt: Prompt) -> str:
        try:
            system_content = ""
            user_content = ""
            for msg in prompt.messages:
                if msg["role"] == "system":
                    system_content = msg["content"]
                elif msg["role"] == "user":
                    user_content = msg["content"]

            payload = {
                "model": ollama_model,
                "prompt": user_content,
                "system": system_content,
                "grammar": grammar,
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "num_predict": 1500,
                },
            }

            response = requests.post(
                f"{ollama_base_url}/api/generate",
                json=payload,
                timeout=180,
            )
            response.raise_for_status()
            return response.json()["response"]

        except Exception as e:
            return f"Error generating response: {str(e)}"

    return llm_function


def create_simple_llm_function(model_name: str) -> Callable:
    def llm_function(prompt: Prompt) -> str:
        try:
            request_params = {
                "model": model_name,
                "messages": prompt.messages,
                "max_tokens": 1500,
                "temperature": 0.2,
            }

            if prompt.tools:
                request_params["tools"] = prompt.tools
            
            response = completion(**request_params)

            if hasattr(response.choices[0].message, "tool_calls") and response.choices[0].message.tool_calls:
                tool_call = response.choices[0].message.tool_calls[0]
                result = {
                    "tool_name": tool_call.function.name,
                    "args": json.loads(tool_call.function.arguments)
                }
                return json.dumps(result)
            else:
                return response.choices[0].message.content
                
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    return llm_function
