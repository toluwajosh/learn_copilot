from typing import Any, Dict
from langchain.chat_models import ChatOpenAI
from langchain.llms import GPT4All, LlamaCpp

MODEL_FACTORY = {
    "chat_openai": ChatOpenAI,
    "gpt4all": GPT4All,
    "llama_cpp": LlamaCpp,
}


def get_model(model_name=str, model_params=Dict) -> Any:
    return MODEL_FACTORY[model_name](**model_params)
