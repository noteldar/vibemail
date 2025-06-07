import os
from pydantic_ai.models.openai import OpenAIModel

def get_openai_model(model_name="o3"):
    """Get OpenAI model for pydantic_ai"""
    # pydantic_ai will automatically use OPENAI_API_KEY from environment
    return OpenAIModel(model_name) 