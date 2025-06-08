import os

from pydantic_ai.models.openai import OpenAIModel


def get_openai_model(model_name="o3"):
    """Get OpenAI model for pydantic_ai"""
    # pydantic_ai will automatically use OPENAI_API_KEY from environment
    return OpenAIModel(model_name)


def get_openai_image_model(model_name="gpt-image-1"):
    """Get OpenAI image model for pydantic_ai"""
    # pydantic_ai will automatically use OPENAI_API_KEY from environment
    return OpenAIModel(model_name, base_url="https://api.openai.com/v1")
