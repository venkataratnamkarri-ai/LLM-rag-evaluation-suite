"""
Custom LLM-as-a-Judge adapter for DeepEval using Cohere models.
Inherits from DeepEvalBaseLLM and integrates json_repair + Pydantic validation.
"""

import json
import os
import cohere
from dotenv import load_dotenv
from json_repair import repair_json
from pydantic import BaseModel
from deepeval.models import DeepEvalBaseLLM

load_dotenv()

class CohereModel(DeepEvalBaseLLM):
    """DeepEval-compatible custom LLM judge powered by Cohere Command models."""

    def __init__(self, model: str = "command-r-plus-08-2024"):
        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            raise ValueError("COHERE_API_KEY not found. Please set it in your .env file.")
        self.model = model
        self.client = cohere.ClientV2(api_key=api_key)

    def load_model(self):
        return self.client

    def generate(self, prompt: str, schema: BaseModel = None):
        """Synchronously generates a response and optionally parses it into a Pydantic schema."""
        response = self.client.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0
        )

        answer = ""
        for item in response.message.content:
            if hasattr(item, "text"):
                answer += item.text

        if schema is None:
            return answer

        # Repair any imperfect JSON produced by the model
        data = repair_json(answer, return_objects=True)
        return schema(**data)

    async def a_generate(self, prompt: str, schema: BaseModel = None):
        """Asynchronous generation interface required by DeepEvalBaseLLM."""
        return self.generate(prompt, schema)

    def get_model_name(self) -> str:
        return f"Cohere ({self.model})"
