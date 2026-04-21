"""
Shared OpenAI client instance.
Import this instead of creating new OpenAI() instances.
"""
from openai import OpenAI
from .config import OPENAI_API_KEY

# Single shared client instance
client = OpenAI(api_key=OPENAI_API_KEY)
