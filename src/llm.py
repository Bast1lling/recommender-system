import numpy as np
from openai import OpenAI
from anthropic import Anthropic
from dotenv import load_dotenv
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from datetime import datetime
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv(os.path.join(os.path.expanduser("~"), ".env"))

OPENAPI_TYPE = os.getenv("OPENAPI_TYPE", None)
DEBUG = os.getenv("DEBUG", True)

if OPENAPI_TYPE == "openai":
    API_KEY = os.getenv("OPENAI_API_KEY", "non-existing")
    client = OpenAI(
        api_key=API_KEY
    )
    model_name = "gpt-4.1"
elif OPENAPI_TYPE == "anthropic":
    API_KEY = os.getenv("ANTHROPIC_API_KEY", "non-existing")
    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    model_name = "claude-3.7-sonnet"
else:
    raise ValueError(f"Your API type {OPENAPI_TYPE} is invalid.")


def _save(text: str, prefix: str, dir_name: str) -> None:
    """Helper function to save response to a file."""
    if not DEBUG:
        return
    # Create responses directory if it doesn't exist
    os.makedirs(dir_name, exist_ok=True)

    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.txt"
    filepath = os.path.join(dir_name, filename)

    # Save the response
    with open(filepath, "w", encoding="utf-8") as file:
        file.write(text)
    print(f"Response saved to {filepath}")


def generate_structured_completion_openai(
    user_prompt: str,
    system_prompt: str,
    schema: BaseModel,
    model: str = model_name,
    save_prefix: str = None,
) -> BaseModel | None:
    """Generate a structured response without streaming."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    try:
        completion = client.beta.chat.completions.parse(
            model=model,  # Use your deployed model name
            messages=messages,
            response_format=schema,  # Here's the magic!
        )
        parsed_response = completion.choices[0].message.parsed
    except Exception as e:
        print(f"Response failed: {e}")
        return None
    print("Response generated")

    if save_prefix:
        _save(
            str(parsed_response),
            save_prefix,
            dir_name=os.path.join("output", "responses"),
        )
        _save(system_prompt, save_prefix, dir_name=os.path.join("output", "prompts"))

    return parsed_response


def generate_structured_completion_anthropic(
    user_prompt: str,
    system_prompt: str,
    schema: BaseModel,
    model: str = model_name,
    save_prefix: str = None,
) -> BaseModel | None:
    messages = [
        {"role": "user", "content": user_prompt},
    ]

    response = client.messages.create(
        model=model,
        max_tokens=1024,
        messages=messages,
        system=system_prompt,
        tools=[
            {
                "name": schema.__name__,
                "description": "Your response",
                "input_schema": schema.model_json_schema(),
            }
        ]
    )
    result = None
    for content_block in response.content:
        if content_block.type == "tool_use":
            # The tool_use content block has direct access to input
            data_dict = content_block.input
            result = schema(**data_dict)

    if result is None:
        return result

    if save_prefix:
        _save(
            str(result),
            save_prefix,
            dir_name=os.path.join("output", "responses"),
        )
        _save(system_prompt, save_prefix, dir_name=os.path.join("output", "prompts"))
    return result


def generate_structured_completion(
    user_prompt: str,
    system_prompt: str,
    schema: BaseModel,
    model: str = model_name,
    save_prefix: str = None,
) -> BaseModel | None:
    if OPENAPI_TYPE == "anthropic":
        return generate_structured_completion_anthropic(user_prompt, system_prompt, schema, save_prefix=save_prefix)
    elif OPENAPI_TYPE == "openai":
        return generate_structured_completion_openai(user_prompt, system_prompt, schema, model=model, save_prefix=save_prefix)


def generate_embeddings_for_list(
    inputs: list[str], embedding_model: str = "all-MiniLM-L6-v2"
) -> dict[str, np.ndarray]:
    """
    Generates embeddings for a list of names and returns a dictionary
    mapping each name to its embedding vector using a local sentence-transformers model.
    """

    # Load the local model
    model = SentenceTransformer(embedding_model)

    # Generate embeddings for all inputs at once
    embeddings = model.encode(inputs)

    return {name: embedding for name, embedding in zip(inputs, embeddings)}


def rank_with_cosine_similarity(
    keys_dict: dict[str, np.ndarray], value: np.ndarray
) -> dict[str, float]:
    """
    Ranks a value based on its cosine similarity to the keys in a dictionary.
    Works with numpy arrays directly.
    """
    # Convert keys_dict values to a matrix
    keys_matrix = np.stack(list(keys_dict.values()))

    # Get value array and compute cosine similarity
    similarities = np.dot(keys_matrix, value) / (
        np.linalg.norm(keys_matrix, axis=1) * np.linalg.norm(value)
    )

    # Create dictionary mapping keys to their similarity scores
    similarity_dict = dict(zip(keys_dict.keys(), similarities))

    # Sort dictionary by similarity scores in descending order
    return dict(sorted(similarity_dict.items(), key=lambda x: x[1], reverse=True))
