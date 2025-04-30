import numpy as np
from openai import BaseModel

import math
import os
import sys
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm import (
    generate_structured_completion,
    generate_embeddings_for_list,
    rank_with_cosine_similarity,
)


class DataScientistResponse(BaseModel):
    thoughts: str
    preference_function: str
    constraint_function: str
    model_config = {
        "extra": "forbid",  # or 'allow' or 'ignore'
    }

    def __str__(self):
        return f"""Thoughts:{self.thoughts}
Preference:
{self.preference_function}
Constraint:
{self.constraint_function}"""


class RepairResponse(BaseModel):
    code_with_correct_syntax: str
    model_config = {
        "extra": "forbid",  # or 'allow' or 'ignore'
    }

    def __str__(self):
        return f"""Corrected code:
{self.code_with_correct_syntax}"""


def get_features_prompt(features: dict[str, any]) -> str:
    features_prompt = "Variable name: domain (explanation)\n"
    for key, value in features.items():
        if isinstance(value, set):
            features_prompt += f"{key}: {value}\n"
        elif isinstance(value, tuple):
            features_prompt += f"{key}: from {value[0]} to {value[1]}\n"
        elif isinstance(value, bool):
            features_prompt += f"{key}: bool\n"
        elif isinstance(value, str):
            features_prompt += f"{key}: string\n"
        else:
            raise ValueError(f"Unknown feature type: {type(value)}")
    return features_prompt


def get_combined_prompt(features_prompt: str) -> str:
    return f"""You are a Python data analyst that analyzes hotel data.
Your boss gave you the task to write two Python functions that can be used to find appropriate hotels corresponding to the user's wishes.

You have access to the following global variables which contain the hotel data:
{features_prompt}

Furthermore, you can use numpy as np and math as math.
Now, first think about which variables to use and which user requirements are constraints vs. which are only preferences
Then, formalize these thoughts into a preference function which calculates a score to rank the hotel, and a constraint function which decides if the hotel is valid.
We will use both functions in the end to first filter the hotels and then return the top results.
"""


def get_few_shots(user_query: str) -> list[tuple[str, str]]:
    """Get few-shot examples for the LLM."""
    # Load cold start examples from memory
    memory_dir = os.path.join(os.path.dirname(__file__), "memory", "cold-start")
    few_shots = []
    # Load constraints example
    with open(os.path.join(memory_dir, "constraints.json"), "r") as f:
        constraints_examples = json.load(f)
        for example in constraints_examples:
            few_shots.append((example["user"], example["assistant"]))
    
    # Load preferences example
    with open(os.path.join(memory_dir, "preferences.json"), "r") as f:
        preferences_examples = json.load(f)
        for example in preferences_examples:
            few_shots.append((example["user"], example["assistant"]))
    
    return few_shots


def query_functions(
    query: str, features: dict[str, any]
) -> tuple[str, str]:
    features_prompt = get_features_prompt(features)
    user_prompt = query
    few_shots = get_few_shots(query)

    combined_prompt = get_combined_prompt(features_prompt)
    response: DataScientistResponse | None = generate_structured_completion(
        user_prompt,
        combined_prompt,
        DataScientistResponse,
        few_shots=few_shots,
        save_prefix="combined",
    )
    if response is None:
        f_p = """def preference_function() -> float:
    return 1."""
        f_c = """def constraint_function() -> bool:
    return True"""
    else:
        f_p = response.preference_function
        f_c = response.constraint_function
    return f_p, f_c


def evaluate_function(
    function_string: str,
    hotel_data: dict[str, any],
    parameter_embeddings,
    repaired_parameters: dict[str, any] = None,
    no_args: bool = False,
    syntax_fixed=False,
) -> tuple[float | bool, str, dict[str, any]]:
    # Create a namespace with numpy and math modules
    function_namespace = {"np": np, "math": math}
    # Add hotel data as global variables
    function_namespace.update(hotel_data)
    if repaired_parameters is None:
        repaired_parameters = {}
    for name, value in repaired_parameters.items():
        function_namespace[name] = value

    try:
        # Execute the function definition
        exec(function_string, function_namespace)
        function_name = function_string.split("def ")[1].split("(")[0].strip()
        function = function_namespace[function_name]
        result = function()
    except TypeError as e:
        if no_args and "required positional argument" in str(e):
            # Remove all arguments from the function definition
            function_string = (
                function_string.split("(")[0]
                + "():\n"
                + "\n".join(function_string.split("\n")[1:])
            )
            return evaluate_function(
                function_string,
                hotel_data,
                parameter_embeddings,
                repaired_parameters,
                no_args=True,
            )
        else:
            raise e
    except NameError as e:
        wrong_parameter = e.name
        if wrong_parameter not in repaired_parameters.keys():
            wrong_parameter_embedding = generate_embeddings_for_list([wrong_parameter])[
                wrong_parameter
            ]
            similarity_ordering = rank_with_cosine_similarity(
                parameter_embeddings, wrong_parameter_embedding
            )
            actual_name = list(similarity_ordering.keys())[0]
            print(
                f"Repaired parameter {wrong_parameter} to {actual_name} with similariy {similarity_ordering[actual_name]:.2f}"
            )
            repaired_parameters[wrong_parameter] = function_namespace[actual_name]
            return evaluate_function(
                function_string,
                hotel_data,
                parameter_embeddings,
                repaired_parameters,
            )
        else:
            raise
    except SyntaxError as e:
        if syntax_fixed:
            raise e
        error_description = str(e)
        improved_code = generate_structured_completion(
            "Here is the code:\n"
            + function_string
            + "\nHere is the error:\n"
            + error_description,
            """You are an expert at writing correct Python syntax.
You will get a Python method and an error message. You can assume that all variables exist. 
Return the improved function.""",
            RepairResponse,
            model="gpt-4o-mini-0718-eu",
            save_prefix="repair",
        )
        return evaluate_function(
            improved_code.code_with_correct_syntax,
            hotel_data,
            parameter_embeddings,
            repaired_parameters,
            no_args=True,
        )
    return result, function_string, repaired_parameters


def test_evaluate_function(
    hotel_data: dict[str, any],
    parameter_embeddings: list[str],
):
    function_string = """def preference_function() -> float:
    return code"""
    hotel_data = list(hotel_data.values())[0]
    evaluate_function(function_string, hotel_data, parameter_embeddings)
