import numpy as np
from pydantic import BaseModel

import math
import threading
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llm import (
    generate_structured_completion,
    generate_embeddings_for_list,
    rank_with_cosine_similarity,
)


class PreferenceResponse(BaseModel):
    brain_storming: str
    preference_function: str


class ConstraintResponse(BaseModel):
    brain_storming: str
    constraint_function: str


class DataScientistResponse(BaseModel):
    thoughts: str
    preference_function: str
    constraint_function: str


class RepairResponse(BaseModel):
    code_with_correct_syntax: str


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


def get_preference_prompt(features_prompt: str, examples: list[str] = None) -> str:
    if examples is not None:
        example_prompt = "\n".join([example for example in examples])
    else:
        example_prompt = ""
    return f"""You are a Python data analyst that analyzes hotel data.
Your boss gave you the task to write a Python preference function.
It rates a hotel between 0 and 1 according to a user's natural language query.
So your task is to first identify which preferences the user has.
Then, formalize them as a Python function.
You have access to the following global variables which contain the hotel data:
{features_prompt}

Furthermore, you can use numpy as np and math as math.

Here are some examples:
{example_prompt}

Now, return valid, well-formed Python code. If there is no need for a function, simply return:
def preference_function() -> float:
    return 1.
"""


def get_constraint_prompt(features_prompt: str, examples: list[str] = None) -> str:
    if examples is not None:
        example_prompt = "\n".join([example for example in examples])
    else:
        example_prompt = ""
    return f"""You are a Python data analyst that analyzes hotel data.
Your boss gave you the task to write a Python constraint function.
It decides if a hotel should be considered based on a user's natural language query.
So your goal is to first identify the constraints the user has.
Then, formalize the user's constraints as a Python function.
You have access to the following global variables which contain the hotel data:
{features_prompt}

Furthermore, you can use numpy as np and math as math.

Here are some examples:
{example_prompt}

Now, return valid, well-formed Python code. If there is no need for a function, simply return:
def constraint_function() -> bool:
    return True
"""


def get_combined_prompt(user_query: str, features_prompt: str, examples: list[str] = None) -> str:
    if examples is not None:
        example_prompt = "\n".join([example for example in examples])
    else:
        example_prompt = ""
    return f"""You are a Python data analyst that analyzes hotel data.
Your boss gave you the task to write two Python functions that can be used to find appropriate hotels corresponding to the user's wishes.
This is the user query:
{user_query}

You have access to the following global variables which contain the hotel data:
{features_prompt}

Furthermore, you can use numpy as np and math as math.
Now, first think about which variables to use and which user requirements are constraints vs. which are only preferences
Then, formalize these thoughts into a preference function which calculates a score to rank the hotel, and a constraint function which decides if the hotel is valid.
We will use both functions in the end to first filter the hotels and then return the top results.

Here are some examples:
{example_prompt}

Now, return valid, well-formed Python code. Also, stick to the example function signatures.
"""


def query_functions(query: str, features: dict[str, any], combined=True) -> tuple[str, str]:
    preference_example = """def preference_function() -> float:
    price_score = 1 / (1 + np.exp(pricepernight / 100 - 2))
    baby_friendly_variables = [
        baby_or_kinderbetreuung,
        babybadewanne,
        kinder_orbabybetten_auf_anfrage,
        kinderbetreuung_im_zimmer,
        kinderhochstuhl,
        kindermahlzeiten,
        wickeltisch,
        betreute_kinderaktivitäten,
        buggys,
        geschirr_für_kinder
    ]
    baby_score = sum(baby_friendly_variables) / (len(baby_friendly_variables) + 1e-9)
    meal_scores = {
        'only_stay': 0.0,  # No meals, doesn't meet requirement
        'breakfast': 1.0,  # Meets minimum requirement
        'half_board': 1.2,  # Better than minimum
        'all_inclusive': 1.5  # Best option
    }
    meal_score = meal_scores[mealtype]
    total_score = (0.3 * price_score) + (0.4 * baby_score) + (0.3 * meal_score)
    return total_score"""
    constrain_example = """def constraint_function() -> bool:
    has_kids_amenity = kinderbecken == True
    has_minimum_meal = mealtype in ['breakfast', 'half_board', 'all_inclusive']
    is_affordable = pricepernight <= 300
    return has_kids_amenity or (has_minimum_meal and is_affordable)"""
    features_prompt = get_features_prompt(features)

    preference_response: PreferenceResponse | None = None
    constraint_response: ConstraintResponse | None = None

    if not combined:
        user_prompt = f"This is the user's query:\n{query}"
        preference_prompt = get_preference_prompt(features_prompt, [preference_example])
        constraint_prompt = get_constraint_prompt(features_prompt, [constrain_example])

        def get_preference(retry_num=2):
            nonlocal preference_response
            while retry_num > 0:
                preference_response = generate_structured_completion(
                    user_prompt,
                    preference_prompt,
                    PreferenceResponse,
                    save_prefix="preference",
                )
                if preference_response is not None:
                    break
                retry_num -= 1

        def get_constraint(retry_num=2):
            nonlocal constraint_response
            while retry_num > 0:
                constraint_response = generate_structured_completion(
                    user_prompt,
                    constraint_prompt,
                    ConstraintResponse,
                    save_prefix="constraint",
                )
                if constraint_response is not None:
                    break
                retry_num -= 1

        # Create and start threads
        pref_thread = threading.Thread(target=get_preference)
        constr_thread = threading.Thread(target=get_constraint)

        pref_thread.start()
        constr_thread.start()

        # Wait for both threads to complete
        pref_thread.join()
        constr_thread.join()

        if preference_response is None:
            f_p = """def preference_function() -> float:
        return 1."""
        else:
            f_p = preference_response.preference_function

        if constraint_response is None:
            f_c = """def constraint_function() -> bool:
        return True"""
        else:
            f_c = constraint_response.constraint_function

        return f_p, f_c
    else:
        user_prompt = "Your response:"
        combined_prompt = get_combined_prompt(query, features_prompt, [constrain_example, preference_example])

        response: DataScientistResponse | None = generate_structured_completion(
            user_prompt,
            combined_prompt,
            DataScientistResponse,
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
            function_string = function_string.split("(")[0] + "():\n" + "\n".join(function_string.split("\n")[1:])
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
            print(f"Repaired parameter {wrong_parameter} to {actual_name} with similariy {similarity_ordering[actual_name]:.2f}")
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
            "Here is the code:\n" + function_string + "\nHere is the error:\n" + error_description,
            """You are an expert at writing correct Python syntax.
You will get a Python method and an error message. You can assume that all variables exist. 
Return the improved function.""",
            RepairResponse,
            model="gpt-4o-mini-0718-eu",
            save_prefix="repair"
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
