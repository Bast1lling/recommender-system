import pandas as pd
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from code_act import evaluate_function, query_functions, test_evaluate_function
from llm import generate_embeddings_for_list
from parse_hotel_data import generate_features, generate_values


def find_matching_hotels(
        query: str, hotels: dict[str, dict[str, object]]
) -> list[str] | None:
    """
    Find matching hotels based on the given query.

    Args:
        query (str): The search query from the user.
        hotels (dict[str, dict[str, object]]): Dictionary containing hotel information.
            Format: {
                "hotel_name": {
                    "name": str,
                    "rating": float,
                    "distance_to_beach": float,
                    ...
                },
                ...
            }

    Returns:
        list[str] | None: List of hotel_names that match the query, or None if the query is not hotel related.
    """
    if not hotels:
        return []
    features = generate_features(hotels)
    hotel_values = generate_values(hotels, features)
    assert set(hotels.keys()) == set(
        hotel_values.keys()
    ), "Hotels and hotel_values have different keys"
    parameters = list(list(hotel_values.values())[0].keys())
    parameter_embeddings = generate_embeddings_for_list(parameters)
    preference_function, constraint_function = query_functions(query, features)
    repaired_parameters = {}

    # filter hotels based on constraint function
    filtered_hotels = {}
    for hotel_name, hotel in hotels.items():
        keep, constraint_function, repaired_parameters = evaluate_function(
            constraint_function, hotel_values[hotel_name], parameter_embeddings, repaired_parameters=repaired_parameters
        )
        if keep:
            filtered_hotels[hotel_name] = hotel

    repaired_parameters = {}
    score_dict = {}
    for hotel_name, hotel in hotels.items():
        score, preference_function, repaired_parameters = evaluate_function(
            preference_function, hotel_values[hotel_name], parameter_embeddings, repaired_parameters=repaired_parameters
        )
        score_dict[hotel_name] = score

    print(score_dict)
    # sort hotels based on preference function
    sorted_hotels = sorted(
        filtered_hotels.items(),
        key=lambda x: score_dict[x[0]],
        reverse=True,
    )
    return [
        hotel_name for hotel_name, _ in sorted_hotels[: min(len(sorted_hotels), 10)]
    ]


if __name__ == "__main__":
    user_query = "I want a popular hotel that is very highly ranked. Furthermore my budget in total is 2000€. I want to either play some blackjack or go shopping."
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "hotels")
    mallorca_df = pd.read_parquet(os.path.join(data_dir, "resultlist_New York.parquet"))
    mallorca_dict = {
        row["hotel_name"]: row.to_dict() for _, row in mallorca_df.iterrows()
    }
    hotels = mallorca_dict
    print(find_matching_hotels(user_query, hotels))
