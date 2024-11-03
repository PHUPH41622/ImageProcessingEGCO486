from typing import (List, Dict, Any, TypeAlias)
import yaml

def read_yaml(file_path: str) -> Dict:
    """Get the key, value in the config.

    Args:
        file_path: The path to the yaml file.

    Returns:
        Dictionary of object.
    """
    with open(file_path, 'r') as file:
        data: Dict = yaml.safe_load(file)
    
    return data

def get_price_from_index(index_result: int) -> int:
    """Get the price from the index result.

    Args:
        index: Dict of index object.

    Returns:
        Price of each index.
    """
    
    if index_result == 0:       # snack
        return 10
    elif index_result == 1:     # water
        return 7
    elif index_result == 2:     # milk
        return 12
    elif index_result == 3:     # crackers
        return 20
    elif index_result == 4:     # candy
        return 15

def map_result_to_price(result: List) -> int:
    """Map the result to the price.

    Args:
        result: List of result.

    Returns:
        Total price.
    """
    total_price = 0
    for index in result:
        index_price = get_price_from_index(index)
        total_price += index_price

    return total_price



# ===== for testing =====

# if __name__ == "__main__":
#     config = read_yaml("config.yaml")
#     object_data = config.get("names")
#     result = map_result_to_price([0, 1, 2, 3, 4])
#     print(result)