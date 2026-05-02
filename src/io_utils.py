import argparse
import json
from typing import Dict, List, Literal, Any

from pydantic import BaseModel, Field


FormatPossible = Literal["number", "string", "boolean"]


class PromptItem(BaseModel):
    prompt: str


class Parameter(BaseModel):
    type: FormatPossible


class FunDef(BaseModel):
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    parameters: Dict[str, Parameter]
    returns: Parameter


def load_json_list(path: str) -> List[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, list):
        raise ValueError(f"{path} must contain a JSON list.")

    for i, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Item {i} in {path} is not a JSON object.")

    return data


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Call Me Maybe: parsing input files"
        )
    parser.add_argument(
        "--functions_definition",
        type=str,
        default="data/input/functions_definition.json",
        help="Path to the functions definition JSON file",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/input/function_calling_tests.json",
        help="Path to the test prompts JSON file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/output/function_calling_results.json",
        help="Path where the results will be saved later",
    )
    return parser.parse_args()


def parse_files(
    input_path: str,
    definition_path: str,
) -> tuple[List[PromptItem], List[FunDef]]:
    prompts_raw = load_json_list(input_path)
    prompts = [PromptItem(**item) for item in prompts_raw]

    functions_raw = load_json_list(definition_path)
    functions = [FunDef(**item) for item in functions_raw]

    return prompts, functions
