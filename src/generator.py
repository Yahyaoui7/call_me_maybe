from typing import Any
from .prompts import PromptBuilder
from .decoder import Decoder
import json


class FunctionCallGenerator:

    def __init__(
        self,
        functions: list,
        prompt_builder: PromptBuilder,
        decoder: Decoder,
    ) -> None:
        self.functions = functions
        self.prompt_builder = prompt_builder
        self.decoder = decoder

    def generate_function_name(
        self,
    ) -> str:
        function_names = [function.name for function in self.functions]

        full_prompt = self.prompt_builder.build_function_name_prompt(
            self.functions)

        return self.decoder.constrained_generate_from_choices(
            full_prompt=full_prompt,
            choices=function_names,
        )

    def find_function_by_name(self, function_name: str):

        for function in self.functions:
            if function.name == function_name:
                return function
        raise ValueError(f"Function not found {function_name}")

    def generate_parameters(
        self,
        function_name: str,
    ) -> dict[str, Any]:
        """Generate parameters for the selected function."""

        func = self.find_function_by_name(function_name)
        full_prompt = self.prompt_builder.build_parameters_prompt(func)

        json_text = self.decoder.generate_json_text(full_prompt)
        print("JSON TEXT:", repr(json_text))
        parameters = json.loads(json_text)

        if not isinstance(parameters, dict):
            raise ValueError("Generated parameters are not a JSON object.")

        return parameters
