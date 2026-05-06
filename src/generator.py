from typing import Any
from .prompts import PromptBuilder
from .decoder import Decoder
import json


class FunctionCallGenerator:

    def __init__(
        self,
        functions: list[Any],
        prompt_builder: PromptBuilder,
        decoder: Decoder,
        prompt: str
    ) -> None:
        self.functions = functions
        self.prompt_builder = prompt_builder
        self.decoder = decoder
        self.prompt = prompt

    def generate_function_name(
        self,
    ) -> str:
        function_names = [function.name for function in self.functions]
        function_names.append("fn_none__")
        full_prompt = self.prompt_builder.build_function_name_prompt(
            self.functions
        )

        chosen_name = self.decoder.constrained_generate_from_choices(
            full_prompt=full_prompt,
            choices=function_names)
        if chosen_name == "fn_none__":
            raise TypeError(
                f"No matching function found for prompt: {self.prompt}"
            )
        return chosen_name

    def find_function_by_name(self, function_name: str) -> Any:
        """Return the function object matching the given function name."""

        for function in self.functions:
            if function.name == function_name:
                return function

        raise ValueError(f"Function not found: {function_name}")

    def generate_parameters(
        self,
        function_name: str,
    ) -> dict[str, Any]:
        """Generate parameters for the selected function."""

        func = self.find_function_by_name(function_name)
        full_prompt = self.prompt_builder.build_parameters_prompt(func)

        json_text = self.decoder.generate_json_text(full_prompt)
        try:
            parameters = json.loads(json_text)
        except json.JSONDecodeError:
            raise ValueError(f"Parameters are not valid JSON: {json_text}")
        if not isinstance(parameters, dict):
            raise ValueError("Generated parameters are not a JSON object.")

        return parameters
