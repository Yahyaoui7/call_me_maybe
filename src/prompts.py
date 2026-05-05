from typing import Any
from .io_utils import FunDef


class PromptBuilder:
    def __init__(self, prompt: str) -> None:
        self.prompt = prompt


    def build_function_name_prompt(
        self,
        functions: list[FunDef],
    ) -> str:
        text = "Choose the best function for the user request.\n\n"

        text += "Rules:\n"
        text += (
            "- Choose exactly one function name from the available functions.\n"
        )
        text += '- Choose "__none__" only if no available function can do the request.\n'
        text += "- Do not guess if the request is unrelated.\n"
        text += "- Write only the function name.\n\n"

        text += "Examples:\n"
        text += "Request: What is the weather today?\n"
        text += "Answer: __none__\n\n"

        text += "Request: Open Google Chrome\n"
        text += "Answer: __none__\n\n"

        text += "Request: What is the sum of 2 and 3?\n"
        text += "Answer: fn_add_numbers\n\n"

        text += "Request: Greet John\n"
        text += "Answer: fn_greet\n\n"

        text += 'Request: Reverse the string "hello"\n'
        text += "Answer: fn_reverse_string\n\n"

        text += "Available functions:\n"
        for function in functions:
            text += f"- {function.name}: {function.description}\n"

        text += "\nUser request:\n"
        text += self.prompt
        text += "\n\nAnswer:"

        return text

    def build_parameters_prompt(self, function: Any) -> str:
        text = "You extract only function parameters from user requests.\n"

        text += "\nRules:\n"
        text += "- Extract the function inputs, not the function result.\n"
        text += "- Do not execute the function.\n"
        text += (
            '- For fn_reverse_string, parameter "s" must be the original '
            "string from the request.\n"
        )
        text += (
            "- If parameter type is number, return a number without quotes.\n"
        )
        text += (
            "- Convert simple number words to numbers: "
            "one=1, two=2, three=3.\n"
        )
        text += '- If the user says "half of X", use X / 2.\n'
        text += '- If the user says "all numbers", use regex "\\\\d+".\n'
        text += '- If the user says "all vowels", use regex "[aeiouAEIOU]".\n'
        text += "- Return only a JSON object.\n"
        text += "- Do not include markdown.\n"
        text += "- Do not include explanation.\n"

        text += "\nExamples:\n"

        text += "Function: fn_reverse_string(s: string)\n"
        text += 'Request: Reverse the string "hello"\n'
        text += 'Answer:\n{"s": "hello"}\n\n'

        text += "Function: fn_reverse_string(s: string)\n"
        text += 'Request: Reverse the string "world"\n'
        text += 'Answer:\n{"s": "world"}\n\n'

        text += "Function: fn_add_numbers(a: number, b: number)\n"
        text += "Request: What is the sum of half of two and 3?\n"
        text += 'Answer:\n{"a": 1, "b": 3}\n\n'

        text += (
            "Function: fn_substitute_string_with_regex("
            "source_string: string, "
            "regex: string, "
            "replacement: string)\n"
        )
        text += (
            "Request: Replace all numbers in "
            '"Hello 34 I\'m 233 years old" with NUMBERS\n'
        )
        text += (
            'Answer:\n{"source_string": "Hello 34 I\'m 233 years old", '
            '"regex": "\\\\d+", '
            '"replacement": "NUMBERS"}\n\n'
        )

        text += (
            "Function: fn_substitute_string_with_regex("
            "source_string: string, "
            "regex: string, "
            "replacement: string)\n"
        )
        text += 'Request: Replace all vowels in "Programming is fun" with *\n'
        text += (
            'Answer:\n{"source_string": "Programming is fun", '
            '"regex": "[aeiouAEIOU]", '
            '"replacement": "*"}\n\n'
        )

        text += "Now extract parameters for this request.\n"

        text += "Function: "
        text += f"{function.name}("
        text += ", ".join(
            f"{name}: {schema.type}"
            for name, schema in function.parameters.items()
        )
        text += ")\n"

        text += f"Request: {self.prompt}\n"
        text += "Answer:\n"

        return text
