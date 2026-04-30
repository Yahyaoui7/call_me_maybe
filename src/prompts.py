
class PromptBuilder:
    def __init__(self, prompt: str) -> None:
        self.prompt = prompt

    def build_function_name_prompt(self, functions: list) -> str:
        text = "Task: match the user request to exactly one function name from the list.\n"
        text += "Available functions:\n"

        for function in functions:
            text += f"- {function.name}: {function.description}\n"
            text += "Parameters:\n"
            for name, schema in function.parameters.items():
                text += f"- {name}: {schema.type}\n"
            text += "\n"

        text += f"User request:\n{self.prompt}\n"
        text += "Function name is: "
        return text

    def build_parameters_prompt(self, function) -> str:

        text = "You extract only function parameters from user requests.\n"

        text += "\nRules:\n"
        text += "- Extract the function inputs, not the function result.\n"
        text += "- Do not execute the function.\n"
        text += '- For fn_reverse_string, parameter "s" must be the original string from the request.\n'
        text += "- If parameter type is number, return a number without quotes.\n"
        text += "- Convert simple number words to numbers: one=1, two=2, three=3.\n"
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

        text += "Function: fn_substitute_string_with_regex(source_string: string, regex: string, replacement: string)\n"
        text += 'Request: Replace all numbers in "Hello 34 I\'m 233 years old" with NUMBERS\n'
        text += 'Answer:\n{"source_string": "Hello 34 I\'m 233 years old", "regex": "\\\\d+", "replacement": "NUMBERS"}\n\n'

        text += "Function: fn_substitute_string_with_regex(source_string: string, regex: string, replacement: string)\n"
        text += 'Request: Replace all vowels in "Programming is fun" with *\n'
        text += 'Answer:\n{"source_string": "Programming is fun", "regex": "[aeiouAEIOU]", "replacement": "*"}\n\n'

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






base_parameter_prompt = """<|im_start|>system
You extract function call parameters from user requests and output JSON inside <tool_call> tags.
Extract string values exactly as   they appear in the request.<|im_end|>

<|im_start|>user
Function: fn_greet(name: string)
Request: Say hello to Alice<|im_end|>
<|im_start|>assistant
<tool_call>
{"function_name": "fn_greet", "parameters": {"name": "Alice"}}
</tool_call><|im_end|>

<|im_start|>user
Function: fn_read_file(path: string, encoding: string)
Request: Read the file at /home/user/data.txt using utf-8<|im_end|>
<|im_start|>assistant
<tool_call>
{"function_name": "fn_read_file", "parameters": {"path": "/home/user/data.txt", "encoding": "utf-8"}}
</tool_call><|im_end|>

<|im_start|>user
Function: fn_format_template(template: string)
Request: Format the template "Hello, {name}! You have {count} messages."<|im_end|>
<|im_start|>assistant
<tool_call>
{"function_name": "fn_format_template", "parameters": {"template": "Hello, {name}! You have {count} messages."}}
</tool_call><|im_end|>
<|im_start|>user
Function: fn_reverse_string(s: string)
Request: Reverse the string "hello world"<|im_end|>
<|im_start|>assistant
<tool_call>
{"function_name": "fn_reverse_string", "parameters": {"s": "hello world"}}
</tool_call><|im_end|>

<|im_start|>user
Function: fn_substitute_string_with_regex(source_string: string, regex: string, replacement: string)
Request: Replace all numbers in "Hello 34 I'm 233 years old" with NUMBER<|im_end|>
<|im_start|>assistant
<tool_call>
{"function_name": "fn_substitute_string_with_regex", "parameters": {"source_string": "Hello 34 I'm 233 years old", "regex": "\d+", "replacement": "NUMBER"}}
</tool_call><|im_end|>

<|im_start|>user
Function: fn_execute_sql_query(query: string, database: string)
Request: Run "SELECT * FROM users" on the production database<|im_end|>
<|im_start|>assistant
<tool_call>
{"function_name": "fn_execute_sql_query", "parameters": {"query": "SELECT * FROM users", "database": "production"}}
</tool_call><|im_end|>

<|im_start|>user
Function: """
