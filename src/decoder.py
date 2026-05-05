from typing import Any
import json


class Decoder:

    def __init__(self, model: Any) -> None:
        self.model = model

    def encode_to_ids(self, text: str) -> Any:
        ids_obj = self.model.encode(text)
        ids_obj = ids_obj.tolist()
        ids_obj = ids_obj[0]
        return ids_obj

    def get_allowed_next_ids(
        self,
        generated_ids: list[int],
        choice_ids: list[list[int]],
    ) -> set[int]:
        allowed_ids: set[int] = set()

        for ids in choice_ids:
            if ids[: len(generated_ids)] == generated_ids:
                if len(ids) > len(generated_ids):
                    allowed_ids.add(ids[len(generated_ids)])

        return allowed_ids

    def choose_best_allowed_token(
        self,
        logits: Any,
        allowed_ids: set[int],
    ) -> int:
        best_id: int | None = None
        best_score: float | None = None

        for token_id in allowed_ids:
            score = logits[token_id]

            if best_score is None or score > best_score:
                best_score = score
                best_id = token_id

        if best_id is None:
            raise ValueError("Could not choose token.")

        return best_id

    def constrained_generate_from_choices(
        self,
        full_prompt: str,
        choices: list[str],
    ) -> str:
        prompt_ids = self.encode_to_ids(full_prompt)
        choice_ids = [self.encode_to_ids(choice) for choice in choices]
        generated_ids: list[int] = []

        while True:
            for choice, ids in zip(choices, choice_ids):
                if generated_ids == ids:
                    return choice

            allowed_ids = self.get_allowed_next_ids(
                generated_ids,
                choice_ids,
            )

            if not allowed_ids:
                raise ValueError("No valid token allowed.")

            input_ids = prompt_ids + generated_ids
            logits = self.model.get_logits_from_input_ids(input_ids)

            next_id = self.choose_best_allowed_token(
                logits=logits,
                allowed_ids=allowed_ids,
            )

            generated_ids.append(next_id)

    #   this bolw about decode prompt ?

    def choose_best_token(self, logits: Any) -> int:

        best_id = 0
        best_score = logits[0]

        for token_id, score in enumerate(logits):
            if score > best_score:
                best_score = score
                best_id = token_id

        return best_id

    def extract_json_object(self, text: str) -> str:
        start = text.find("{")

        if start == -1:
            raise ValueError("No JSON object found.")

        decoder = json.JSONDecoder()
        _, end = decoder.raw_decode(text[start:])

        return text[start:start + end]

    def generate_json_text(
        self,
        full_prompt: str,
        max_tokens: int = 200,
    ) -> Any:
        prompt_ids = self.encode_to_ids(full_prompt)
        generated_ids: list[int] = []

        for _ in range(max_tokens):
            input_ids = prompt_ids + generated_ids
            logits = self.model.get_logits_from_input_ids(input_ids)
            next_id = self.choose_best_token(logits)
            generated_ids.append(next_id)

            generated_text = self.model.decode(generated_ids)

            try:
                return self.extract_json_object(generated_text)
            except json.JSONDecodeError:
                pass
            except ValueError:
                pass

        raise ValueError("Could not generate complete JSON.")
