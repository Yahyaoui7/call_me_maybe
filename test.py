from llm_sdk import Small_LLM_Model

def main() -> None:
    model = Small_LLM_Model()

    prompt = "What is the sum of 2 and 3?"
    generated_text = ""

    for _ in range(20):
        full_text = prompt + generated_text

        input_ids = model.encode(full_text)[0].tolist()
        logits = model.get_logits_from_input_ids(input_ids)

        next_token_id = logits.index(max(logits))
        next_token = model.decode([next_token_id])

        if not next_token:
            break

        generated_text += next_token
        print("next token:", next_token)
        print("generated:", generated_text)
        print()

    print("final answer:", generated_text)

if __name__ == "__main__":
    main()





# model = Small_LLM_Model()

# full_prompt = build_prompt(user_prompt, functions)
# input_ids = model.encode(full_prompt)[0].tolist()

# generated_ids = []

# while not done:
#     current_ids = input_ids + generated_ids
#     logits = model.get_logits_from_input_ids(current_ids)

#     allowed_ids = get_allowed_token_ids(current_text, functions, vocab)

#     next_id = pick_best_valid_token(logits, allowed_ids)
#     generated_ids.append(next_id)

# result = model.decode(generated_ids)
