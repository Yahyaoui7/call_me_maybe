from llm_sdk import Small_LLM_Model

def ft_encode(full_prompts):
    model = Small_LLM_Model()
    for prompt in full_prompts:
        input_ids = model.encode(prompt)
        logits = model.get_logits_from_input_ids(input_ids[0].tolist())
    print (logits)
