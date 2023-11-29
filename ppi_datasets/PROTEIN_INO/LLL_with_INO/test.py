import torch
from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer

if __name__ == '__main__':
    tmp_model_path = "/results/llama2/final_checkpoint/"
    print("Loading the checkpoint in a Llama model.")
    model = LlamaForCausalLM.from_pretrained(tmp_model_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True)
    tokenizer = LlamaTokenizer.from_pretrained(tmp_model_path)
    # Specify input
    text = "What is OVHcloud?"

    # Specify device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt").to(device)

    # Get answer
    # (Adjust max_new_tokens variable as you wish (maximum number of tokens the model can generate to answer the input))
    outputs = model.generate(input_ids=inputs["input_ids"].to(device), attention_mask=inputs["attention_mask"],
                             max_new_tokens=50, pad_token_id=tokenizer.eos_token_id)

    # Decode output & print it
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))