from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("SEBIS/code_trans_t5_large_source_code_summarization_python_multitask_finetune")

with open("D:\\openai.py", "r") as f:
    text = f.read()

tokens = tokenizer.tokenize(text)
num_tokens = len(tokens)
print(num_tokens)