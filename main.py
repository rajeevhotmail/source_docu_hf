from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, SummarizationPipeline

pipeline = SummarizationPipeline(
    model=AutoModelForSeq2SeqLM.from_pretrained(
        "SEBIS/code_trans_t5_large_source_code_summarization_python_multitask_finetune"
    ),
    tokenizer=AutoTokenizer.from_pretrained(
        "SEBIS/code_trans_t5_large_source_code_summarization_python_multitask_finetune",
        skip_special_tokens=True,
    ),
    device=-1,  # Use CPU

)

def summarize_code_from_file(filename, max_new_tokens=50):  # Add max_new_tokens parameter
    """Reads Python code from a file and summarizes it."""
    try:
        with open(filename, "r") as in_file:
            code = in_file.read()

        tokenized_code = pipeline.tokenizer.tokenize(code)
        tokenized_code = " ".join(tokenized_code)

        result = pipeline([tokenized_code], max_new_tokens=max_new_tokens)  # Pass max_new_tokens
        print(result[0]['summary_text'])

    except FileNotFoundError:
        print(f"Error: The input file '{filename}' was not found.")

# Example usage with a longer summary:
summarize_code_from_file("input.py", max_new_tokens=300)