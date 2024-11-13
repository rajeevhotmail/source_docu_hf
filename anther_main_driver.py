import os
import base64
import requests
import argparse
from transformers import AutoTokenizer, AutoModelWithLMHead, SummarizationPipeline

def get_github_file_content(owner, repo, filepath, branch="main"):
    """Fetches the content of a file from a GitHub repository."""
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{filepath}?ref={branch}"
    headers = {}

    # Add GitHub token if available for authentication
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"token {token}"

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        file_data = response.json()
        # Decode the base64-encoded content
        file_content = base64.b64decode(file_data["content"]).decode("utf-8")
        return file_content
    else:
        print("Error fetching file:", response.status_code, response.json())
        return None

# Initialize the pipeline
pipeline = SummarizationPipeline(
    model=AutoModelWithLMHead.from_pretrained("SEBIS/code_trans_t5_large_source_code_summarization_python_multitask_finetune"),
    tokenizer=AutoTokenizer.from_pretrained("SEBIS/code_trans_t5_large_source_code_summarization_python_multitask_finetune", skip_special_tokens=True),
    device=-1,
)

def tokenize_code(code):
    # Simple tokenization: replace strings with CODE_STRING
    return code.replace('"', ' CODE_STRING ').replace("'", ' CODE_STRING ')

def summarize_code(code, max_length=300):
    tokenized_code = tokenize_code(code)
    result = pipeline([tokenized_code], max_length=max_length)
    return result[0]['summary_text']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize Python code files from local path or GitHub")
    parser.add_argument("--github", action="store_true", help="Fetch file from GitHub")
    parser.add_argument("--local", help="Path to local Python file")
    parser.add_argument("--owner", default="pypa", help="GitHub repository owner")
    parser.add_argument("--repo", default="sampleproject", help="GitHub repository name")
    parser.add_argument("--filepath", default="src/sample/simple.py", help="File path in the GitHub repository")
    parser.add_argument("--branch", default="main", help="GitHub branch")
    parser.add_argument("--max_length", type=int, default=300, help="Maximum length of the summary")
    args = parser.parse_args()

    try:
        if args.github:
            code_content = get_github_file_content(args.owner, args.repo, args.filepath, args.branch)
            if code_content:
                summary = summarize_code(code_content, args.max_length)
                print(f"Summary of {args.filepath} from {args.owner}/{args.repo}:")
                print(summary)
            else:
                print("Failed to fetch the file from GitHub")
        elif args.local:
            if os.path.exists(args.local):
                with open(args.local, 'r') as file:
                    code_content = file.read()
                summary = summarize_code(code_content, args.max_length)
                print(f"Summary of local file {args.local}:")
                print(summary)
            else:
                print(f"Error: Local file {args.local} not found")
        else:
            print("Please use either --github to fetch from GitHub or --local to specify a local file path")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Usage examples:
# python anther_main_driver.py --github --owner pypa --repo sampleproject --filepath src/sample/simple.py --branch main
# python anther_main_driver.py --local path/to/your/local/file.py