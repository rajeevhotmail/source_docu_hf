import os
import base64
import requests
import argparse
import ast
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

def get_github_repo_files(owner, repo, branch="main"):
    """Fetches the list of files from a GitHub repository."""
    url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
    headers = {}

    # Add GitHub token if available for authentication
    token = os.getenv("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"token {token}"

    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        file_data = response.json()
        # Filter only Python files
        py_files = [file['path'] for file in file_data['tree'] if file['path'].endswith('.py')]
        return py_files
    else:
        print("Error fetching repo files:", response.status_code, response.json())
        return []

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
        print(f"Error fetching file {filepath}:", response.status_code, response.json())
        return None

# Initialize the pipeline with `facebook/bart-large-cnn`
summarization_pipeline = pipeline("summarization",
                                  model="facebook/bart-large-cnn",
                                  tokenizer="facebook/bart-large-cnn")

def extract_functions(code):
    """Extracts functions and their signatures from the code."""
    tree = ast.parse(code)
    functions = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            func_code = ast.get_source_segment(code, node)
            func_name = node.name
            func_args = [arg.arg for arg in node.args.args]
            func_return = "N/A"  # Simplification: More complex parsing can be added
            functions.append((func_name, func_code, func_args, func_return))
    return functions

def summarize_function(name, func_code, func_args, func_return, max_length=100, min_length=30):
    """Generates a summary for a single function."""
    prompt = (
        f"Summarize the following Python function:\n\n"
        f"Function Name: {name}\n"
        f"Arguments: {func_args}\n"
        f"Return: {func_return}\n"
        f"Code:\n{func_code}\n\nSummary:"
    )
    result = summarization_pipeline([prompt], max_length=max_length, min_length=min_length)
    return result[0]['summary_text']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize Python code files from local path or GitHub")
    parser.add_argument("--github", action="store_true", help="Fetch files from GitHub")
    parser.add_argument("--local", help="Path to local Python project directory")
    parser.add_argument("--owner", default="pypa", help="GitHub repository owner")
    parser.add_argument("--repo", default="sampleproject", help="GitHub repository name")
    parser.add_argument("--branch", default="main", help="GitHub branch")
    parser.add_argument("--max_length", type=int, default=100, help="Maximum length of the summary")
    args = parser.parse_args()

    try:
        if args.github:
            py_files = get_github_repo_files(args.owner, args.repo, args.branch)
            summaries = []
            for py_file in py_files:
                code_content = get_github_file_content(args.owner, args.repo, py_file, args.branch)
                if code_content:
                    functions = extract_functions(code_content)
                    summaries.extend(
                        [summarize_function(name, func_code, func_args, func_return, args.max_length)
                         for name, func_code, func_args, func_return in functions]
                    )
            print(f"Summary of {args.owner}/{args.repo} repository:")
            print("\n\n".join(summaries))
        elif args.local:
            for root, dirs, files in os.walk(args.local):
                for file in files:
                    if file.endswith(".py"):
                        file_path = os.path.join(root, file)
                        with open(file_path, 'r') as f:
                            code_content = f.read()
                        functions = extract_functions(code_content)
                        summaries = [
                            summarize_function(name, func_code, func_args, func_return, args.max_length)
                            for name, func_code, func_args, func_return in functions
                        ]
                        print(f"Summary of {file_path}:")
                        print("\n\n".join(summaries))
        else:
            print("Please use either --github to fetch from GitHub or --local to specify a local project directory")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Usage examples:
# python hf_working_reas.py --github --owner pypa --repo sampleproject --branch main
# python hf_working_reas.py --local path/to/your/local/project_directory
