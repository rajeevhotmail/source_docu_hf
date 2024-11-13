import os
import base64
import requests
import argparse
import ast
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

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
        if file_data["type"] == "file":
            file_content = base64.b64decode(file_data["content"]).decode("utf-8")
            return file_content
        else:
            print(f"Skipping non-file item: {filepath}")
            return None
    else:
        print("Error fetching file:", response.status_code, response.json())
        return None

def get_github_files_in_directory(owner, repo, directory_path, branch="main"):
    """Recursively fetches all Python files in a GitHub directory."""
    url = f"https://api.github.com/repos/{owner}/{repo}/contents/{directory_path}?ref={branch}"
    headers = {"Authorization": f"token {os.getenv('GITHUB_TOKEN')}"}
    response = requests.get(url, headers=headers)
    files = []

    if response.status_code == 200:
        contents = response.json()
        for item in contents:
            if item["type"] == "file" and item["name"].endswith(".py"):
                files.append(item["path"])
            elif item["type"] == "dir":
                # Recursive call for subdirectory
                files.extend(get_github_files_in_directory(owner, repo, item["path"], branch))
    return files

def get_local_files_in_directory(directory_path):
    """Recursively fetches all Python files in a local directory."""
    files = []
    for root, _, filenames in os.walk(directory_path):
        for filename in filenames:
            if filename.endswith(".py"):
                files.append(os.path.join(root, filename))
    return files

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
            functions.append((node.name, func_code))
    return functions

def summarize_function(name, func_code, max_length=100, min_length=30):
    """Generates a summary for a single function."""
    prompt = f"Python function: {name}\nCode:\n{func_code}\nSummary:"
    result = summarization_pipeline([prompt], max_length=max_length, min_length=min_length)
    return result[0]['summary_text']

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize Python code files from local path or GitHub")
    parser.add_argument("--github", action="store_true", help="Fetch files from a GitHub repository")
    parser.add_argument("--local", help="Path to local directory or file")
    parser.add_argument("--owner", default="pypa", help="GitHub repository owner")
    parser.add_argument("--repo", default="sampleproject", help="GitHub repository name")
    parser.add_argument("--filepath", default="src/sample", help="File or directory path in the GitHub repository")
    parser.add_argument("--branch", default="main", help="GitHub branch")
    parser.add_argument("--max_length", type=int, default=100, help="Maximum length of the summary")
    args = parser.parse_args()

    try:
        if args.github:
            code_content = get_github_file_content(args.owner, args.repo, args.filepath, args.branch)
            if code_content:
                functions = extract_functions(code_content)
                summaries = [summarize_function(name, func_code, args.max_length) for name, func_code in functions]
                print(f"Summary of {args.filepath} from {args.owner}/{args.repo}:")
                print("\n\n".join(summaries))
            else:
                print("Failed to fetch the file from GitHub")

            # Uncomment this block if you want to list all files in a GitHub directory.
            # files = get_github_files_in_directory(args.owner, args.repo, args.filepath, args.branch)
            # for filepath in files:
            #     code_content = get_github_file_content(args.owner, args.repo, filepath, args.branch)
            #     if code_content:
            #         functions = extract_functions(code_content)
            #         summaries = [summarize_function(name, func_code, args.max_length) for name, func_code in functions]
            #         print(f"Summary of {filepath} from {args.owner}/{args.repo}:")
            #         print("\n\n".join(summaries))
            #     else:
            #         print(f"Failed to fetch file: {filepath}")

        elif args.local:
            if os.path.isdir(args.local):
                files = get_local_files_in_directory(args.local)
            elif os.path.isfile(args.local) and args.local.endswith(".py"):
                files = [args.local]
            else:
                print(f"Invalid local path: {args.local}")
                files = []

            for filepath in files:
                with open(filepath, 'r') as file:
                    code_content = file.read()
                functions = extract_functions(code_content)
                summaries = [summarize_function(name, func_code, args.max_length) for name, func_code in functions]
                print(f"Summary of local file {filepath}:")
                print("\n\n".join(summaries))
        else:
            print("Please specify either --github or --local to proceed")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
