import argparse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import requests
import os
import base64
import google.generativeai as genai
import requests
import os
import base64

#genai.configure(api_key=os.environ["GEMINI_API_KEY"])
genai.configure(api_key="AIzaSyBvND7YoOuNVlmLNdawYPRbJKYfcVQNnaU")


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



def summarize_code_with_gemini(code_content):
    """Generates a summary for the provided code content using Gemini."""
    prompt = f"""
    You are a helpful AI assistant that can summarize Python code.

    Please summarize the following Python code:

    ```python
    {code_content}
    ```

    Provide a summary that includes:

    * The purpose of the code.
    * The main functions and their functionality.
    * Any important classes or variables.
    * External dependencies.

    The summary should be clear, concise, and easy for a non-programmer to understand.
    """

    model = genai.GenerativeModel('gemini-pro')
    response = model.generate_content(prompt)
    summary = response.text
    return summary



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize Python code files from local path or GitHub")
    parser.add_argument("--github", action="store_true", help="Fetch file from GitHub")
    parser.add_argument("--local", help="Path to local Python file")
    parser.add_argument("--owner", default="pypa", help="GitHub repository owner")
    parser.add_argument("--repo", default="sampleproject", help="GitHub repository name")
    parser.add_argument("--filepath", default="src/sample/simple.py", help="File path in the GitHub repository")
    parser.add_argument("--branch", default="main", help="GitHub branch")
    parser.add_argument("--max_length", type=int, default=100, help="Maximum length of the summary")
    args = parser.parse_args()

    try:
        if args.github:
            code_content = get_github_file_content(args.owner, args.repo, args.filepath, args.branch)
            if code_content:
                summary = summarize_code_with_gemini(code_content)
                print("Generated Summary:\n", summary)
            else:
                print("Failed to fetch the file from GitHub")
        elif args.local:
            if os.path.exists(args.local):
                with open(args.local, 'r') as file:
                    code_content = file.read()
                summary = summarize_code_with_gemini(code_content)
                print("Generated Summary:\n", summary)
            else:
                print(f"Error: Local file {args.local} not found")
        else:
            print("Please use either --github to fetch from GitHub or --local to specify a local file path")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

# Usage examples:
# python hf_working_reas.py --github --owner pypa --repo sampleproject --filepath src/sample/simple.py --branch main
# python hf_working_reas.py --local path/to/your/local/file.py
