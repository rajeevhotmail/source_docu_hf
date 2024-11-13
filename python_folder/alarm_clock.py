import os
import base64
import requests
import argparse
import ast
from transformers import pipeline

# Initialize the pipeline with `facebook/bart-large-cnn`
summarization_pipeline = pipeline("summarization", model="facebook/bart-large-cnn", tokenizer="facebook/bart-large-cnn")

# Function to extract functions and their signatures from the code
def extract_functions(code):
	"""Extracts functions and their signatures from the code."""
	tree = ast.parse(code)
	functions = []
	for node in ast.walk(tree):
		if isinstance(node, ast.FunctionDef):
			func_code = ast.get_source_segment(code, node)
			functions.append((node.name, func_code))
	return functions

# Function to generate a summary for a single function
def summarize_function(name, func_code):
	"""Generates a summary for a single function."""
	prompt = f"Python function: {name}\nCode:\n{func_code}\nSummary:"
	result = summarization_pipeline([prompt], max_length=100, min_length=30)
	return result[0]['summary_text']

# Main function to process local directory
if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Summarize Python code files from a local path")
	parser.add_argument("--local", help="Path to local Python project directory")
	args = parser.parse_args()

	try:
		if args.local:
			summaries = []
			print(f"Scanning directory: {args.local}")
			for root, dirs, files in os.walk(args.local):
				print(f"Checking directory: {root}")  # Debug statement
				for file in files:
					if file.endswith(".py"):
						print(f"Found Python file: {file}")  # Debug statement
						file_path = os.path.join(root, file)
						with open(file_path, 'r') as f:
							code_content = f.read()
						print(f"Code content of {file_path}:\n{code_content}\n")  # Debug statement
						functions = extract_functions(code_content)
						for name, func_code in functions:
							print(f"Extracted function: {name}")  # Debug statement
						summaries.extend(
							[summarize_function(name, func_code)
							 for name, func_code in functions]
						)
			print("Summary of local project:")
			if summaries:
				print("\n\n".join(summaries))
			else:
				print("No Python files found in the specified directory.")
		else:
			print("Please specify a local project directory using --local")
	except Exception as e:
		print(f"An error occurred: {str(e)}")

# Usage example:
# python reading_from_multiple_files.py --local "D:\\python-mini-projects-master\\python-mini-projects-master\\projects\\Alarm clock"
