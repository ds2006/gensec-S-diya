import os
import requests
from bs4 import BeautifulSoup
import torch
import warnings
from transformers import pipeline
from langchain_google_genai import GoogleGenerativeAI


cache_dir = ".cache"

def get_github_repo_content(username, repo_name):
    """Fetches content of all .py and .md files in a GitHub repository."""
    base_url = f"https://api.github.com/repos/{username}/{repo_name}/contents/"
    headers = {"Accept": "application/vnd.github.v3+json"}

    response = requests.get(base_url, headers=headers)
    response.raise_for_status()

    content_data = response.json()

    all_content = ""
    for item in content_data:
        if item["type"] == "file" and item["name"].endswith((".py", ".md")):
            file_url = item["download_url"]
            file_response = requests.get(file_url)
            file_response.raise_for_status()
            all_content += file_response.text + "\n\n"
    return all_content

def summarize_content(text, max_length=500):
    """Summarizes text using a pre-trained model."""
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    # Chunk the text for summarization  
    chunk_size = 2000  # Adjust as needed                                                                                                                                                         
    chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
    summaries = []
    for chunk in chunks:
        summary = summarizer(
            chunk, max_length=max_length, min_length=30, do_sample=False, truncation=True
        )
        summaries.append(summary[0]["summary_text"])
    return "\n".join(summaries)


def get_readme_content(username, repo_name):
    """Fetches the raw README content from the GitHub repository."""
    readme_url = f"https://raw.githubusercontent.com/{username}/{repo_name}/main/README.md"
    response = requests.get(readme_url)
    if response.status_code == 200:
        return response.text
    else:
        return ""


def get_repo_content(username, repo_name):
    """Fetches content from README and code files."""

    readme_content = get_readme_content(username, repo_name)
    code_content = get_github_repo_content(username, repo_name)
    # pdf_content = get_pdf_content(username, repo_name)                                                                                                                                          

    return readme_content, code_content


def analyze_repo_structure(username, repo_name):
    """Fetches the repository content and analyzes its structure."""
    base_url = f"https://api.github.com/repos/{username}/{repo_name}/contents/"
    headers = {"Accept": "application/vnd.github.v3+json"}

    response = requests.get(base_url, headers=headers)
    response.raise_for_status()

    content_data = response.json()

    repo_files = []
    for item in content_data:
        if item["type"] == "file":
            file_name = item["name"]
            file_type = file_name.split(".")[-1] if "." in file_name else "unknown"
            repo_files.append({"name": file_name, "type": file_type})

    return repo_files
def ask_question(question, context, username, repo_name):
    """Asks a question about the repository content and returns the answer."""
    import os
    prompt = f"""                                                                                                                                                                                 
    You are a helpful AI assistant. Answer the following question based on the provided GitHub repository information.                                                                            
    Repository Owner: {username}                                                                                                                                                                  
    Repository Name: {repo_name}                                                                                                                                                                  
    Repository Content: {context}                                                                                                                                                                 
                                                                                                                                                                                                  
    Question: {question}                                                                                                                                                                          
    """
    llm = GoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0, api_key=os.environ.get("GOOGLE_API_KEY"))
    response = llm.invoke(prompt)
    return response
    
def main():
    """Main function to run the summarization and Q&A."""
    github_link = input(
        "Enter the GitHub link (e.g., https://github.com/username/repo): "
    ).strip()
    github_link = github_link.replace(".git", "")
    if github_link.endswith("/"):
        github_link = github_link[:-1]
    username = github_link.split("/")[-2]
    repo_name = github_link.split("/")[-1]

    # Get README and code content                                                                                                                                                                 
    readme_content = get_readme_content(username, repo_name)
    code_content = get_github_repo_content(username, repo_name)
    print(f"## README Summary:\n{summarize_content(readme_content)}\n")
    run_question_answering(username, repo_name, readme_content, code_content)
    
def run_question_answering(username, repo_name, readme_content, code_content):
    """Runs the question answering loop."""
    repo_files = analyze_repo_structure(username, repo_name)
    file_list_string = ', '.join([file['name'] for file in repo_files])
    while True:
        question = input("Ask a question about the repository (or type 'exit'): ")
        if question.lower() == 'exit':
            break
        # Combine README and code content for context                                                                                                                                             
        context = f"```\n{readme_content}\n```\n```\n{code_content}\n```\nFile list: {file_list_string}"
        answer = ask_question(question, context, username, repo_name)
        if answer:
            print(f"Answer: {answer}\n")

if __name__ == "__main__":
    main()
