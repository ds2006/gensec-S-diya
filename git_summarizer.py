import os
import requests
from bs4 import BeautifulSoup
import torch
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer
from utils import find_pdfs

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
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    summaries = []
    for chunk in chunks:
        summary = summarizer(chunk, max_length=max_length, min_length=30, do_sample=False)
        summaries.append(summary[0]['summary_text'])
    return "\n".join(summaries)

def get_repo_content(username, repo_name):
    """Fetches content from README and code files."""

    def get_readme_content(username, repo_name):
        """Fetches and extracts content from the README file."""
        readme_urls = [
            f"https://github.com/{username}/{repo_name}/blob/master/README.md",
            f"https://github.com/{username}/{repo_name}/blob/main/README.md",
            f"https://github.com/{username}/{repo_name}",
        ]
        response = None
        for url in readme_urls:
            response = requests.get(url)
            if response.status_code == 200:
                readme_url = url
                break
       if response is None or response.status_code != 200:
            print("README not found. Skipping...")
            return ""

        soup = BeautifulSoup(response.text, 'html.parser')
        possible_elements = ['article', 'div', 'section']
        possible_classes = ['markdown-body', 'entry-content', 'container-lg', 'Box-body', 'px-5', 'pb-5']
        for element in possible_elements:
            for class_name in possible_classes:
                readme_element = soup.find(element, class_=class_name)
                if readme_element:
                    readme_content = readme_element.get_text(separator='\n', strip=True)
                    summary = summarize_content(readme_content)
                    print(f"Got summary: {repr(summary)}")
                    return summary
        print(f"Could not find README content. readme_element: {readme_element}")
        return ""
    readme_content = get_readme_content(username, repo_name)
    code_content = get_github_repo_content(username, repo_name)
    pdf_content = get_pdf_content(username, repo_name)

    return readme_content, code_content, pdf_content

def get_pdf_content(username, repo_name):
    """Fetches and extracts content from PDF files in the repository."""
    return "" # TODO: implement this                                                                                                                                                              

def main():
    """Main function to run the summarization."""
    github_link = input("Enter the GitHub link (e.g., https://github.com/username/repo): ").strip()
    github_link = github_link.replace(".git", "")
    if github_link.endswith("/"):
        github_link = github_link[:-1]
    username = github_link.split("/")[-2]
    repo_name = github_link.split("/")[-1]

    # Get README and code content                                                                                                                                                                 
    readme_content, _, _ = get_repo_content(username, repo_name)
    print(f"## README Summary:\n{readme_content}\n")

if __name__ == "__main__":
    main()
