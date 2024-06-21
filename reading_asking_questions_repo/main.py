#main.py  
"""import statements"""
import os
import tempfile
from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAI
from langchain import PromptTemplate, LLMChain
from langchain.llms import OpenAI
from config import WHITE, GREEN, RESET_COLOR, model_name
from utils import format_user_question
from file_processing import clone_github_repo, load_and_index_files
from questions import ask_question, QuestionContext

load_dotenv()
GOOGLE_API_KEY = "ENTER_API_KEY_HERE"

def main():
  """This is the main program which clones the repo and uses the llm to answer questions about it. 
      While this program does use an LLM, many of the questions asked are returned with incorrect answers or an unsure response. """
  """cloning repo and making sure that there is something in it"""
    github_url = input("Enter the GitHub URL of the repository: ")
    repo_name = github_url.split("/")[-1]
    print("Cloning the repository...")
    with tempfile.TemporaryDirectory() as local_path:
        if clone_github_repo(github_url, local_path):
            index, documents, file_type_counts, filenames = load_and_index_files(local_path)
            if index is None:
                print("No documents were found to index. Exiting.")
                exit()
            """though printing"""
            print("Repository cloned. Indexing files...")
            """connecting to llm"""
            llm = GoogleGenerativeAI(model=("gemini-pro"),api_key=GOOGLE_API_KEY)

            template = """                                                                                                                                                                        
            Repo: {repo_name} ({github_url}) | Conv: {conversation_history} | Docs: {numbered_documents} | Q: {question} | FileCount: {file_type_counts} | FileNames: {filenames}     
            Instr:                                                                                                                                                                                
            1. Answer based on context/docs.                                                                                                                                                      
            2. Focus on repo/code.                                                                                                                                                                
            3. Consider:                                                                                                                                                                          
                a. Purpose/features - describe.                                                                                                                                                   
                b. Functions/code - provide details/samples.                                                                                                                                      
                c. Setup/usage - give instructions.                                                                                                                                               
            4. Unsure? Say "I am not sure".                                                                                                                                                       
                                                                                                                                                                                                  
            Answer:                                                                                                                                                                               
            """

            prompt = PromptTemplate(
                template=template,
                input_variables=["repo_name", "github_url", "conversation_history", "question", "numbered_documents", "file_type_counts", "filenames"]
            )

            llm_chain = LLMChain(prompt=prompt, llm=llm)

            conversation_history = ""
            question_context = QuestionContext(index, documents, llm_chain, model_name, repo_name, github_url, conversation_history, file_type_counts, filenames)
            while True:
                try:
                    """ask a question about the repo or exit"""
                    user_question = input("\n" + WHITE + "Ask a question about the repository (type 'exit()' to quit): " + RESET_COLOR)
                    if user_question.lower() == "exit()":
                        break
                    print('Thinking...')
                    user_question = format_user_question(user_question)
                    """Answer question in a different color"""
                    answer = ask_question(user_question, question_context)
                    print(GREEN + '\nANSWER\n' + answer + RESET_COLOR + '\n')
                    conversation_history += f"Question: {user_question}\nAnswer: {answer}\n"
                except Exception as e:
                    print(f"An error occurred: {e}")
                    break

        else:
          """catch if input is not a valid repo"""
            print("Failed to clone the repository.")


if __name__ == "__main__":
    main()
