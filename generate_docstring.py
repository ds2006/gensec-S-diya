from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import GoogleGenerativeAI
import readline

llm = GoogleGenerativeAI(model="gemini-1.5-pro-latest",temperature=0)

# reads in a python file and uses NumPy Style docstrings to comment the program
while True:
    try:
        filename= input("Give me a python file name>> ")
        with open(filename, 'r') as f:
            line = f.read()
        if line:
            result = llm.invoke(f"Write a NumPy Style docstring for the following program: {line}")
            print(result)
        else:
            break
    except:
        break
