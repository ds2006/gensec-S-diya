from langchain_google_genai import GoogleGenerativeAI
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_experimental.tools import PythonREPLTool
import readline
from langchain.agents import tool
import wikipedia

"""Different Custom Tools"""
@tool
def get_word_length(word: str) -> int:
    """Returns the length of a word."""
    return len(word)
@tool
def searchWikipedia(input):
    """ gives you a summary of the phrase you enter or answers a question based on the first few sentences in a wikipedia page"""
    # getting suggestions                                                                                                                                                                         
    result = wikipedia.summary(input, sentences = 2)
    return result

"""All Accesible Tools"""
tools = [PythonREPLTool(),get_word_length,searchWikipedia]


llm = GoogleGenerativeAI(model="gemini-1.5-pro-latest",temperature=0)

instructions = """You are an agent designed to write and execute python code to                                                                                                                   
answer questions.  You have access to a python REPL, get_word_length, and searchWikipedia                                                                                                         
which you can use to execute python code, get the length of an answer, or search wikipedia.                                                                                                       
If you get an error, debug your code and try again.  Only use the                                                                                                                                 
output of your code to answer the question. If it does                                                                                                                
not seem like you can write code to answer the question, just return "I don't know" as the answer
"""
base_prompt = hub.pull("langchain-ai/react-agent-template")
prompt = base_prompt.partial(instructions=instructions)

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

print(f"Welcome to my application.  I am configured with these tools")
for tool in tools:
  print(f'  Tool: {tool.name} = {tool.description}')

while True:
    try:
        line = input("llm>> ")
        if line:
            result = agent_executor.invoke({"input":line})
            print(result)
        else:
            break
    except Exception as e:
        print(e)
