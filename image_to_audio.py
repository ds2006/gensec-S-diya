from dotenv import find_dotenv, load_dotenv
from transformers import pipeline
from langchain import PromptTemplate, LLMChain
from langchain_google_genai import GoogleGenerativeAI
import requests
import os
import urllib
import streamlit as st

load_dotenv(find_dotenv())
HUGGINGFACEHUB_API_TOKEN=os.getenv("HUGGINGFACEHUB_API_TOKEN")

"""cconverts the image to text"""
def img2text(url):
    image_to_text=pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")

    text=image_to_text(url)[0]["generated_text"]

    print(text)

    return text
"""uses a prompt template and a language model to create a story based on that image"""
def generate_story (scenario):
    template= """                                                                                                                                                                                 
    You are a story teller;                                                                                                                                                                       
    You can generate a short story based on a simple narrative, the story should be no mor                                                                                                        
    than 50 words;                                                                                                                                                                                
                                                                                                                                                                                                  
    CONTEXT: {scenario}                                                                                                                                                                           
    STORY:                                                                                                                                                                                        
    """
    prompt=PromptTemplate(template=template,input_variables=["scenario"])

    story_llm=LLMChain(llm=GoogleGenerativeAI(
        model=("gemini-pro")), prompt=prompt, verbose=True)

    story=story_llm.predict(scenario=scenario)

    print(story)
    return story


"""creates an audio file based on the story"""
def text2speech(message):
    API_URL = "https://api-inference.huggingface.co/models/espnet/kan-bayashi_ljspeech_vits"
    headers = {"Authorization": f"Bearer {HUGGINGFACEHUB_API_TOKEN}"}
    payloads={        
      "inputs":message
    }

    response=requests.post(API_URL,headers=headers,json=payloads)
    with open ('story.mp3','wb') as file:
        file.write(response.content)

"""responsible for running the program"""
url  = input("Enter a URL:")
urllib.request.urlretrieve(url, "file.jpg")
scenario=img2text("file.jpg")
story=generate_story(scenario)
text2speech(story)

