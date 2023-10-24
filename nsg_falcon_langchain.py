import os

from dotenv import load_dotenv
from langchain.llms import HuggingFaceHub 
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate 

load_dotenv()


repo_id = "tiiuae/falcon-7b-instruct"
llm = HuggingFaceHub(huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"], 
                     repo_id=repo_id, 
                     model_kwargs={"temperature":0.7, "max_new_tokens":700})

template = """
You are a helpful AI assistant and provide the answer for the question asked politely.

{question}
Answer: Let's think step by step.
"""
prompt = PromptTemplate(template=template, input_variables=["question"])
llm_chain = LLMChain(prompt=prompt, llm=llm)

question = "How to cook Pizza ?"

print(llm_chain.run(question))