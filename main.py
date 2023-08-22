from fastapi import FastAPI
from langchain import PromptTemplate, HuggingFaceHub, LLMChain
import os
import json


with open('config.json', 'r') as file:
    data = json.load(file)
    api_token = data['api_token']


app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/generate_response/{param1}/{param2}")
async def dummy_route(param1: str, param2: str):
    response_text = get_response(param1, param2)
    return {"response": response_text}

def get_response(param1: str, param2: str) -> str:

    os.environ["HUGGINGFACEHUB_API_TOKEN"] = api_token

    template = """Question: {query}

    Answer: Let's think step by step. Based on the information in the paragraph, {answer}.

    Here is the complete paragraph:

    {data}"""

    prompt = PromptTemplate(template=template, input_variables=["query", "answer", "data"])

    llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.001})

    llm_chain = LLMChain(prompt=prompt, llm=llm)

    inputs = {"query": param1, "answer": param2, "data": param2}
    answer = llm_chain.run(inputs)
    print(f"Answer: {answer}")
    return answer

