import streamlit as st
from langchain import HuggingFaceHub, PromptTemplate, LLMChain

huggingfacehub_api_token = "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"
repo_id = "tiiuae/falcon-7b-instruct"


template_rephrase = """
You are an artificial intelligence assistant. Your task is to generate a text that is similar to the one provided, in other words, rephrase it. 
Rephrase the text below, delimited by triple backticks.
Text: '''{text}'''
answer:
"""

# Text correction
llm_correct = HuggingFaceHub(
    huggingfacehub_api_token=huggingfacehub_api_token,
    repo_id=repo_id,
    model_kwargs={"temperature": 0.1, "max_new_tokens": 500},
)

template_correct = """
You are an artificial intelligence assistant. Your task is to correct the following text and rewrite the corrected version without Rephrasing the text,
only dont use it correct the text below , delimited by triple backticks .
Text: '''{text}'''
answer:
"""

def rephrase_text(text, temperature):
    llm_rephrase = HuggingFaceHub(
    huggingfacehub_api_token=huggingfacehub_api_token,
    repo_id=repo_id,
    model_kwargs={"temperature": temperature, "max_new_tokens": 500},
    )
    prompt = PromptTemplate(template=template_rephrase, input_variables=["text"])
    llm_chain = LLMChain(prompt=prompt, llm=llm_rephrase)
    return llm_chain.run(text)

def correct_text(text):
    prompt = PromptTemplate(template=template_correct, input_variables=["text"])
    llm_chain = LLMChain(prompt=prompt, llm=llm_correct)
    return llm_chain.run(text)

# Streamlit app
st.title("Text Transformation App")

# Choose transformation type
transformation_type = st.selectbox("Choose Transformation Type", ["Rephrase", "Correct"])

if transformation_type == "Rephrase":
    text_to_transform = st.text_area("Enter Text to Rephrase")
    temperature = st.slider("Select Creativity Level (Temperature)", 0.1, 1.0, 0.6, 0.1)
    if st.button("Rephrase"):
        result = rephrase_text(text_to_transform, temperature)
        st.markdown(result)

elif transformation_type == "Correct":
    text_to_transform = st.text_area("Enter Text to Correct")
    if st.button("Correct"):
        result = correct_text(text_to_transform)
        st.markdown(result)
