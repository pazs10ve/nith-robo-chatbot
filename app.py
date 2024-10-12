import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import textwrap

token = "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"
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", token=token)
model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0", token=token)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def generate_response(prompt, min_length = 1000, max_length=100000, temperature=0.7):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)    
    output = model.generate(
        **inputs, 
        min_length = min_length,
        max_length=max_length, 
        temperature=temperature, 
        do_sample=True, 
        top_k=50
    )
    
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

template_question = "User: {question} \n Assistant:"

def generate_response_from_template(question):
    prompt = template_question.format(question=question)
    response = generate_response(prompt)
    return response

st.title("Robosoc Chatbot")

st.write("Ask anything ")

user_input = st.text_input("Your question:", "")

if st.button("Get Response"):
    if user_input:
        with st.spinner("Generating response..."):
            response = generate_response_from_template(user_input)
            wrapped_response = textwrap.fill(response, width=80)
            st.write("**Assistant:**")
            st.write(wrapped_response)
    else:
        st.write("Please enter a question.")

