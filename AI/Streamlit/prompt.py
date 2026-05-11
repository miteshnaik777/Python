import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
import getpass


api_key = st.text_input("Enter your api key:")    

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=api_key
)


st.title("AI Assistant")
prompt = st.text_input("Enter your prompt:")

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory()

conversation = ConversationChain(
    llm=llm,
    verbose=False,
    memory=st.session_state.memory,
)



if st.button("Generate Prompt"):
    st.write(conversation.predict(input=prompt))  
