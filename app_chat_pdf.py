import os
from key import openai_api_key
import streamlit as st
from PyPDF2 import PdfReader  # Reads pdf file.
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain

os.environ['OPENAI_API_KEY']= openai_api_key

st.set_page_config(page_title="Ask your PDF")
st.header("💬Chat with your PDF🗒️")

#upload file
pdf= st.file_uploader("Upload your pdf", type="pdf")

#extracting the text from pdf
if pdf is not None:
    pdf_reader= PdfReader(pdf)
    text=""
    for page in pdf_reader.pages:
        text +=page.extract_text()

    text_splitter= CharacterTextSplitter(
        separator="\n", 
        chunk_size=1000, 
        chunk_overlap=200, 
        #length_function=len
      )
    chunks= text_splitter.split_text(text)
    embedddings= OpenAIEmbeddings()
    knowledge_base= FAISS.from_texts(chunks, embedddings)


    user_question= st.text_input("Ask any question about the data in pdf: ")
    if user_question:
        docs= knowledge_base.similarity_search(user_question)
        llm= OpenAI()
        chain= load_qa_chain(llm, chain_type="stuff")
        response= chain.run(input_documents=docs, question=user_question)
        st.write(response)
