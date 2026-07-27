import os
from PIL import Image
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
import shutil


vector_space_dir = os.path.join(os.getcwd(), "vector_DB")
if not os.path.exists(vector_space_dir):
    os.makedirs(vector_space_dir)


icon = Image.open("chatbot.jpg")

st.set_page_config(page_title="RAG Chatbot", page_icon=icon, layout="centered")
st.title("RAG Chatbot Ask your documents anything!")

if 'vectorstore' not in st.session_state:
    st.session_state['vectorstore'] = None
if 'memory' not in st.session_state:
    st.session_state['memory'] = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
if 'retriever' not in st.session_state:
    st.session_state['retriever'] = None                                                                                                                                 

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"], key="pdf_uploader")
embedding_model = OllamaEmbeddings(model="embeddinggemma")

if uploaded_file is not None and st.session_state['vectorstore'] is None:
    with st.spinner("Loading PDF and creating vector DB...."):
        pdf_path = os.path.join(os.getcwd(), uploaded_file.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state['pdf_file_path'] = pdf_path
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        vectorstore = FAISS.from_documents(documents, embedding_model)
        vectorstore.save_local(vector_space_dir)
        st.session_state['vectorstore'] = vectorstore
        st.session_state['retriever'] = vectorstore.as_retriever(search_kwargs={"k": 3})
        st.success("Vector DB created successfully!")

llm = OllamaLLM(model="llama2")
if st.session_state['retriever'] is not None:
    qa_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=st.session_state['retriever'], memory=st.session_state['memory'], return_source_documents=False)
    user_question = st.text_input("Ask a question about the document:", key="text")
    if user_question:
        with st.spinner("Thinking...."):
            result = qa_chain.run({"question": user_question})
            st.markdown(f"**You**: {user_question}")
            st.markdown(f"**Assistant**: {result}")

def del_vectordb(vector_space_dir):
    if os.path.exists(vector_space_dir):
        shutil.rmtree(vector_space_dir)

def del_pdf(path):
    if os.path.exists(path) and path.endswith(".pdf"):
        os.remove(path)

# def del_uploaded_file(original_pdf):
#     if uploaded_file is not None:
#         os.remove(original_pdf)

if st.button("Clear Chat History"):
    st.session_state['memory'].clear()
    st.session_state['vectorstore'] = None
    st.session_state['retriever'] = None
    del_vectordb(vector_space_dir)
    pdf = st.session_state.get('pdf_file_path', None)
    del_pdf(pdf)
    st.session_state['pdf_file_path'] = None
    for key in ['pdf_uploader', 'text']:
        if key in st.session_state:
            del st.session_state[key]
    # del_uploaded_file(uploaded_file)
    st.success("Chat history and vector DB cleared successfully!")
    st.rerun()
