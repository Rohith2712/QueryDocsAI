import os
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import streamlit as st

class DocumentHandler:
    @staticmethod
    def load_document(file):
        name, extension = os.path.splitext(file)

        if extension == '.txt':
            loader = TextLoader(file)
        elif extension == '.pdf':
            loader = PyPDFLoader(file)
        return loader.load()

    @staticmethod
    def process_uploaded_file(upload_file, chunk_size, k):
        bytes_data = upload_file.read()
        file_name = os.path.join('./', upload_file.name)
        with open(file_name, 'wb') as f:
            f.write(bytes_data)

        data = DocumentHandler.load_document(file_name)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=k, length_function=len)
        texts = text_splitter.split_documents(data)
        st.write('Chunk size:', len(texts))

        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        from langchain_community.vectorstores import FAISS
        embedding = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        db = FAISS.from_documents(texts, embedding)
        st.session_state.vs = db
