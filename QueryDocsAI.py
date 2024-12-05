import streamlit as st
from document_handler import DocumentHandler
from vectorstore_handler import VectorStoreHandler
from ask_handler import AskHandler

st.set_page_config(page_title="QueryDocsAI", page_icon="📝")


with st.sidebar:
    upload_file = st.file_uploader('Upload a file:', type=['pdf', 'txt'])
    chunk_size = st.number_input('Chunk size:', min_value=100, max_value=2048, value=100)
    k = st.number_input('Chunk overlap', min_value=1, max_value=20, value=3)
    add_data = st.button('Add Data')
    if upload_file and add_data:
        DocumentHandler.process_uploaded_file(upload_file, chunk_size, k)
        

st.title("AskMyDocs")
question = st.text_input("Question: ")
btn = st.button("Find Answer")

if btn:
    if 'vs' in st.session_state:
        vector_store = st.session_state.vs
        with st.spinner("Generating answer... please wait!"):
            answer = AskHandler.ask_and_get_answer(vector_store, question, k)
        answer_length = len(answer)
        height = max(200, min(500, answer_length // 2)) 
        st.text_area('Answer:', value=answer, height=height)
        st.divider()

    if 'history' not in st.session_state:
        st.session_state.history = ''
