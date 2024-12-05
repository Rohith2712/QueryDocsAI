from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS

# chromadb.api.client.SharedSystemClient.clear_system_cache()
# db_path = 'db'

# def remove_db_with_retry(db_path, retries=5, delay=1):
#     for i in range(retries):
#         try:
#             if os.path.exists(db_path):
#                 shutil.rmtree(db_path)
#             break
#         except PermissionError:
#             print(f"Retrying deletion of {db_path}... Attempt {i+1}/{retries}")
#             time.sleep(delay)
#     else:
#         print(f"Failed to delete {db_path} after {retries} retries.")


# remove_db_with_retry(db_path)

os.environ['GOOGLE_API_KEY']= st.secrets["GOOGLE_API_KEY"]
st.set_page_config(page_title="QueryDocsAI",page_icon="📝")
llm = ChatGoogleGenerativeAI(model="gemini-pro",convert_system_message_to_human=True)

if "vs" not in st.session_state:
    st.session_state.vs = {}

def format_docs(docs):
    return "\n\n".join([d.page_content for d in docs])

template = """Answer the question based only on the below context:
{context}

Question:{question}"""

prompt = ChatPromptTemplate.from_template(template)

def load_document(file):
    name,extension = os.path.splitext(file)

    if(extension == '.txt'):
        from langchain_community.document_loaders import TextLoader
        loader=TextLoader(file)
    elif(extension == '.pdf'):
        from langchain_community.document_loaders import PyPDFLoader
        loader=PyPDFLoader(file)
    return loader.load()
with st.sidebar:
    upload_file = st.file_uploader('Upload a file:',type=['pdf','txt'])
    chunk_size = st.number_input('Chunk size:',min_value=100,max_value=2048,value=100)

    k = st.number_input('Chunk overlap',min_value=1,max_value=20,value=3)

    add_data=st.button('Add Data')

    if upload_file and add_data:
        with st.spinner('Reading and chunking and embedding file'):
            bytes_data = upload_file.read()
            file_name=os.path.join('./',upload_file.name)
            with open(file_name,'wb') as f:
                f.write(bytes_data)
            data = load_document(file_name)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=k,length_function=len)
            texts = text_splitter.split_documents(data)
            st.write('chunk size',len(texts))
            embedding = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
            db = FAISS.from_documents(texts, embedding)
            st.session_state.vs=db

            

st.title("AskMyDocs")
question = st.text_input("Question: ")
btn = st.button("Find Answer")

def ask_and_get_answer(store,q,k):
    retriever = store.as_retriever(search_type="similarity", search_kwargs={"k": k})
    
    chain = ({"context":retriever | format_docs, "question":RunnablePassthrough() } | prompt | llm | StrOutputParser())
    doc = chain.invoke(q)
    return doc

if btn:
    if 'vs' in st.session_state:
        vector_store = st.session_state.vs
        with st.spinner("Generating answer... please wait!"):
            answer=ask_and_get_answer(vector_store,question,k)
        answer_length = len(answer)
        height = max(200, min(500, answer_length // 2)) 
        st.text_area('Answer:', value=answer, height=height)
        st.divider()

        if 'history' not in st.session_state:
            st.session_state.history=''
