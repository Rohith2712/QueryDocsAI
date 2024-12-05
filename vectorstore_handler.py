from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate

class VectorStoreHandler:
    @staticmethod
    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    @staticmethod
    def create_prompt():
        template = """Answer the question based only on the below context:
        {context}

        Question:{question}"""
        return ChatPromptTemplate.from_template(template)
