from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables.history import RunnableWithMessageHistory
from vectorstore_handler import VectorStoreHandler
from langchain_google_genai import ChatGoogleGenerativeAI
import config

class AskHandler:
    @staticmethod
    def ask_and_get_answer(store, q, k):
        prompt = VectorStoreHandler.create_prompt()
        
        retriever = store.as_retriever(search_type="similarity", search_kwargs={"k": k})

        chain = ({"context": retriever | VectorStoreHandler.format_docs, "question": RunnablePassthrough()} 
                 | prompt | ChatGoogleGenerativeAI(model="gemini-pro", convert_system_message_to_human=True) 
                 | StrOutputParser())
        doc = chain.invoke(q)
        return doc
