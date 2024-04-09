__import__("pysqlite3")
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

DB_DIRECTORY = "./chroma_db"

class LLMChatBot:

    def __init__(self, api_key) -> None:
        llm = ChatOpenAI(model="gpt-3.5-turbo-0125", api_key=api_key)
        
        vectorstore = Chroma(persist_directory=DB_DIRECTORY, embedding_function=OpenAIEmbeddings(api_key=api_key))
        retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

        prompt = hub.pull("rlm/rag-prompt")

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        self.rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    def ask_question(self, question):
        return self.rag_chain.invoke(question)