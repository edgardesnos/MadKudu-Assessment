from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage
from langchain_community.vectorstores import FAISS

DB_DIRECTORY = "./document_db"

CONTEXTUALIZE_Q_SYSTEM_PROMPT = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""
CONTEXTUALIZE_Q_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", CONTEXTUALIZE_Q_SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

QA_SYSTEM_PROMPT = """You are an assistant for question-answering tasks about the MadKudu platform. \
You speak using a professional and respectful tone. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise but clear.\

{context}"""
QA_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", QA_SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


class LLMChatBot:

    def __init__(self, api_key) -> None:
        llm = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0, api_key=api_key)

        vectorstore = FAISS.load_local(
            folder_path=DB_DIRECTORY,
            embeddings=OpenAIEmbeddings(api_key=api_key),
            allow_dangerous_deserialization=True,
        )
        retriever = vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={"score_threshold": 0.3, "k": 6},
        )

        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, CONTEXTUALIZE_Q_PROMPT
        )

        question_answer_chain = create_stuff_documents_chain(llm, QA_PROMPT)

        self.rag_chain = create_retrieval_chain(
            history_aware_retriever, question_answer_chain
        )
        self.chat_history = []

    def ask_question(self, question):
        chain_response = self.rag_chain.invoke(
            {"input": question, "chat_history": self.chat_history}
        )
        self.chat_history.extend(
            [HumanMessage(content=question), chain_response["answer"]]
        )
        return chain_response
