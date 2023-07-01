import os

from dotenv import load_dotenv
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma

from agents.settings import PARAMS

load_dotenv()


def get_agent(
    collection_name: str,
    collection_description: str,
    collection_path: str,
    persist_path: str,
    rerun_indexing: bool,
    persist: bool,
):
    llm = ChatOpenAI(temperature=0.5, model=PARAMS.model)
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True
    )

    # tool
    if persist and os.path.exists(persist_path):
        if rerun_indexing:
            print("Rerunning index...\n")
            loader = TextLoader(collection_path)
            index = VectorstoreIndexCreator(
                vectorstore_kwargs={"persist_directory": persist_path}
            ).from_loaders([loader])
        else:
            print("Reusing index...\n")
            vectorstore = Chroma(
                persist_directory=persist_path,
                embedding_function=OpenAIEmbeddings(),
            )
            index = VectorStoreIndexWrapper(vectorstore=vectorstore)
    else:
        loader = TextLoader(collection_path)
        if persist:
            index = VectorstoreIndexCreator(
                vectorstore_kwargs={"persist_directory": persist_path}
            ).from_loaders([loader])
        else:
            index = VectorstoreIndexCreator().from_loaders([loader])

    doc_agent = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=index.vectorstore.as_retriever(),
    )

    tools = [
        Tool(
            name=collection_name,
            func=doc_agent.run,
            description=collection_description,
        ),
    ]

    return initialize_agent(
        tools,
        llm,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        memory=memory,
        handle_parsing_errors=True,
    )
