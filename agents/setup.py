import os

from dotenv import load_dotenv
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.memory import ConversationBufferMemory
from langchain.tools import DuckDuckGoSearchRun
from langchain.vectorstores import Chroma
from langchain.agents import load_tools

from agents.settings import PARAMS

from .models import get_model

load_dotenv()


def get_agent(
    collection_name: str,
    collection_description: str,
    collection_path: str,
    persist_path: str,
    persist: bool,
    enable_search: bool = False,
):
    llm = get_model(PARAMS.model["name"], PARAMS.model["params"])
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True
    )

    if os.path.isdir(collection_path):
        loader = DirectoryLoader(collection_path)
    else:
        loader = TextLoader(collection_path)

    # tool
    if persist and os.path.exists(persist_path):
        print("Reusing index...\n")
        vectorstore = Chroma(
            persist_directory=persist_path,
            embedding_function=OpenAIEmbeddings(),
        )
        index = VectorStoreIndexWrapper(vectorstore=vectorstore)
    else:
        if persist:
            print("Creating index...\n")
            index = VectorstoreIndexCreator(
                vectorstore_kwargs={"persist_directory": persist_path}
            ).from_loaders([loader])
        else:
            index = VectorstoreIndexCreator().from_loaders([loader])

    doc_agent = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=index.vectorstore.as_retriever(
            search_kwargs={
                # "reduce_k_below_max_tokens": True,
                # "max_tokens_limit": 4097,
            },
            return_source_documents=True,
        ),
        # max_tokens_limit=4097
        # reduce_k_below_max_tokens=True,
    )

    # option 2
    # doc_agent = ConversationalRetrievalChain.from_llm(
    #     llm,
    #     index.vectorstore.as_retriever(search_kwargs={"k": 1}),
    #     max_tokens_limit=4097,
    # )

    tools = [
        Tool(
            name=collection_name,
            func=doc_agent.run,
            description=collection_description,
        ),
    ]

    if enable_search:
        tools.append(
            Tool(
                name="search",
                func=DuckDuckGoSearchRun().run,
                description="Useful to search for general information and current events.",
            )
        )
        # tools += load_tools(
        #     ["searx-search"], searx_host="http://localhost:8888", llm=llm
        # )

    print("tools: ", tools)
    return initialize_agent(
        tools,
        llm,
        # agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        # agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        verbose=True,
        memory=memory,
        handle_parsing_errors=True,
    )
