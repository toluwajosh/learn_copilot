import os

from dotenv import load_dotenv
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.chains import (
    ConversationalRetrievalChain,
    RetrievalQA,
    LLMChain,
)
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.memory import ConversationBufferMemory
from langchain.tools import DuckDuckGoSearchRun
from langchain.vectorstores import Chroma
from langchain.agents import load_tools
from langchain import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain import PromptTemplate
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from functools import partial

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
    model: str = "OpenAI",
):
    llm = get_model(
        PARAMS.models[model]["name"], PARAMS.models[model]["params"]
    )
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
    )

    if collection_name == "Chat":
        template = "You are a helpful assistant who can answer question about anything. Explain with examples as much as you can and ask questions where necessary."

        # template = """You are Assistant, a large language model trained by OpenAI.

        # Assistant is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, Assistant is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

        # Assistant is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, Assistant is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

        # Overall, Assistant is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, Assistant is here to assist."""
        system_message_prompt = SystemMessagePromptTemplate.from_template(
            template
        )
        human_template = "{input}"
        human_message_prompt = HumanMessagePromptTemplate.from_template(
            human_template
        )
        chat_prompt = ChatPromptTemplate.from_messages(
            [system_message_prompt, human_message_prompt]
        )
        # chain = LLMChain(llm=llm, prompt=chat_prompt, return_final_only=False)
        tools = [
            Tool(
                name=collection_name,
                func=DuckDuckGoSearchRun().run,
                description=collection_description,
            ),
        ]
        # tools=[DuckDuckGoSearchRun(name="Search")]
    else:
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
            # chain_type="refine",
            retriever=index.vectorstore.as_retriever(
                search_kwargs={
                    # "reduce_k_below_max_tokens": True,
                    # "max_tokens_limit": 4097,
                    "k": 4,
                },
                return_source_documents=True,
            ),
            # max_tokens_limit=4097
            # reduce_k_below_max_tokens=True,
        )

        # # option 2
        # doc_agent = ConversationalRetrievalChain.from_llm(
        #     llm,
        #     index.vectorstore.as_retriever(search_kwargs={"k": 4}),
        #     # return_source_documents=True, # `run` not supported when there is not exactly one output
        #     # max_tokens_limit=4097,
        #     # chain expects multiple inputs ({'chat_history', 'question'})
        # )

        tools = [
            Tool(
                name=collection_name,
                func=doc_agent.run,
                description=collection_description,
            ),
        ]
    print("collection_name: ", collection_name)
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

    # print("tools: ", tools)
    return initialize_agent(
        tools,
        llm,
        # agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
        # agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        # agent=AgentType.CONVERSATIONAL_REACT_DESCRIPTION,
        memory=memory,
        handle_parsing_errors=True,
        # max_iterations=5,
        verbose=True,
        # return_intermediate_steps=True,
    )
