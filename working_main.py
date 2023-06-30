"""
Chat with a document
"""


import os

import openai
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

from settings import PARAMS

load_dotenv()


# run params
COLLECTION_PATH = PARAMS["collection_path"]
COLLECTION_NAME = PARAMS["collection_name"]
COLLECTION_DESCRIPTION = PARAMS["collection_description"]
PERSIST = PARAMS["persist"]
PERSIST_PATH = PARAMS["persist_path"]
RERUN_INDEXING = PARAMS["rerun_indexing"]


llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
memory = ConversationBufferMemory(
    memory_key="chat_history", return_messages=True
)

# Tools #######################################################################

# search
# search = SerpAPIWrapper() # need api

# math
# llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)

# database
# db = SQLDatabase.from_uri("sqlite:///../../../../../notebooks/Chinook.db")
# db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)

# document with persistent index
if PERSIST and os.path.exists(PERSIST_PATH):
    if RERUN_INDEXING:
        print("Rerunning index...\n")
        loader = TextLoader(COLLECTION_PATH)
        index = VectorstoreIndexCreator(
            vectorstore_kwargs={"persist_directory": PERSIST_PATH}
        ).from_loaders([loader])
    else:
        print("Reusing index...\n")
        vectorstore = Chroma(
            persist_directory=PERSIST_PATH,
            embedding_function=OpenAIEmbeddings(),
        )
        index = VectorStoreIndexWrapper(vectorstore=vectorstore)
else:
    loader = TextLoader(COLLECTION_PATH)
    if PERSIST:
        index = VectorstoreIndexCreator(
            vectorstore_kwargs={"persist_directory": PERSIST_PATH}
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
        name=COLLECTION_NAME,
        func=doc_agent.run,
        description=COLLECTION_DESCRIPTION,
        # return_direct=True, # I dont understand this
    ),
    # Tool(
    #     name="Calculator",
    #     func=llm_math_chain.run,
    #     description="useful for when you need to answer questions about math",
    # ),
    # Tool(
    #     name="Search",
    #     func=search.run,
    #     description="useful for when you need to answer questions about current events. You should ask targeted questions",
    # ),
    # Tool(
    #     name="FooBar-DB",
    #     func=db_chain.run,
    #     description="useful for when you need to answer questions about FooBar. Input should be in the form of a question containing full context",
    # ),
]

agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=True,
    # agent_kwargs=agent_kwargs,
    memory=memory,
    handle_parsing_errors=True,
)

while True:
    input_text = input(">>> ")

    if input_text.upper() == "Q":
        print("Quitting...")
        break

    try:
        output = agent.run(input=input_text)
        # print("Chain output: ", output)
    except (ValueError, openai.error.InvalidRequestError) as e:
        print(e)
