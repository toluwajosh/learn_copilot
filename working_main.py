"""
Chat with a document
"""


import json
import os

import openai
from dotenv import load_dotenv
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader,
    JSONLoader,
)
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.memory import ConversationBufferMemory
from langchain.vectorstores import Chroma

load_dotenv()

with open("config/params.json") as f:
    session_params = json.load(f)

doc_loaders = {
    "pdf": PyPDFLoader,
    "txt": TextLoader,
    "dir": DirectoryLoader,
    "json": JSONLoader,
}


# run params
DOC_PATH = session_params["doc_path"]
COLLECTION_NAME = session_params["collection_name"]
PERSIST = session_params["persist"]
LOADER = doc_loaders[session_params["loader"]]


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
if PERSIST and os.path.exists("persist"):
    print("Reusing index...\n")
    vectorstore = Chroma(
        persist_directory="persist", embedding_function=OpenAIEmbeddings()
    )
    index = VectorStoreIndexWrapper(vectorstore=vectorstore)
else:
    loader = LOADER(DOC_PATH)
    if PERSIST:
        index = VectorstoreIndexCreator(
            vectorstore_kwargs={"persist_directory": "persist"}
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
        name="English language syllabus",
        func=doc_agent.run,
        description="Useful for when asking about the English language syllabus, or just simply syllabus",
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
)

while True:
    input_text = input(">>> ")
    try:
        agent.run(input=input_text)
    except (ValueError, openai.error.InvalidRequestError) as e:
        print(e)
