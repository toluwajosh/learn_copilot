import json
import os

import openai
from dotenv import load_dotenv
from langchain import (LLMMathChain, OpenAI, SerpAPIWrapper, SQLDatabase,
                       SQLDatabaseChain)
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.prompts import MessagesPlaceholder
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

load_dotenv()

with open("config/params.json") as f:
    session_params = json.load(f)


# run params
DOC_PATH = session_params["doc_path"]
COLLECTION_NAME = session_params["collection_name"]
PERSIST = session_params["persist"]


llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-0613")
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Tools >>>

# search
# search = SerpAPIWrapper()

# # math
# llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)

# database
# db = SQLDatabase.from_uri("sqlite:///../../../../../notebooks/Chinook.db")
# db_chain = SQLDatabaseChain.from_llm(llm, db, verbose=True)

# # document alone

# loader = TextLoader(DOC_PATH)
# loader = PyPDFLoader(DOC_PATH)
# documents = loader.load()
# text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
# texts = text_splitter.split_documents(documents)

# embeddings = OpenAIEmbeddings()
# docsearch = Chroma.from_documents(texts, embeddings, collection_name=COLLECTION_NAME)

# # agent_kwargs = {
# #     "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
# # }

# document with persistent index
if PERSIST and os.path.exists("persist"):
    print("Reusing index...\n")
    vectorstore = Chroma(persist_directory="persist", embedding_function=OpenAIEmbeddings())
    index = VectorStoreIndexWrapper(vectorstore=vectorstore)
else:
    #loader = TextLoader("data/data.txt") # Use this line if you only need data.txt
    # loader = DirectoryLoader("data/")
    loader = PyPDFLoader(DOC_PATH)
    if PERSIST:
        index = VectorstoreIndexCreator(vectorstore_kwargs={"persist_directory":"persist"}).from_loaders([loader])
    else:
        index = VectorstoreIndexCreator().from_loaders([loader])

doc_agent = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff", 
    # retriever=docsearch.as_retriever(),
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
    tools,  # tools
    llm, 
    # agent=AgentType.OPENAI_FUNCTIONS, 
    # agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
    # agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION, 
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, 
    verbose=True, 
    # agent_kwargs=agent_kwargs, 
    memory=memory
)

while True:
    input_text = input(">>> ")
    try:
        agent.run(input=input_text)
    except (ValueError, openai.error.InvalidRequestError) as e:
        print(e)