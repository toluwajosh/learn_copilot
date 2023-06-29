
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
# from langchain.document_loaders import TextLoader
# from langchain.document_loaders import JSONLoader

load_dotenv()


loader = PyPDFLoader("data/english_language_syllabus.pdf")
pages = loader.load_and_split()


faiss_index = FAISS.from_documents(pages, OpenAIEmbeddings())
docs = faiss_index.similarity_search("AIMS AND OBJECTIVES", k=2)
for doc in docs:
    print(str(doc.metadata["page"]) + ":", doc.page_content[:300])