FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
  build-essential \
  curl \
  software-properties-common \
  git

# RUN git clone https://github.com/streamlit/streamlit-example.git .

# RUN pip3 install -r requirements.ini
# Or
RUN pip3 install \
  langchain \
  openai \
  chromadb \
  tiktoken \
  unstructured \
  pypdf \
  faiss-cpu \
  docx2txt \
  streamlit \
  duckduckgo-search \
  pymupdf

WORKDIR /app/

RUN rm -rf /var/lib/apt/lists/*

# clone repo, or
COPY . /app/

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

ENTRYPOINT ["streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]