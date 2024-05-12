import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.retrievers import EnsembleRetriever
from sentence_transformers import CrossEncoder
from langchain import hub
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
import numpy as np
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import pandas as pd

load_dotenv()


llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")

model = CrossEncoder("models/cross-encoder-original")
prompt = hub.pull("rlm/rag-prompt")
llm_chain = prompt | llm | StrOutputParser()

st.title("RAG with Human Feedback Project")

# File upload
st.sidebar.title("Upload PDF Files")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF files", type="pdf", accept_multiple_files=True
)

# Input query
query = st.text_input("Enter your query:")


def process_pdf_batch(pdf_files):
    batch_docs = []
    for pdf_file_path in pdf_files:
        temp_file = pdf_file_path.name
        with open(temp_file, "wb") as file:
            file.write(pdf_file_path.getvalue())
        loader = PyPDFLoader(temp_file)
        pages = loader.load_and_split()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=300,
            chunk_overlap=50,
            is_separator_regex=False,
        )
        pages = text_splitter.split_documents(pages)
        os.remove(temp_file)
        batch_docs.extend(pages)
    return batch_docs


def get_retrieved_documents(loader, query):
    with st.spinner("Multiple pdf are being processed. Please wait....."):
        loader = process_pdf_batch(uploaded_files)
    with st.spinner("Ensemble Retrieval is being creating. Please wait....."):
        Bm25_retriever = BM25Retriever.from_documents(loader)
        Bm25_retriever.k = 5
        Faiss_retriever = FAISS.from_documents(loader, OpenAIEmbeddings())
        Faiss_retriever = Faiss_retriever.as_retriever(search_kwargs={"k": 5})
        ensemble_retriever = EnsembleRetriever(
            retrievers=[Bm25_retriever, Faiss_retriever], weights=[0.5, 0.5]
        )
        retrievered = ensemble_retriever.invoke(query)
    with st.spinner("Crossencoder reranking is being proceesing. Please wait....."):
        corpus = [[query, doc.page_content] for doc in retrievered]
        reranker = model.predict(corpus)
        data = pd.DataFrame(
            [[query, doc.page_content, 0] for doc in retrievered],
            columns=["question", "context", "score"],
        )
        data["reranker"] = reranker
        data.sort_values(by="reranker", ascending=False, inplace=True)
        directory = "testing"
        if not os.path.exists(directory):
            os.makedirs(directory)
        data.to_csv(f".\{directory}\{query}_feedback_.csv", index=False)
    return data



# document=[]
if st.button("Search"):
    st.session_state["searched"] = True
    if uploaded_files:
        st.session_state["uploaded_files"] = True

        data = get_retrieved_documents(uploaded_files, query)
        data["answer"] = ""
        for idx in range(len(data)):
            data.iloc[idx, 4] = llm_chain.invoke(
                    {"context": data.iloc[idx, 1], "question": query}
                )

        directory = "training"
        if not os.path.exists(directory):
            os.makedirs(directory)
        data.to_csv(f".\{directory}\{query}_feedback_.csv", index=False)
        # st.table(data)
        st.success(f"created a feedback at .\{directory}\{query}_feedback_.csv.    Done!...")
