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


llm = ChatOpenAI(temperature=0,model_name="gpt-3.5-turbo")

model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
prompt = hub.pull("rlm/rag-prompt")
llm_chain= ( prompt
    | llm
    | StrOutputParser()
)

st.title("RAG with Human Feedback Project")

# File upload
st.sidebar.title("Upload PDF Files")
uploaded_files = st.sidebar.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

# Input query
query = st.text_input("Enter your query:")


def process_pdf_batch(pdf_files):
    batch_docs = []
    for pdf_file_path in pdf_files:
        temp_file = pdf_file_path.name
        with open(temp_file, "wb") as file:
            file.write(pdf_file_path.getvalue())
            # file_name = pdf_file_path.name
        loader = PyPDFLoader(temp_file)
        pages = loader.load_and_split()
        text_splitter = RecursiveCharacterTextSplitter(
                                    # Set a really small chunk size, just to show.
                                    chunk_size=300,
                                    chunk_overlap=50,
                                    is_separator_regex=False,
                                    )
        pages = text_splitter.split_documents(pages)
        os.remove(temp_file)
        batch_docs.extend(pages)
    return batch_docs
document=[]
if st.button("Search"):
    if uploaded_files:
        with st.spinner("Multiple pdf are being processed. Please wait....."):
            loader = process_pdf_batch(uploaded_files)
        with st.spinner("Ensemble Retrieval is being creating. Please wait....."):
            Bm25_retriever=BM25Retriever.from_documents(loader)
            Bm25_retriever.k=5
            Faiss_retriever = FAISS.from_documents(loader, OpenAIEmbeddings())
            Faiss_retriever=Faiss_retriever.as_retriever(search_kwargs={"k": 5})
            ensemble_retriever = EnsembleRetriever(retrievers=[Bm25_retriever, Faiss_retriever], 
                                                weights=[0.5, 0.5])
            retrievered=ensemble_retriever.invoke(query)
        with st.spinner("Crossencoder reranking is being proceesing. Please wait....."):
            corpus = [[query, doc.page_content] for doc in retrievered]
            reranker=model.predict(corpus)
            # Sort the scores in decreasing order to get the corpus indices
            ranked_indices = np.argsort(reranker)[::-1]
            # for idx in ranked_indices:
            #     st.write(f"{reranker[idx]:.2f}\t{corpus[idx][1]}")
        
        
        for idx in ranked_indices:
            ans=llm_chain.invoke({"context":corpus[idx][1], "question":query})
            text=corpus[idx][1]
            st.write(f"Document :{idx+1}\t{corpus[idx][1]}")
            genre = st.radio(f"Answer : \n {ans}",[":thumbsup:",":thumbdown:"],horizontal=True)
            if genre == ":thumbsup:":
                score=1
                document.append({"Text":text,"Query":query,"Score":score})
            else:
                score=0
                document.append({"Text":text,"Query":query,"Score":score})
        


        
        st.write(prompt)
