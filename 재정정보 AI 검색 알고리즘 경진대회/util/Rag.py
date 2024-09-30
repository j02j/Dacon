import os
import unicodedata
import torch
import pandas as pd
from tqdm import tqdm
import fitz  # PyMuPDF

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    pipeline,
)

from accelerate import Accelerator

# Langchain 관련
from langchain.llms import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

def process_pdf(file_path, chunk_size=800, chunk_overlap=50):
    doc = fitz.open(file_path)
    text = ' '.join([page.get_text() for page in doc])
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_text(text)
    return [Document(page_content=t) for t in chunks]

def create_vector_db(chunks, model_path="sentence-transformers/all-MiniLM-L6-v2"):
    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': True}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    db = FAISS.from_documents(chunks, embedding=embeddings)
    return db

def normalize_path(path):
    return unicodedata.normalize('NFC', path)

def process_pdfs_from_dataframe(df, base_directory):
    pdf_databases = {}
    unique_paths = df['Source_path'].unique()

    for path in tqdm(unique_paths, desc="Processing PDFs"):
        normalized_path = normalize_path(path)
        full_path = os.path.normpath(os.path.join(base_directory, normalized_path.lstrip('./')))
        pdf_title = os.path.splitext(os.path.basename(full_path))[0]
        print(f"Processing {pdf_title}...")

        chunks = process_pdf(full_path)
        db = create_vector_db(chunks)
        retriever = db.as_retriever(search_type="mmr", search_kwargs={'k': 3, 'fetch_k': 8})
        pdf_databases[pdf_title] = {'db': db, 'retriever': retriever}

    return pdf_databases