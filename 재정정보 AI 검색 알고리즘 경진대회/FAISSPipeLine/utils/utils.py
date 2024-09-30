import os
import unicodedata
import warnings
warnings.filterwarnings('ignore')

import torch
import pandas as pd
from tqdm.autonotebook import tqdm
import fitz  # PyMuPDF


# Langchain 관련
from langchain.llms import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.vectorstores import Chroma
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores.utils import DistanceStrategy

import pickle

def normalize_path(path):
    """경로 유니코드 정규화"""
    return unicodedata.normalize('NFC', path)

################################## 1. PDF 처리 방법 실험하기 ##########################################################
def process_pdf(file_path, chunk_size=800, chunk_overlap=50):
    """PDF 텍스트 추출 후 chunk 단위로 나누기"""
    # PDF 파일 열기
    doc = fitz.open(file_path)
    text = ''
    # 모든 페이지의 텍스트 추출
    for page in doc:
        text += page.get_text()
    # 텍스트를 chunk로 분할
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunk_temp = splitter.split_text(text)
    # Document 객체 리스트 생성
    chunks = [Document(page_content=t) for t in chunk_temp]
    return chunks


################################## 2. 임베딩모델 바꿔보기 ##########################################################
def get_embedding(model_path="intfloat/multilingual-e5-small"):
    # 임베딩 모델 설정
    model_kwargs = {'device': 'cuda'}
    encode_kwargs = {'normalize_embeddings': True}
    embeddings = HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs
    )
    
    return embeddings

################################## 3. Chroma로 할건지, FAISS로 할건지 등 ##########################################################
def create_chroma_db(pdf_title, chunks, model_path="intfloat/multilingual-e5-small"):
    """Chroma DB 생성"""
    # 임베딩 모델 설정
    embeddings = get_embedding(model_path=model_path)
    # Chroma DB 생성 및 반환 persist_directory=DB_PATH
    Chroma.from_documents(chunks, embedding=embeddings, persist_directory=pdf_title)

    
################################## 3. Chroma로 할건지, FAISS로 할건지 등 ##########################################################
def create_faiss_db(save_file, chunks, model_path="intfloat/multilingual-e5-small"):
    """Chroma DB 생성"""
    # 임베딩 모델 설정
    embeddings = get_embedding(model_path=model_path)
    # Chroma DB 생성 및 반환 persist_directory=DB_PATH
    db = FAISS.from_documents(chunks, embedding=embeddings, distance_strategy = DistanceStrategy.COSINE)
    db.save_local(save_file)

################################## Dense and Sparse dataset ##########################################################
def process_pdfs_from_dataframe(df, base_directory, save_path):
    """딕셔너리에 pdf명을 키로해서 DB, retriever 저장"""

    unique_paths = df['Source_path'].unique()
    print('가지고 있는 pdf파일들:', unique_paths)
    
    for path in tqdm(unique_paths, desc="Processing PDFs"):
        # 경로 정규화 및 절대 경로 생성
        normalized_path = normalize_path(path)
        full_path = os.path.normpath(os.path.join(base_directory, normalized_path.lstrip('./'))) if not os.path.isabs(normalized_path) else normalized_path
        
        pdf_title = os.path.splitext(os.path.basename(full_path))[0]
        print(f"Processing {pdf_title}...")
        
        # PDF 처리 및 벡터 DB 생성
        chunks = process_pdf(full_path)
        save_file = f'{save_path}/{pdf_title}'
        create_chroma_db(save_file, chunks, model_path="intfloat/multilingual-e5-small")
        with open(f'{save_file}.pkl', 'wb') as f:
            pickle.dump(chunks, f)
            
            
################################## FAISS로 만들기 Dense and Sparse dataset ##########################################################
def process_pdfs_from_dataframe_faiss(df, base_directory, save_path, file_mapping):
    """딕셔너리에 pdf명을 키로해서 DB, retriever 저장"""

    unique_paths = df['Source_path'].unique()
    print('가지고 있는 pdf파일들:', unique_paths)
    
    for path in tqdm(unique_paths, desc="Processing PDFs"):
        # 경로 정규화 및 절대 경로 생성
        normalized_path = normalize_path(path)
        full_path = os.path.normpath(os.path.join(base_directory, normalized_path.lstrip('./'))) if not os.path.isabs(normalized_path) else normalized_path
        
        pdf_title = os.path.splitext(os.path.basename(full_path))[0]
        print(f"Processing {pdf_title}...")
        
        # PDF 처리 및 벡터 DB 생성
        chunks = process_pdf(full_path)
        new_title = file_mapping[pdf_title]
        save_file = f'{save_path}/{new_title}'
        create_faiss_db(save_file, chunks, model_path="intfloat/multilingual-e5-small")
        with open(f'{save_file}.pkl', 'wb') as f:
            pickle.dump(chunks, f)