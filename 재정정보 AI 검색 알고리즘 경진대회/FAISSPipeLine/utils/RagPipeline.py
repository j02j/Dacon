
from langchain_community.llms import Ollama
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate

from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain_community.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

import pickle

from utils.prompt import template
from utils.utils import get_embedding

def format_docs(docs):
    """검색된 문서들을 하나의 문자열로 포맷팅"""
    context = ""
    for doc in docs:
        context += doc.page_content
        context += '\n'
    return context


class Ragpipeline:
    def __init__(self, source):
        # 1. LLM 바꿔보면서 실험하기 
        self.llm = ChatOllama(model="llama3.1", temperature=0.1)
        # self.llm = ChatOllama(model="llama3-ko-instruct", temperature=0)
        
        self.base_retriever = self.init_retriever(source)
        self.ensemble_retriever = self.init_ensemble_retriever(source)
        self.chain = self.init_chain()
        
        
    def init_retriever(self, source):
        # vector_store = Chroma(persist_directory=source, embedding_function=get_embedding())
        embeddings_model = get_embedding()
        vector_store = FAISS.load_local(source, embeddings_model, allow_dangerous_deserialization=True)
        retriever = vector_store.as_retriever(
                search_type="mmr",                                              # mmr 검색 방법으로 
                search_kwargs={'fetch_k': 5, "k": 3, 'lambda_mult': 0.4},      # 상위 10개의 관련 context에서 최종 5개를 추리고 'lambda_mult'는 관련성과 다양성 사이의 균형을 조정하는 파라메타 default 값이 0.5
                
            )
        
        return retriever
    
    def init_ensemble_retriever(self, source):
        retriever = self.base_retriever
        
        all_docs = pickle.load(open(f'{source}.pkl', 'rb'))
        bm25_retriever = BM25Retriever.from_documents(all_docs)
        bm25_retriever.k = 1   
        
        ensemble_retriever = EnsembleRetriever(
                retrievers=[bm25_retriever, retriever],
                weights=[0.4, 0.6],
                search_type='mmr'
            )
        
        return ensemble_retriever
        
    def init_chain(self):
        prompt = PromptTemplate.from_template(template)
        # 2. Chroma 할지, FAISS 할지, 앙상블 리트리버를 할지, 리트리버의 하이퍼파라메타 바꿔보면서 하기 
        retriever = self.base_retriever       # get_retriever()
        
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        return rag_chain
        
    
    def answer_generation(self, question: str) -> dict:
        full_response = self.chain.invoke(question)
        return full_response
