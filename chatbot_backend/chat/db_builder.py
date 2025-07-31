# chat/db_builder.py - 참고 모델 통합된 개선 버전
import os
import json
import unicodedata
from datetime import datetime
from typing import List, Dict, Any
import torch
from tqdm import tqdm
from dotenv import load_dotenv
from django.conf import settings

# LangChain 관련 import
try:
    from langchain_community.document_loaders import JSONLoader
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain.schema import Document
    from langchain.retrievers import EnsembleRetriever
    from langchain_community.retrievers import BM25Retriever
    from langchain_openai import ChatOpenAI
    LANGCHAIN_AVAILABLE = True
except ImportError:
    print("⚠️ LangChain 라이브러리 미설치 - RAG 기능 비활성화")
    LANGCHAIN_AVAILABLE = False

# Transformers 관련 import (참고 모델에서 가져옴)
try:
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        pipeline,
        BitsAndBytesConfig
    )
    from accelerate import Accelerator
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("⚠️ Transformers 라이브러리 미설치")
    TRANSFORMERS_AVAILABLE = False

# 환경 변수 설정 (참고 모델 기반)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# .env 파일을 찾아 환경변수로 로드

load_dotenv()  
# 필수 키 확인
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("환경변수 GROQ_API_KEY가 설정되지 않았습니다.")
# 설정값 (참고 모델에서 가져옴)
CHUNK_SIZE = 256
CHUNK_OVERLAP = 128
EMBEDDING_MODEL = "intfloat/e5-large"
K = 3
LLM = "gemma2-9b-it"  # 기본 모델
QUANTIZATION = "bf16"  # "qlora", "bf16", "fp16"
MAX_NEW_TOKENS = 512

PROMPT_TEMPLATE = """
You are an AI visual assistant that can analyze video content and provide detailed insights.

Using the provided video analysis information, answer the user's question accurately.
Be careful not to answer with false information.

Context from video analysis:
{context}

Question: {question}

Answer:
"""

class EnhancedVideoRAGSystem:
    """개선된 비디오 분석 결과를 활용한 RAG 시스템"""
    
    def __init__(self, use_gpu=False):
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = "cuda" if self.use_gpu else "cpu"
        
        # 초기화 상태 추적
        self._embeddings_initialized = False
        self._llm_initialized = False
        
        print(f"🚀 Enhanced VideoRAG 시스템 초기화 (디바이스: {self.device})")
        
        # LangChain 사용 가능 여부 확인
        if not LANGCHAIN_AVAILABLE:
            print("⚠️ LangChain 미설치 - 기본 RAG 기능만 사용")
            return
        
        try:
            # 임베딩 모델 초기화
            self._init_embeddings()
            # LLM 초기화
            self._init_llm()
        except Exception as e:
            print(f"⚠️ RAG 시스템 초기화 부분 실패: {e}")
        
        self.video_databases = {}
        print("✅ Enhanced VideoRAG 시스템 초기화 완료")
    
    def _init_embeddings(self):
        """임베딩 모델 초기화"""
        try:
            model_kwargs = {"device": self.device}
            encode_kwargs = {'normalize_embeddings': True}
            
            self.embeddings = HuggingFaceEmbeddings(
                model_name=EMBEDDING_MODEL,
                model_kwargs=model_kwargs,
                encode_kwargs=encode_kwargs
            )
            self._embeddings_initialized = True
            print(f"✅ 임베딩 모델 로드 완료: {EMBEDDING_MODEL}")
        except Exception as e:
            print(f"⚠️ 임베딩 모델 초기화 실패: {e}")
            self._embeddings_initialized = False
    
    def _init_llm(self):
        """LLM 초기화 (참고 모델 방식 적용)"""
        try:
            self.llm = ChatOpenAI(
                model=LLM,
                openai_api_key=os.environ["GROQ_API_KEY"],
                openai_api_base="https://api.groq.com/openai/v1",
                temperature=0.2,
                max_tokens=MAX_NEW_TOKENS
            )
            self._llm_initialized = True
            print(f"✅ LLM 초기화 완료: {LLM}")
        except Exception as e:
            print(f"⚠️ LLM 초기화 실패: {e}")
            self._llm_initialized = False
    
    def process_json(self, file_path, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
        """JSON 파일 처리 (참고 모델 기반)"""
        try:
            loader = JSONLoader(
                file_path=file_path,
                jq_schema=".frame_results.[].caption",  # 수정된 스키마
                text_content=False,
            )
            docs = loader.load()
            chunks = docs.copy()
            return chunks
        except Exception as e:
            print(f"⚠️ JSON 처리 실패: {e}")
            return []
    
    def process_total_json(self, file_path, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
        """전체 JSON을 하나의 문서로 처리 (참고 모델 기반)"""
        try:
            # 직접 JSON 파일 읽어서 처리
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 프레임 결과에서 캡션 추출
            video_doc = ""
            frame_results = data.get('frame_results', [])
            
            for i, frame in enumerate(frame_results):
                caption = frame.get('caption', '') or frame.get('final_caption', '') or frame.get('enhanced_caption', '')
                if caption:
                    video_doc += f"Frame {i}: {caption}\n"
                
                # 객체 정보 추가
                objects = frame.get('objects', [])
                if objects:
                    object_names = [obj.get('class', '') for obj in objects]
                    video_doc += f"Objects in frame {i}: {', '.join(object_names)}\n"
            
            # 메타데이터 설정
            meta_data = {
                'source': file_path,
                'video_id': data.get('metadata', {}).get('video_id', 'unknown'),
                'analysis_type': data.get('metadata', {}).get('analysis_type', 'unknown'),
                'total_frames': len(frame_results)
            }
            
            chunks = [Document(page_content=video_doc, metadata=meta_data)]
            return chunks
            
        except Exception as e:
            print(f"⚠️ 전체 JSON 처리 실패: {e}")
            return []
    
    def create_vector_db(self, chunks, model_path=EMBEDDING_MODEL):
        """FAISS 벡터 DB 생성 (참고 모델 기반)"""
        if not self._embeddings_initialized or not chunks:
            print("⚠️ 임베딩 모델이 초기화되지 않았거나 문서가 없음")
            return None
        
        try:
            db = FAISS.from_documents(chunks, embedding=self.embeddings)
            print(f"✅ 벡터 DB 생성 완료: {len(chunks)}개 문서")
            return db
        except Exception as e:
            print(f"⚠️ 벡터 DB 생성 실패: {e}")
            return None
    
    def process_video_analysis_json(self, json_file_path: str, video_id: str):
        """비디오 분석 JSON을 처리하여 벡터 DB 생성 (개선됨)"""
        try:
            if not os.path.exists(json_file_path):
                print(f"⚠️ JSON 파일을 찾을 수 없음: {json_file_path}")
                return False
            
            print(f"📄 JSON 분석 파일 처리 중: {json_file_path}")
            
            with open(json_file_path, 'r', encoding='utf-8') as f:
                analysis_data = json.load(f)
            
            documents = []
            
            # 프레임별 분석 결과를 문서로 변환
            frame_results = analysis_data.get('frame_results', [])
            video_metadata = analysis_data.get('metadata', {})
            
            for frame_result in frame_results:
                frame_id = frame_result.get('image_id', 0)
                timestamp = frame_result.get('timestamp', 0)
                
                # 다양한 캡션 소스 시도
                caption = (frame_result.get('final_caption') or 
                          frame_result.get('enhanced_caption') or 
                          frame_result.get('caption') or '')
                
                objects = frame_result.get('objects', [])
                scene_analysis = frame_result.get('scene_analysis', {})
                
                # 문서 내용 구성
                content_parts = []
                
                if caption:
                    content_parts.append(f"Frame {frame_id} at {timestamp:.1f}s: {caption}")
                
                if objects:
                    object_list = [obj.get('class', '') for obj in objects if obj.get('class')]
                    if object_list:
                        content_parts.append(f"Objects detected: {', '.join(object_list)}")
                
                # Scene 분석 정보 추가
                if scene_analysis:
                    scene_class = scene_analysis.get('scene_classification', {})
                    if scene_class:
                        location = scene_class.get('location', {}).get('label', '')
                        time_of_day = scene_class.get('time', {}).get('label', '')
                        if location or time_of_day:
                            content_parts.append(f"Scene: {location} {time_of_day}".strip())
                    
                    # OCR 텍스트 추가
                    ocr_text = scene_analysis.get('ocr_text', '')
                    if ocr_text:
                        content_parts.append(f"Text found: {ocr_text}")
                
                # 모든 내용 결합
                content = '. '.join(content_parts)
                
                if content:  # 빈 내용이 아닌 경우만 추가
                    metadata = {
                        'video_id': video_id,
                        'frame_id': frame_id,
                        'timestamp': timestamp,
                        'objects': [obj.get('class', '') for obj in objects],
                        'analysis_type': video_metadata.get('analysis_type', 'unknown')
                    }
                    
                    documents.append(Document(page_content=content, metadata=metadata))
            
            # 벡터 DB 생성
            if documents:
                db = self.create_vector_db(documents)
                if not db:
                    return False
                
                # Retriever 설정 (참고 모델 방식)
                retriever_similarity = db.as_retriever(
                    search_type="similarity",
                    search_kwargs={'k': K}
                )
                
                retriever_mmr = db.as_retriever(
                    search_type="mmr",
                    search_kwargs={'k': K}
                )
                
                try:
                    retriever_bm25 = BM25Retriever.from_documents(documents)
                    retriever_bm25.k = K
                    
                    ensemble_retriever = EnsembleRetriever(
                        retrievers=[retriever_similarity, retriever_mmr, retriever_bm25],
                        weights=[0.5, 0.3, 0.2]
                    )
                except Exception as e:
                    print(f"⚠️ BM25 retriever 생성 실패, similarity만 사용: {e}")
                    ensemble_retriever = retriever_similarity
                
                self.video_databases[video_id] = {
                    'db': db,
                    'retriever': ensemble_retriever,
                    'documents': documents,
                    'created_at': datetime.now(),
                    'json_path': json_file_path
                }
                
                print(f"✅ 비디오 {video_id} RAG DB 생성 완료: {len(documents)}개 문서")
                return True
            else:
                print(f"⚠️ 처리할 문서가 없음: {json_file_path}")
                return False
            
        except Exception as e:
            print(f"❌ RAG DB 생성 실패: {e}")
            import traceback
            print(f"상세 오류: {traceback.format_exc()}")
            return False
    
    def search_video_content(self, video_id: str, query: str, top_k: int = 5):
        """비디오 내용 검색 (개선됨)"""
        if video_id not in self.video_databases:
            print(f"⚠️ 비디오 {video_id}의 RAG DB가 없음")
            return []
        
        try:
            retriever = self.video_databases[video_id]['retriever']
            documents = retriever.get_relevant_documents(query)
            
            results = []
            for doc in documents[:top_k]:
                results.append({
                    'content': doc.page_content,
                    'metadata': doc.metadata,
                    'frame_id': doc.metadata.get('frame_id'),
                    'timestamp': doc.metadata.get('timestamp'),
                    'objects': doc.metadata.get('objects', [])
                })
            
            print(f"🔍 검색 완료: {len(results)}개 결과")
            return results
            
        except Exception as e:
            print(f"❌ 비디오 검색 실패: {e}")
            return []
    
    def answer_question(self, video_id: str, question: str):
        """질문에 대한 답변 생성 (개선됨)"""
        if not self._llm_initialized:
            return "LLM이 초기화되지 않아 답변을 생성할 수 없습니다."
        
        # 관련 문서 검색
        search_results = self.search_video_content(video_id, question)
        
        if not search_results:
            return "관련된 비디오 내용을 찾을 수 없습니다."
        
        # 컨텍스트 구성 (참고 모델 방식)
        context = self.format_docs_for_prompt(search_results)
        
        # LLM에 질문
        try:
            prompt = PROMPT_TEMPLATE.format(context=context, question=question)
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            print(f"❌ LLM 답변 생성 실패: {e}")
            return "답변 생성 중 오류가 발생했습니다."
    
    def format_docs_for_prompt(self, search_results):
        """검색 결과를 프롬프트용으로 포맷 (참고 모델 방식)"""
        context = ""
        for i, result in enumerate(search_results):
            frame_id = result['metadata'].get('frame_id', i)
            context += f"Frame {frame_id}: {result['content']}\n"
        return context
    
    def get_database_info(self, video_id: str = None):
        """데이터베이스 정보 조회"""
        if video_id:
            if video_id in self.video_databases:
                db_info = self.video_databases[video_id]
                return {
                    'video_id': video_id,
                    'document_count': len(db_info['documents']),
                    'created_at': db_info['created_at'].isoformat(),
                    'json_path': db_info.get('json_path', 'unknown')
                }
            else:
                return None
        else:
            return {
                'total_videos': len(self.video_databases),
                'videos': list(self.video_databases.keys()),
                'embeddings_initialized': self._embeddings_initialized,
                'llm_initialized': self._llm_initialized
            }

# 전역 RAG 시스템 인스턴스 (개선됨)
_global_rag_system = None

def get_video_rag_system():
    """전역 RAG 시스템 인스턴스 반환 (싱글톤 패턴)"""
    global _global_rag_system
    if _global_rag_system is None:
        _global_rag_system = EnhancedVideoRAGSystem()
    return _global_rag_system

# 호환성을 위해 기존 변수명 유지
rag_system = get_video_rag_system()