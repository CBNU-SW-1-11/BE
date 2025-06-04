# import logging
# import re
# import math
# import numpy as np
# from collections import Counter
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from sentence_transformers import SentenceTransformer

# logger = logging.getLogger(__name__)

# class SimilarityAnalyzer:
#     """
#     AI 모델 응답 간의 유사도를 분석하고 응답 특성을 추출하는 클래스
#     """
    
#     def __init__(self, threshold=0.05):  # 임계값을 0.01로 설정
#         """
#         초기화
        
#         Args:
#             threshold (float): 유사 응답으로 분류할 임계값 (0~1)
#                               여기서는 유사도가 0.01 이상이면 동일한 그룹으로 분류합니다.
#         """
#         self.threshold = threshold
#         self.vectorizer = TfidfVectorizer(
#             min_df=1, 
#             analyzer='word',
#             ngram_range=(1, 2),
#             stop_words='english'
#         )
        
#         # Sentence-BERT 모델 로드 - 다국어 지원 모델 사용
#         try:
#             self.sentence_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
#             self.use_semantic = True
#             logger.info("Sentence-BERT model loaded successfully")
#         except Exception as e:
#             logger.error(f"Error loading Sentence-BERT model: {str(e)}")
#             self.use_semantic = False
    
#     def preprocess_text(self, text):
#         """
#         텍스트 전처리
        
#         Args:
#             text (str): 전처리할 텍스트
            
#         Returns:
#             str: 전처리된 텍스트
#         """
#         # 소문자 변환
#         text = text.lower()
        
#         # 코드 블록 제거 (분석에서 제외)
#         text = re.sub(r'```.*?```', ' CODE_BLOCK ', text, flags=re.DOTALL)
        
#         # HTML 태그 제거
#         text = re.sub(r'<.*?>', '', text)
        
#         # 여러 공백을 하나로 치환
#         text = re.sub(r'\s+', ' ', text).strip()
        
#         return text
    
#     def calculate_similarity_matrix(self, responses):
#         """
#         BERTScore 기반으로 모델 응답 간의 유사도 행렬 계산 (F1 점수를 유사도로 사용)
#         보정 계수를 적용하여 의미적 유사도가 좀더 높게 측정되도록 함.
        
#         Args:
#             responses (dict): 모델 ID를 키로, 응답 텍스트를 값으로 하는 딕셔너리
#         Returns:
#             dict: 모델 간 유사도 행렬
#         """
#         try:
#             import bert_score
#             model_ids = list(responses.keys())
#             # 전처리된 텍스트 목록 생성
#             texts = [self.preprocess_text(responses[mid]) for mid in model_ids]
#             n = len(texts)
#             similarity_matrix = np.zeros((n, n))
            
#             for i in range(n):
#                 for j in range(n):
#                     if i == j:
#                         similarity_matrix[i][j] = 1.0
#                     else:
#                         # BERTScore는 (P, R, F1)를 반환합니다.
#                         _, _, F1 = bert_score.score([texts[i]], [texts[j]], lang="ko", verbose=False)
#                         # F1 점수에 2.0을 곱해 보정한 후, 최대값을 1.0으로 제한
#                         boosted_score = min(F1.item() * 2.0, 1.0)
#                         similarity_matrix[i][j] = boosted_score
            
#             # 결과를 딕셔너리 형태로 변환
#             result = {}
#             for i, model1 in enumerate(model_ids):
#                 result[model1] = {}
#                 for j, model2 in enumerate(model_ids):
#                     result[model1][model2] = float(similarity_matrix[i][j])
#             return result

#         except Exception as e:
#             logger.error(f"BERTScore 기반 유사도 계산 중 오류: {str(e)}")
#             return {mid: {other: 0.0 for other in responses} for mid in responses}

#     def cluster_responses(self, responses):
#         """
#         응답을 유사도에 따라 군집화
        
#         Args:
#             responses (dict): 모델 ID를 키로, 응답 텍스트를 값으로 하는 딕셔너리
            
#         Returns:
#             dict: 군집화 결과
#         """
#         try:
#             model_ids = list(responses.keys())
#             if len(model_ids) <= 1:
#                 return {
#                     "similarGroups": [model_ids],
#                     "outliers": [],
#                     "similarityMatrix": {}
#                 }
            
#             # 유사도 행렬 계산 (BERTScore 기반)
#             similarity_matrix = self.calculate_similarity_matrix(responses)
            
#             # 계층적 클러스터링 수행 (유사도가 0.01 이상이면 병합)
#             clusters = [[model_id] for model_id in model_ids]
            
#             merge_happened = True
#             while merge_happened and len(clusters) > 1:
#                 merge_happened = False
#                 max_similarity = -1
#                 merge_indices = [-1, -1]
                
#                 # 가장 유사한 두 클러스터 찾기
#                 for i in range(len(clusters)):
#                     for j in range(i + 1, len(clusters)):
#                         # 두 클러스터 간 평균 유사도 계산
#                         cluster_similarity = 0
#                         pair_count = 0
                        
#                         for model1 in clusters[i]:
#                             for model2 in clusters[j]:
#                                 cluster_similarity += similarity_matrix[model1][model2]
#                                 pair_count += 1
                        
#                         avg_similarity = cluster_similarity / max(1, pair_count)
                        
#                         if avg_similarity > max_similarity:
#                             max_similarity = avg_similarity
#                             merge_indices = [i, j]
                
#                 # 임계값(0.01)보다 유사도가 높으면 클러스터 병합
#                 if max_similarity >= self.threshold:
#                     i, j = merge_indices
#                     clusters[i].extend(clusters[j])
#                     clusters.pop(j)
#                     merge_happened = True
            
#             # 클러스터 크기에 따라 정렬
#             clusters.sort(key=lambda x: -len(x))
            
#             # 주요 그룹과 이상치 구분
#             main_group = clusters[0] if clusters else []
#             outliers = [model for cluster in clusters[1:] for model in cluster]
            
#             # 의미 유사도 태그 추가
#             semantic_tags = {}
#             if len(model_ids) >= 2:
#                 for i, cluster in enumerate(clusters):
#                     if i == 0:
#                         tag = "주요 의미 그룹"
#                     elif i == 1:
#                         tag = "보조 의미 그룹"
#                     else:
#                         tag = f"의미 그룹 {i+1}"
                    
#                     for model in cluster:
#                         semantic_tags[model] = tag
            
#             # 응답 특성 추출
#             response_features = {model_id: self.extract_response_features(responses[model_id]) 
#                                 for model_id in model_ids}
            
#             return {
#                 "similarGroups": clusters,
#                 "mainGroup": main_group,
#                 "outliers": outliers,
#                 "similarityMatrix": similarity_matrix,
#                 "responseFeatures": response_features,
#                 "semanticTags": semantic_tags
#             }
            
#         except Exception as e:
#             logger.error(f"응답 군집화 중 오류: {str(e)}")
#             return {
#                 "similarGroups": [model_ids],
#                 "mainGroup": model_ids,
#                 "outliers": [],
#                 "similarityMatrix": {},
#                 "responseFeatures": {}
#             }
    
#     def extract_response_features(self, text):
#         """
#         응답 텍스트에서 특성 추출
        
#         Args:
#             text (str): 응답 텍스트
            
#         Returns:
#             dict: 응답 특성 정보
#         """
#         try:
#             # 응답 길이
#             length = len(text)
            
#             # 코드 블록 개수
#             code_blocks = re.findall(r'```[\s\S]*?```', text)
#             code_block_count = len(code_blocks)
            
#             # 링크 개수
#             links = re.findall(r'\[.*?\]\(.*?\)', text) or re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
#             link_count = len(links)
            
#             # 목록 항목 개수
#             list_items = re.findall(r'^[\s]*[-*+] |^[\s]*\d+\. ', text, re.MULTILINE)
#             list_item_count = len(list_items)
            
#             # 문장 분리
#             sentences = re.split(r'[.!?]+', text)
#             sentences = [s.strip() for s in sentences if s.strip()]
            
#             # 평균 문장 길이
#             avg_sentence_length = sum(len(s) for s in sentences) / max(1, len(sentences))
            
#             # 어휘 다양성 (고유 단어 비율)
#             words = re.findall(r'\b\w+\b', text.lower())
#             unique_words = set(words)
#             vocabulary_diversity = len(unique_words) / max(1, len(words))
            
#             # 자주 사용된 단어들 (stopwords 제외)
#             word_counter = Counter(words)
#             common_words = [word for word, count in word_counter.most_common(5)]
            
#             return {
#                 "length": length,
#                 "codeBlockCount": code_block_count,
#                 "linkCount": link_count,
#                 "listItemCount": list_item_count,
#                 "sentenceCount": len(sentences),
#                 "avgSentenceLength": avg_sentence_length,
#                 "vocabularyDiversity": vocabulary_diversity,
#                 "commonWords": common_words,
#                 "hasCode": code_block_count > 0
#             }
            
#         except Exception as e:
#             logger.error(f"응답 특성 추출 중 오류: {str(e)}")
#             return {
#                 "length": len(text),
#                 "codeBlockCount": 0,
#                 "linkCount": 0,
#                 "listItemCount": 0,
#                 "sentenceCount": 1,
#                 "avgSentenceLength": len(text),
#                 "vocabularyDiversity": 0,
#                 "hasCode": False
#             }

# similarity_analyzer.py
import logging
import re
import math
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

# 선택적 모듈 임포트
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMER_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMER_AVAILABLE = False
    logger.warning("sentence_transformers package not available. Semantic similarity will be disabled.")

try:
    import bert_score
    BERT_SCORE_AVAILABLE = True
except ImportError:
    BERT_SCORE_AVAILABLE = False
    logger.warning("bert_score package not available. BERT-based similarity will be disabled.")

class SimilarityAnalyzer:
    """
    AI 모델 응답 간의 유사도를 분석하고 응답 특성을 추출하는 클래스
    """
    
    def __init__(self, threshold=0.85):
        """
        초기화
        
        Args:
            threshold (float): 유사 응답으로 분류할 임계값 (0~1)
        """
        self.threshold = threshold
        self.vectorizer = TfidfVectorizer(
            min_df=1, 
            analyzer='word',
            ngram_range=(1, 2),
            stop_words='english'
        )
        
        # Sentence-BERT 모델 로드 (가능한 경우)
        self.use_semantic = False
        if SENTENCE_TRANSFORMER_AVAILABLE:
            try:
                self.sentence_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                self.use_semantic = True
                logger.info("Sentence-BERT model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading Sentence-BERT model: {str(e)}")
    
    def preprocess_text(self, text):
        """
        텍스트 전처리
        
        Args:
            text (str): 전처리할 텍스트
            
        Returns:
            str: 전처리된 텍스트
        """
        # 소문자 변환
        text = text.lower()
        
        # 코드 블록 제거 (분석에서 제외)
        text = re.sub(r'```.*?```', ' CODE_BLOCK ', text, flags=re.DOTALL)
        
        # HTML 태그 제거
        text = re.sub(r'<.*?>', '', text)
        
        # 특수 문자 제거 (기 ㅇ본 버전에서만 사용)
        # text = re.sub(r'[^\w\s]', ' ', text)
        
        # 여러 공백을 하나로 치환
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def calculate_similarity_matrix(self, responses):
        """
        모델 응답 간의 유사도 행렬 계산
        BERT 모델이 가능하면 BERTScore 기반으로, 아니면 TF-IDF 기반으로 계산
        
        Args:
            responses (dict): 모델 ID를 키로, 응답 텍스트를 값으로 하는 딕셔너리
            
        Returns:
            dict: 모델 간 유사도 행렬
        """
        # BERT 기반 유사도 계산 시도
        if BERT_SCORE_AVAILABLE:
            try:
                model_ids = list(responses.keys())
                # 전처리된 텍스트 목록 생성
                texts = [self.preprocess_text(responses[mid]) for mid in model_ids]
                n = len(texts)
                similarity_matrix = np.zeros((n, n))
                
                for i in range(n):
                    for j in range(n):
                        if i == j:
                            similarity_matrix[i][j] = 1.0
                        else:
                            # BERTScore는 (P, R, F1)를 반환합니다.
                            _, _, F1 = bert_score.score([texts[i]], [texts[j]], lang="ko", verbose=False)
                            # F1 점수에 2.0을 곱해 보정한 후, 최대값을 1.0으로 제한
                            boosted_score = min(F1.item() * 2.0, 1.0)
                            similarity_matrix[i][j] = boosted_score
                
                # 결과를 딕셔너리 형태로 변환
                result = {}
                for i, model1 in enumerate(model_ids):
                    result[model1] = {}
                    for j, model2 in enumerate(model_ids):
                        result[model1][model2] = float(similarity_matrix[i][j])
                return result

            except Exception as e:
                logger.error(f"BERTScore 기반 유사도 계산 중 오류: {str(e)}")
                # 오류 발생 시 TF-IDF로 폴백
                logger.info("Falling back to TF-IDF similarity")
        
        # TF-IDF 기반 유사도 계산
        try:
            model_ids = list(responses.keys())
            
            # 텍스트 전처리
            preprocessed_texts = [self.preprocess_text(responses[model_id]) for model_id in model_ids]
            
            # TF-IDF 벡터화
            tfidf_matrix = self.vectorizer.fit_transform(preprocessed_texts)
            
            # 코사인 유사도 계산
            similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # 결과를 딕셔너리로 변환
            result = {}
            for i, model1 in enumerate(model_ids):
                result[model1] = {}
                for j, model2 in enumerate(model_ids):
                    result[model1][model2] = float(similarity_matrix[i][j])
            
            return result
            
        except Exception as e:
            logger.error(f"유사도 행렬 계산 중 오류: {str(e)}")
            # 오류 발생 시 빈 행렬 반환
            return {model_id: {other_id: 0.0 for other_id in responses} for model_id in responses}
    
    def cluster_responses(self, responses):
        """
        응답을 유사도에 따라 군집화
        
        Args:
            responses (dict): 모델 ID를 키로, 응답 텍스트를 값으로 하는 딕셔너리
            
        Returns:
            dict: 군집화 결과
        """
        try:
            model_ids = list(responses.keys())
            if len(model_ids) <= 1:
                return {
                    "similarGroups": [model_ids],
                    "outliers": [],
                    "similarityMatrix": {}
                }
            
            # 유사도 행렬 계산
            similarity_matrix = self.calculate_similarity_matrix(responses)
            
            # 계층적 클러스터링 수행
            clusters = [[model_id] for model_id in model_ids]
            
            merge_happened = True
            while merge_happened and len(clusters) > 1:
                merge_happened = False
                max_similarity = -1
                merge_indices = [-1, -1]
                
                # 가장 유사한 두 클러스터 찾기
                for i in range(len(clusters)):
                    for j in range(i + 1, len(clusters)):
                        # 두 클러스터 간 평균 유사도 계산
                        cluster_similarity = 0
                        pair_count = 0
                        
                        for model1 in clusters[i]:
                            for model2 in clusters[j]:
                                cluster_similarity += similarity_matrix[model1][model2]
                                pair_count += 1
                        
                        avg_similarity = cluster_similarity / max(1, pair_count)
                        
                        if avg_similarity > max_similarity:
                            max_similarity = avg_similarity
                            merge_indices = [i, j]
                
                # 임계값보다 유사도가 높으면 클러스터 병합
                if max_similarity >= self.threshold:
                    i, j = merge_indices
                    clusters[i].extend(clusters[j])
                    clusters.pop(j)
                    merge_happened = True
            
            # 클러스터 크기에 따라 정렬
            clusters.sort(key=lambda x: -len(x))
            
            # 주요 그룹과 이상치 구분
            main_group = clusters[0] if clusters else []
            outliers = [model for cluster in clusters[1:] for model in cluster]
            
            # 의미 유사도 태그 추가 (확장 기능)
            semantic_tags = {}
            if len(model_ids) >= 2:
                for i, cluster in enumerate(clusters):
                    if i == 0:
                        tag = "주요 의미 그룹"
                    elif i == 1:
                        tag = "보조 의미 그룹"
                    else:
                        tag = f"의미 그룹 {i+1}"
                    
                    for model in cluster:
                        semantic_tags[model] = tag
            
            # 응답 특성 추출
            response_features = {model_id: self.extract_response_features(responses[model_id]) 
                                for model_id in model_ids}
            
            result = {
                "similarGroups": clusters,
                "mainGroup": main_group,
                "outliers": outliers,
                "similarityMatrix": similarity_matrix,
                "responseFeatures": response_features
            }
            
            # 의미 태그 추가 (확장 기능)
            if semantic_tags:
                result["semanticTags"] = semantic_tags
                
            return result
            
        except Exception as e:
            logger.error(f"응답 군집화 중 오류: {str(e)}")
            # 오류 발생 시 모든 모델을 하나의 그룹으로 반환
            return {
                "similarGroups": [model_ids],
                "mainGroup": model_ids,
                "outliers": [],
                "similarityMatrix": {},
                "responseFeatures": {}
            }
    
    def extract_response_features(self, text):
        """
        응답 텍스트에서 특성 추출
        
        Args:
            text (str): 응답 텍스트
            
        Returns:
            dict: 응답 특성 정보
        """
        try:
            # 응답 길이
            length = len(text)
            
            # 코드 블록 개수
            code_blocks = re.findall(r'```[\s\S]*?```', text)
            code_block_count = len(code_blocks)
            
            # 링크 개수
            links = re.findall(r'\[.*?\]\(.*?\)', text) or re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
            link_count = len(links)
            
            # 목록 항목 개수
            list_items = re.findall(r'^[\s]*[-*+] |^[\s]*\d+\. ', text, re.MULTILINE)
            list_item_count = len(list_items)
            
            # 문장 분리
            sentences = re.split(r'[.!?]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            # 평균 문장 길이
            avg_sentence_length = sum(len(s) for s in sentences) / max(1, len(sentences))
            
            # 어휘 다양성 (고유 단어 비율)
            words = re.findall(r'\b\w+\b', text.lower())
            unique_words = set(words)
            vocabulary_diversity = len(unique_words) / max(1, len(words))
            
            # 자주 사용된 단어들 (확장 기능)
            result = {
                "length": length,
                "codeBlockCount": code_block_count,
                "linkCount": link_count,
                "listItemCount": list_item_count,
                "sentenceCount": len(sentences),
                "avgSentenceLength": avg_sentence_length,
                "vocabularyDiversity": vocabulary_diversity,
                "hasCode": code_block_count > 0
            }
            
            # 자주 사용된 단어들 추가 (확장 기능)
            word_counter = Counter(words)
            common_words = [word for word, count in word_counter.most_common(5)]
            if common_words:
                result["commonWords"] = common_words
                
            return result
            
        except Exception as e:
            logger.error(f"응답 특성 추출 중 오류: {str(e)}")
            # 오류 발생 시 기본값 반환
            return {
                "length": len(text),
                "codeBlockCount": 0,
                "linkCount": 0,
                "listItemCount": 0,
                "sentenceCount": 1,
                "avgSentenceLength": len(text),
                "vocabularyDiversity": 0,
                "hasCode": False
            }


# views.py (ChatView)
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import AllowAny
import logging
import json
from django.http import StreamingHttpResponse
from .similarity_analyzer import SimilarityAnalyzer
import time
import numpy as np

logger = logging.getLogger(__name__)

# JSON 직렬화 유틸리티 함수 추가
def convert_to_serializable(obj):
    """모든 객체를 JSON 직렬화 가능한 형태로 변환합니다"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):  # 정수형 타입 처리
        return int(obj)
    elif isinstance(obj, (np.float16, np.float32, np.float64)):  # float_ 제거하고 구체적인 타입만 사용
        return float(obj)
    elif isinstance(obj, (set, tuple)):
        return list(obj)
    elif hasattr(obj, 'isoformat'):  # datetime 객체 처리
        return obj.isoformat()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    else:
        try:
            # str() 사용 시도
            return str(obj)
        except:
            return repr(obj)

class ChatView(APIView):
    permission_classes = [AllowAny]
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # SimilarityAnalyzer 인스턴스 생성
        self.similarity_analyzer = SimilarityAnalyzer(threshold=0.85)

    def post(self, request, preferredModel):
        try:
            logger.info(f"Received chat request for {preferredModel}")
            
            data = request.data
            user_message = data.get('message')
            compare_responses = data.get('compare', True)
            
            # 선택된 모델들
            selected_models = data.get('selectedModels', ['gpt', 'claude', 'mixtral'])
            
            # 선택된 모델 로그
            logger.info(f"Selected models: {selected_models}")
            
            # 토큰 유무에 따른 언어 및 선호 모델 처리
            token = request.headers.get('Authorization')
            if not token:
                # 비로그인: 기본 언어는 ko, 선호 모델은 GPT로 고정
                user_language = 'ko'
                preferredModel = 'gpt'
            else:
                # 로그인: 요청 데이터의 언어 사용 (혹은 사용자의 설정을 따름)
                user_language = data.get('language', 'ko')
                # URL에 전달된 preferredModel을 그대로 사용 (프론트엔드에서 사용자 설정 반영)

            logger.info(f"Received language setting: {user_language}")

            if not user_message:
                return Response({'error': 'No message provided'}, 
                                status=status.HTTP_400_BAD_REQUEST)

            # 비동기 응답을 위한 StreamingHttpResponse 사용
            def stream_responses():
                try:
                    system_message = {
                        "role": "system",
                        "content": f"사용자가 선택한 언어는 '{user_language}'입니다. 반드시 모든 응답을 이 언어({user_language})로 제공해주세요."
                    }
                    
                    responses = {}
                    
                    # 현재 요청에 대한 고유 식별자 생성 (타임스탬프 활용)
                    request_id = str(time.time())
                    
                    # 선택된 모델들만 대화에 참여시킴
                    selected_chatbots = {model: chatbots.get(model) for model in selected_models if model in chatbots}
                    
                    # 각 봇의 응답을 개별적으로 처리하고 즉시 응답
                    for bot_id, bot in selected_chatbots.items():
                        if bot is None:
                            logger.warning(f"Selected model {bot_id} not available in chatbots")
                            yield json.dumps({
                                'type': 'bot_error',
                                'botId': bot_id,
                                'error': f"Model {bot_id} is not available"
                            }) + '\n'
                            continue
                            
                        try:
                            # 매번 새로운 대화 컨텍스트 생성 (이전 내용 초기화)
                            bot.conversation_history = [system_message]
                            response = bot.chat(user_message)
                            responses[bot_id] = response
                            
                            # 각 봇 응답을 즉시 전송
                            yield json.dumps({
                                'type': 'bot_response',
                                'botId': bot_id,
                                'response': response,
                                'requestId': request_id  # 요청 ID 추가
                            }) + '\n'
                            
                        except Exception as e:
                            logger.error(f"Error from {bot_id}: {str(e)}")
                            responses[bot_id] = f"Error: {str(e)}"
                            
                            # 에러도 즉시 전송
                            yield json.dumps({
                                'type': 'bot_error',
                                'botId': bot_id,
                                'error': str(e),
                                'requestId': request_id  # 요청 ID 추가
                            }) + '\n'
                    
                    # 응답이 2개 이상일 때만 유사도 분석 수행
                    if len(responses) >= 2:
                        try:
                            # 유사도 분석 결과 계산
                            similarity_result = self.similarity_analyzer.cluster_responses(responses)
                            
                            # 결과를 직렬화 가능한 형태로 변환
                            serializable_result = convert_to_serializable(similarity_result)
                            
                            # 디버깅을 위한 유사도 분석 결과 로깅
                            logger.info(f"Similarity analysis result: {serializable_result}")
                            
                            # 유사도 분석 결과 전송
                            yield json.dumps({
                                'type': 'similarity_analysis',
                                'result': serializable_result,
                                'requestId': request_id,
                                'timestamp': time.time(),
                                'userMessage': user_message  # 사용자 메시지 포함
                            }) + '\n'
                            
                        except Exception as e:
                            logger.error(f"Error in similarity analysis: {str(e)}", exc_info=True)
                            yield json.dumps({
                                'type': 'similarity_error',
                                'error': f"Similarity analysis error: {str(e)}",
                                'requestId': request_id
                            }) + '\n'
                    
                    # 선택된 모델이 있고 응답이 있을 때만 분석 수행
                    if selected_models and responses:
                        # 분석(비교)은 로그인 시 사용자의 선호 모델을, 비로그인 시 GPT를 사용
                        if token:
                            analyzer_bot = chatbots.get(preferredModel) or chatbots.get('gpt')
                        else:
                            analyzer_bot = chatbots.get('gpt')
                        
                        # 분석용 봇도 새로운 대화 컨텍스트로 초기화
                        analyzer_bot.conversation_history = [system_message]
                        
                        # 분석 실행 (항상 새롭게 실행)
                        analysis = analyzer_bot.analyze_responses(responses, user_message, user_language, selected_models)
                        
                        # 분석 결과 전송
                        yield json.dumps({
                            'type': 'analysis',
                            'preferredModel': analyzer_bot and analyzer_bot.api_type.upper(),
                            'best_response': analysis.get('best_response', ''),
                            'analysis': analysis.get('analysis', {}),
                            'reasoning': analysis.get('reasoning', ''),
                            'language': user_language,
                            'requestId': request_id,  # 요청 ID 추가
                            'timestamp': time.time(),  # 타임스탬프 추가
                            'userMessage': user_message  # 사용자 메시지 포함
                        }) + '\n'
                    else:
                        logger.warning("No selected models or responses to analyze")
                    
                except Exception as e:
                    logger.error(f"Stream processing error: {str(e)}", exc_info=True)
                    yield json.dumps({
                        'type': 'error',
                        'error': f"Stream processing error: {str(e)}"
                    }) + '\n'

            # StreamingHttpResponse 반환
            response = StreamingHttpResponse(
                streaming_content=stream_responses(),
                content_type='text/event-stream'
            )
            response['Cache-Control'] = 'no-cache'
            response['X-Accel-Buffering'] = 'no'
            return response
                
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}", exc_info=True)
            return Response({
                'error': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)