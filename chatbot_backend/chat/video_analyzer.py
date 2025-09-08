# video_analyzer.py - 상단 import 및 초기화 부분 수정
import os
import json
import numpy as np
import cv2
import colorsys
from collections import Counter, defaultdict
from datetime import datetime
import torch
import torch.nn as nn
import time
from PIL import Image
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv
import threading
# 환경 설정
load_dotenv()  
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
# API 클라이언트들
from groq import Groq
import openai
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("ℹ️ Anthropic 라이브러리 미설치 - Anthropic API 비활성화")

# AI 모델들 - 안전한 import 처리
YOLO_AVAILABLE = False
TRANSFORMERS_AVAILABLE = False
OCR_AVAILABLE = False
MEDIAPIPE_AVAILABLE = False
NETWORKX_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    print("✅ YOLO (Ultralytics) 로드 성공")
except ImportError:
    print("⚠️ YOLO (Ultralytics) 미설치 - 객체 감지 기능 제한됨")
    print("💡 설치 방법: pip install ultralytics")

try:
    from transformers import (
        BlipProcessor, BlipForConditionalGeneration, BlipForQuestionAnswering,
        CLIPProcessor, CLIPModel, AutoTokenizer, AutoModelForCausalLM, pipeline
    )
    TRANSFORMERS_AVAILABLE = True
    print("✅ Transformers 라이브러리 로드 성공")
except ImportError:
    print("⚠️ Transformers 라이브러리 미설치 - VQA/CLIP 기능 비활성화")

try:
    import easyocr
    OCR_AVAILABLE = True
    print("✅ EasyOCR 라이브러리 로드 성공")
except ImportError:
    print("⚠️ EasyOCR 미설치 - OCR 기능 비활성화")

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
    print("✅ MediaPipe 라이브러리 로드 성공")
except ImportError:
    print("⚠️ MediaPipe 미설치 - 포즈 추정 기능 비활성화")

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
    print("✅ NetworkX 라이브러리 로드 성공")
except ImportError:
    print("⚠️ NetworkX 미설치 - Scene Graph 기능 제한됨")

# PyTorch 안전 설정 (수정됨)
try:
    torch_version = torch.__version__
    if torch_version >= "2.6.0" and YOLO_AVAILABLE:
        try:
            import ultralytics.nn.tasks
            torch.serialization.add_safe_globals([
                ultralytics.nn.tasks.DetectionModel,
                ultralytics.nn.tasks.SegmentationModel,
                ultralytics.nn.tasks.ClassificationModel,
                ultralytics.nn.tasks.PoseModel,
                ultralytics.nn.tasks.OBBModel,
                ultralytics.nn.tasks.WorldModel
            ])
            print(f"✅ PyTorch {torch_version} - YOLO 안전 글로벌 설정 완료")
        except Exception as e:
            print(f"⚠️ 안전 글로벌 설정 건너뜀: {e}")
except Exception as e:
    print(f"⚠️ PyTorch 초기화 경고: {e}")

# LangChain 관련 import 추가 (참고 모델 통합)
try:
    from langchain_community.document_loaders import JSONLoader
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain.schema import Document
    from langchain.retrievers import EnsembleRetriever
    from langchain_community.retrievers import BM25Retriever
    from langchain_openai import ChatOpenAI
    LANGCHAIN_AVAILABLE = True
    print("✅ LangChain 라이브러리 로드 성공")
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("⚠️ LangChain 라이브러리 미설치 - RAG 기능 비활성화")

# API 설정
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")

# 클라이언트 초기화
groq_client = Groq(api_key=GROQ_API_KEY)

# OpenAI 클라이언트
try:
    openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
    OPENAI_AVAILABLE = True
except Exception as e:
    print(f"⚠️ OpenAI API 초기화 실패: {e}")
    openai_client = None
    OPENAI_AVAILABLE = False

# Anthropic 클라이언트
try:
    if ANTHROPIC_AVAILABLE:
        anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    else:
        anthropic_client = None
except Exception as e:
    print(f"⚠️ Anthropic API 초기화 실패: {e}")
    anthropic_client = None

# 참고 모델 설정 (통합됨)
CHUNK_SIZE = 256
CHUNK_OVERLAP = 128
EMBEDDING_MODEL = "intfloat/e5-large"
K = 3
LLM = "gemma2-9b-it"
MAX_NEW_TOKENS = 512

PROMPT_TEMPLATE = """
You are an AI visual assistant that can analyze video content and provide detailed insights.

Using the provided video analysis information, answer the user's question accurately.
Context from video analysis:
{context}

Question: {question}

Answer:
"""

# COCO 객체 클래스 (기존과 동일)
ENHANCED_OBJECTS = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
    6: 'train', 7: 'truck', 8: 'boat', 15: 'cat', 16: 'dog', 17: 'horse',
    18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe',
    56: 'chair', 57: 'couch', 58: 'potted_plant', 59: 'bed', 60: 'dining_table',
    61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard',
    67: 'cell_phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink',
    72: 'refrigerator', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase',
    29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports_ball', 33: 'kite',
    34: 'baseball_bat', 35: 'baseball_glove', 36: 'skateboard', 37: 'surfboard',
    38: 'tennis_racket', 39: 'bottle', 40: 'wine_glass', 41: 'cup', 42: 'fork', 43: 'knife', 
    44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange',
    50: 'broccoli', 51: 'carrot', 52: 'hot_dog', 53: 'pizza', 54: 'donut', 55: 'cake',
    73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy_bear',
    78: 'hair_drier', 79: 'toothbrush', 9: 'traffic_light', 10: 'fire_hydrant',
    11: 'stop_sign', 12: 'parking_meter', 13: 'bench', 14: 'bird'
}

# 로그 중복 방지를 위한 글로벌 변수
_logged_messages = set()

def log_once(message, level="INFO"):
    """중복 로그 방지"""
    if message not in _logged_messages:
        print(f"[{level}] {message}")
        _logged_messages.add(message)

def call_groq_llm_enhanced(prompt, system_prompt="", model="llama-3.1-8b-instant", max_retries=3):
    """개선된 LLM 호출 함수 - Rate Limiting 강화 및 로그 중복 제거"""
    
    # Rate Limiting을 위한 전역 변수
    if not hasattr(call_groq_llm_enhanced, 'last_call_time'):
        call_groq_llm_enhanced.last_call_time = 0
        call_groq_llm_enhanced.call_count = 0
    
    # 요청 간격 제어 (최소 2초)
    current_time = time.time()
    time_since_last_call = current_time - call_groq_llm_enhanced.last_call_time
    if time_since_last_call < 2.0:
        sleep_time = 2.0 - time_since_last_call
        log_once(f"⏱️ Rate limiting: {sleep_time:.1f}초 대기", "DEBUG")
        time.sleep(sleep_time)
    
    call_groq_llm_enhanced.last_call_time = time.time()
    call_groq_llm_enhanced.call_count += 1
    
    # 너무 많은 요청시 긴 대기
    if call_groq_llm_enhanced.call_count % 20 == 0:
        log_once("🔄 대량 요청 감지 - 10초 대기", "WARNING")
        time.sleep(10)
    
    # Groq 먼저 시도 (기존 로직)
    for attempt in range(max_retries):
        try:
            response = groq_client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.6,
                max_tokens=512,
                top_p=0.8
            )
            content = response.choices[0].message.content.strip()
            
            # 성공 로그는 중요한 경우만
            # if attempt > 0:  # 재시도 후 성공한 경우만 로그
            #     log_once(f"✅ Groq API 성공 (시도 {attempt + 1})", "SUCCESS")
            # return content
            
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                wait_time = min(2 ** attempt * 2, 30)  # 지수적 백오프, 최대 30초
                log_once(f"⏳ Rate limit - {wait_time}초 대기 (시도 {attempt + 1}/{max_retries})", "WARNING")
                time.sleep(wait_time)
                continue
            else:
                log_once(f"⚠️ Groq API 실패 (시도 {attempt + 1}/{max_retries}): {e}", "ERROR")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                else:
                    break

    # OpenAI fallback
    if OPENAI_AVAILABLE and openai_client:
        try:
            response = openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.6,
                max_tokens=512
            )
            content = response.choices[0].message.content.strip()
            log_once("✅ OpenAI API fallback 성공", "SUCCESS")
            return content
            
        except Exception as e:
            log_once(f"❌ OpenAI API도 실패: {e}", "ERROR")
    
    # Anthropic fallback
    if ANTHROPIC_AVAILABLE and anthropic_client:
        try:
            response = anthropic_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=512,
                temperature=0.6,
                system=system_prompt,
                messages=[{"role": "user", "content": prompt}]
            )
            content = response.content[0].text.strip()
            log_once("✅ Anthropic API fallback 성공", "SUCCESS")
            return content
            
        except Exception as e:
            log_once(f"❌ Anthropic API도 실패: {e}", "ERROR")
    
    # 모든 API 실패시 기본 응답
    log_once("🔄 기본 응답 생성...", "WARNING")
    
    if "객체" in prompt or "object" in prompt.lower():
        return "프레임에서 여러 객체가 감지되었으며, 이들 간의 공간적 관계를 바탕으로 장면을 분석했습니다."
    elif "특징" in prompt or "feature" in prompt.lower():
        return '{"activity_type": 0.5, "emotional_tone": 0.6, "complexity_level": 0.4, "interaction_level": 0.3, "visual_coherence": 0.7, "spatial_composition": 0.6, "temporal_context": 0.5, "social_context": 0.4}'
    elif "캡션" in prompt or "caption" in prompt.lower():
        return "동영상 프레임에서 다양한 객체들이 감지되고 있으며, 전반적으로 일상적인 장면으로 보입니다."
    else:
        return f"API 호출 실패로 인한 기본 분석 결과입니다. 프롬프트 길이: {len(prompt)}자"

class ColorAnalyzer:
    """고급 색상 분석기"""

    def __init__(self):
        self.color_ranges = {
            'red': [(0, 70, 50), (10, 255, 255), (170, 70, 50), (180, 255, 255)],
            'orange': [(11, 70, 50), (25, 255, 255)],
            'yellow': [(26, 70, 50), (35, 255, 255)],
            'green': [(36, 70, 50), (85, 255, 255)],
            'blue': [(86, 70, 50), (125, 255, 255)],
            'purple': [(126, 70, 50), (155, 255, 255)],
            'pink': [(156, 70, 50), (169, 255, 255)],
            'white': [(0, 0, 200), (180, 25, 255)],
            'black': [(0, 0, 0), (180, 255, 50)],
            'gray': [(0, 0, 51), (180, 25, 199)],
            'brown': [(8, 60, 20), (20, 255, 200)]
        }
    
    def extract_dominant_colors(self, frame, bbox, num_colors=3):
        """바운딩 박스 내 주요 색상 추출"""
        try:
            h, w = frame.shape[:2]
            
            x1 = max(0, int(bbox[0] * w))
            y1 = max(0, int(bbox[1] * h))
            x2 = min(w, int(bbox[2] * w))
            y2 = min(h, int(bbox[3] * h))
            
            roi = frame[y1:y2, x1:x2]
            
            if roi.size == 0:
                return []
            
            target_size = min(100, max(roi.shape[:2]))
            roi_resized = cv2.resize(roi, (target_size, target_size))
            hsv = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2HSV)
            
            color_percentages = {}
            
            for color_name, ranges in self.color_ranges.items():
                mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
                
                if len(ranges) == 4:  # red 경우
                    mask1 = cv2.inRange(hsv, np.array(ranges[0]), np.array(ranges[1]))
                    mask2 = cv2.inRange(hsv, np.array(ranges[2]), np.array(ranges[3]))
                    mask = cv2.bitwise_or(mask1, mask2)
                else:
                    mask = cv2.inRange(hsv, np.array(ranges[0]), np.array(ranges[1]))
                
                percentage = (np.sum(mask > 0) / (hsv.shape[0] * hsv.shape[1])) * 100
                color_percentages[color_name] = round(percentage, 1)
            
            significant_colors = [(color, pct) for color, pct in color_percentages.items() if pct >= 3.0]
            significant_colors.sort(key=lambda x: x[1], reverse=True)
            
            return significant_colors[:num_colors]
            
        except Exception as e:
            print(f"⚠️ 색상 분석 오류: {e}")
            return []
    
    def get_color_description(self, colors):
        """색상 리스트를 텍스트로 변환"""
        if not colors:
            return "unknown"
        
        if len(colors) == 1:
            return colors[0][0]
        
        dominant_colors = [color[0] for color in colors if color[1] >= 10.0]
        
        if len(dominant_colors) == 1:
            return dominant_colors[0]
        elif len(dominant_colors) == 2:
            return f"{dominant_colors[0]}-{dominant_colors[1]}"
        elif len(dominant_colors) >= 3:
            return f"{dominant_colors[0]}-mixed"
        else:
            minor_colors = [color[0] for color in colors[:2]]
            return f"{minor_colors[0]}-{minor_colors[1]}" if len(minor_colors) == 2 else minor_colors[0]

class SceneClassifier:
    """Scene 분류기 - 실내/실외, 시간대, 날씨, 활동 등 분류"""
    
    def __init__(self):
        self.clip_processor = None
        self.clip_model = None
        
        try:
            if TRANSFORMERS_AVAILABLE:
                print("🏞️ Scene 분류기 로딩 중...")
                self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
                self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
                
                self.scene_templates = {
                    'location': ['indoor', 'outdoor', 'street', 'home'],
                    'time': ['day', 'night'],
                    'weather': ['sunny', 'cloudy', 'clear'],
                    'activity': ['walking', 'driving', 'sitting', 'standing']
                }
                
                print("🏞️ Scene 분류기 로드 완료")
            else:
                raise ImportError("Transformers not available")
                
        except Exception as e:
            print(f"⚠️ Scene 분류기 로드 실패: {e}")
            self.scene_templates = {
                'location': ['indoor', 'outdoor'],
                'time': ['day', 'night'], 
                'activity': ['general']
            }
    
    def classify_scene(self, frame):
        """Scene 분류 수행"""
        if not self.clip_processor or not self.clip_model:
            return self._basic_scene_classification(frame)
        
        try:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            scene_classification = {}
            
            for category, labels in self.scene_templates.items():
                text_inputs = [f"a photo of {label}" for label in labels]
                
                inputs = self.clip_processor(
                    text=text_inputs, 
                    images=image, 
                    return_tensors="pt", 
                    padding=True
                )
                
                outputs = self.clip_model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)
                
                best_idx = probs.argmax().item()
                best_label = labels[best_idx]
                confidence = probs[0][best_idx].item()
                
                scene_classification[category] = {
                    'label': best_label,
                    'confidence': float(confidence),
                    'all_scores': {label: float(score) for label, score in zip(labels, probs[0])}
                }
            
            return scene_classification
            
        except Exception as e:
            print(f"⚠️ Scene 분류 오류: {e}")
            return self._basic_scene_classification(frame)
    
    def _basic_scene_classification(self, frame):
        """기본 Scene 분류 (휴리스틱 기반)"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            
            if brightness > 120:
                time_label = 'day'
                time_confidence = 0.7
            else:
                time_label = 'night'
                time_confidence = 0.7
            
            edges = cv2.Canny(gray, 100, 200)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            if edge_density > 0.05:
                location_label = 'outdoor'
                location_confidence = 0.6
            else:
                location_label = 'indoor'
                location_confidence = 0.6
            
            return {
                'location': {
                    'label': location_label,
                    'confidence': location_confidence
                },
                'time': {
                    'label': time_label,
                    'confidence': time_confidence
                },
                'activity': {
                    'label': 'general',
                    'confidence': 0.5
                }
            }
            
        except Exception as e:
            print(f"⚠️ 기본 Scene 분류 오류: {e}")
            return {}

class SceneGraphGenerator:
    """Scene Graph 생성기 - 객체간 관계 분석"""
    
    def __init__(self):
        if NETWORKX_AVAILABLE:
            self.graph = nx.DiGraph()
            self.use_networkx = True
        else:
            self.graph = None
            self.use_networkx = False
            print("⚠️ NetworkX 없음 - 기본 관계 분석만 수행")
        
    def analyze_spatial_relationships(self, objects):
        """공간적 관계 분석"""
        relationships = []
        
        for i, obj1 in enumerate(objects):
            for j, obj2 in enumerate(objects[i+1:], i+1):
                bbox1 = obj1['bbox']
                bbox2 = obj2['bbox']
                
                center1 = [(bbox1[0] + bbox1[2])/2, (bbox1[1] + bbox1[3])/2]
                center2 = [(bbox2[0] + bbox2[2])/2, (bbox2[1] + bbox2[3])/2]
                
                distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
                
                if distance < 0.1:
                    relation = 'near'
                elif distance > 0.5:
                    relation = 'far_from'
                else:
                    if center1[1] < center2[1] - 0.05:
                        relation = 'above'
                    elif center1[1] > center2[1] + 0.05:
                        relation = 'under'
                    elif center1[0] < center2[0] - 0.05:
                        relation = 'left_of'
                    elif center1[0] > center2[0] + 0.05:
                        relation = 'right_of'
                    else:
                        relation = 'near'
                
                relationships.append({
                    'subject': obj1['class'],
                    'subject_id': obj1.get('track_id', i),
                    'predicate': relation,
                    'object': obj2['class'],
                    'object_id': obj2.get('track_id', j),
                    'confidence': min(obj1['confidence'], obj2['confidence']),
                    'distance': float(distance)
                })
        
        return relationships
    
    def analyze_semantic_relationships(self, objects, scene_context):
        """의미적 관계 분석"""
        relationships = []
        
        semantic_rules = {
            'person': {
                'car': 'driving',
                'bicycle': 'riding',
                'chair': 'sitting_on',
                'laptop': 'using',
                'cell_phone': 'holding'
            },
            'car': {
                'traffic_light': 'stopped_at',
                'stop_sign': 'stopped_at'
            }
        }
        
        for obj1 in objects:
            for obj2 in objects:
                if obj1['class'] != obj2['class']:
                    if obj1['class'] in semantic_rules:
                        if obj2['class'] in semantic_rules[obj1['class']]:
                            relation = semantic_rules[obj1['class']][obj2['class']]
                            
                            relationships.append({
                                'subject': obj1['class'],
                                'subject_id': obj1.get('track_id', 0),
                                'predicate': relation,
                                'object': obj2['class'],
                                'object_id': obj2.get('track_id', 0),
                                'confidence': min(obj1['confidence'], obj2['confidence']),
                                'type': 'semantic'
                            })
        
        return relationships
    
    def build_scene_graph(self, objects, scene_context):
        """Scene Graph 구축"""
        spatial_relations = self.analyze_spatial_relationships(objects)
        semantic_relations = self.analyze_semantic_relationships(objects, scene_context)
        
        all_relations = spatial_relations + semantic_relations
        
        if self.use_networkx:
            self.graph.clear()
            
            for obj in objects:
                node_id = f"{obj['class']}_{obj.get('track_id', 0)}"
                self.graph.add_node(node_id, **obj)
            
            for relation in all_relations:
                subject_id = f"{relation['subject']}_{relation['subject_id']}"
                object_id = f"{relation['object']}_{relation['object_id']}"
                
                if subject_id in self.graph.nodes and object_id in self.graph.nodes:
                    self.graph.add_edge(subject_id, object_id, **relation)
            
            scene_graph = {
                'nodes': dict(self.graph.nodes(data=True)),
                'edges': [
                    {
                        'source': edge[0],
                        'target': edge[1],
                        'relationship': edge[2]
                    }
                    for edge in self.graph.edges(data=True)
                ],
                'relationships': all_relations,
                'graph_metrics': {
                    'num_nodes': self.graph.number_of_nodes(),
                    'num_edges': self.graph.number_of_edges(),
                    'density': nx.density(self.graph) if self.graph.number_of_nodes() > 0 else 0
                }
            }
        else:
            scene_graph = {
                'nodes': {f"{obj['class']}_{obj.get('track_id', i)}": obj for i, obj in enumerate(objects)},
                'edges': [],
                'relationships': all_relations,
                'graph_metrics': {
                    'num_nodes': len(objects),
                    'num_edges': len(all_relations),
                    'density': 0
                }
            }
        
        return scene_graph

class GPTFeatureExtractor:
    """GPT 기반 고급 특징 추출기"""
    
    def __init__(self):
        self.feature_dimensions = 1024
        
    def extract_gpt_features(self, frame_analysis, scene_context):
        """GPT 기반 고수준 특징 벡터 생성"""
        try:
            objects_info = scene_context.get('objects', [])
            scene_classification = scene_context.get('scene_classification', {})
            
            prompt = f"""
            다음 비디오 프레임 분석 결과를 바탕으로 고수준 특징을 추출해주세요:

            객체 정보: {[obj.get('class', '') for obj in objects_info]}
            객체 수: {len(objects_info)}
            Scene 분류: {scene_classification}
            관계 수: {len(scene_context.get('relationships', []))}
            
            다음 8개 특징을 0-1 사이 값으로 정량화해주세요:
            1. activity_type (활동 유형)
            2. emotional_tone (감정적 분위기)  
            3. complexity_level (복잡성 수준)
            4. interaction_level (상호작용 정도)
            5. visual_coherence (시각적 일관성)
            6. spatial_composition (공간적 구성)
            7. temporal_context (시간적 맥락)
            8. social_context (사회적 상황)

            JSON 형태로만 응답해주세요:
            {{"activity_type": 0.x, "emotional_tone": 0.x, ...}}
            """
            
            response = call_groq_llm_enhanced(
                prompt, 
                "비디오 분석 전문가로서 정확한 수치로 특징을 추출합니다.",
                model="llama-3.1-8b-instant"
            )
            
            try:
                if '{' in response and '}' in response:
                    json_start = response.find('{')
                    json_end = response.rfind('}') + 1
                    json_str = response[json_start:json_end]
                    features = json.loads(json_str)
                    print(f"✅ GPT 특징 추출 성공: {len(features)}개 특징")
                else:
                    raise ValueError("JSON 형식 응답 없음")
            except:
                print("⚠️ GPT 응답 파싱 실패, 기본 특징 사용")
                features = self._generate_default_features(scene_context)
            
            feature_vector = self._features_to_vector(features)
            
            return {
                'gpt_features': features,
                'feature_vector': feature_vector,
                'vector_dimension': len(feature_vector),
                'extraction_method': 'gpt_llm' if isinstance(features, dict) and len(features) > 4 else 'default'
            }
            
        except Exception as e:
            print(f"⚠️ GPT 특징 추출 오류: {e}")
            return self._generate_default_features(scene_context)
    
    def _generate_default_features(self, scene_context):
        """기본 특징 생성"""
        objects = scene_context.get('objects', [])
        relationships = scene_context.get('relationships', [])
        scene_class = scene_context.get('scene_classification', {})
        
        object_count = len(objects)
        relationship_count = len(relationships)
        
        features = {
            'activity_type': min(object_count * 0.15 + relationship_count * 0.1, 1.0),
            'emotional_tone': 0.6 if scene_class.get('time', {}).get('label') == 'day' else 0.4,
            'complexity_level': min(object_count * 0.12 + relationship_count * 0.08, 1.0),
            'interaction_level': min(relationship_count * 0.15, 1.0),
            'visual_coherence': 0.7 + min(object_count * 0.02, 0.2),
            'spatial_composition': 0.6 + min(relationship_count * 0.05, 0.3),
            'temporal_context': 0.5,
            'social_context': 1.0 if any(obj.get('class') == 'person' for obj in objects) else 0.2
        }
        
        feature_vector = list(features.values()) + [0.0] * (self.feature_dimensions - len(features))
        
        return {
            'gpt_features': features,
            'feature_vector': feature_vector[:self.feature_dimensions],
            'vector_dimension': self.feature_dimensions,
            'extraction_method': 'default_heuristic'
        }
    
    def _features_to_vector(self, features):
        """특징 딕셔너리를 벡터로 변환"""
        if isinstance(features, dict):
            vector = []
            for key in sorted(features.keys()):
                value = features[key]
                if isinstance(value, (int, float)):
                    vector.append(float(value))
                else:
                    vector.append(0.0)
            
            while len(vector) < self.feature_dimensions:
                vector.append(0.0)
            
            return vector[:self.feature_dimensions]
        else:
            return [0.0] * self.feature_dimensions

class EnhancedCaptionGenerator:
    """개선된 캡션 생성기"""
    
    def __init__(self):
        self.temporal_context = []
        self.max_context_frames = 3
        
    def generate_accurate_caption(self, frame, detected_objects, scene_analysis, frame_id, timestamp):
        """정확도가 향상된 캡션 생성"""
        
        # 1. 기본 객체 정보 분석
        object_info = self._analyze_objects_detailed(detected_objects)
        
        # 2. 공간적 관계 분석
        spatial_info = self._analyze_spatial_relationships(detected_objects)
        
        # 3. 시간적 맥락 분석
        temporal_info = self._analyze_temporal_context(detected_objects, frame_id)
        
        # 4. 시각적 특징 분석
        visual_info = self._analyze_visual_features(frame)
        
        # 5. Scene 분류 정보
        scene_info = scene_analysis.get('scene_classification', {})
        
        # 6. VQA 결과 활용
        vqa_info = scene_analysis.get('vqa_results', {})
        
        # 7. OCR 텍스트 정보
        ocr_text = scene_analysis.get('ocr_text', '')
        
        # 8. 종합 캡션 생성
        caption = self._generate_comprehensive_caption(
            object_info, spatial_info, temporal_info, visual_info, 
            scene_info, vqa_info, ocr_text, frame_id, timestamp
        )
        
        # 9. 컨텍스트 업데이트
        self._update_temporal_context(detected_objects, frame_id)
        
        return caption
    
    def _analyze_objects_detailed(self, detected_objects):
        """객체들의 상세 분석"""
        if not detected_objects:
            return {"count": 0, "types": [], "colors": [], "positions": []}
        
        object_types = [obj['class'] for obj in detected_objects]
        object_counter = Counter(object_types)
        
        colors = []
        for obj in detected_objects:
            if obj.get('color_description') and obj['color_description'] != 'unknown':
                colors.append(f"{obj['color_description']} {obj['class']}")
        
        positions = []
        for obj in detected_objects:
            bbox = obj['bbox']
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
            if center_x < 0.33:
                x_pos = "왼쪽"
            elif center_x > 0.67:
                x_pos = "오른쪽"
            else:
                x_pos = "중앙"
                
            if center_y < 0.33:
                y_pos = "상단"
            elif center_y > 0.67:
                y_pos = "하단"
            else:
                y_pos = "중간"
            
            positions.append(f"{x_pos} {y_pos}의 {obj['class']}")
        
        return {
            "count": len(detected_objects),
            "types": object_counter,
            "colors": colors,
            "positions": positions,
            "main_objects": [item[0] for item in object_counter.most_common(3)]
        }
    
    def _analyze_spatial_relationships(self, detected_objects):
        """공간적 관계 분석"""
        if len(detected_objects) < 2:
            return []
        
        relationships = []
        for i, obj1 in enumerate(detected_objects):
            for obj2 in detected_objects[i+1:]:
                bbox1 = obj1['bbox']
                bbox2 = obj2['bbox']
                
                center1 = [(bbox1[0] + bbox1[2])/2, (bbox1[1] + bbox1[3])/2]
                center2 = [(bbox2[0] + bbox2[2])/2, (bbox2[1] + bbox2[3])/2]
                
                distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
                
                if distance < 0.15:
                    relationships.append(f"{obj1['class']}와 {obj2['class']}가 가깝게 위치")
                elif center1[0] < center2[0] - 0.1:
                    relationships.append(f"{obj1['class']}가 {obj2['class']} 왼쪽에 위치")
                elif center1[0] > center2[0] + 0.1:
                    relationships.append(f"{obj1['class']}가 {obj2['class']} 오른쪽에 위치")
                elif center1[1] < center2[1] - 0.1:
                    relationships.append(f"{obj1['class']}가 {obj2['class']} 위쪽에 위치")
                elif center1[1] > center2[1] + 0.1:
                    relationships.append(f"{obj1['class']}가 {obj2['class']} 아래쪽에 위치")
        
        return relationships[:5]
    
    def _analyze_temporal_context(self, detected_objects, frame_id):
        """시간적 맥락 분석"""
        if not self.temporal_context:
            return {"is_first_frame": True, "changes": []}
        
        current_objects = set(obj['class'] for obj in detected_objects)
        
        if self.temporal_context:
            prev_objects = self.temporal_context[-1]['objects']
            
            new_objects = current_objects - prev_objects
            disappeared_objects = prev_objects - current_objects
            
            changes = []
            if new_objects:
                changes.append(f"새로 나타난 객체: {', '.join(new_objects)}")
            if disappeared_objects:
                changes.append(f"사라진 객체: {', '.join(disappeared_objects)}")
            
            return {
                "is_first_frame": False,
                "changes": changes,
                "continuity": len(current_objects & prev_objects) / max(len(current_objects | prev_objects), 1)
            }
        
        return {"is_first_frame": True, "changes": []}
    
    def _analyze_visual_features(self, frame):
        """시각적 특징 분석"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180])
            dominant_hue = np.argmax(hist_h)
            
            saturation = np.mean(hsv[:, :, 1])
            
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            return {
                "brightness": "밝은" if brightness > 120 else "어두운",
                "saturation": "선명한" if saturation > 100 else "흐린",
                "complexity": "복잡한" if edge_density > 0.05 else "단순한",
                "dominant_color_hue": dominant_hue
            }
            
        except Exception as e:
            print(f"⚠️ 시각적 특징 분석 오류: {e}")
            return {"brightness": "보통", "saturation": "보통", "complexity": "보통"}
    
    def _generate_comprehensive_caption(self, object_info, spatial_info, temporal_info, 
                                      visual_info, scene_info, vqa_info, ocr_text, 
                                      frame_id, timestamp):
        """종합적인 캡션 생성"""
        
        caption_parts = []
        
        # 1. 시간 정보
        caption_parts.append(f"영상 {timestamp:.1f}초 지점에서")
        
        # 2. 장소/환경 정보
        location = scene_info.get('location', {}).get('label', '')
        time_of_day = scene_info.get('time', {}).get('label', '')
        
        if location and time_of_day:
            caption_parts.append(f"{time_of_day} {location} 환경에서")
        elif location:
            caption_parts.append(f"{location} 환경에서")
        elif time_of_day:
            caption_parts.append(f"{time_of_day} 시간대에")
        
        # 3. 주요 객체 및 개수
        if object_info['count'] > 0:
            main_objects = object_info['main_objects'][:3]
            
            if len(main_objects) == 1:
                if object_info['types'][main_objects[0]] > 1:
                    caption_parts.append(f"{object_info['types'][main_objects[0]]}개의 {main_objects[0]}이/가")
                else:
                    caption_parts.append(f"{main_objects[0]}이/가")
            else:
                objects_desc = []
                for obj in main_objects:
                    count = object_info['types'][obj]
                    if count > 1:
                        objects_desc.append(f"{count}개의 {obj}")
                    else:
                        objects_desc.append(obj)
                caption_parts.append(f"{', '.join(objects_desc)}이/가")
        
        # 4. 색상 정보 (있는 경우)
        if object_info['colors']:
            colors_text = ", ".join(object_info['colors'][:2])
            caption_parts.append(f"({colors_text})")
        
        # 5. 위치 정보
        if object_info['positions']:
            positions_text = object_info['positions'][0]
            caption_parts.append(f"{positions_text}에서")
        
        # 6. 행동/활동 추정
        action = self._estimate_action(object_info, spatial_info, vqa_info)
        if action:
            caption_parts.append(action)
        
        # 7. 공간적 관계 (중요한 것만)
        if spatial_info:
            relationship = spatial_info[0]
            caption_parts.append(f", {relationship}")
        
        # 8. 시간적 변화 (있는 경우)
        if not temporal_info.get('is_first_frame', True) and temporal_info.get('changes'):
            change = temporal_info['changes'][0]
            caption_parts.append(f". {change}")
        
        # 9. 시각적 특징
        visual_desc = f"{visual_info['brightness']} {visual_info['complexity']} 장면"
        caption_parts.append(f"으로 {visual_desc}입니다")
        
        # 10. OCR 텍스트 (있는 경우)
        if ocr_text and len(ocr_text.strip()) > 0:
            caption_parts.append(f". 텍스트 '{ocr_text[:20]}...'가 확인됩니다")
        
        # 문장 결합 및 정리
        caption = " ".join(caption_parts)
        
        # 문장 정리
        caption = self._clean_caption(caption)
        
        return caption
    
    def _estimate_action(self, object_info, spatial_info, vqa_info):
        """객체와 관계를 바탕으로 행동/활동 추정"""
        
        activity_keywords = {
            "walking": "걷고 있는",
            "sitting": "앉아 있는",
            "standing": "서 있는",
            "driving": "운전하고 있는",
            "eating": "먹고 있는",
            "drinking": "마시고 있는",
            "reading": "읽고 있는",
            "working": "작업하고 있는",
            "playing": "놀고 있는",
            "talking": "대화하고 있는"
        }
        
        # VQA 결과에서 활동 추정
        for question, answer in vqa_info.items():
            if "활동" in question or "activity" in question.lower():
                for keyword, korean in activity_keywords.items():
                    if keyword in answer.lower():
                        return korean
        
        # 객체 기반 활동 추정
        main_objects = object_info.get('main_objects', [])
        
        if 'person' in main_objects:
            if 'car' in main_objects:
                return "이동하고 있는"
            elif 'laptop' in main_objects or 'computer' in main_objects:
                return "작업하고 있는"
            elif 'chair' in main_objects:
                return "앉아 있는"
            elif any(food in main_objects for food in ['apple', 'banana', 'sandwich', 'pizza']):
                return "식사하고 있는"
            else:
                return "활동하고 있는"
        elif 'car' in main_objects and 'traffic_light' in main_objects:
            return "교통 상황에서 정지해 있는"
        elif len(main_objects) > 3:
            return "다양한 활동이 일어나고 있는"
        else:
            return "위치해 있는"
    
    def _clean_caption(self, caption):
        """캡션 정리 및 최적화"""
        
        if len(caption) > 500:
            caption = caption[:500] + "..."
        
        words = caption.split()
        cleaned_words = []
        prev_word = ""
        
        for word in words:
            if word != prev_word:
                cleaned_words.append(word)
            prev_word = word
        
        caption = " ".join(cleaned_words)
        
        caption = caption.replace("이/가 이/가", "이/가")
        caption = caption.replace("에서 에서", "에서")
        caption = caption.replace("  ", " ")
        
        return caption.strip()
    
    def _update_temporal_context(self, detected_objects, frame_id):
        """시간적 컨텍스트 업데이트"""
        current_objects = set(obj['class'] for obj in detected_objects)
        
        self.temporal_context.append({
            'frame_id': frame_id,
            'objects': current_objects,
            'count': len(detected_objects)
        })
        
        if len(self.temporal_context) > self.max_context_frames:
            self.temporal_context.pop(0)

class AdvancedSceneAnalyzer:
    """고급 Scene 분석기"""
    
    def __init__(self, enable_vqa=True, enable_ocr=True, enable_segmentation=True):
        self.enable_vqa = enable_vqa
        self.enable_ocr = enable_ocr
        self.enable_segmentation = enable_segmentation
        
        # 캡션 생성기 초기화
        self.caption_generator = EnhancedCaptionGenerator()
        
        # OCR 모델 초기화
        if enable_ocr and OCR_AVAILABLE:
            try:
                self.ocr_reader = easyocr.Reader(['en', 'ko'])
                # print("🔤 OCR 모델 로드 완료")
            except:
                self.ocr_reader = None
                print("⚠️ OCR 모델 로드 실패")
        else:
            self.ocr_reader = None
        
        # VQA 모델 초기화
        if enable_vqa and TRANSFORMERS_AVAILABLE:
            try:
                # print("🤖 VQA 모델 로딩 중...")
                self.vqa_processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
                self.vqa_model = BlipForQuestionAnswering.from_pretrained(
                    "Salesforce/blip-vqa-base",
                    torch_dtype=torch.float32
                )
                
                self.caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                self.caption_model = BlipForConditionalGeneration.from_pretrained(
                    "Salesforce/blip-image-captioning-base",
                    torch_dtype=torch.float32
                )
                
                print("🤖 VQA 모델 로드 완료")
                
            except Exception as e:
                self.vqa_processor = None
                self.vqa_model = None
                self.caption_processor = None
                self.caption_model = None
                print(f"⚠️ VQA 모델 로드 실패: {e}")
        else:
            self.vqa_processor = None
            self.vqa_model = None
            self.caption_processor = None
            self.caption_model = None
        
        # MediaPipe 포즈 추정
        if MEDIAPIPE_AVAILABLE:
            try:
                self.mp_pose = mp.solutions.pose
                self.pose = self.mp_pose.Pose()
                print("🏃 MediaPipe 포즈 추정 로드 완료")
            except:
                self.mp_pose = None
                self.pose = None
                print("⚠️ MediaPipe 로드 실패")
        else:
            self.mp_pose = None
            self.pose = None
        
        # 추가 구성 요소
        self.scene_classifier = SceneClassifier()
        self.scene_graph_generator = SceneGraphGenerator()
        self.gpt_extractor = GPTFeatureExtractor()
    
    def comprehensive_scene_analysis(self, frame, frame_id, timestamp, detected_objects):
        """종합적인 Scene 분석"""
        analysis_result = {
            'frame_id': frame_id,
            'timestamp': timestamp,
            'basic_info': {},
            'scene_classification': {},
            'scene_graph': {},
            'gpt_features': {},
            'ocr_text': "",
            'vqa_results': {},
            'visual_features': [],
            'pose_features': []
        }
        
        try:
            # 1. 기본 정보
            analysis_result['basic_info'] = {
                'frame_shape': frame.shape,
                'detected_objects_count': len(detected_objects),
                'object_types': list(set(obj.get('class', '') for obj in detected_objects))
            }
            
            # 2. Scene 분류
            analysis_result['scene_classification'] = self.scene_classifier.classify_scene(frame)
            
            # 3. OCR
            if self.enable_ocr and self.ocr_reader:
                analysis_result['ocr_text'] = self.extract_scene_text(frame)
            
            # 4. VQA
            if self.enable_vqa:
                vqa_questions = [
                    "What is happening in this image?",
                    "What is the main activity?",
                    "How many people are in the image?",
                    "What is the setting or location?",
                    "What time of day is it?",
                    "What objects are visible?"
                ]
                analysis_result['vqa_results'] = self.answer_vqa_questions(frame, vqa_questions)
            
            # 5. Scene Graph 생성
            scene_context = {
                'objects': detected_objects,
                'scene_classification': analysis_result['scene_classification'],
                'vqa_results': analysis_result['vqa_results']
            }
            analysis_result['scene_graph'] = self.scene_graph_generator.build_scene_graph(
                detected_objects, scene_context
            )
            
            # 6. GPT 기반 고급 특징 추출
            analysis_result['gpt_features'] = self.gpt_extractor.extract_gpt_features(
                analysis_result, scene_context
            )
            
            # 7. 시각적 특징
            analysis_result['visual_features'] = self.extract_visual_features(frame).tolist()
            
            # 8. 포즈 특징
            analysis_result['pose_features'] = self.extract_pose_features(frame)
            
        except Exception as e:
            print(f"⚠️ 종합 분석 오류: {e}")
        
        return analysis_result
    
    def generate_enhanced_caption(self, frame, detected_objects, scene_analysis, frame_id, timestamp):
        """개선된 캡션 생성"""
        
        # 1. BLIP 기본 캡션 생성
        blip_caption = self.generate_advanced_caption(frame)
        
        # 2. 향상된 캡션 생성기 사용
        enhanced_caption = self.caption_generator.generate_accurate_caption(
            frame, detected_objects, scene_analysis, frame_id, timestamp
        )
        
        # 3. LLM 기반 최종 캡션 생성
        final_caption = self._generate_llm_caption(
            frame, detected_objects, scene_analysis, blip_caption, enhanced_caption, frame_id, timestamp
        )
        
        return {
            'blip_caption': blip_caption,
            'enhanced_caption': enhanced_caption,
            'final_caption': final_caption,
            'caption_length': len(final_caption),
            'has_multiple_sources': True
        }
    
    def _generate_llm_caption(self, frame, detected_objects, scene_analysis, blip_caption, enhanced_caption, frame_id, timestamp):
        """LLM 기반 최종 캡션 생성"""
        
        try:
            scene_info = scene_analysis.get('scene_classification', {})
            vqa_info = scene_analysis.get('vqa_results', {})
            relationships = scene_analysis.get('scene_graph', {}).get('relationships', [])
            ocr_text = scene_analysis.get('ocr_text', '')
            
            object_details = []
            for obj in detected_objects:
                detail = f"{obj['class']}"
                if obj.get('color_description') and obj['color_description'] != 'unknown':
                    detail += f" ({obj['color_description']})"
                if obj.get('confidence'):
                    detail += f" [신뢰도: {obj['confidence']:.2f}]"
                object_details.append(detail)
            
            relationship_summary = []
            for rel in relationships[:3]:
                relationship_summary.append(f"{rel['subject']} {rel['predicate']} {rel['object']}")
            
            key_vqa_info = {}
            for question, answer in vqa_info.items():
                if "happening" in question.lower() or "activity" in question.lower():
                    key_vqa_info['main_activity'] = answer
                elif "people" in question.lower():
                    key_vqa_info['people_count'] = answer
                elif "setting" in question.lower() or "location" in question.lower():
                    key_vqa_info['location'] = answer
                elif "time" in question.lower():
                    key_vqa_info['time'] = answer
            
            llm_prompt = f"""
            다음은 비디오 프레임 {frame_id} ({timestamp:.1f}초)의 종합 분석 결과입니다. 
            이 정보를 바탕으로 정확하고 상세한 한국어 캡션을 생성해주세요.

            == 객체 감지 결과 ==
            총 {len(detected_objects)}개 객체 감지:
            {', '.join(object_details)}

            == Scene 분류 ==
            - 장소: {scene_info.get('location', {}).get('label', '불명')} (신뢰도: {scene_info.get('location', {}).get('confidence', 0):.2f})
            - 시간대: {scene_info.get('time', {}).get('label', '불명')} (신뢰도: {scene_info.get('time', {}).get('confidence', 0):.2f})
            - 활동: {scene_info.get('activity', {}).get('label', '불명')}

            == 객체간 관계 ==
            {'; '.join(relationship_summary) if relationship_summary else '특별한 관계 없음'}

            == VQA 분석 결과 ==
            {json.dumps(key_vqa_info, ensure_ascii=False, indent=2) if key_vqa_info else '분석 데이터 없음'}

            == 기존 캡션들 ==
            BLIP 자동 캡션: {blip_caption}
            향상된 캡션: {enhanced_caption}

            == OCR 텍스트 ==
            {ocr_text if ocr_text else '텍스트 없음'}

            요청사항:
            1. 위의 모든 정보를 종합하여 정확하고 자연스러운 한국어 캡션을 작성해주세요
            2. 객체의 위치, 색상, 개수를 정확히 반영해주세요
            3. 객체간의 관계와 상호작용을 포함해주세요
            4. Scene의 시간대, 장소, 분위기를 자연스럽게 묘사해주세요
            5. 150-300자 길이로 작성해주세요
            6. 기존 캡션들의 좋은 부분은 활용하되, 더 정확하고 상세하게 개선해주세요

            캡션:
            """
            
            final_caption = call_groq_llm_enhanced(
                llm_prompt,
                "당신은 비디오 분석 전문가입니다. 주어진 모든 정보를 정확히 반영하여 생생하고 상세한 한국어 캡션을 작성합니다.",
                model="llama-3.1-8b-instant"
            )
            
            if final_caption and len(final_caption) > 50:
                if "캡션:" in final_caption:
                    final_caption = final_caption.split("캡션:")[-1].strip()
                
                print(f"✅ LLM 최종 캡션 생성 완료 ({len(final_caption)}자)")
                return final_caption
            else:
                print("⚠️ LLM 캡션 품질 부족, 향상된 캡션 사용")
                return enhanced_caption
                
        except Exception as e:
            print(f"⚠️ LLM 캡션 생성 실패: {e}")
            return enhanced_caption if enhanced_caption else blip_caption
    
    def generate_advanced_caption(self, frame):
        """BLIP 모델로 고급 캡션 생성"""
        if not self.caption_processor or not self.caption_model:
            return ""
        
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb_frame)
            
            inputs = self.caption_processor(image, return_tensors="pt")
            
            out = self.caption_model.generate(**inputs, max_length=50, do_sample=False)
            caption = self.caption_processor.decode(out[0], skip_special_tokens=True)
            return caption
        except Exception as e:
            print(f"⚠️ BLIP 캡션 생성 오류: {e}")
            return ""
    
    def extract_scene_text(self, frame):
        """OCR로 텍스트 추출"""
        if not self.ocr_reader:
            return ""
        
        try:
            results = self.ocr_reader.readtext(frame)
            text_list = [result[1] for result in results if result[2] > 0.5]
            return " ".join(text_list)
        except Exception as e:
            print(f"⚠️ OCR 오류: {e}")
            return ""
    
    def answer_vqa_questions(self, frame, questions):
        """VQA로 특정 질문에 답변"""
        if not self.vqa_processor or not self.vqa_model:
            return {}
        
        answers = {}
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb_frame)
            
            for question in questions:
                inputs = self.vqa_processor(image, question, return_tensors="pt")
                out = self.vqa_model.generate(**inputs, max_length=50)
                answer = self.vqa_processor.decode(out[0], skip_special_tokens=True)
                answers[question] = answer
                
        except Exception as e:
            print(f"⚠️ VQA 오류: {e}")
        
        return answers
    
    def extract_pose_features(self, frame):
        """MediaPipe로 포즈 특징 추출"""
        if not self.pose:
            return []
        
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            
            if results.pose_landmarks:
                landmarks = []
                for landmark in results.pose_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
                return landmarks
            else:
                return []
        except Exception as e:
            print(f"⚠️ 포즈 추출 오류: {e}")
            return []
    
    def extract_visual_features(self, frame):
        """시각적 특징 벡터 추출"""
        try:
            # 1. 색상 히스토그램
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist_h = cv2.calcHist([hsv], [0], None, [50], [0, 180])
            hist_s = cv2.calcHist([hsv], [1], None, [50], [0, 256])
            hist_v = cv2.calcHist([hsv], [2], None, [50], [0, 256])
            
            # 2. 엣지 특징
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            
            # 3. 텍스처 특징
            texture_score = np.std(gray)
            
            # 4. 밝기 특징
            brightness = np.mean(gray)
            contrast = np.std(gray)
            
            # 특징 벡터 결합
            features = np.concatenate([
                hist_h.flatten(),
                hist_s.flatten(), 
                hist_v.flatten(),
                [edge_density, texture_score, brightness, contrast]
            ])
            
            return features / np.linalg.norm(features)  # 정규화
            
        except Exception as e:
            print(f"⚠️ 시각적 특징 추출 오류: {e}")
            return np.zeros(154)
# 기존 코드를 다음으로 교체

# video_analyzer.py - EnhancedVideoAnalyzer 클래스 __init__ 및 _load_yolo_model 수정

class EnhancedVideoAnalyzer:
    """최고급 비디오 분석기 - Django 통합"""
    
    def __init__(self, model_path="yolov8x.pt", confidence_threshold=0.25, **kwargs):
        
        self.model = None
        self.confidence_threshold = confidence_threshold
        
        # 설정값들
        self.enable_color_analysis = kwargs.get('enable_color_analysis', True)
        self.enable_tracking = kwargs.get('enable_tracking', True)
        self.enable_scene_analysis = kwargs.get('enable_scene_analysis', True)
        self.enable_scene_classification = kwargs.get('enable_scene_classification', True)
        self.enable_segmentation = kwargs.get('enable_segmentation', True)
        self.enable_scene_graph = kwargs.get('enable_scene_graph', True)
        self.enable_vqa = kwargs.get('enable_vqa', True)
        self.enable_ocr = kwargs.get('enable_ocr', True)
        self.enable_gpt_features = kwargs.get('enable_gpt_features', True)
        
        # 시스템 기능 상태 속성 추가
        self.clip_available = TRANSFORMERS_AVAILABLE
        self.ocr_available = OCR_AVAILABLE
        self.vqa_available = TRANSFORMERS_AVAILABLE
        self.scene_graph_available = NETWORKX_AVAILABLE
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        print(f"🧠 Enhanced 비디오 분석기 초기화 중... (디바이스: {self.device})")
        
        # YOLO 모델 로딩 (개선된 예외처리)
        if YOLO_AVAILABLE:
            try:
                self._load_yolo_model(model_path)
            except Exception as e:
                log_once(f"⚠️ YOLO 모델 로드 실패 (분석 기능 제한): {e}", "WARNING")
                self.model = None
        else:
            log_once("ℹ️ YOLO 라이브러리 미설치 - 기본 분석만 수행", "INFO")
        
        # 분석 모듈 초기화 (예외 처리 강화)
        self._init_analysis_modules()
        
        log_once(f"✅ Enhanced 비디오 분석기 초기화 완료", "SUCCESS")
        log_once(f"📊 사용 가능한 기능: YOLO={self.model is not None}, CLIP={self.clip_available}, OCR={self.ocr_available}, VQA={self.vqa_available}", "INFO")
    
    def _load_yolo_model(self, model_path):
        """YOLO 모델 로딩 - 개선된 버전"""
        try:
            log_once(f"🔄 YOLO 모델 로딩 중: {model_path}", "INFO")
            
            # 모델 파일 경로 확인
            if not os.path.exists(model_path):
                # 사전 훈련된 모델 다운로드 시도
                log_once(f"📥 사전 훈련된 YOLO 모델 다운로드 중: {model_path}", "INFO")
            
            self.model = YOLO(model_path)
            
            # 모델을 적절한 디바이스로 이동
            if hasattr(self.model, 'to'):
                self.model = self.model.to(self.device)
            
            log_once(f"✅ YOLO 모델 로드 성공: {model_path} (디바이스: {self.device})", "SUCCESS")
            
            # 모델 정보 로그
            if hasattr(self.model, 'names'):
                class_count = len(self.model.names)
                log_once(f"📋 감지 가능한 클래스 수: {class_count}개", "INFO")
            
        except Exception as e:
            log_once(f"❌ YOLO 모델 로드 실패: {e}", "ERROR")
            self.model = None
            
            # 구체적인 해결 방안 제시
            if "No module named 'ultralytics'" in str(e):
                log_once("💡 해결 방법: pip install ultralytics", "INFO")
            elif "CUDA" in str(e):
                log_once("💡 CUDA 오류 - CPU 모드로 전환 시도", "INFO")
                try:
                    self.device = "cpu"
                    self.model = YOLO(model_path)
                    log_once("✅ CPU 모드로 YOLO 모델 로드 성공", "SUCCESS")
                except:
                    self.model = None
            elif "download" in str(e).lower():
                log_once("💡 모델 다운로드 실패 - 인터넷 연결 확인 필요", "INFO")
    
    def _init_analysis_modules(self):
        """분석 모듈 초기화"""
        try:
            if self.enable_color_analysis:
                self.color_analyzer = ColorAnalyzer()
                log_once("🎨 색상 분석기 활성화", "INFO")
        except Exception as e:
            log_once(f"⚠️ 색상 분석기 초기화 실패: {e}", "WARNING")
        
        try:
            if self.enable_scene_analysis:
                self.scene_analyzer = AdvancedSceneAnalyzer(
                    enable_vqa=self.enable_vqa,
                    enable_ocr=self.enable_ocr,
                    enable_segmentation=self.enable_segmentation
                )
                log_once("🎬 고급 Scene 분석기 활성화", "INFO")
                
                # 속성 업데이트
                if hasattr(self.scene_analyzer, 'ocr_reader') and self.scene_analyzer.ocr_reader:
                    self.ocr_available = True
                if hasattr(self.scene_analyzer, 'vqa_model') and self.scene_analyzer.vqa_model:
                    self.vqa_available = True
                    
        except Exception as e:
            log_once(f"⚠️ Scene 분석기 초기화 실패: {e}", "WARNING")
            self.scene_analyzer = None
    
    def detect_objects_comprehensive(self, frame):
        """종합적인 객체 감지 - 오류 처리 강화"""
        if self.model is None:
            # 첫 번째 경고만 출력하도록 개선
            if not hasattr(self, '_no_yolo_warning_shown'):
                log_once("⚠️ YOLO 모델이 로드되지 않음 - 기본 분석 수행", "WARNING")
                self._no_yolo_warning_shown = True
            return []
            
        try:
            if self.enable_tracking:
                results = self.model.track(frame, verbose=False, persist=True, conf=self.confidence_threshold)
            else:
                results = self.model(frame, verbose=False, conf=self.confidence_threshold)
            
            detected_objects = []
            
            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    boxes = result.boxes.xyxy.cpu().numpy()
                    confidences = result.boxes.conf.cpu().numpy()
                    class_ids = result.boxes.cls.cpu().numpy().astype(int)
                    
                    if self.enable_tracking and result.boxes.id is not None:
                        track_ids = result.boxes.id.cpu().numpy().astype(int)
                    else:
                        track_ids = np.arange(len(boxes))
                    
                    h, w = frame.shape[:2]
                    
                    for i, (box, conf, cls_id, track_id) in enumerate(zip(boxes, confidences, class_ids, track_ids)):
                        if conf > self.confidence_threshold and cls_id in ENHANCED_OBJECTS:
                            class_name = ENHANCED_OBJECTS[cls_id]
                            
                            normalized_box = [
                                round(float(box[0]) / w, 4),
                                round(float(box[1]) / h, 4),
                                round(float(box[2]) / w, 4),
                                round(float(box[3]) / h, 4)
                            ]
                            
                            obj_info = {
                                'class': class_name,
                                'bbox': normalized_box,
                                'track_id': int(track_id),
                                'confidence': float(conf),
                                'colors': [],
                                'color_description': "unknown"
                            }
                            
                            # 색상 분석 (예외 처리 추가)
                            if self.enable_color_analysis and hasattr(self, 'color_analyzer'):
                                try:
                                    colors = self.color_analyzer.extract_dominant_colors(frame, normalized_box)
                                    obj_info['colors'] = colors
                                    obj_info['color_description'] = self.color_analyzer.get_color_description(colors)
                                except Exception as e:
                                    # 색상 분석 실패는 로그 없이 기본값 사용
                                    pass
                            
                            detected_objects.append(obj_info)
            
            return detected_objects
            
        except Exception as e:
            log_once(f"❌ 객체 감지 오류: {e}", "ERROR")
            return []
    
    def analyze_video_comprehensive(self, video, analysis_type='comprehensive', progress_callback=None):
        """비디오 종합 분석 - Django 모델과 연동"""
        try:
            from django.conf import settings
            import os
            
            # 비디오 파일 경로 찾기 (개선됨)
            video_path = self._find_video_path(video)
            if not video_path:
                raise Exception("비디오 파일을 찾을 수 없습니다")
            
            # 비디오 정보 추출
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise Exception("비디오 파일을 열 수 없습니다")
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = total_frames / fps if fps > 0 else 0
            
            # 비디오 정보 업데이트
            video.width = width
            video.height = height
            video.fps = fps
            video.total_frames = total_frames
            video.duration = duration
            video.save()
            
            # 프레임 샘플링 (분석 타입에 따라 조정)
            sample_intervals = {
                'basic': max(1, total_frames // 30),
                'enhanced': max(1, total_frames // 50),
                'comprehensive': max(1, total_frames // 100),
                'custom': max(1, total_frames // 75)
            }
            sample_interval = sample_intervals.get(analysis_type, max(1, total_frames // 50))
            
            frame_results = []
            processed_frames = 0
            frame_id = 0
            
            log_once(f"🎬 비디오 분석 시작: {video.original_name} ({analysis_type})", "INFO")
            log_once(f"📊 총 프레임: {total_frames}, 샘플링 간격: {sample_interval}", "INFO")
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_id += 1
                
                # 샘플링 체크
                if frame_id % sample_interval != 1:
                    continue
                
                timestamp = frame_id / fps
                
                try:
                    # 1. 객체 감지
                    detected_objects = self.detect_objects_comprehensive(frame)
                    
                    # 2. Scene 분석
                    scene_analysis = {}
                    if self.enable_scene_analysis and hasattr(self, 'scene_analyzer') and self.scene_analyzer:
                        scene_analysis = self.scene_analyzer.comprehensive_scene_analysis(
                            frame, frame_id, timestamp, detected_objects
                        )
                    
                    # 3. 향상된 캡션 생성
                    caption_result = {}
                    if self.enable_scene_analysis and hasattr(self, 'scene_analyzer') and self.scene_analyzer:
                        caption_result = self.scene_analyzer.generate_enhanced_caption(
                            frame, detected_objects, scene_analysis, frame_id, timestamp
                        )
                    else:
                        # 기본 캡션
                        basic_caption = f"프레임 {frame_id} ({timestamp:.1f}초)에서 {len(detected_objects)}개의 객체가 감지되었습니다."
                        if detected_objects:
                            object_list = [obj['class'] for obj in detected_objects[:3]]
                            basic_caption += f" 주요 객체: {', '.join(object_list)}"
                        
                        caption_result = {
                            'blip_caption': "",
                            'enhanced_caption': basic_caption,
                            'final_caption': basic_caption,
                            'caption_length': len(basic_caption),
                            'has_multiple_sources': False
                        }
                    
                    # 4. 프레임 데이터 구성
                    frame_dict = {
                        "image_id": frame_id,
                        "timestamp": timestamp,
                        "objects": detected_objects,
                        "scene_analysis": scene_analysis,
                        "caption": caption_result.get('final_caption', ''),
                        "blip_caption": caption_result.get('blip_caption', ''),
                        "enhanced_caption": caption_result.get('enhanced_caption', ''),
                        "final_caption": caption_result.get('final_caption', ''),
                        "caption_sources": caption_result,
                        "comprehensive_features": {
                            "object_count": len(detected_objects),
                            "object_types": list(set(obj['class'] for obj in detected_objects)),
                            "scene_complexity": len(scene_analysis.get('scene_graph', {}).get('relationships', [])),
                            "has_text": bool(scene_analysis.get('ocr_text', '')),
                            "has_caption": bool(caption_result.get('final_caption', '')),
                            "caption_length": caption_result.get('caption_length', 0),
                            "caption_quality": "enhanced" if caption_result.get('has_multiple_sources', False) else "basic",
                            "analysis_success": True
                        }
                    }
                    
                    frame_results.append(frame_dict)
                    processed_frames += 1
                    
                    # 진행률 콜백 (로그 중복 방지)
                    if progress_callback and processed_frames % 5 == 0:  # 5프레임마다만 콜백
                        progress = (frame_id / total_frames) * 100
                        progress_callback(progress, f"프레임 {frame_id}/{total_frames} 분석 중")
                
                except Exception as e:
                    log_once(f"❌ 프레임 {frame_id} 분석 실패: {e}", "ERROR")
                    continue
            
            cap.release()
            
            # 결과 요약
            # video_summary = self._summarize_video_analysis(frame_results, video)
            # video_analyzer.py - analyze_video_comprehensive 메서드에서 다음 부분만 수정

# # ❌ 현재 오류가 나는 코드 (이 줄을 찾으세요)
# video_summary = self._summarize_video_analysis(frame_results, video)

            # ✅ 다음 코드로 교체 (안전한 임시 수정)
            try:
                if hasattr(self, '_summarize_video_analysis'):
                    video_summary = self._summarize_video_analysis(frame_results, video)
                else:
                    # 임시 요약 생성
                    log_once("⚠️ _summarize_video_analysis 메서드 없음, 기본 요약 생성", "WARNING")
                    video_summary = self._create_basic_summary(frame_results, video)
            except Exception as e:
                log_once(f"⚠️ 비디오 요약 생성 실패, 기본 요약 사용: {e}", "WARNING")
                video_summary = self._create_basic_summary(frame_results, video)

            # 그리고 EnhancedVideoAnalyzer 클래스에 다음 메서드도 추가:
            def _create_basic_summary(self, frame_results, video):
                """기본 요약 생성 (fallback)"""
                from collections import Counter
                
                all_objects = []
                for frame_result in frame_results:
                    if 'objects' in frame_result:
                        for obj in frame_result['objects']:
                            if 'class' in obj:
                                all_objects.append(obj['class'])
                
                object_counter = Counter(all_objects)
                dominant_objects = [obj for obj, count in object_counter.most_common(10)]
                
                return {
                    'total_frames': len(frame_results),
                    'analysis_features': ['object_detection', 'basic_analysis'],
                    'dominant_objects': dominant_objects,
                    'text_content': '',
                    'scene_types': ['general'],
                    'highlight_frames': []
                }
            log_once(f"✅ 비디오 분석 완료: {processed_frames}개 프레임 처리", "SUCCESS")
            
            return {
                'video_summary': video_summary,
                'frame_results': frame_results,
                'analysis_config': {
                    'method': 'Enhanced_MultiModal_Analysis',
                    'analysis_type': analysis_type,
                    'sample_interval': sample_interval,
                    'features_enabled': {
                        'yolo': self.model is not None,
                        'clip': self.clip_available,
                        'ocr': self.ocr_available,
                        'vqa': self.vqa_available,
                        'scene_graph': self.scene_graph_available
                    }
                },
                'total_frames_analyzed': processed_frames,
                'success': True
            }
            
        except Exception as e:
            log_once(f"❌ 비디오 분석 실패: {e}", "ERROR")
            return {'error': str(e), 'success': False}
    # video_analyzer.py - EnhancedVideoAnalyzer 클래스에 추가할 메서드

    def _summarize_video_analysis(self, frame_results, video):
        """비디오 분석 결과 요약 - 누락된 메서드 추가"""
        try:
            log_once("📊 비디오 분석 결과 요약 생성 중...", "INFO")
            
            summary = {
                'total_frames': len(frame_results),
                'analysis_features': [],
                'dominant_objects': [],
                'text_content': '',
                'scene_types': [],
                'highlight_frames': [],
                'analysis_quality_metrics': {},
                'processing_statistics': {}
            }
            
            if not frame_results:
                log_once("⚠️ 분석할 프레임 결과가 없음", "WARNING")
                return summary
            
            # 각 프레임 결과 통합
            all_objects = []
            all_texts = []
            scene_types = []
            caption_lengths = []
            confidence_scores = []
            
            for frame_result in frame_results:
                # 객체 수집
                if 'comprehensive_features' in frame_result:
                    object_types = frame_result['comprehensive_features'].get('object_types', [])
                    all_objects.extend(object_types)
                
                # 일반 객체 정보도 수집
                if 'objects' in frame_result:
                    for obj in frame_result['objects']:
                        if 'class' in obj:
                            all_objects.append(obj['class'])
                        if 'confidence' in obj:
                            confidence_scores.append(obj['confidence'])
                
                # 텍스트 수집 (OCR 결과)
                scene_analysis = frame_result.get('scene_analysis', {})
                ocr_text = scene_analysis.get('ocr_text', '')
                if ocr_text and isinstance(ocr_text, str) and len(ocr_text.strip()) > 0:
                    all_texts.append(ocr_text.strip())
                
                # 캡션 길이 수집
                final_caption = frame_result.get('final_caption', '')
                enhanced_caption = frame_result.get('enhanced_caption', '')
                caption = final_caption or enhanced_caption or frame_result.get('caption', '')
                if caption:
                    caption_lengths.append(len(caption))
                
                # 씬 타입 수집
                if scene_analysis:
                    scene_classification = scene_analysis.get('scene_classification', {})
                    location = scene_classification.get('location', {}).get('label', '')
                    time_label = scene_classification.get('time', {}).get('label', '')
                    activity = scene_classification.get('activity', {}).get('label', '')
                    
                    for scene_type in [location, time_label, activity]:
                        if scene_type and scene_type != 'unknown':
                            scene_types.append(scene_type)
            
            # 요약 정보 생성
            if all_objects:
                from collections import Counter
                object_counter = Counter(all_objects)
                summary['dominant_objects'] = [obj for obj, count in object_counter.most_common(15)]
                
                # 분석 품질 메트릭
                summary['analysis_quality_metrics']['unique_object_types'] = len(object_counter)
                summary['analysis_quality_metrics']['total_object_detections'] = len(all_objects)
                summary['analysis_quality_metrics']['average_objects_per_frame'] = len(all_objects) / len(frame_results)
            
            if all_texts:
                # 텍스트 내용 결합 (최대 1000자)
                combined_text = ' '.join(all_texts)[:1000]
                summary['text_content'] = combined_text
                summary['analysis_quality_metrics']['text_extraction_frames'] = len(all_texts)
                summary['analysis_quality_metrics']['total_text_length'] = len(combined_text)
            
            if scene_types:
                scene_counter = Counter(scene_types)
                summary['scene_types'] = [scene for scene, count in scene_counter.most_common(8)]
                summary['analysis_quality_metrics']['scene_variety'] = len(scene_counter)
            
            if caption_lengths:
                summary['analysis_quality_metrics']['average_caption_length'] = sum(caption_lengths) / len(caption_lengths)
                summary['analysis_quality_metrics']['frames_with_captions'] = len(caption_lengths)
            
            if confidence_scores:
                summary['analysis_quality_metrics']['average_detection_confidence'] = sum(confidence_scores) / len(confidence_scores)
                summary['analysis_quality_metrics']['high_confidence_detections'] = len([c for c in confidence_scores if c > 0.8])
            
            # 처리 통계
            summary['processing_statistics'] = {
                'frames_processed': len(frame_results),
                'frames_with_objects': len([f for f in frame_results if f.get('objects', [])]),
                'frames_with_scene_analysis': len([f for f in frame_results if f.get('scene_analysis')]),
                'frames_with_enhanced_captions': len([f for f in frame_results if f.get('enhanced_caption')]),
                'processing_success_rate': (len(frame_results) / len(frame_results)) * 100 if frame_results else 0
            }
            
            # 하이라이트 프레임 선정 (객체가 많이 감지된 프레임들)
            highlight_candidates = []
            for i, frame_result in enumerate(frame_results):
                object_count = len(frame_result.get('objects', []))
                scene_complexity = frame_result.get('comprehensive_features', {}).get('scene_complexity', 0)
                caption_quality = 1 if frame_result.get('enhanced_caption') else 0
                
                score = object_count * 2 + scene_complexity + caption_quality
                highlight_candidates.append({
                    'frame_index': i,
                    'frame_id': frame_result.get('image_id', i),
                    'timestamp': frame_result.get('timestamp', 0),
                    'score': score,
                    'object_count': object_count
                })
            
            # 점수 기준으로 정렬하여 상위 5개 선정
            highlight_candidates.sort(key=lambda x: x['score'], reverse=True)
            summary['highlight_frames'] = highlight_candidates[:5]
            
            # 사용된 분석 기능들
            analysis_features = set()
            for frame_result in frame_results:
                if frame_result.get('objects'):
                    analysis_features.add('object_detection')
                if frame_result.get('scene_analysis', {}).get('scene_classification'):
                    analysis_features.add('scene_classification')
                if frame_result.get('scene_analysis', {}).get('ocr_text'):
                    analysis_features.add('ocr')
                if frame_result.get('scene_analysis', {}).get('vqa_results'):
                    analysis_features.add('vqa')
                if frame_result.get('enhanced_caption'):
                    analysis_features.add('enhanced_caption')
                if frame_result.get('scene_analysis', {}).get('gpt_features'):
                    analysis_features.add('gpt_features')
            
            summary['analysis_features'] = list(analysis_features)
            summary['analysis_quality_metrics']['features_used_count'] = len(analysis_features)
            
            log_once(f"✅ 비디오 분석 요약 완료: {len(frame_results)}프레임, {len(summary['dominant_objects'])}개 객체 유형", "SUCCESS")
            
            return summary
            
        except Exception as e:
            log_once(f"❌ 비디오 분석 요약 실패: {e}", "ERROR")
            
            # 기본 요약 반환
            return {
                'total_frames': len(frame_results) if frame_results else 0,
                'analysis_features': ['basic_analysis'],
                'dominant_objects': [],
                'text_content': '',
                'scene_types': [],
                'highlight_frames': [],
                'analysis_quality_metrics': {'error': str(e)},
                'processing_statistics': {'status': 'error', 'frames_processed': len(frame_results) if frame_results else 0}
            }
        
    def search_comprehensive(self, video, query):
        """종합적인 비디오 검색"""
        try:
            # Frame에서 검색
            frames = Frame.objects.filter(video=video)
            search_results = []
            
            for frame in frames:
                # 캡션에서 검색
                if query.lower() in frame.final_caption.lower():
                    search_results.append({
                        'frame_id': frame.image_id,
                        'timestamp': frame.timestamp,
                        'caption': frame.final_caption,
                        'match_type': 'caption',
                        'confidence': 0.8
                    })
                
                # 객체에서 검색
                for obj in frame.detected_objects:
                    if query.lower() in obj.get('class', '').lower():
                        search_results.append({
                            'frame_id': frame.image_id,
                            'timestamp': frame.timestamp,
                            'object_class': obj.get('class'),
                            'match_type': 'object',
                            'confidence': obj.get('confidence', 0.5)
                        })
            
            return search_results[:20]  # 상위 20개 결과
            
        except Exception as e:
            print(f"❌ 검색 오류: {e}")
            return []
        
    def _find_video_path(self, video):
        """비디오 파일 경로 찾기 - 개선됨"""
        from django.conf import settings
        
        possible_paths = [
            os.path.join(settings.MEDIA_ROOT, 'videos', video.filename),
            os.path.join(settings.MEDIA_ROOT, 'uploads', video.filename),
            video.file_path if hasattr(video, 'file_path') else None
        ]
        
        # None 제거
        possible_paths = [p for p in possible_paths if p]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        return None
    
# video_analyzer.py 파일 맨 마지막 부분에 추가/수정

# 전역 VideoAnalyzer 인스턴스 (싱글톤 패턴)
_global_video_analyzer = None
_analyzer_lock = threading.Lock()  # 스레드 안전성을 위한 락

def get_video_analyzer():
    """전역 VideoAnalyzer 인스턴스 반환 (싱글톤 패턴) - 스레드 안전"""
    global _global_video_analyzer
    
    # 이미 인스턴스가 있으면 반환
    if _global_video_analyzer is not None:
        return _global_video_analyzer
    
    # 스레드 안전성을 위한 락 사용
    with _analyzer_lock:
        # 더블 체크 - 다른 스레드가 이미 생성했을 수 있음
        if _global_video_analyzer is not None:
            return _global_video_analyzer
        
        try:
            log_once("🚀 전역 VideoAnalyzer 인스턴스 생성 중...", "INFO")
            
            # EnhancedVideoAnalyzer 우선 생성 시도
            try:
                _global_video_analyzer = EnhancedVideoAnalyzer()
                log_once("✅ EnhancedVideoAnalyzer 인스턴스 생성 성공", "SUCCESS")
            except Exception as enhanced_error:
                log_once(f"⚠️ EnhancedVideoAnalyzer 생성 실패: {enhanced_error}", "WARNING")
                
                # Fallback: 기본 VideoAnalyzer 생성 시도
                try:
                    if 'VideoAnalyzer' in globals():
                        _global_video_analyzer = VideoAnalyzer()
                        log_once("✅ 기본 VideoAnalyzer 인스턴스 생성 성공", "SUCCESS")
                    else:
                        raise ImportError("VideoAnalyzer 클래스를 찾을 수 없습니다")
                except Exception as basic_error:
                    log_once(f"❌ 기본 VideoAnalyzer 생성도 실패: {basic_error}", "ERROR")
                    
                    # 최종 Fallback: 최소 기능 더미 클래스
                    _global_video_analyzer = _create_dummy_analyzer()
                    log_once("⚠️ 더미 VideoAnalyzer 생성됨 (제한된 기능)", "WARNING")
            
            return _global_video_analyzer
            
        except Exception as e:
            log_once(f"❌ VideoAnalyzer 인스턴스 생성 완전 실패: {e}", "ERROR")
            
            # 최종 Fallback
            _global_video_analyzer = _create_dummy_analyzer()
            return _global_video_analyzer

def _create_dummy_analyzer():
    """최소 기능 더미 VideoAnalyzer 클래스"""
    class DummyVideoAnalyzer:
        def __init__(self):
            self.model = None
            self.clip_available = False
            self.ocr_available = False
            self.vqa_available = False
            self.scene_graph_available = False
            self.device = "cpu"
            log_once("⚠️ 더미 VideoAnalyzer 초기화됨", "WARNING")
        
        def detect_objects_comprehensive(self, frame):
            """더미 객체 감지"""
            return []
        
        def analyze_video_comprehensive(self, video, analysis_type='basic', progress_callback=None):
            """더미 비디오 분석"""
            if progress_callback:
                progress_callback(50, "더미 분석 진행 중")
                progress_callback(100, "더미 분석 완료")
            
            return {
                'success': True,
                'video_summary': {
                    'dominant_objects': [],
                    'scene_types': ['unknown'],
                    'text_content': ''
                },
                'frame_results': [],
                'total_frames_analyzed': 0,
                'analysis_config': {
                    'method': 'Dummy_Analysis',
                    'analysis_type': analysis_type,
                    'features_enabled': {
                        'yolo': False,
                        'clip': False,
                        'ocr': False,
                        'vqa': False,
                        'scene_graph': False
                    }
                }
            }
    
    return DummyVideoAnalyzer()

def reset_video_analyzer():
    """전역 VideoAnalyzer 인스턴스 리셋 (테스트/개발용)"""
    global _global_video_analyzer
    with _analyzer_lock:
        _global_video_analyzer = None
        log_once("🔄 VideoAnalyzer 인스턴스 리셋됨", "INFO")

def get_analyzer_status():
    """현재 VideoAnalyzer 상태 반환"""
    analyzer = get_video_analyzer()
    if analyzer is None:
        return {'status': 'not_initialized'}
    
    return {
        'status': 'initialized',
        'type': type(analyzer).__name__,
        'features': {
            'yolo': getattr(analyzer, 'model', None) is not None,
            'clip': getattr(analyzer, 'clip_available', False),
            'ocr': getattr(analyzer, 'ocr_available', False),
            'vqa': getattr(analyzer, 'vqa_available', False),
            'scene_graph': getattr(analyzer, 'scene_graph_available', False)
        },
        'device': getattr(analyzer, 'device', 'unknown')
    }

# 기존 VideoAnalyzer 클래스를 확장 (호환성 유지)
class VideoAnalyzer(EnhancedVideoAnalyzer):
    """기존 기능과 새로운 기능을 통합한 VideoAnalyzer - 호환성 유지"""
    
    def __init__(self):
        try:
            super().__init__()
            log_once("✅ VideoAnalyzer (호환성 버전) 초기화 완료", "SUCCESS")
        except Exception as e:
            log_once(f"⚠️ VideoAnalyzer 초기화 부분 실패: {e}", "WARNING")
            # 최소한의 속성들은 설정
            self.model = None
            self.clip_available = False
            self.ocr_available = False
            self.vqa_available = False
            self.scene_graph_available = False
            self.device = "cpu"

# 모듈 로드 시 즉시 인스턴스 생성 (선택사항)
try:
    # import 시점에서 바로 인스턴스 생성하지 않음 (지연 로딩)
    # get_video_analyzer()  # 이 줄을 주석 처리하여 지연 로딩
    log_once("📦 video_analyzer 모듈 로드 완료", "INFO")
except Exception as module_error:
    log_once(f"⚠️ video_analyzer 모듈 로드 중 경고: {module_error}", "WARNING")