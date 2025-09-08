# chat/multi_llm_service.py - 완전 구현 버전
import asyncio
import json
import time
import os
import base64
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import cv2
import numpy as np
from PIL import Image
import io

# API 클라이언트들 - 안전한 import
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("⚠️ OpenAI 라이브러리 미설치")

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("⚠️ Anthropic 라이브러리 미설치")

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("⚠️ Groq 라이브러리 미설치")

try:
    import google.generativeai as genai
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False
    print("⚠️ Google AI 라이브러리 미설치")

logger = logging.getLogger(__name__)

@dataclass
class LLMResponse:
    model_name: str
    response_text: str
    confidence_score: float
    processing_time: float
    token_usage: Dict[str, int]
    error: str = None
    success: bool = True

class MultiLLMVideoAnalyzer:
    """멀티 LLM 영상 분석 서비스 - 완전 구현 버전"""
    
    def __init__(self):
        print("🚀 MultiLLMVideoAnalyzer 초기화 중...")
        
        # API 클라이언트 초기화 (안전하게)
        self.openai_client = None
        self.anthropic_client = None
        self.groq_client = None
        self.gemini_model = None
        
        self.available_models = []
        
        # OpenAI 초기화
        if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
            try:
                self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
                self.available_models.append("gpt-4v")
                print("✅ OpenAI GPT-4V 클라이언트 초기화 완료")
            except Exception as e:
                print(f"⚠️ OpenAI 초기화 실패: {e}")
        
        # Anthropic 초기화
        if ANTHROPIC_AVAILABLE and os.getenv("ANTHROPIC_API_KEY"):
            try:
                self.anthropic_client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
                self.available_models.append("claude-3.5")
                print("✅ Anthropic Claude-3.5 클라이언트 초기화 완료")
            except Exception as e:
                print(f"⚠️ Anthropic 초기화 실패: {e}")
        
        # Groq 초기화 (텍스트 전용)
        if GROQ_AVAILABLE and os.getenv("GROQ_API_KEY"):
            try:
                self.groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
                self.available_models.append("groq-llama")
                print("✅ Groq Llama 클라이언트 초기화 완료")
            except Exception as e:
                print(f"⚠️ Groq 초기화 실패: {e}")
        
        # Google Gemini 초기화
        if GOOGLE_AVAILABLE and os.getenv("GOOGLE_API_KEY"):
            try:
                genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
                self.gemini_model = genai.GenerativeModel('gemini-1.5-pro-vision-latest')
                self.available_models.append("gemini-pro")
                print("✅ Google Gemini Pro Vision 클라이언트 초기화 완료")
            except Exception as e:
                print(f"⚠️ Google Gemini 초기화 실패: {e}")
        
        print(f"🎯 사용 가능한 모델: {self.available_models}")
        
        if not self.available_models:
            print("⚠️ 사용 가능한 LLM 모델이 없습니다. 환경 변수를 확인하세요.")
    
    def analyze_video_multi_llm(
        self, 
        frame_images: List[str], 
        question: str, 
        video_context: Dict = None
    ) -> Dict[str, LLMResponse]:
        """
        여러 LLM으로 동시에 영상 분석 (동기 버전)
        """
        print(f"🚀 멀티 LLM 분석 시작: '{question}'")
        print(f"📊 분석할 프레임 수: {len(frame_images)}")
        print(f"🤖 사용 가능한 모델: {self.available_models}")
        
        if not self.available_models:
            return {
                "error": LLMResponse(
                    model_name="system",
                    response_text="사용 가능한 LLM 모델이 없습니다.",
                    confidence_score=0.0,
                    processing_time=0.0,
                    token_usage={},
                    error="No models available",
                    success=False
                )
            }
        
        results = {}
        start_time = time.time()
        
        # 각 모델별 분석 실행
        with ThreadPoolExecutor(max_workers=len(self.available_models)) as executor:
            future_to_model = {}
            
            # GPT-4V
            if "gpt-4v" in self.available_models:
                future_to_model[executor.submit(self._analyze_with_gpt4v, frame_images, question, video_context)] = "gpt-4v"
            
            # Claude-3.5
            if "claude-3.5" in self.available_models:
                future_to_model[executor.submit(self._analyze_with_claude, frame_images, question, video_context)] = "claude-3.5"
            
            # Groq (텍스트 기반)
            if "groq-llama" in self.available_models:
                future_to_model[executor.submit(self._analyze_with_groq, frame_images, question, video_context)] = "groq-llama"
            
            # Gemini Pro
            if "gemini-pro" in self.available_models:
                future_to_model[executor.submit(self._analyze_with_gemini, frame_images, question, video_context)] = "gemini-pro"
            
            # 결과 수집
            for future in as_completed(future_to_model, timeout=60):
                model_name = future_to_model[future]
                try:
                    result = future.result(timeout=30)
                    results[model_name] = result
                    print(f"✅ {model_name} 분석 완료 ({result.processing_time:.1f}초)")
                except Exception as e:
                    print(f"❌ {model_name} 분석 실패: {e}")
                    results[model_name] = LLMResponse(
                        model_name=model_name,
                        response_text="",
                        confidence_score=0.0,
                        processing_time=0.0,
                        token_usage={},
                        error=str(e),
                        success=False
                    )
        
        total_time = time.time() - start_time
        print(f"🎯 멀티 LLM 분석 완료: {total_time:.1f}초")
        
        return results
    
    def _analyze_with_gpt4v(self, frame_images, question, video_context):
        """GPT-4V 분석"""
        start_time = time.time()
        
        if not self.openai_client:
            return LLMResponse(
                model_name="gpt-4v",
                response_text="",
                confidence_score=0.0,
                processing_time=0.0,
                token_usage={},
                error="OpenAI client not available",
                success=False
            )
        
        try:
            messages = self._build_gpt_messages(frame_images, question, video_context)
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",  # 비용 절약을 위해 mini 사용
                messages=messages,
                max_tokens=1000,
                temperature=0.7
            )
            
            response_text = response.choices[0].message.content
            processing_time = time.time() - start_time
            
            return LLMResponse(
                model_name="gpt-4v",
                response_text=response_text,
                confidence_score=self._calculate_confidence(response_text),
                processing_time=processing_time,
                token_usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens
                },
                success=True
            )
            
        except Exception as e:
            return LLMResponse(
                model_name="gpt-4v",
                response_text="",
                confidence_score=0.0,
                processing_time=time.time() - start_time,
                token_usage={},
                error=str(e),
                success=False
            )
    
    def _analyze_with_claude(self, frame_images, question, video_context):
        """Claude-3.5 분석"""
        start_time = time.time()
        
        if not self.anthropic_client:
            return LLMResponse(
                model_name="claude-3.5",
                response_text="",
                confidence_score=0.0,
                processing_time=0.0,
                token_usage={},
                error="Anthropic client not available",
                success=False
            )
        
        try:
            content = self._build_claude_content(frame_images, question, video_context)
            
            message = self.anthropic_client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                temperature=0.7,
                messages=[{
                    "role": "user",
                    "content": content
                }]
            )
            
            response_text = message.content[0].text
            processing_time = time.time() - start_time
            
            return LLMResponse(
                model_name="claude-3.5",
                response_text=response_text,
                confidence_score=self._calculate_confidence(response_text),
                processing_time=processing_time,
                token_usage={
                    "input_tokens": message.usage.input_tokens,
                    "output_tokens": message.usage.output_tokens
                },
                success=True
            )
            
        except Exception as e:
            return LLMResponse(
                model_name="claude-3.5",
                response_text="",
                confidence_score=0.0,
                processing_time=time.time() - start_time,
                token_usage={},
                error=str(e),
                success=False
            )
    
    def _analyze_with_groq(self, frame_images, question, video_context):
        """Groq 분석 (텍스트 기반)"""
        start_time = time.time()
        
        if not self.groq_client:
            return LLMResponse(
                model_name="groq-llama",
                response_text="",
                confidence_score=0.0,
                processing_time=0.0,
                token_usage={},
                error="Groq client not available",
                success=False
            )
        
        try:
            # Groq는 이미지를 직접 처리할 수 없으므로 텍스트 설명 기반으로 처리
            prompt = self._build_groq_prompt(question, video_context)
            
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "당신은 영상 분석 전문가입니다. 주어진 정보를 바탕으로 정확하고 상세한 답변을 반드시 한국어로 제공해주세요."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.7
            )
            
            response_text = response.choices[0].message.content
            processing_time = time.time() - start_time
            
            return LLMResponse(
                model_name="groq-llama",
                response_text=response_text,
                confidence_score=self._calculate_confidence(response_text),
                processing_time=processing_time,
                token_usage={
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens
                },
                success=True
            )
            
        except Exception as e:
            return LLMResponse(
                model_name="groq-llama",
                response_text="",
                confidence_score=0.0,
                processing_time=time.time() - start_time,
                token_usage={},
                error=str(e),
                success=False
            )
    
    def _analyze_with_gemini(self, frame_images, question, video_context):
        """Gemini Pro Vision 분석"""
        start_time = time.time()
        
        if not self.gemini_model:
            return LLMResponse(
                model_name="gemini-pro",
                response_text="",
                confidence_score=0.0,
                processing_time=0.0,
                token_usage={},
                error="Gemini model not available",
                success=False
            )
        
        try:
            prompt = self._build_gemini_prompt(question, video_context)
            
            # 첫 번째 프레임만 사용 (Gemini 제한)
            if frame_images:
                image_data = base64.b64decode(frame_images[0])
                
                response = self.gemini_model.generate_content([
                    prompt,
                    {"mime_type": "image/jpeg", "data": image_data}
                ])
                
                response_text = response.text
            else:
                # 이미지가 없으면 텍스트만으로 처리
                response = self.gemini_model.generate_content(prompt)
                response_text = response.text
            
            processing_time = time.time() - start_time
            
            return LLMResponse(
                model_name="gemini-pro",
                response_text=response_text,
                confidence_score=self._calculate_confidence(response_text),
                processing_time=processing_time,
                token_usage={
                    "prompt_tokens": 0,  # Gemini는 상세 토큰 정보 제공 안함
                    "completion_tokens": len(response_text.split())
                },
                success=True
            )
            
        except Exception as e:
            return LLMResponse(
                model_name="gemini-pro",
                response_text="",
                confidence_score=0.0,
                processing_time=time.time() - start_time,
                token_usage={},
                error=str(e),
                success=False
            )
    
    def _build_gpt_messages(self, frame_images, question, video_context):
        """GPT-4V용 메시지 구성"""
        messages = [{
            "role": "system",
            "content": "당신은 영상 분석 전문가입니다. 주어진 영상 프레임들을 분석하고 사용자의 질문에 정확하고 상세하게 답변해주세요."
        }]
        
        content = [{"type": "text", "text": f"영상 분석 질문: {question}"}]
        
        # 비디오 컨텍스트 추가
        if video_context:
            context_text = "\n\n추가 정보:\n"
            if video_context.get('duration'):
                context_text += f"- 영상 길이: {video_context['duration']}초\n"
            if video_context.get('detected_objects'):
                context_text += f"- 감지된 객체: {', '.join(video_context['detected_objects'][:5])}\n"
            if video_context.get('ocr_text'):
                context_text += f"- 추출된 텍스트: {video_context['ocr_text']}\n"
            
            content[0]["text"] += context_text
        
        # 프레임 이미지들 추가 (최대 3개)
        for i, frame_image in enumerate(frame_images[:3]):
            content.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{frame_image}",
                    "detail": "high"
                }
            })
        
        messages.append({"role": "user", "content": content})
        return messages
    
    def _build_claude_content(self, frame_images, question, video_context):
        """Claude용 컨텐츠 구성"""
        content = []
        
        prompt_text = f"영상 분석 질문: {question}\n\n"
        
        if video_context:
            prompt_text += "추가 정보:\n"
            if video_context.get('duration'):
                prompt_text += f"- 영상 길이: {video_context['duration']}초\n"
            if video_context.get('detected_objects'):
                prompt_text += f"- 감지된 객체: {', '.join(video_context['detected_objects'][:5])}\n"
            if video_context.get('ocr_text'):
                prompt_text += f"- 추출된 텍스트: {video_context['ocr_text']}\n"
        
        prompt_text += "\n주어진 영상 프레임들을 분석하고 정확하고 상세한 답변을 해주세요:"
        
        content.append({"type": "text", "text": prompt_text})
        
        # 이미지들 추가 (최대 3개)
        for frame_image in frame_images[:3]:
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": frame_image
                }
            })
        
        return content
    
    def _build_groq_prompt(self, question, video_context):
        """Groq용 프롬프트 구성 (텍스트 기반)"""
        prompt = f"영상 분석 질문: {question}\n\n"
        
        if video_context:
            prompt += "영상 정보:\n"
            if video_context.get('duration'):
                prompt += f"- 영상 길이: {video_context['duration']}초\n"
            if video_context.get('detected_objects'):
                prompt += f"- 감지된 객체들: {', '.join(video_context['detected_objects'][:10])}\n"
            if video_context.get('ocr_text'):
                prompt += f"- 영상 내 텍스트: {video_context['ocr_text']}\n"
            if video_context.get('filename'):
                prompt += f"- 파일명: {video_context['filename']}\n"
        
        prompt += "\n위 정보를 바탕으로 영상에 대한 질문에 답변해주세요. 영상을 직접 볼 수는 없지만, 제공된 정보를 활용하여 최대한 도움이 되는 답변을 제공해주세요."
        
        return prompt
    
    def _build_gemini_prompt(self, question, video_context):
        """Gemini용 프롬프트 구성"""
        prompt = f"영상 분석 질문: {question}\n\n"
        
        if video_context:
            prompt += "추가 정보:\n"
            if video_context.get('duration'):
                prompt += f"- 영상 길이: {video_context['duration']}초\n"
            if video_context.get('detected_objects'):
                prompt += f"- 감지된 객체: {', '.join(video_context['detected_objects'][:5])}\n"
            if video_context.get('ocr_text'):
                prompt += f"- 추출된 텍스트: {video_context['ocr_text']}\n"
        
        prompt += "\n주어진 영상을 분석하고 정확하고 상세한 답변을 해주세요."
        
        return prompt
    
    def _calculate_confidence(self, response_text):
        """응답 텍스트 기반 신뢰도 계산"""
        if not response_text:
            return 0.0
        
        confidence_keywords = ["확실", "명확", "분명", "확인", "정확", "확실히"]
        uncertainty_keywords = ["아마", "추측", "가능성", "보임", "것 같음", "추정"]
        
        confidence_score = 0.5  # 기본값
        
        text_lower = response_text.lower()
        
        for keyword in confidence_keywords:
            if keyword in text_lower:
                confidence_score += 0.1
        
        for keyword in uncertainty_keywords:
            if keyword in text_lower:
                confidence_score -= 0.1
        
        # 응답 길이 고려
        if len(response_text) < 50:
            confidence_score -= 0.2
        elif len(response_text) > 200:
            confidence_score += 0.1
        
        return max(0.0, min(1.0, confidence_score))
    
    def compare_responses(self, responses: Dict[str, LLMResponse]) -> Dict[str, Any]:
        """여러 LLM 응답 비교 분석"""
        
        valid_responses = [r for r in responses.values() if r.success and r.response_text]
        
        comparison_result = {
            "responses": responses,
            "comparison": {
                "response_count": len(valid_responses),
                "total_models": len(responses),
                "success_rate": len(valid_responses) / len(responses) if responses else 0,
                "average_confidence": 0.0,
                "processing_times": {},
                "similarities": {},
                "best_response": None,
                "consensus": None,
                "recommendation": {}
            }
        }
        
        if not valid_responses:
            comparison_result["comparison"]["consensus"] = "failed"
            comparison_result["comparison"]["recommendation"] = {
                "reliability": "none",
                "message": "모든 AI 모델이 응답에 실패했습니다.",
                "recommended_action": "retry"
            }
            return comparison_result
        
        # 평균 신뢰도 계산
        comparison_result["comparison"]["average_confidence"] = \
            sum(r.confidence_score for r in valid_responses) / len(valid_responses)
        
        # 처리 시간 비교
        for model_name, response in responses.items():
            comparison_result["comparison"]["processing_times"][model_name] = response.processing_time
        
        # 최고 응답 선정
        best_response = max(valid_responses, 
                          key=lambda r: r.confidence_score * 0.7 + (len(r.response_text) / 1000) * 0.3)
        comparison_result["comparison"]["best_response"] = best_response.model_name
        
        # 유사도 및 합의도 계산
        if len(valid_responses) >= 2:
            similarity_score = self._calculate_response_similarity(valid_responses)
            comparison_result["comparison"]["similarities"]["average"] = similarity_score
            
            if similarity_score > 0.7:
                comparison_result["comparison"]["consensus"] = "high"
            elif similarity_score > 0.4:
                comparison_result["comparison"]["consensus"] = "medium"
            else:
                comparison_result["comparison"]["consensus"] = "low"
        else:
            comparison_result["comparison"]["consensus"] = "single"
        
        # 추천 생성
        comparison_result["comparison"]["recommendation"] = self._generate_recommendation(
            comparison_result["comparison"]
        )
        
        return comparison_result
    
    def _calculate_response_similarity(self, responses: List[LLMResponse]) -> float:
        """응답 간 유사도 계산"""
        if len(responses) < 2:
            return 1.0
        
        # 키워드 기반 유사도 계산
        response_keywords = []
        for response in responses:
            keywords = set(response.response_text.lower().split())
            # 불용어 제거
            stop_words = {'은', '는', '이', '가', '을', '를', '의', '에', '에서', '으로', '로', '와', '과', '한', '하는', '있는', '같은'}
            keywords = keywords - stop_words
            response_keywords.append(keywords)
        
        similarities = []
        for i in range(len(response_keywords)):
            for j in range(i + 1, len(response_keywords)):
                intersection = len(response_keywords[i] & response_keywords[j])
                union = len(response_keywords[i] | response_keywords[j])
                if union > 0:
                    similarities.append(intersection / union)
        
        return sum(similarities) / len(similarities) if similarities else 0.0
    
    def _generate_recommendation(self, comparison_data):
        """추천 생성"""
        consensus = comparison_data["consensus"]
        success_rate = comparison_data["success_rate"]
        avg_confidence = comparison_data["average_confidence"]
        
        if success_rate < 0.5:
            return {
                "reliability": "low",
                "message": "대부분의 AI 모델이 응답에 실패했습니다. 다른 방식으로 질문해보세요.",
                "recommended_action": "retry"
            }
        elif consensus == "high" and avg_confidence > 0.7:
            return {
                "reliability": "high",
                "message": "모든 AI 모델이 높은 신뢰도로 유사한 답변을 제공했습니다.",
                "recommended_action": "accept"
            }
        elif consensus == "medium":
            return {
                "reliability": "medium",
                "message": "AI 모델들 간에 일부 차이가 있습니다. 추가 확인이 도움될 수 있습니다.",
                "recommended_action": "review"
            }
        else:
            return {
                "reliability": "low",
                "message": "AI 모델들의 답변이 크게 다릅니다. 더 구체적인 질문을 시도해보세요.",
                "recommended_action": "clarify"
            }

# 전역 인스턴스
_multi_llm_analyzer = None

def get_multi_llm_analyzer():
    """전역 MultiLLMVideoAnalyzer 인스턴스 반환"""
    global _multi_llm_analyzer
    if _multi_llm_analyzer is None:
        _multi_llm_analyzer = MultiLLMVideoAnalyzer()
    return _multi_llm_analyzer