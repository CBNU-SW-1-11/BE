# chat/multi_ai_views.py - 완전히 수정된 안전한 버전
import os
import json
import time
import logging
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import AllowAny

# AI API 클라이언트들
import openai
from groq import Groq
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

logger = logging.getLogger(__name__)

class MultiAIService:
    """다중 AI 서비스 - 완전히 수정된 안전한 버전"""
    
    def __init__(self):
        self.openai_client = None
        self.groq_client = None
        self.anthropic_client = None
        self._initialize_clients()
        print("다중 AI 서비스 초기화 완료")
    
    def _initialize_clients(self):
        """API 클라이언트 초기화"""
        # OpenAI 초기화
        try:
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key:
                self.openai_client = openai.OpenAI(api_key=openai_key)
                print("OpenAI 클라이언트 초기화 완료")
        except Exception as e:
            print(f"OpenAI 초기화 실패: {e}")
        
        # Groq 초기화
        try:
            groq_key = os.getenv("GROQ_API_KEY")
            if groq_key:
                self.groq_client = Groq(api_key=groq_key)
                print("Groq 클라이언트 초기화 완료")
        except Exception as e:
            print(f"Groq 초기화 실패: {e}")
        
        # Anthropic 초기화
        try:
            if ANTHROPIC_AVAILABLE:
                anthropic_key = os.getenv("ANTHROPIC_API_KEY")
                if anthropic_key:
                    self.anthropic_client = anthropic.Anthropic(api_key=anthropic_key)
                    print("Anthropic 클라이언트 초기화 완료")
        except Exception as e:
            print(f"Anthropic 초기화 실패: {e}")
    
    def get_single_response(self, model, query, context=None):
        """단일 모델 응답 - 메인 메서드"""
        start_time = time.time()
        
        try:
            print(f"{model} 모델 응답 요청: {query[:50]}...")
            
            if not self._is_model_available(model):
                return {
                    'status': 'error',
                    'error': f'Model {model} is not available',
                    'response_time_ms': 0
                }
            
            # 모델별 API 호출
            if model == 'gpt-4':
                result = self._call_openai(query, context, start_time)
            elif model == 'claude-3':
                result = self._call_anthropic(query, context, start_time)
            elif model == 'groq-llama':
                result = self._call_groq(query, context, start_time)
            else:
                result = {
                    'status': 'error',
                    'error': f'Unknown model: {model}',
                    'response_time_ms': (time.time() - start_time) * 1000
                }
            
            print(f"{model} 응답 완료: {result.get('status')}")
            return result
            
        except Exception as e:
            print(f"{model} 응답 실패: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'response_time_ms': (time.time() - start_time) * 1000
            }
    
    def _get_single_response(self, model, query, context=None):
        """내부 메서드 - get_single_response와 동일"""
        return self.get_single_response(model, query, context)
    
    def get_multiple_responses(self, query, models=None, context=None):
        """여러 모델에서 순차적으로 응답 받기"""
        if not models:
            models = ['gpt-4', 'claude-3', 'groq-llama']
        
        print(f"다중 모델 응답 요청: {models}")
        
        responses = {}
        start_time = time.time()
        
        for model in models:
            try:
                result = self.get_single_response(model, query, context)
                responses[model] = result
            except Exception as e:
                responses[model] = {
                    'status': 'error',
                    'error': str(e),
                    'response_time_ms': 0
                }
        
        total_time = (time.time() - start_time) * 1000
        
        return {
            'responses': responses,
            'metadata': {
                'total_time_ms': total_time,
                'successful_models': len([r for r in responses.values() if r.get('status') == 'success']),
                'total_models': len(responses)
            }
        }
    
    def _call_openai(self, query, context, start_time):
        """OpenAI API 호출"""
        try:
            if not self.openai_client:
                raise Exception("OpenAI 클라이언트가 초기화되지 않음")
            
            user_prompt = self._build_prompt(query, context)
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "당신은 비디오 분석 전문가입니다. 사용자 질문에 정확하고 상세하게 답변해주세요."},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=1000,
                timeout=30
            )
            
            return {
                'status': 'success',
                'content': response.choices[0].message.content,
                'model': 'gpt-4',
                'provider': 'OpenAI',
                'response_time_ms': (time.time() - start_time) * 1000,
                'confidence': 0.9
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': f"OpenAI API 오류: {str(e)}",
                'response_time_ms': (time.time() - start_time) * 1000
            }
    
    def _call_anthropic(self, query, context, start_time):
        """Anthropic API 호출"""
        try:
            if not self.anthropic_client:
                raise Exception("Anthropic 클라이언트가 초기화되지 않음")
            
            user_prompt = self._build_prompt(query, context)
            
            response = self.anthropic_client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1000,
                temperature=0.7,
                system="당신은 비디오 분석 전문가입니다. 분석적이고 상세한 답변을 제공해주세요.",
                messages=[
                    {"role": "user", "content": user_prompt}
                ],
                timeout=30
            )
            
            return {
                'status': 'success',
                'content': response.content[0].text,
                'model': 'claude-3-haiku',
                'provider': 'Anthropic',
                'response_time_ms': (time.time() - start_time) * 1000,
                'confidence': 0.85
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': f"Anthropic API 오류: {str(e)}",
                'response_time_ms': (time.time() - start_time) * 1000
            }
    
    def _call_groq(self, query, context, start_time):
        """Groq API 호출"""
        try:
            if not self.groq_client:
                raise Exception("Groq 클라이언트가 초기화되지 않음")
            
            user_prompt = self._build_prompt(query, context)
            
            response = self.groq_client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": "당신은 비디오 분석 전문가입니다. 실용적이고 명확한 답변을 제공해주세요."},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=1000,
                timeout=30
            )
            
            return {
                'status': 'success',
                'content': response.choices[0].message.content,
                'model': 'llama-3.1-8b-instant',
                'provider': 'Groq',
                'response_time_ms': (time.time() - start_time) * 1000,
                'confidence': 0.8
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'error': f"Groq API 오류: {str(e)}",
                'response_time_ms': (time.time() - start_time) * 1000
            }
    
    def _build_prompt(self, query, context):
        """프롬프트 구성"""
        user_prompt = query
        
        if context and context.get('videoInfo'):
            video_info = context['videoInfo']
            video_name = video_info.get('name', '')
            analysis_type = video_info.get('analysis_type', '')
            user_prompt += f"\n\n[비디오 정보: {video_name} - {analysis_type} 분석 완료]"
        
        return user_prompt
    
    def _is_model_available(self, model):
        """모델 사용 가능 여부 확인"""
        if model == 'gpt-4':
            return self.openai_client is not None
        elif model == 'claude-3':
            return self.anthropic_client is not None
        elif model == 'groq-llama':
            return self.groq_client is not None
        return False
    
    def get_available_models(self):
        """사용 가능한 모델 목록 반환"""
        models = []
        if self.openai_client:
            models.append({
                'id': 'gpt-4',
                'name': 'GPT-4',
                'provider': 'OpenAI'
            })
        if self.anthropic_client:
            models.append({
                'id': 'claude-3',
                'name': 'Claude-3',
                'provider': 'Anthropic'
            })
        if self.groq_client:
            models.append({
                'id': 'groq-llama',
                'name': 'Llama-3',
                'provider': 'Groq'
            })
        return models
    
    # 호환성을 위한 별칭 메서드들
    def _get_available_models(self):
        return self.get_available_models()
    
    def _generate_response_summary(self, responses):
        """응답 요약 생성"""
        successful = [r for r in responses.values() if r.get('status') == 'success']
        total = len(responses)
        
        if not successful:
            return {
                'status': 'all_failed',
                'message': '모든 AI 모델에서 응답 실패'
            }
        
        return {
            'status': 'success',
            'successful_models': len(successful),
            'total_models': total,
            'message': f'{len(successful)}/{total} 모델이 성공적으로 응답했습니다.'
        }


class SimpleMultiLLMAnalyzer:
    """간단한 멀티 LLM 분석기"""
    
    def __init__(self):
        self.multi_ai_service = MultiAIService()
        print("멀티 LLM 분석기 초기화 완료")
    
    def analyze_video_multi_llm(self, frame_images, user_query, video_context):
        """비디오 멀티 LLM 분석"""
        print(f"멀티 LLM 분석 시작: {user_query[:50]}...")
        
        models = ['gpt-4', 'claude-3', 'groq-llama']
        context = {
            'videoInfo': video_context,
            'frame_count': len(frame_images)
        }
        
        responses = {}
        
        for model in models:
            try:
                result = self.multi_ai_service.get_single_response(model, user_query, context)
                responses[model] = self._convert_to_response_object(result)
            except Exception as e:
                print(f"{model} 분석 실패: {e}")
                responses[model] = self._create_error_response(str(e))
        
        print(f"멀티 LLM 분석 완료: {len(responses)}개 응답")
        return responses
    
    def compare_responses(self, responses):
        """응답 비교"""
        successful_responses = [r for r in responses.values() if hasattr(r, 'success') and r.success]
        
        if not successful_responses:
            return {
                'comparison': {
                    'status': 'failed',
                    'recommendation': '모든 AI 모델에서 응답 생성에 실패했습니다. API 키와 네트워크 연결을 확인해주세요.'
                }
            }
        
        # 가장 높은 신뢰도의 응답 선택
        best_response = max(successful_responses, key=lambda x: getattr(x, 'confidence_score', 0))
        
        return {
            'comparison': {
                'status': 'success',
                'total_responses': len(responses),
                'successful_responses': len(successful_responses),
                'best_model': getattr(best_response, 'model', 'unknown'),
                'recommendation': getattr(best_response, 'response_text', '')[:200]
            }
        }
    
    def _convert_to_response_object(self, result):
        """결과를 응답 객체로 변환"""
        class ResponseObject:
            def __init__(self, result):
                self.success = result.get('status') == 'success'
                self.response_text = result.get('content', '')
                self.confidence_score = result.get('confidence', 0)
                self.processing_time = result.get('response_time_ms', 0)
                self.error = result.get('error', None)
                self.model = result.get('model', 'unknown')
        
        return ResponseObject(result)
    
    def _create_error_response(self, error_msg):
        """에러 응답 생성"""
        class ErrorResponse:
            def __init__(self, error_msg):
                self.success = False
                self.response_text = ''
                self.confidence_score = 0
                self.processing_time = 0
                self.error = error_msg
                self.model = 'unknown'
        
        return ErrorResponse(error_msg)


# 전역 인스턴스 (싱글톤)
_multi_ai_service_instance = None
_multi_llm_analyzer_instance = None

def get_multi_ai_service():
    """멀티 AI 서비스 인스턴스 반환"""
    global _multi_ai_service_instance
    if _multi_ai_service_instance is None:
        _multi_ai_service_instance = MultiAIService()
    return _multi_ai_service_instance

def get_multi_llm_analyzer():
    """멀티 LLM 분석기 인스턴스 반환"""
    global _multi_llm_analyzer_instance
    if _multi_llm_analyzer_instance is None:
        _multi_llm_analyzer_instance = SimpleMultiLLMAnalyzer()
    return _multi_llm_analyzer_instance


# Django Views
class MultiAIChatView(APIView):
    """다중 AI 채팅 API"""
    permission_classes = [AllowAny]
    
    def post(self, request):
        try:
            data = request.data
            query = data.get('query', '').strip()
            context = data.get('context', {})
            models = data.get('models', ['gpt-4', 'claude-3'])
            
            print(f"다중 AI 채팅 요청: {query[:50]}...")
            
            if not query:
                return Response({
                    'error': '질문을 입력해주세요.'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # 다중 AI 서비스 사용
            multi_ai_service = get_multi_ai_service()
            result = multi_ai_service.get_multiple_responses(query, models, context)
            
            return Response({
                'query': query,
                'responses': result['responses'],
                'metadata': result['metadata'],
                'timestamp': time.time()
            })
            
        except Exception as e:
            logger.error(f"다중 AI 채팅 오류: {e}")
            return Response({
                'error': f'다중 AI 채팅 처리 실패: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class MultiLLMChatView(APIView):
    """멀티 LLM 전용 채팅 뷰"""
    permission_classes = [AllowAny]
    
    def post(self, request):
        try:
            user_query = request.data.get('message', '').strip()
            video_id = request.data.get('video_id')
            
            if not user_query:
                return Response({'error': '메시지를 입력해주세요.'}, status=400)
            
            # 비디오 정보 준비
            video_context = {}
            if video_id:
                try:
                    from .models import Video
                    video = Video.objects.get(id=video_id)
                    video_context = {
                        'duration': getattr(video, 'duration', 0),
                        'filename': getattr(video, 'original_name', 'Unknown')
                    }
                except:
                    pass
            
            # 멀티 LLM 분석 실행
            analyzer = get_multi_llm_analyzer()
            multi_responses = analyzer.analyze_video_multi_llm([], user_query, video_context)
            comparison_result = analyzer.compare_responses(multi_responses)
            
            return Response({
                'response_type': 'multi_llm_result',
                'query': user_query,
                'video_info': {'id': video_id, 'name': video_context.get('filename', 'Unknown')} if video_id else None,
                'llm_responses': {
                    model: {
                        'response': resp.response_text,
                        'confidence': resp.confidence_score,
                        'processing_time': resp.processing_time,
                        'success': resp.success,
                        'error': resp.error
                    }
                    for model, resp in multi_responses.items()
                },
                'comparison_analysis': comparison_result['comparison'],
                'recommendation': comparison_result['comparison']['recommendation']
            })
            
        except Exception as e:
            print(f"MultiLLM 채팅 오류: {e}")
            return Response({'error': str(e)}, status=500)


class AvailableModelsView(APIView):
    """사용 가능한 AI 모델 목록 API"""
    permission_classes = [AllowAny]
    
    def get(self, request):
        try:
            multi_ai_service = get_multi_ai_service()
            models = multi_ai_service.get_available_models()
            
            return Response({
                'available_models': models,
                'total_count': len(models)
            })
            
        except Exception as e:
            return Response({
                'error': f'모델 목록 조회 실패: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# 함수형 뷰들
@csrf_exempt
@require_http_methods(["POST"])
def multi_ai_chat(request):
    """다중 AI 채팅 (함수형 뷰)"""
    try:
        data = json.loads(request.body)
        query = data.get('query', '').strip()
        
        if not query:
            return JsonResponse({'error': '질문을 입력해주세요.'}, status=400)
        
        # 간단한 폴백 응답
        response_data = {
            'query': query,
            'responses': {
                'system': {
                    'status': 'success',
                    'content': f'안녕하세요! "{query}"에 대한 질문을 받았습니다. AI 시스템이 현재 처리 중입니다.',
                    'model': 'system',
                    'provider': 'System'
                }
            },
            'metadata': {
                'successful_models': 1,
                'total_models': 1
            }
        }
        
        return JsonResponse(response_data)
        
    except Exception as e:
        return JsonResponse({
            'error': f'다중 AI 채팅 오류: {str(e)}'
        }, status=500)

@csrf_exempt  
@require_http_methods(["GET"])
def available_models(request):
    """사용 가능한 모델 목록 (함수형 뷰)"""
    try:
        multi_ai_service = get_multi_ai_service()
        models = multi_ai_service.get_available_models()
        
        return JsonResponse({
            'available_models': models,
            'total_count': len(models)
        })
        
    except Exception as e:
        return JsonResponse({
            'error': f'모델 목록 조회 오류: {str(e)}'
        }, status=500)