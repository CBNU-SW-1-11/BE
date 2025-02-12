from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import openai
import anthropic
from groq import Groq
import json
import logging

logger = logging.getLogger(__name__)

class ChatBot:
    def __init__(self, api_key, model, api_type):
        self.conversation_history = []
        self.model = model
        self.api_type = api_type
        self.api_key = api_key
        
        if api_type == 'openai':
            openai.api_key = api_key
        elif api_type == 'anthropic':
            self.client = anthropic.Anthropic(api_key=api_key)
        elif api_type == 'groq':
            self.client = Groq(api_key=api_key)
    
    def chat(self, user_input):
        try:
            logger.info(f"Processing chat request for {self.api_type}")
            
            # 대화 시작 시 시스템 메시지 추가
            if not self.conversation_history:
                self.conversation_history.append({
                    "role": "system",
                    "content": """당신은 사용자가 입력한 언어에 맞춰 답변하는 AI 어시스턴트입니다.
                    1. 사용자가 한국어로 입력하면, 모든 답변을 한국어로 작성합니다.
                    2. 사용자가 다른 언어(예: 영어)로 입력하면, 해당 언어로 답변을 작성합니다.
                    3. 전문 용어가 필요할 경우, 한글로 추가 설명을 덧붙입니다.
                    Your task is to respond in the same language as the user's input and answer their question or address their request.
                    Once you have identified the language, formulate your response in that same language.
                    Ensure that your grammar, vocabulary, and style are appropriate for the detected language.
                    Answer the user's question or address their request to the best of your ability. If you need clarification or additional information, ask for it in the same language as the user's input.

                    """
                })

            logger.debug(f"User input: {user_input}")
            self.conversation_history.append({"role": "user", "content": user_input})
            
            if self.api_type == 'openai':
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=self.conversation_history,
                    temperature=0.7,
                    max_tokens=1024
                )
                assistant_response = response['choices'][0]['message']['content']
            
            elif self.api_type == 'anthropic':
                try:
                    messages = []
                    # 시스템 메시지와 이전 대화 내용 포함
                    for msg in self.conversation_history:
                        if msg["role"] != "system":  # system 메시지 제외
                            messages.append({
                                "role": msg["role"],
                                "content": msg["content"]
                            })
                    
                    message = self.client.messages.create(
                        model=self.model,
                        max_tokens=4096,
                        temperature=0,
                        messages=messages
                    )
                    
                    assistant_response = message.content[0].text
                    logger.info(f"Anthropic response received successfully")
                    
                except Exception as e:
                    logger.error(f"Anthropic API error: {str(e)}")
                    raise
            
            elif self.api_type == 'groq':
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.conversation_history,
                    temperature=0.7,
                    max_tokens=1024
                )
                assistant_response = response.choices[0].message.content
            
            self.conversation_history.append({
                "role": "assistant", 
                "content": assistant_response
            })
            return assistant_response
            
        except Exception as e:
            logger.error(f"Error in chat method: {str(e)}")
            raise
        
    def analyze_responses(self, responses, query):
        """AI 응답들을 분석하고 최적의 답변을 생성"""
        try:
            logger.info("\n" + "="*100)
            logger.info("📊 분석 시작")
            logger.info(f"🤖 분석 수행 AI: {self.api_type.upper()}")
            logger.info("="*100)

            analysis_prompt = f"""아래는 동일한 질문에 대한 세 가지 AI의 응답입니다. 
            각 응답을 비교하여 가장 신뢰할 수 있는 정보를 조합하여 최적의 답변을 만들어 주세요.
            
            질문: {query}

            GPT 응답: {responses.get('gpt', '응답 없음')}
            Claude 응답: {responses.get('claude', '응답 없음')}
            Mixtral 응답: {responses.get('mixtral', '응답 없음')}

            [최적의 응답을 만들 때 고려할 사항]
            - 답변을 조합하여 최적의 답변을 재작성
            - 가장 정확하고 관련성이 높은 정보를 선택
            - 여러 AI의 정보를 조합하여 더욱 풍부한 답변을 생성
            - 중복된 내용은 제거하고, 표현이 명확한 쪽을 선택
            - 논리적 흐름을 고려하여 자연스럽고 이해하기 쉬운 답변 작성
            
            [출력 형식]
            {{
                "bot_name": "{self.api_type.upper()}",
                "best_response": "최적의 답변",
                "analysis": {{
                    "gpt": {{
                        "장점": "GPT 답변의 장점",
                        "단점": "GPT 답변의 단점"
                    }},
                    "claude": {{
                        "장점": "Claude 답변의 장점",
                        "단점": "Claude 답변의 단점"
                    }},
                    "mixtral": {{
                        "장점": "Mixtral 답변의 장점",
                        "단점": "Mixtral 답변의 단점"
                    }}
                }},
                "reasoning": "최적의 응답을 만들 때 각 AI 응답에서 어떤 정보를 사용했는지 설명"
            }}"""

            logger.info("🔍 각 AI 응답 분석 중...")

            # API 타입에 따른 분기 처리
            if self.api_type == 'openai':
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[{"role": "user", "content": analysis_prompt}],
                    temperature=0,
                    max_tokens=4096
                )
                analysis_text = response['choices'][0]['message']['content']
                
            elif self.api_type == 'anthropic':
                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    temperature=0,
                    system="You must respond with valid JSON only. No other text or formatting.",
                    messages=[{
                        "role": "user", 
                        "content": analysis_prompt
                    }]
                )
                analysis_text = message.content[0].text.strip()
                                
            elif self.api_type == 'groq':
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": analysis_prompt}],
                    temperature=0,
                    max_tokens=4096
                )
                analysis_text = response.choices[0].message.content

            logger.info("✅ 분석 완료\n")
            
            try:
                
                # 모든 제어 문자 제거 및 텍스트 정리
                analysis_text = analysis_text.strip()
                # 제어 문자 및 특수 문자 처리
                analysis_text = analysis_text.replace('\_', '_')  # 백슬래시 언더스코어 처리
                analysis_text = analysis_text.replace('\\"', '"')  # 백슬래시 따옴표 처리
                analysis_text = analysis_text.replace('\\n', ' ')  # 백슬래시 줄바꿈 처리
                analysis_text = analysis_text.replace('\\t', ' ')  # 백슬래시 탭 처리
                analysis_text = analysis_text.replace('\\r', '')  # 백슬래시 캐리지리턴 처리
                
                analysis_text = ''.join(char for char in analysis_text if ord(char) >= 32)
                analysis_text = analysis_text.replace('\n', ' ').replace('\r', '').replace('\t', ' ')
                
                # JSON 시작과 끝 찾기
                start_idx = analysis_text.find('{')
                end_idx = analysis_text.rfind('}') + 1
                if start_idx != -1 and end_idx != -1:
                    analysis_text = analysis_text[start_idx:end_idx]
                
                # 따옴표 정리
                analysis_text = analysis_text.replace('"', '"').replace('"', '"')
                
                # JSON 파싱
                analysis_result = json.loads(analysis_text)
                analysis_result['bot_name'] = self.api_type.upper()
                        
                # 분석 결과 로깅
                logger.info("📊 분석 결과:")
                logger.info("="*100)
                logger.info(f"✨ 최적의 답변:\n{analysis_result['best_response']}\n")
                
                logger.info("📌 각 AI 분석:")
                for ai, analysis in analysis_result['analysis'].items():
                    logger.info(f"\n{ai.upper()}:")
                    logger.info(f"장점: {analysis.get('장점', '정보 없음')}")
                    logger.info(f"단점: {analysis.get('단점', '정보 없음')}")
                
                logger.info(f"\n💡 분석 근거:\n{analysis_result['reasoning']}")
                logger.info("="*100 + "\n")

                return analysis_result

            except json.JSONDecodeError as je:
                logger.error(f"❌ JSON 파싱 실패. 원본 응답:")
                logger.error(analysis_text)
                logger.error(f"파싱 에러: {str(je)}\n")
                
                default_result = {
                    "bot_name": self.api_type.upper(),
                    "best_response": max(responses.values(), key=len),
                    "analysis": {
                        "gpt": {"장점": "분석 실패", "단점": "분석 실패"},
                        "claude": {"장점": "분석 실패", "단점": "분석 실패"},
                        "mixtral": {"장점": "분석 실패", "단점": "분석 실패"}
                    },
                    "reasoning": "응답 분석 중 오류가 발생하여 최적의 답변을 생성하지 못했습니다."
                }
                
                # 에러 결과 로깅
                logger.info("❌ 분석 실패 - 기본 응답 사용:")
                logger.info("="*100)
                logger.info(f"✨ 최적의 답변:\n{default_result['best_response']}\n")
                logger.info(f"💡 분석 실패 사유:\n{default_result['reasoning']}")
                logger.info("="*100 + "\n")
                
                return default_result
        
        except Exception as e:
            logger.error(f"❌ Analysis error: {str(e)}")
            default_result = {
                "bot_name": self.api_type.upper(),
                "best_response": max(responses.values(), key=len),
                "analysis": {
                    "gpt": {"장점": "분석 실패", "단점": "분석 실패"},
                    "claude": {"장점": "분석 실패", "단점": "분석 실패"},
                    "mixtral": {"장점": "분석 실패", "단점": "분석 실패"}
                },
                "reasoning": f"응답 분석 중 오류가 발생했습니다: {str(e)}"
            }
            
            # 에러 결과 로깅
            logger.info("❌ 분석 실패 - 기본 응답 사용:")
            logger.info("="*100)
            logger.info(f"✨ 최적의 답변:\n{default_result['best_response']}\n")
            logger.info(f"💡 분석 실패 사유:\n{default_result['reasoning']}")
            logger.info("="*100 + "\n")
            
            return default_result

import json
import logging

logger = logging.getLogger(__name__)

class ChatView(APIView):
    def post(self, request, bot_name):
        try:
            logger.info(f"Received chat request for {bot_name}")
            data = request.data
            user_message = data.get('message')
            compare_responses = data.get('compare', True)
            
            if not user_message:
                return Response({
                    'error': 'No message provided'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            chatbot = chatbots.get(bot_name)
            if not chatbot:
                return Response({
                    'error': 'Invalid bot name'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            try:
                # 모든 AI에게 메시지 전송
                responses = {}
                for name, bot in chatbots.items():
                    try:
                        response = bot.chat(user_message)
                        responses[name] = response
                    except Exception as e:
                        logger.error(f"Error getting response from {name}: {str(e)}")
                        responses[name] = f"Error: {str(e)}"

                # if compare_responses:
                #     analysis = chatbot.analyze_responses(responses, user_message)
                    
                #     return Response({
                #         'responses': responses,
                #         'bot_name': bot_name.upper(),
                #         'best_response': analysis.get('best_response', ''),
                #         'analysis': {
                #             'scores': analysis.get('scores', {}),
                #             'reasoning': analysis.get('reasoning', '')
                #         }
                #     })
                # ChatView 클래스의 post 메서드에서
                if compare_responses:
                    analysis = chatbot.analyze_responses(responses, user_message)
                    
                    return Response({
                        'responses': responses,
                        'bot_name': bot_name.upper(),
                        'best_response': analysis.get('best_response', ''),
                        'analysis': analysis.get('analysis', {}),  # scores 대신 analysis 직접 전달
                        'reasoning': analysis.get('reasoning', '')
                    })
                else:
                    return Response({'responses': responses})
                
                    
            except Exception as e:
                logger.error(f"Error processing chat request: {str(e)}")
                return Response({
                    'error': f"Chat processing error: {str(e)}"
                }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
                
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return Response({
                'error': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# Logging 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('chat_server.log')
    ]
)





# API 키 설정
OPENAI_API_KEY = "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"
ANTHROPIC_API_KEY = "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"
GROQ_API_KEY = "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"

chatbots = {
    'gpt': ChatBot(OPENAI_API_KEY, 'gpt-3.5-turbo', 'openai'),
    'claude': ChatBot(ANTHROPIC_API_KEY, 'claude-3-5-haiku-20241022', 'anthropic'), 
    'mixtral': ChatBot(GROQ_API_KEY, 'mixtral-8x7b-32768', 'groq'),
}
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
import requests
import logging
from django.contrib.auth import get_user_model
from .models import SocialAccount
from .serializers import UserSerializer
import uuid

logger = logging.getLogger(__name__)
User = get_user_model()

# def generate_unique_username(email, name=None):
#     """고유한 username 생성"""
#     base = name or email.split('@')[0]
#     username = base
#     suffix = 1
    
#     # username이 고유할 때까지 숫자 추가
#     while User.objects.filter(username=username).exists():
#         username = f"{base}_{suffix}"
#         suffix += 1
    
#     return username

def generate_unique_username(email, name=None):
    """username 생성 - 이메일 앞부분 또는 이름 사용"""
    if name:
        return name  # 이름이 있으면 그대로 사용
    return email.split('@')[0]  # 이름이 없으면 이메일 앞부분 사용

@api_view(['GET'])
@permission_classes([AllowAny])
def google_callback(request):
    try:
        # 액세스 토큰 추출
        auth_header = request.headers.get('Authorization', '')
        if not auth_header.startswith('Bearer '):
            return Response(
                {'error': '잘못된 인증 헤더'}, 
                status=status.HTTP_401_UNAUTHORIZED
            )
        
        access_token = auth_header.split(' ')[1]

        # Google API로 사용자 정보 요청
        user_info_response = requests.get(
            'https://www.googleapis.com/oauth2/v3/userinfo',
            headers={'Authorization': f'Bearer {access_token}'}
        )

        if user_info_response.status_code != 200:
            return Response(
                {'error': 'Google에서 사용자 정보를 가져오는데 실패했습니다'}, 
                status=status.HTTP_400_BAD_REQUEST
            )

        user_info = user_info_response.json()
        email = user_info.get('email')
        name = user_info.get('name')
        
        if not email:
            return Response(
                {'error': '이메일이 제공되지 않았습니다'}, 
                status=status.HTTP_400_BAD_REQUEST
            )

        try:
            # 기존 사용자 검색
            user = User.objects.get(email=email)
        except User.DoesNotExist:
            # 새로운 사용자 생성
            username = generate_unique_username(email, name)
            user = User.objects.create(
                username=username,
                email=email,
                is_active=True
            )
            
            # 기본 비밀번호 설정 (선택적)
            random_password = uuid.uuid4().hex
            user.set_password(random_password)
            user.save()

        # 소셜 계정 정보 생성 또는 업데이트
        social_account, created = SocialAccount.objects.get_or_create(
            email=email,
            provider='google',
            defaults={'user': user}
        )

        if not created and social_account.user != user:
            social_account.user = user
            social_account.save()

        # 사용자 데이터 반환
        serializer = UserSerializer(user)
        return Response({
            'user': serializer.data,
            'access_token': access_token,
        })

    except Exception as e:
        logger.error(f"Error in google_callback: {str(e)}")
        return Response(
            {'error': str(e)}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
import requests
import json
import requests
from django.conf import settings
from django.http import JsonResponse
from django.shortcuts import redirect
from .models import User  # User 모델을 임포트



@api_view(['GET'])
@permission_classes([AllowAny])
def kakao_callback(request):
    try:
        auth_code = request.GET.get('code')
        logger.info(f"Received Kakao auth code: {auth_code}")
        
        # 카카오 토큰 받기
        token_url = "https://kauth.kakao.com/oauth/token"
        data = {
            "grant_type": "authorization_code",
            "client_id": settings.KAKAO_CLIENT_ID,
            "redirect_uri": settings.KAKAO_REDIRECT_URI,
            "code": auth_code,
        }
        
        token_response = requests.post(token_url, data=data)
        
        if not token_response.ok:
            return Response({
                'error': '카카오 토큰 받기 실패',
                'details': token_response.text
            }, status=status.HTTP_400_BAD_REQUEST)
        
        token_data = token_response.json()
        access_token = token_data.get('access_token')
        
        if not access_token:
            return Response({
                'error': '액세스 토큰 없음',
                'details': token_data
            }, status=status.HTTP_400_BAD_REQUEST)

        # 카카오 사용자 정보 받기
        user_info_url = "https://kapi.kakao.com/v2/user/me"
        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-type": "application/x-www-form-urlencoded;charset=utf-8"
        }
        
        user_info_response = requests.get(
            user_info_url,
            headers=headers,
            params={
                'property_keys': json.dumps([
                    "kakao_account.email",
                    "kakao_account.profile",
                    "kakao_account.name"
                ])
            }
        )
        
        if not user_info_response.ok:
            return Response({
                'error': '사용자 정보 받기 실패',
                'details': user_info_response.text
            }, status=status.HTTP_400_BAD_REQUEST)
        
        user_info = user_info_response.json()
        kakao_account = user_info.get('kakao_account', {})
        email = kakao_account.get('email')
        profile = kakao_account.get('profile', {})
        nickname = profile.get('nickname')
        
        logger.info(f"Kakao user info - email: {email}, nickname: {nickname}")
        
        if not email:
            return Response({
                'error': '이메일 정보 없음',
                'details': '카카오 계정의 이메일 정보가 필요합니다.'
            }, status=status.HTTP_400_BAD_REQUEST)

        # 사용자 생성 또는 업데이트
        try:
            user = User.objects.get(email=email)
            # 기존 사용자의 경우 닉네임 업데이트
            logger.info(f"Updated existing user with nickname: {nickname}")
        except User.DoesNotExist:
            # 새 사용자 생성
            unique_username = generate_unique_username(email, nickname)  # 수정
            user = User.objects.create(
                email=email,
                username=unique_username,
                is_active=True
            )            
            logger.info(f"Created new user with nickname: {nickname}")

        # 소셜 계정 생성 또는 업데이트
        social_account, _ = SocialAccount.objects.update_or_create(
            email=email,
            provider='kakao',
            defaults={
                'user': user,
                'nickname': nickname
            }
        )
        logger.info(f"Social account updated - email: {email}, nickname: {nickname}")

        serializer = UserSerializer(user)
        response_data = {
            'user': {
                **serializer.data,
                'loginType': 'kakao'
            },
            'access_token': access_token
        }
        logger.info(f"Response data: {response_data}")
        
        return Response(response_data)
        
    except Exception as e:
        logger.exception("Unexpected error in kakao_callback")
        return Response({
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    


# views.py
import requests
from django.conf import settings
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from rest_framework import status
from django.contrib.auth import get_user_model
from .models import SocialAccount
import logging

logger = logging.getLogger(__name__)
User = get_user_model()


@api_view(['GET'])
@permission_classes([AllowAny])
def naver_callback(request):
    try:
        code = request.GET.get('code')
        state = request.GET.get('state')
        logger.info(f"Received Naver auth code: {code}")

        # 네이버 토큰 받기
        token_url = "https://nid.naver.com/oauth2.0/token"
        token_params = {
            "grant_type": "authorization_code",
            "client_id": settings.NAVER_CLIENT_ID,
            "client_secret": settings.NAVER_CLIENT_SECRET,
            "code": code,
            "state": state
        }

        token_response = requests.get(token_url, params=token_params)

        if not token_response.ok:
            return Response({
                'error': '네이버 토큰 받기 실패',
                'details': token_response.text
            }, status=status.HTTP_400_BAD_REQUEST)

        token_data = token_response.json()
        access_token = token_data.get('access_token')

        if not access_token:
            return Response({
                'error': '액세스 토큰 없음',
                'details': token_data
            }, status=status.HTTP_400_BAD_REQUEST)

        # 네이버 사용자 정보 받기
        user_info_url = "https://openapi.naver.com/v1/nid/me"
        headers = {
            "Authorization": f"Bearer {access_token}"
        }

        user_info_response = requests.get(user_info_url, headers=headers)

        if not user_info_response.ok:
            return Response({
                'error': '사용자 정보 받기 실패',
                'details': user_info_response.text
            }, status=status.HTTP_400_BAD_REQUEST)

        user_info = user_info_response.json()
        response = user_info.get('response', {})
        email = response.get('email')
        nickname = response.get('nickname')
        username = email.split('@')[0]

        if not email:
            return Response({
                'error': '이메일 정보 없음',
                'details': '네이버 계정의 이메일 정보가 필요합니다.'
            }, status=status.HTTP_400_BAD_REQUEST)

        # 사용자 생성 또는 업데이트
        try:
            user = User.objects.get(email=email)
            logger.info(f"Found existing user with email: {email}")
        except User.DoesNotExist:
            unique_username = generate_unique_username(email, username)  # 수정
            user = User.objects.create(
                username=unique_username,
                email=email,
                is_active=True
            )
            logger.info(f"Created new user with username: {unique_username}")

        # 소셜 계정 생성 또는 업데이트
        social_account, _ = SocialAccount.objects.update_or_create(
            email=email,
            provider='naver',
            defaults={
                'user': user,
                'nickname': username
            }
        )

        logger.info(f"Social account updated - email: {email}, nickname: {nickname}")

        serializer = UserSerializer(user)
        response_data = {
            'user': {
                **serializer.data,
                'loginType': 'naver'
            },
            'access_token': access_token
        }

        return Response(response_data)

    except Exception as e:
        logger.exception("Unexpected error in naver_callback")
        return Response({
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)