from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from chat.serializers import UserSerializer
import openai
import anthropic
from groq import Groq
import anthropic

class ChatBot:
    def __init__(self, api_key, model, api_type):
        self.conversation_history = []
        self.model = model
        self.api_type = api_type
        self.api_key = api_key  # api_key 속성 추가
        
        if api_type == 'openai':
            openai.api_key = api_key
        elif api_type == 'anthropic':
            self.client = anthropic.Client(api_key=api_key)
        elif api_type == 'groq':
            self.client = Groq(api_key=api_key)
    
    def chat(self, user_input):
        try:
            # 대화 시작 시 시스템 메시지 추가
            if not self.conversation_history:
                self.conversation_history.append({
                    "role": "system",
                    "content": """당신은 사용자가 입력한 언어에 맞춰 답변하는 AI 어시스턴트입니다.
                    1. 사용자가 한국어로 입력하면, 모든 답변을 한국어로 작성합니다.
                    2. 사용자가 다른 언어(예: 영어)로 입력하면, 해당 언어로 답변을 작성합니다.
                    3. 전문 용어가 필요할 경우, 한글로 추가 설명을 덧붙입니다.
                    """
                })

            # 사용자 입력 출력
            print(f"User input: {user_input}")
            
            self.conversation_history.append({"role": "user", "content": user_input})
            
            if self.api_type == 'openai':
                # OpenAI 방식 처리
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=self.conversation_history,
                    temperature=0.7,
                    max_tokens=1024
                )
                assistant_response = response['choices'][0]['message']['content']
            
            elif self.api_type == 'anthropic':
    # Anthropic Messages API 방식 처리
                client = anthropic.Client(api_key=self.api_key)
                
                # 메시지 구조 수정
                message = client.messages.create(
                    model="claude-3-5-haiku-20241022",  # 올바른 모델명 사용
                    max_tokens=4096,
                    temperature=0,
                    messages=[
                        {
                            "role": "user",
                            "content": f"""You are an AI assistant capable of communicating in multiple languages.
                            Your task is to respond in the same language as the user's input and answer their question or address their request.

                            Once you have identified the language, formulate your response in that same language.
                            Ensure that your grammar, vocabulary, and style are appropriate for the detected language.

                            Answer the user's question or address their request to the best of your ability. If you need clarification or additional information, ask for it in the same language as the user's input.

                            Here is the user's input:

                            <user_input>
                            {user_input}
                            </user_input>"""
                        }
                    ]
                )
                
                # 응답 추출 및 할당
           
                assistant_response = message.content[0].text 
                
                # 응답 확인
                print(f"Anthropic response: {assistant_response}")  # 응답 출력


            
            elif self.api_type == 'groq':
                # Groq 방식 처리
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=self.conversation_history,
                    temperature=0.7,
                    max_tokens=1024
                )
                assistant_response = response.choices[0].message.content
            
            # 대화 이력에 추가
            self.conversation_history.append({"role": "assistant", "content": assistant_response})
            return assistant_response
        except Exception as e:
            print(f"Error: {str(e)}")
            return f"오류가 발생했습니다: {str(e)}"

OPENAI_API_KEY = ""
ANTHROPIC_API_KEY = ""
GROQ_API_KEY = ""



chatbots = {
    'gpt': ChatBot(OPENAI_API_KEY, 'gpt-3.5-turbo', 'openai'),
    'claude': ChatBot(ANTHROPIC_API_KEY, 'claude-3-5-haiku-20241022', 'anthropic'), 
    'mixtral': ChatBot(GROQ_API_KEY, 'mixtral-8x7b-32768', 'groq'),
}

class ChatView(APIView):
    def post(self, request, bot_name):
        try:
            data = request.data
            user_message = data.get('message')
            
            if not user_message:
                return Response({'error': 'No message provided'}, status=status.HTTP_400_BAD_REQUEST)
            
            chatbot = chatbots.get(bot_name)
            if not chatbot:
                return Response({'error': 'Invalid bot name'}, status=status.HTTP_400_BAD_REQUEST)

            response = chatbot.chat(user_message)
            return Response({'response': response})
        except Exception as e:
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
     # chat/views.py
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

def generate_unique_username(email, name=None):
    """고유한 username 생성"""
    base = name or email.split('@')[0]
    username = base
    suffix = 1
    
    # username이 고유할 때까지 숫자 추가
    while User.objects.filter(username=username).exists():
        username = f"{base}_{suffix}"
        suffix += 1
    
    return username

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
    # chat/views.py
import json
import logging
from django.conf import settings  # settings import 추가
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
import requests
from django.contrib.auth import get_user_model
from .serializers import UserSerializer
from .models import SocialAccount
# chat/views.py
import json
import logging
from django.conf import settings
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
import requests
from django.contrib.auth import get_user_model
from .serializers import UserSerializer
from .models import SocialAccount

logger = logging.getLogger(__name__)
User = get_user_model()

@api_view(['GET'])
@permission_classes([AllowAny])
def google_callback(request):
    try:
        auth_header = request.headers.get('Authorization', '')
        if not auth_header.startswith('Bearer '):
            return Response(
                {'error': '잘못된 인증 헤더'}, 
                status=status.HTTP_401_UNAUTHORIZED
            )
        
        access_token = auth_header.split(' ')[1]

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

        # 사용자 생성 또는 가져오기
        user, created = User.objects.get_or_create(
            email=email,
            defaults={
                'username': name or email.split('@')[0],
                'is_active': True
            }
        )



        # 소셜 계정 생성 또는 업데이트
        social_account, _ = SocialAccount.objects.update_or_create(
            email=email,
            provider='google',
            defaults={
                'user': user,
                'nickname': name
            }
        )

        serializer = UserSerializer(user)
        return Response({
            'user': {
                **serializer.data,
                'loginType': 'google'  # 구글 로그인 타입 추가
            },
            'access_token': access_token,
        })

    except Exception as e:
        logger.exception("Error in google_callback")
        return Response(
            {'error': str(e)}, 
            status=status.HTTP_500_INTERNAL_SERVER_ERROR
        )
    
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
            return Response({'error': '카카오 토큰 받기 실패'}, status=status.HTTP_400_BAD_REQUEST)

        token_data = token_response.json()
        access_token = token_data.get('access_token')
        if not access_token:
            return Response({'error': '액세스 토큰 없음'}, status=status.HTTP_400_BAD_REQUEST)

        # 카카오 사용자 정보 받기
        user_info_url = "https://kapi.kakao.com/v2/user/me"
        headers = {"Authorization": f"Bearer {access_token}"}
        user_info_response = requests.get(user_info_url, headers=headers)
        if not user_info_response.ok:
            return Response({'error': '사용자 정보 받기 실패'}, status=status.HTTP_400_BAD_REQUEST)

        user_info = user_info_response.json()
        kakao_account = user_info.get('kakao_account', {})
        email = kakao_account.get('email')
        profile = kakao_account.get('profile', {})
        nickname = profile.get('nickname')

        logger.info(f"Kakao user info - email: {email}, nickname: {nickname}")

        if not email:
            return Response({'error': '이메일 정보 없음'}, status=status.HTTP_400_BAD_REQUEST)

        # 사용자 생성 또는 가져오기
        user, created = User.objects.get_or_create(
            email=email,
            defaults={'username': nickname, 'is_active': True}
        )
        if not created:
            user.username = nickname  # 닉네임 업데이트
            user.save()

        # 소셜 계정 생성 또는 업데이트
        social_account, _ = SocialAccount.objects.update_or_create(
            user=user,
            provider='kakao',
            defaults={'email': email, 'nickname': nickname}
        )

        logger.info(f"Social account updated - email: {email}, nickname: {nickname}")

        serializer = UserSerializer(user)
        response_data = {
            'user': {**serializer.data, 'loginType': 'kakao'},
            'access_token': access_token
        }

        return Response(response_data)

    except Exception as e:
        logger.exception("Unexpected error in kakao_callback")
        return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
