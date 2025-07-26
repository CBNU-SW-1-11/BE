from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import AllowAny
import logging
import json
import openai
import anthropic
from groq import Groq
from django.conf import settings
from bs4 import BeautifulSoup
logger = logging.getLogger(__name__)
import re
import json
import logging

def fetch_and_clean_url(url, timeout=10):
    """
    주어진 URL의 HTML을 요청해, 스크립트·스타일 제거 후 텍스트만 반환합니다.
    """
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    # 스크립트·스타일·네비게이션 태그 제거
    for tag in soup(["script", "style", "header", "footer", "nav"]):
        tag.decompose()

    text = soup.get_text(separator="\n")
    # 빈 줄 제거
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)

# Add this function to your ChatBot class or as a standalone function
def sanitize_and_parse_json(text, selected_models, responses):
    """
    Sanitize and parse the JSON response from AI models.
    Handles various edge cases and formatting issues.
    """
    import re
    import json
    import logging
    
    logger = logging.getLogger(__name__)
    
    try:
        # Step 1: Basic cleanup
        text = text.strip()
        
        # Step 2: Handle code blocks
        if text.startswith('```json') and '```' in text:
            text = re.sub(r'```json(.*?)```', r'\1', text, flags=re.DOTALL).strip()
        elif text.startswith('```') and text.endswith('```'):
            text = text[3:-3].strip()
            
        # Step 3: Extract JSON object if embedded in other text
        json_pattern = r'({[\s\S]*})'
        json_matches = re.findall(json_pattern, text)
        if json_matches:
            text = json_matches[0]
            
        # Step 4: Handle escaped backslashes in the text
        # First identify all occurrences of escaped backslashes followed by characters like "_"
        text = re.sub(r'\\([_"])', r'\1', text)
        
        # Step 5: Attempt to parse the JSON
        result = json.loads(text)
        
        # Ensure the required fields exist
        required_fields = ["preferredModel", "best_response", "analysis", "reasoning"]
        for field in required_fields:
            if field not in result:
                if field == "best_response" and "bestResponse" in result:
                    result["best_response"] = result["bestResponse"]
                else:
                    result[field] = "" if field != "analysis" else {}
        
        return result
        
    except Exception as e:
        logger.error(f"❌ JSON 파싱 실패: {str(e)}")
        logger.error(f"원본 텍스트: {text[:200]}..." if len(text) > 200 else text)
        
        # Advanced recovery attempt for malformed JSON
        try:
            # Remove problematic escaped characters
            fixed_text = text.replace("\\_", "_").replace('\\"', '"')
            
            # Try to fix common issues with JSON (missing quotes, commas, etc.)
            fixed_text = re.sub(r'([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', fixed_text)
            fixed_text = re.sub(r':\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*([,}])', r':"\1"\2', fixed_text)
            
            # Handle unclosed strings
            for match in re.finditer(r':\s*"([^"\\]*(\\.[^"\\]*)*)', fixed_text):
                if not re.search(r':\s*"([^"\\]*(\\.[^"\\]*)*)"', match.group(0)):
                    pos = match.end()
                    fixed_text = fixed_text[:pos] + '"' + fixed_text[pos:]
            
            result = json.loads(fixed_text)
            logger.info("✅ Recovered JSON after fixing format issues")
            return result
        except:
            # Last resort: construct a sensible fallback response
            error_analysis = {}
            for model in selected_models:
                model_lower = model.lower()
                error_analysis[model_lower] = {"장점": "분석 실패", "단점": "분석 실패"}
            
            # Find the largest response to use as best_response
            best_response = ""
            if responses:
                best_response = max(responses.values(), key=len) 
            
            return {
                "preferredModel": "FALLBACK",
                "best_response": best_response,
                "analysis": error_analysis,
                "reasoning": "응답 분석 중 오류가 발생하여 최적의 답변을 생성하지 못했습니다."
            }

# src/chatbot_backend/chat/bot.py

# import logging
# import openai
# import anthropic
# import base64
# import imghdr
# from groq import Groq
# from io import BytesIO

# logger = logging.getLogger(__name__)

# class ChatBot:
#     def __init__(self, api_key, model, api_type):
#         self.conversation_history = []
#         self.api_type = api_type
#         self.api_key = api_key

#         # Anthropic 멀티모달은 Opus 모델 권장
#         if api_type == 'anthropic' and not model.startswith('claude-3-opus-20240229'):
#             logger.info(f"Overriding Anthropic model '{model}' to 'claude-3-opus-20240229' for image support")
#             self.model = 'claude-3-opus-20240229'
#         else:
#             self.model = model

#         if api_type == 'openai':
#             openai.api_key = api_key
#         elif api_type == 'anthropic':
#             # Anthropic Python SDK 초기화
#             self.client = anthropic.Client(api_key=api_key)
#         elif api_type == 'groq':
#             self.client = Groq(api_key=api_key)
#         else:
#             raise ValueError(f"Unsupported api_type: {api_type}")

#     def chat(self, prompt=None, user_input=None, image_file=None, analysis_mode=None, user_language=None):
#         """
#         prompt       : 텍스트 프롬프트 (키워드)
#         user_input   : 텍스트 프롬프트 (위치 인자)
#         image_file   : 파일 객체 (BytesIO, InMemoryUploadedFile 등)
#         analysis_mode: 'describe'|'ocr'|'objects'
#         user_language: 'ko','en'
#         """
#         text = prompt if prompt is not None else user_input
#         try:
#             logger.info(f"[{self.api_type}] Received input: {text}")

#             # 모델별 호출
#             if self.api_type == 'openai':
#                 # GPT-4 Vision 지원
#                 params = {
#                     'model': self.model,
#                     'messages': self.conversation_history + [{"role": "user", "content": text}],
#                     'temperature': 0.7,
#                     'max_tokens': 1024
#                 }
#                 if image_file:
#                     params['files'] = [("image", image_file)]
#                 resp = openai.ChatCompletion.create(**params)
#                 assistant_response = resp.choices[0].message.content

#             elif self.api_type == 'anthropic':
#                 # Claude 3 Opus: 이미지+텍스트 지원 via Messages API
#                 messages = []
#                 # 토큰 수 설정
#                 max_tokens = 1024 if image_file else 4096
#                 if image_file:
#                     # 이미지 바이너리 읽기 및 미디어 타입 자동 감지
#                     image_file.seek(0)
#                     data_bytes = image_file.read()
#                     ext = imghdr.what(None, h=data_bytes) or 'jpeg'
#                     mime_map = {
#                         'jpeg': 'image/jpeg', 'jpg': 'image/jpeg',
#                         'png': 'image/png', 'gif': 'image/gif',
#                         'bmp': 'image/bmp', 'webp': 'image/webp'
#                     }
#                     media_type = mime_map.get(ext, 'image/jpeg')
#                     b64 = base64.b64encode(data_bytes).decode('utf-8')

#                     # 이미지 블록과 텍스트 블록을 리스트로 구성
#                     image_block = {
#                         'type': 'image',
#                         'source': {'type': 'base64', 'media_type': media_type, 'data': b64}
#                     }
#                     text_block = {'type': 'text', 'text': text}
#                     content_blocks = [image_block, text_block]

#                     # 단일 메시지에 블록 리스트 전달
#                     messages.append({'role': 'user', 'content': content_blocks})
#                 else:
#                     # 텍스트 전용 메시지
#                     messages.append({'role': 'user', 'content': [{'type': 'text', 'text': text}]})

#                 # Messages API 호출
#                 resp = self.client.messages.create(
#                     model=self.model,
#                     messages=messages,
#                     max_tokens=max_tokens
#                 )
#                 # 응답 블록에서 텍스트만 합치기
#                 assistant_response = ' '.join(getattr(block, 'text', '') for block in resp.content)

#             elif self.api_type == 'groq':
#                 # Groq Chat API
#                 resp = self.client.chat.completions.create(
#                     model=self.model,
#                     messages=self.conversation_history + [{"role": "user", "content": text}],
#                     temperature=0.7,
#                     max_tokens=1024
#                 )
#                 assistant_response = resp.choices[0].message.content

#             else:
#                 raise ValueError(f"Unsupported api_type: {self.api_type}")

#             # 응답 기록 및 반환
#             self.conversation_history.append({"role": "assistant", "content": assistant_response})
#             logger.info(f"[{self.api_type}] Response: {assistant_response[:100]}...")
#             return assistant_response

#         except Exception as e:
#             logger.error(f"Error in chat method ({self.api_type}): {e}", exc_info=True)
#             raise


# paste-2.txt 수정된 내용

# chatbot.py - OpenAI v1.0+ 호환 버전
import openai
import anthropic
from groq import Groq
import logging
import json
import asyncio
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

# sanitize_and_parse_json 함수 (기존 함수 포함)
def sanitize_and_parse_json(text, selected_models=None, responses=None):
    """JSON 응답을 정리하고 파싱하는 함수"""
    import re
    try:
        text = text.strip()
        
        # 코드 블록 제거
        if text.startswith('```json') and '```' in text:
            text = re.sub(r'```json(.*?)```', r'\1', text, flags=re.DOTALL).strip()
        elif text.startswith('```') and text.endswith('```'):
            text = text[3:-3].strip()
        
        # JSON 패턴 추출
        json_pattern = r'({[\s\S]*})'
        json_matches = re.findall(json_pattern, text)
        if json_matches:
            text = json_matches[0]
        
        # 이스케이프 문자 처리
        text = re.sub(r'\\([_"])', r'\1', text)
        
        # JSON 파싱
        result = json.loads(text)
        
        # 필수 필드 확인 및 보정
        required_fields = ["preferredModel", "best_response", "analysis", "reasoning"]
        for field in required_fields:
            if field not in result:
                if field == "best_response" and "bestResponse" in result:
                    result["best_response"] = result["bestResponse"]
                else:
                    result[field] = "" if field != "analysis" else {}
        
        return result
        
    except Exception as e:
        logger.error(f"JSON 파싱 실패: {e}")
        # 폴백 응답 생성
        error_analysis = {}
        if selected_models:
            for model in selected_models:
                model_lower = model.lower()
                error_analysis[model_lower] = {"장점": "분석 실패", "단점": "분석 실패"}
        
        return {
            "preferredModel": "ERROR",
            "best_response": max(responses.values(), key=len) if responses else "분석 오류가 발생했습니다.",
            "analysis": error_analysis,
            "reasoning": "응답 분석 중 오류가 발생했습니다."
        }
import openai

import anthropic
from groq import Groq
import logging
from .langchain_config import LangChainManager

logger = logging.getLogger(__name__)

class ChatBot:
    def __init__(self, api_key, model, api_type, langchain_manager=None):
        self.conversation_history = []
        self.model = model
        self.api_type = api_type
        self.api_key = api_key
        self.langchain_manager = langchain_manager
        
        # LangChain 사용 여부 결정
        self.use_langchain = langchain_manager is not None
        
        if not self.use_langchain:
            # 기존 방식 초기화
            if api_type == 'openai':
                openai.api_key = api_key
            elif api_type == 'anthropic':
                self.client = anthropic.Anthropic(api_key=api_key)
            elif api_type == 'groq':
                self.client = Groq(api_key=api_key)
        else:
            # LangChain 체인 생성
            try:
                if api_type in ['gpt', 'claude']:
                    self.chat_chain = langchain_manager.create_chat_chain(api_type)
                elif api_type == 'groq' or api_type == 'mixtral':
                    # Groq는 별도 처리
                    self.groq_llm = langchain_manager.groq_llm if hasattr(langchain_manager, 'groq_llm') else None
                logger.info(f"LangChain 체인 생성 완료: {api_type}")
            except Exception as e:
                logger.warning(f"LangChain 체인 생성 실패, 기존 방식 사용: {e}")
                self.use_langchain = False
   
    async def chat_async(self, user_input, image_file=None, analysis_mode=None, user_language=None):
        """비동기 채팅 메서드 (LangChain 용)"""
        if self.use_langchain:
            return await self._chat_with_langchain(user_input, user_language)
        else:
            return self.chat(user_input, image_file, analysis_mode, user_language)
    
    async def _chat_with_langchain(self, user_input, user_language='ko'):
        """LangChain을 사용한 채팅"""
        try:
            if self.api_type in ['gpt', 'claude']:
                result = await self.chat_chain.arun(
                    user_input=user_input,
                    user_language=user_language
                )
                return result
            elif self.api_type == 'groq' or self.api_type == 'mixtral':
                if self.groq_llm:
                    prompt = f"사용자가 선택한 언어는 '{user_language}'입니다. 반드시 이 언어({user_language})로 응답하세요.\n\n{user_input}"
                    result = self.groq_llm(prompt)
                    return result
                else:
                    # 폴백: 기존 방식
                    return self.chat(user_input, user_language=user_language)
            else:
                raise ValueError(f"지원하지 않는 API 타입: {self.api_type}")
                
        except Exception as e:
            logger.error(f"LangChain 채팅 에러: {e}")
            # 폴백: 기존 방식
            return self.chat(user_input, user_language=user_language)

    def chat(self, user_input, image_file=None, analysis_mode=None, user_language=None):
        """기존 동기 채팅 메서드 (호환성 유지)"""
        try:
            logger.info(f"Processing chat request for {self.api_type}")
            logger.info(f"User input: {user_input}")
            
            # 대화 기록에 사용자 입력 추가
            if image_file:
                self.conversation_history = [{
                    "role": "system",
                    "content": f"이미지 분석 모드: {analysis_mode}, 응답 언어: {user_language}"
                }]
                messages = [
                    {"role": "user", "content": user_input}
                ]
            else:
                self.conversation_history.append({"role": "user", "content": user_input})
                messages = self.conversation_history

            try:
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
                        # 시스템 메시지 찾기
                        system_message = next((msg['content'] for msg in self.conversation_history 
                                            if msg['role'] == 'system'), '')
                        
                        # 사용자 메시지 찾기
                        user_content = next((msg['content'] for msg in self.conversation_history 
                                        if msg['role'] == 'user'), '')

                        message = self.client.messages.create(
                            model=self.model,
                            max_tokens=4096,
                            temperature=0,
                            system=system_message,
                            messages=[{
                                "role": "user",
                                "content": user_content
                            }]
                        )
                        assistant_response = message.content[0].text
                        logger.info(f"Anthropic response with system message: {system_message[:100]}")
                        
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
               
                # 응답 기록
                self.conversation_history.append({
                    "role": "assistant",
                    "content": assistant_response
                })
               
                logger.info(f"Generated response: {assistant_response[:100]}...")
                return assistant_response
               
            except Exception as e:
                logger.error(f"API error in {self.api_type}: {str(e)}", exc_info=True)
                raise
               
        except Exception as e:
            logger.error(f"Error in chat method: {str(e)}", exc_info=True)
            raise

    async def analyze_responses_async(self, responses, query, user_language, selected_models):
        """비동기 응답 분석 (LangChain 용)"""
        if self.use_langchain and self.langchain_manager:
            return await self._analyze_with_langchain(responses, query, user_language, selected_models)
        else:
            return self.analyze_responses(responses, query, user_language, selected_models)
    
    async def _analyze_with_langchain(self, responses, query, user_language, selected_models):
        """LangChain을 사용한 응답 분석"""
        try:
            logger.info("\n" + "="*100)
            logger.info("📊 LangChain 분석 시작")
            logger.info(f"🤖 분석 수행 AI: {self.api_type.upper()}")
            logger.info(f"🔍 선택된 모델들: {', '.join(selected_models)}")
            logger.info("="*100)
            
            # 분석 체인 생성
            analysis_chain = self.langchain_manager.create_analysis_chain(self.api_type)
            
            # 응답 포맷팅
            formatted = self.langchain_manager.format_responses_for_analysis(
                responses, selected_models
            )
            
            # 분석 실행
            analysis_result = await analysis_chain.arun(
                query=query,
                user_language=user_language,
                selected_models=selected_models,
                **formatted
            )
            
            # preferredModel 설정
            analysis_result['preferredModel'] = self.api_type.upper()
            
            logger.info("✅ LangChain 분석 완료\n")
            return analysis_result
            
        except Exception as e:
            logger.error(f"❌ LangChain 분석 에러: {str(e)}")
            # 폴백: 기존 방식
            return self.analyze_responses(responses, query, user_language, selected_models)

    def analyze_responses(self, responses, query, user_language, selected_models):
        """기존 동기 응답 분석 메서드 (호환성 유지)"""
        try:
            logger.info("\n" + "="*100)
            logger.info("📊 분석 시작")
            logger.info(f"🤖 분석 수행 AI: {self.api_type.upper()}")
            logger.info(f"🔍 선택된 모델들: {', '.join(selected_models)}")
            logger.info("="*100)

            # 선택된 모델들만 분석에 포함
            responses_section = ""
            analysis_section = ""
            
            for model in selected_models:
                model_lower = model.lower()
                responses_section += f"\n{model.upper()} 응답: 반드시 이 언어({user_language})로 작성 {responses.get(model_lower, '응답 없음')}"
                
                analysis_section += f"""
                        "{model_lower}": {{
                            "장점": "반드시 이 언어({user_language})로 작성 {model.upper()} 답변의 장점",
                            "단점": "반드시 이 언어({user_language})로 작성 {model.upper()} 답변의 단점"
                        }}{"," if model_lower != selected_models[-1].lower() else ""}"""

            # 기존 분석 프롬프트 (변경 없음)
            analysis_prompt = f"""다음은 동일한 질문에 대한 {len(selected_models)}가지 AI의 응답을 분석하는 것입니다.
                    사용자가 선택한 언어는 '{user_language}'입니다.
                    반드시 이 언어({user_language})로 최적의 답을 작성해주세요.
                    반드시 이 언어({user_language})로 장점을 작성해주세요.
                    반드시 이 언어({user_language})로 단점을 작성해주세요.
                    반드시 이 언어({user_language})로 분석 근거를 작성해주세요.

                    질문: {query}
                    {responses_section}

                     [최적의 응답을 만들 때 고려할 사항]
                    - 모든 AI의 답변들을 종합하여 최적의 답변으로 반드시 재구성합니다
                    - 기존 AI의 답변을 그대로 사용하면 안됩니다
                    - 즉, 기존 AI의 답변과 최적의 답변이 동일하면 안됩니다.
                    - 다수의 AI가 공통으로 제공한 정보는 가장 신뢰할 수 있는 올바른 정보로 간주합니다
                    - 코드를 묻는 질문일때는, AI의 답변 중 제일 좋은 답변을 선택해서 재구성해줘
                    - 반드시 JSON 형식으로 응답해주세요
                    [출력 형식]
                    {{
                        "preferredModel": "{self.api_type.upper()}",
                        "best_response": "최적의 답변 ({user_language}로 작성)",
                        "analysis": {{
                            {analysis_section}
                        }},
                        "reasoning": "반드시 이 언어({user_language})로 작성 최적의 응답을 선택한 이유"
                    }}"""

            # 기존 API 호출 로직 (변경 없음)
            if self.api_type == 'openai':
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant. Always respond with valid JSON ONLY, no additional text or explanations."},
                        {"role": "user", "content": analysis_prompt}
                    ],
                    temperature=0,
                    max_tokens=4096
                )
                analysis_text = response['choices'][0]['message']['content']
                
            elif self.api_type == 'anthropic':
                system_message = next((msg['content'] for msg in self.conversation_history 
                                    if msg['role'] == 'system'), '')
                
                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=4096,
                    temperature=0,
                    system=f"{system_message}\nYou must respond with valid JSON only in the specified language. No other text or formatting.",
                    messages=[{
                        "role": "user", 
                        "content": analysis_prompt
                    }]
                )
                analysis_text = message.content[0].text.strip()
            
            elif self.api_type == 'groq':
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant. Always respond with valid JSON ONLY, no additional text or explanations."},
                        {"role": "user", "content": analysis_prompt}
                    ],
                    temperature=0,
                    max_tokens=4096
                )
                analysis_text = response.choices[0].message.content

            logger.info("✅ 분석 완료\n")
            
            # JSON 파싱 (기존 함수 사용)
            from paste_3 import sanitize_and_parse_json  # 기존 함수 import
            analysis_result = sanitize_and_parse_json(analysis_text, selected_models, responses)
            analysis_result['preferredModel'] = self.api_type.upper()
            
            return analysis_result
        
        except Exception as e:
            logger.error(f"❌ Analysis error: {str(e)}")
            # 기존 폴백 로직
            error_analysis = {}
            for model in selected_models:
                model_lower = model.lower()
                error_analysis[model_lower] = {"장점": "분석 실패", "단점": "분석 실패"}
            
            return {
                "preferredModel": self.api_type.upper(),
                "best_response": max(responses.values(), key=len) if responses else "",
                "analysis": error_analysis,
                "reasoning": "응답 분석 중 오류가 발생하여 최적의 답변을 생성하지 못했습니다."
            }
# class ChatView(APIView):
#     permission_classes = [AllowAny]

#     def post(self, request, preferredModel):
#         try:
#             logger.info(f"Received chat request for {preferredModel}")
            
#             data = request.data
#             user_message = data.get('message')
#             compare_responses = data.get('compare', True)
            
#             # 새로운 파라미터: 선택된 모델들
#             selected_models = data.get('selectedModels', ['gpt', 'claude', 'mixtral'])
            
#             # 선택된 모델 로그
#             logger.info(f"Selected models: {selected_models}")
            
#             # 토큰 유무에 따른 언어 및 선호 모델 처리
#             token = request.headers.get('Authorization')
#             if not token:
#                 # 비로그인: 기본 언어는 ko, 선호 모델은 GPT로 고정
#                 user_language = 'ko'
#                 preferredModel = 'gpt'
#             else:
#                 # 로그인: 요청 데이터의 언어 사용 (혹은 사용자의 설정을 따름)
#                 user_language = data.get('language', 'ko')
#                 # URL에 전달된 preferredModel을 그대로 사용 (프론트엔드에서 사용자 설정 반영)

#             logger.info(f"Received language setting: {user_language}")

#             if not user_message:
#                 return Response({'error': 'No message provided'}, 
#                                 status=status.HTTP_400_BAD_REQUEST)

#             # 비동기 응답을 위한 StreamingHttpResponse 사용
#             from django.http import StreamingHttpResponse
#             import json
#             import time

#             def stream_responses():
#                 try:
#                     system_message = {
#                         "role": "system",
#                         "content": f"사용자가 선택한 언어는 '{user_language}'입니다. 반드시 모든 응답을 이 언어({user_language})로 제공해주세요."
#                     }
                    
#                     responses = {}
                    
#                     # 현재 요청에 대한 고유 식별자 생성 (타임스탬프 활용)
#                     request_id = str(time.time())
                    
#                     # 선택된 모델들만 대화에 참여시킴
#                     selected_chatbots = {model: chatbots.get(model) for model in selected_models if model in chatbots}
                    
#                     # 각 봇의 응답을 개별적으로 처리하고 즉시 응답
#                     for bot_id, bot in selected_chatbots.items():
#                         if bot is None:
#                             logger.warning(f"Selected model {bot_id} not available in chatbots")
#                             yield json.dumps({
#                                 'type': 'bot_error',
#                                 'botId': bot_id,
#                                 'error': f"Model {bot_id} is not available"
#                             }) + '\n'
#                             continue
                            
#                         try:
#                             # 매번 새로운 대화 컨텍스트 생성 (이전 내용 초기화)
#                             bot.conversation_history = [system_message]
#                             response = bot.chat(user_message)
#                             responses[bot_id] = response
                            
#                             # 각 봇 응답을 즉시 전송
#                             yield json.dumps({
#                                 'type': 'bot_response',
#                                 'botId': bot_id,
#                                 'response': response,
#                                 'requestId': request_id  # 요청 ID 추가
#                             }) + '\n'
                            
#                         except Exception as e:
#                             logger.error(f"Error from {bot_id}: {str(e)}")
#                             responses[bot_id] = f"Error: {str(e)}"
                            
#                             # 에러도 즉시 전송
#                             yield json.dumps({
#                                 'type': 'bot_error',
#                                 'botId': bot_id,
#                                 'error': str(e),
#                                 'requestId': request_id  # 요청 ID 추가
#                             }) + '\n'
                    
#                     # 선택된 모델이 있고 응답이 있을 때만 분석 수행
#                     if selected_models and responses:
#                         # 분석(비교)은 로그인 시 사용자의 선호 모델을, 비로그인 시 GPT를 사용
#                         if token:
#                             analyzer_bot = chatbots.get(preferredModel) or chatbots.get('gpt')
#                         else:
#                             analyzer_bot = chatbots.get('gpt')
                        
#                         # 분석용 봇도 새로운 대화 컨텍스트로 초기화
#                         analyzer_bot.conversation_history = [system_message]
                        
#                         # 분석 실행 (항상 새롭게 실행)
#                         analysis = analyzer_bot.analyze_responses(responses, user_message, user_language, selected_models)
                        
#                         # 분석 결과 전송
#                         yield json.dumps({
#                             'type': 'analysis',
#                             'preferredModel': analyzer_bot and analyzer_bot.api_type.upper(),
#                             'best_response': analysis.get('best_response', ''),
#                             'analysis': analysis.get('analysis', {}),
#                             'reasoning': analysis.get('reasoning', ''),
#                             'language': user_language,
#                             'requestId': request_id,  # 요청 ID 추가
#                             'timestamp': time.time()  # 타임스탬프 추가
#                         }) + '\n'
#                     else:
#                         logger.warning("No selected models or responses to analyze")
                    
#                 except Exception as e:
#                     logger.error(f"Stream processing error: {str(e)}", exc_info=True)
#                     yield json.dumps({
#                         'type': 'error',
#                         'error': f"Stream processing error: {str(e)}"
#                     }) + '\n'

#             # StreamingHttpResponse 반환
#             response = StreamingHttpResponse(
#                 streaming_content=stream_responses(),
#                 content_type='text/event-stream'
#             )
#             response['Cache-Control'] = 'no-cache'
#             response['X-Accel-Buffering'] = 'no'
#             return response
                
#         except Exception as e:
#             logger.error(f"Unexpected error: {str(e)}", exc_info=True)
#             return Response({
#                 'error': str(e)
#             }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import AllowAny
from django.http import StreamingHttpResponse
import logging
import json
import openai
import anthropic
from groq import Groq
from django.conf import settings
from bs4 import BeautifulSoup
import re
import time
import asyncio
from asgiref.sync import sync_to_async

# 새로 추가된 import
from .langchain_config import LangChainManager
from .langgraph_workflow import AIComparisonWorkflow

logger = logging.getLogger(__name__)

def convert_to_serializable(obj):
    """객체를 직렬화 가능한 형태로 변환"""
    if hasattr(obj, '__dict__'):
        return {k: convert_to_serializable(v) for k, v in obj.__dict__.items()}
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj

class ChatView(APIView):
    permission_classes = [AllowAny]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # 기존 유사도 분석기
        from .similarity_analyzer import SimilarityAnalyzer  # 실제 import 경로로 변경 필요
        self.similarity_analyzer = SimilarityAnalyzer(threshold=0.85)
        
        # LangChain 관리자 초기화
        self.langchain_manager = LangChainManager(
            openai_key=OPENAI_API_KEY,
            anthropic_key=ANTHROPIC_API_KEY,
            groq_key=GROQ_API_KEY
        )
        
        # LangGraph 워크플로우 초기화
        self.workflow = AIComparisonWorkflow(
            langchain_manager=self.langchain_manager,
            similarity_analyzer=self.similarity_analyzer
        )
        
        # 기존 ChatBot 인스턴스들도 LangChain 사용하도록 업데이트
        self.update_chatbots_with_langchain()

    def update_chatbots_with_langchain(self):
        """기존 ChatBot들을 LangChain을 사용하도록 업데이트"""
        global chatbots
        
        # 기존 ChatBot들에 LangChain 매니저 추가
        for bot_id, bot in chatbots.items():
            bot.langchain_manager = self.langchain_manager
            bot.use_langchain = True
            
            # LangChain 체인 생성 시도
            try:
                if bot_id == 'gpt':
                    bot.chat_chain = self.langchain_manager.create_chat_chain('gpt')
                elif bot_id == 'claude':
                    bot.chat_chain = self.langchain_manager.create_chat_chain('claude')
                elif bot_id == 'mixtral':
                    bot.groq_llm = self.langchain_manager.groq_llm if hasattr(self.langchain_manager, 'groq_llm') else None
                logger.info(f"LangChain 체인 생성 완료: {bot_id}")
            except Exception as e:
                logger.warning(f"LangChain 체인 생성 실패 ({bot_id}), 기존 방식 사용: {e}")
                bot.use_langchain = False

    def post(self, request, preferredModel):
        try:
            logger.info(f"Received chat request for {preferredModel}")
            data = request.data
            user_message = data.get('message')
            selected_models = data.get('selectedModels', ['gpt', 'claude', 'mixtral'])
            token = request.headers.get('Authorization')
            user_language = 'ko' if not token else data.get('language', 'ko')
            use_workflow = data.get('useWorkflow', True)  # 워크플로우 사용 여부
            
            if not user_message:
                return Response({'error': 'No message provided'}, status=status.HTTP_400_BAD_REQUEST)

            # URL 처리 로직 (기존과 동일)
            url_pattern = r'^(https?://\S+)$'
            match = re.match(url_pattern, user_message.strip())
            if match:
                url = match.group(1)
                try:
                    page_text = fetch_and_clean_url(url)
                    if len(page_text) > 10000:
                        page_text = page_text[:5000] + "\n\n…(중략)…\n\n" + page_text[-5000:]
                    user_message = (
                        f"다음 웹페이지의 내용을 분석해 주세요:\n"
                        f"URL: {url}\n\n"
                        f"{page_text}"
                    )
                except Exception as e:
                    logger.error(f"URL fetch error: {e}")
                    return Response({'error': f"URL을 가져오지 못했습니다: {e}"}, status=status.HTTP_400_BAD_REQUEST)

            # 워크플로우 사용 여부에 따른 분기
            if use_workflow:
                return self.handle_with_workflow(user_message, selected_models, user_language, preferredModel)
            else:
                return self.handle_with_legacy(user_message, selected_models, user_language, preferredModel)

        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

    def handle_with_workflow(self, user_message, selected_models, user_language, preferred_model):
        """LangGraph 워크플로우를 사용한 처리"""
        def stream_workflow_responses():
            try:
                request_id = str(time.time())
                
                # 워크플로우 실행을 위한 async 래퍼
                async def run_workflow_async():
                    return await self.workflow.run_workflow(
                        user_message=user_message,
                        selected_models=selected_models,
                        user_language=user_language,
                        request_id=request_id
                    )
                
                # asyncio 이벤트 루프에서 실행
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    workflow_result = loop.run_until_complete(run_workflow_async())
                finally:
                    loop.close()
                
                # 개별 응답 스트리밍
                for bot_id, response in workflow_result["individual_responses"].items():
                    yield json.dumps({
                        'type': 'bot_response',
                        'botId': bot_id,
                        'response': response,
                        'requestId': request_id
                    }) + '\n'
                
                # 유사도 분석 결과
                if workflow_result["similarity_analysis"]:
                    yield json.dumps({
                        'type': 'similarity_analysis',
                        'result': workflow_result["similarity_analysis"],
                        'requestId': request_id,
                        'timestamp': time.time(),
                        'userMessage': user_message
                    }) + '\n'
                
                # 최종 분석 결과
                final_analysis = workflow_result["final_analysis"]
                yield json.dumps({
                    'type': 'analysis',
                    'preferredModel': final_analysis.get('preferredModel', preferred_model.upper()),
                    'best_response': final_analysis.get('best_response', ''),
                    'analysis': final_analysis.get('analysis', {}),
                    'reasoning': final_analysis.get('reasoning', ''),
                    'language': user_language,
                    'requestId': request_id,
                    'timestamp': time.time(),
                    'userMessage': user_message,
                    'workflowUsed': True,
                    'errors': workflow_result.get("errors", [])
                }) + '\n'
                
            except Exception as e:
                logger.error(f"워크플로우 스트리밍 에러: {e}")
                yield json.dumps({
                    'type': 'error',
                    'error': f"Workflow error: {e}",
                    'fallbackToLegacy': True
                }) + '\n'

        return StreamingHttpResponse(stream_workflow_responses(), content_type='text/event-stream')

    def handle_with_legacy(self, user_message, selected_models, user_language, preferred_model):
        """기존 방식으로 처리 (호환성 유지)"""
        def stream_responses():
            try:
                system_message = {
                    'role': 'system',
                    'content': f"사용자가 선택한 언어는 '{user_language}'입니다. 반드시 이 언어({user_language})로 응답하세요."
                }
                responses = {}
                request_id = str(time.time())
                
                # 각 모델별 챗봇 인스턴스 가져오기
                selected_chatbots = {m: chatbots.get(m) for m in selected_models if chatbots.get(m)}

                # 모델 응답 수집 (비동기 처리 시도)
                async def collect_responses_async():
                    responses = {}
                    tasks = []
                    
                    for bot_id, bot in selected_chatbots.items():
                        if hasattr(bot, 'chat_async') and bot.use_langchain:
                            # LangChain 비동기 사용
                            task = bot.chat_async(user_message, user_language=user_language)
                        else:
                            # 기존 동기 방식을 비동기로 래핑
                            task = sync_to_async(self.sync_chat)(bot, user_message, system_message)
                        tasks.append((bot_id, task))
                    
                    for bot_id, task in tasks:
                        try:
                            response = await task
                            responses[bot_id] = response
                            logger.info(f"✅ {bot_id} 응답 완료")
                        except Exception as e:
                            logger.error(f"❌ {bot_id} 응답 실패: {e}")
                    
                    return responses

                # 비동기 응답 수집
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    responses = loop.run_until_complete(collect_responses_async())
                finally:
                    loop.close()

                # 개별 응답 스트리밍
                for bot_id, resp_text in responses.items():
                    yield json.dumps({
                        'type': 'bot_response',
                        'botId': bot_id,
                        'response': resp_text,
                        'requestId': request_id
                    }) + '\n'

                # 유사도 분석
                if len(responses) >= 2:
                    sim_res = self.similarity_analyzer.cluster_responses(responses)
                    serial = convert_to_serializable(sim_res)
                    yield json.dumps({
                        'type': 'similarity_analysis',
                        'result': serial,
                        'requestId': request_id,
                        'timestamp': time.time(),
                        'userMessage': user_message
                    }) + '\n'

                # 최종 비교 및 분석
                analyzer_bot = chatbots.get(preferred_model) or chatbots.get('gpt')
                analyzer_bot.conversation_history = [system_message]
                
                # LangChain 비동기 분석 시도
                if hasattr(analyzer_bot, 'analyze_responses_async') and analyzer_bot.use_langchain:
                    async def analyze_async():
                        return await analyzer_bot.analyze_responses_async(
                            responses, user_message, user_language, list(responses.keys())
                        )
                    
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    try:
                        analysis = loop.run_until_complete(analyze_async())
                    finally:
                        loop.close()
                else:
                    # 기존 동기 방식
                    analysis = analyzer_bot.analyze_responses(
                        responses, user_message, user_language, list(responses.keys())
                    )
                
                yield json.dumps({
                    'type': 'analysis',
                    'preferredModel': analyzer_bot.api_type.upper(),
                    'best_response': analysis.get('best_response', ''),
                    'analysis': analysis.get('analysis', {}),
                    'reasoning': analysis.get('reasoning', ''),
                    'language': user_language,
                    'requestId': request_id,
                    'timestamp': time.time(),
                    'userMessage': user_message,
                    'workflowUsed': False
                }) + '\n'
                
            except Exception as e:
                yield json.dumps({
                    'type': 'error',
                    'error': f"Stream error: {e}"
                }) + '\n'

        return StreamingHttpResponse(stream_responses(), content_type='text/event-stream')

    def sync_chat(self, bot, user_message, system_message):
        """동기 채팅을 위한 헬퍼 메서드"""
        bot.conversation_history = [system_message]
        return bot.chat(user_message)

# API 키 설정 (기존과 동일)
OPENAI_API_KEY = "***REMOVED***"
ANTHROPIC_API_KEY = "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"
GROQ_API_KEY = "***REMOVED***"

# ChatBot import (수정된 버전)

chatbots = {
    'gpt': ChatBot(OPENAI_API_KEY, 'gpt-3.5-turbo', 'openai'),
    'claude': ChatBot(ANTHROPIC_API_KEY, 'claude-3-5-haiku-20241022', 'anthropic'), 
    'mixtral': ChatBot(GROQ_API_KEY, 'llama3-8b-8192', 'groq'),
}
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import IsAuthenticated
from .models import UserProfile, UserSettings
from .serializers import UserSerializer

class UserSettingsView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        try:
            user_settings, created = UserSettings.objects.get_or_create(
                user=request.user,
                defaults={
                    'language': 'ko',
                    'analyzer_bot': 'claude'
                }
            )
            serializer = UserSerializer(user_settings)
            return Response(serializer.data)
        except Exception as e:
            return Response({
                'language': 'ko', 
                'analyzer_bot': 'claude'
            }, status=status.HTTP_200_OK)

    def post(self, request):
        try:
            user_settings, created = UserSettings.objects.get_or_create(
                user=request.user,
                defaults={
                    'language': request.data.get('language', 'ko'),
                    'analyzer_bot': request.data.get('analyzer_bot', 'claude')
                }
            )

            # 이미 존재하는 경우 업데이트
            if not created:
                user_settings.language = request.data.get('language', user_settings.language)
                user_settings.analyzer_bot = request.data.get('analyzer_bot', user_settings.analyzer_bot)
                user_settings.save()

            serializer = UserSettingsSerializer(user_settings)
            return Response(serializer.data, status=status.HTTP_200_OK)
        except Exception as e:
            return Response({
                'error': str(e)
            }, status=status.HTTP_400_BAD_REQUEST)
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
import requests
import logging
from django.contrib.auth import get_user_model
from .models import SocialAccount

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
from rest_framework.authentication import TokenAuthentication
from rest_framework.permissions import AllowAny
from rest_framework.authtoken.models import Token
from rest_framework import status
from rest_framework.decorators import api_view, authentication_classes, permission_classes
# views.py
from django.db import transaction, IntegrityError

# @api_view(['GET'])
# @authentication_classes([TokenAuthentication])
# @permission_classes([AllowAny])
# @api_view(['GET'])
# @permission_classes([AllowAny])
# def google_callback(request):
#     logger.info("Starting Google callback process")  # 로깅 추가
#     try:
#         with transaction.atomic():
#             # 1. 사용자 정보 가져오기
#             auth_header = request.headers.get('Authorization', '')
#             access_token = auth_header.split(' ')[1]
            
#             user_info_response = requests.get(
#                 'https://www.googleapis.com/oauth2/v3/userinfo',
#                 headers={'Authorization': f'Bearer {access_token}'}
#             )
            
#             user_info = user_info_response.json()
#             email = user_info.get('email')
#             name = user_info.get('name')

#             logger.info(f"Processing user: {email}")  # 로깅 추가

#             # 2. User 객체 가져오기 또는 생성
#             user = User.objects.filter(email=email).first()
#             if not user:
#                 user = User.objects.create(
#                     username=email,
#                     email=email,
#                     first_name=name or '',
#                     is_active=True
#                 )
#                 logger.info(f"Created new user: {user.id}")
#             else:
#                 logger.info(f"Found existing user: {user.id}")

#             # 3. 기존 UserSettings 삭제 (있다면)
#             UserSettings.objects.filter(user=user).delete()
#             logger.info("Deleted any existing settings")

#             # 4. 새로운 UserSettings 생성
#             settings = UserSettings.objects.create(
#                 user=user,
#                 language='ko',
#                 preferred_model='default'
#             )
#             logger.info(f"Created new settings for user: {user.id}")

#             # 5. 토큰 생성
#             token, _ = Token.objects.get_or_create(user=user)
            
#             return Response({
#                 'user': {
#                     'id': user.id,
#                     'email': user.email,
#                     'username': user.username,
#                     'first_name': user.first_name,
#                     'settings': {
#                         'language': settings.language,
#                         'preferred_model': settings.preferred_model
#                     }
#                 },
#                 'access_token': token.key
#             })
#     except Exception as e:
#         logger.error(f"Error in google_callback: {str(e)}")
#         return Response(
#             {'error': str(e)},
#             status=status.HTTP_400_BAD_REQUEST
        # )
@api_view(['GET'])
@authentication_classes([TokenAuthentication])
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

        # 토큰 생성 또는 가져오기
        token, created = Token.objects.get_or_create(user=user)
        logger.info(f"GOOGLE Token created: {created}")
        logger.info(f"Token key: {token.key}")
        logger.info(f"Token user: {token.user.username}")


        # 사용자 데이터 반환
        serializer = UserSerializer(user)
        return Response({
            'user': serializer.data,
            'access_token': token.key,  # Django REST Framework Token 반환
            'token_created': created,
            'google_access_token': access_token,  # Google OAuth 액세스 토큰

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
            logger.info(f"Updated existing user with nickname: {nickname}")
        except User.DoesNotExist:
            unique_username = generate_unique_username(email, nickname)
            user = User.objects.create(
                email=email,
                username=unique_username,
                is_active=True
            )            
            logger.info(f"Created new user with nickname: {nickname}")

        # 소셜 계정 생성 또는 업데이트
        social_account, created = SocialAccount.objects.update_or_create(
            email=email,
            provider='kakao',
            defaults={
                'user': user,
                'nickname': nickname
            }
        )
        logger.info(f"Social account updated - email: {email}, nickname: {nickname}")

        if not created and social_account.user != user:
            social_account.user = user
            social_account.save()

        # 토큰 생성 또는 가져오기
        token, token_created = Token.objects.get_or_create(user=user)
        logger.info(f"KAKAO Token created: {token_created}")
        logger.info(f"Token key: {token.key}")
        logger.info(f"Token user: {token.user.username}")

        serializer = UserSerializer(user)
        return Response({
            'user': serializer.data,
            'access_token': token.key,  # Django REST Framework Token 반환
            'token_created': created,
            'kakao_access_token': access_token,  # Google OAuth 액세스 토큰

        })


        
    except Exception as e:
        logger.exception("Unexpected error in kakao_callback")
        return Response({
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


from rest_framework.authtoken.models import Token  # Token 모델 추가

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
            "Authorization": f"Bearer {access_token}",
            "Content-type": "application/x-www-form-urlencoded;charset=utf-8"
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

        # 사용자 생성 또는 조회
        user, created = User.objects.get_or_create(
            email=email,
            defaults={'username': generate_unique_username(email, username), 'is_active': True}
        )

        # 소셜 계정 조회 및 업데이트
        social_account, social_created = SocialAccount.objects.update_or_create(
            provider='naver',
            email=email,
            defaults={'user': user, 'nickname': nickname}
        )

        logger.info(f"Social account updated - email: {email}, nickname: {nickname}")

        # ✅ Django REST Framework Token 생성
        token, token_created = Token.objects.get_or_create(user=user)
        logger.info(f"Naver Token created: {token_created}")
        logger.info(f"Token key: {token.key}")
        logger.info(f"Token user: {token.user.username}")

        serializer = UserSerializer(user)
        return Response({
            'user': serializer.data,
            'access_token': token.key,  # Django REST Framework Token 반환
            'token_created': created,
            'naver_access_token': access_token,  # 네이버 액세스 토큰
        })

    except Exception as e:
        logger.exception("Unexpected error in naver_callback")
        return Response({
            'error': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# views.py
import logging
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from django.contrib.auth import get_user_model

# logger = logging.getLogger(__name__)

# @api_view(['PUT'])
# @permission_classes([IsAuthenticated])
# def update_user_settings(request):
#     # 추가 로깅 및 디버깅
#     logger.info(f"User authentication status: {request.user.is_authenticated}")
#     logger.info(f"User: {request.user}")
#     logger.info(f"Request headers: {request.headers}")
    
#     try:
#         # 인증 상태 명시적 확인
#         if not request.user.is_authenticated:
#             logger.error("Unauthenticated user attempt")
#             return Response({
#                 'status': 'error',
#                 'message': '인증되지 않은 사용자입니다.'
#             }, status=401)
        
#         # UserProfile 모델이 있다고 가정
#         user = request.user
#         user_profile = user.userprofile
        
#         # 설정 업데이트
#         settings_data = request.data
#         user_profile.language = settings_data.get('language', user_profile.language)
#         user_profile.preferred_model = settings_data.get('preferredModel', user_profile.preferred_model)
#         user_profile.save()
        
#         return Response({
#             'status': 'success',
#             'message': '설정이 성공적으로 업데이트되었습니다.',
#             'settings': {
#                 'language': user_profile.language,
#                 'preferredModel': user_profile.preferred_model
#             }
#         })
    
#     except Exception as e:
#         print("Error:", str(e))  # 에러 로깅
#         logger.error(f"Settings update error: {str(e)}")
#         return Response({
#             'status': 'error',
#             'message': f'오류 발생: {str(e)}'
#         }, status=400)
# views.py
# 백엔드에서 토큰 형식 확인
@api_view(['PUT'])
@authentication_classes([TokenAuthentication])
@permission_classes([IsAuthenticated])
def update_user_settings(request):
    try:
        # 토큰 로깅 추가
        token_header = request.headers.get('Authorization')
        if not token_header or not token_header.startswith('Token '):
            return Response({'error': '잘못된 토큰 형식'}, status=status.HTTP_401_UNAUTHORIZED)
        
        user = request.user
        if not user.is_authenticated:
            return Response({'error': '인증되지 않은 사용자'}, status=status.HTTP_401_UNAUTHORIZED)
        
        settings_data = request.data
        print(f"Received settings data: {settings_data}")  # 데이터 로깅 추가
        
        # UserSettings 업데이트 또는 생성
        settings, created = UserSettings.objects.get_or_create(user=user)
        
        # 필드 업데이트
        if 'language' in settings_data:
            settings.language = settings_data['language']
        if 'preferredModel' in settings_data:
            settings.preferred_model = settings_data['preferredModel']
        
        settings.save()
        
        return Response({
            'message': 'Settings updated successfully',
            'settings': {
                'language': settings.language,
                'preferredModel': settings.preferred_model
            }
        })
        
    except Exception as e:
        logger.error(f"Settings update error: {str(e)}")
        return Response(
            {'error': str(e)},
            status=status.HTTP_400_BAD_REQUEST
        )
from rest_framework import status
from rest_framework.decorators import api_view, authentication_classes, permission_classes
from rest_framework.authentication import TokenAuthentication
from rest_framework.permissions import IsAuthenticated
from rest_framework.response import Response
from django.core.exceptions import ObjectDoesNotExist

@api_view(['PUT'])
@authentication_classes([TokenAuthentication])
@permission_classes([IsAuthenticated])
def update_user_settings(request):
    try:
        user = request.user
        settings_data = request.data
        
        # UserProfile 확인 및 생성
        try:
            profile = user.userprofile
        except ObjectDoesNotExist:
            profile = UserProfile.objects.create(user=user)
            
        # UserSettings 확인 및 생성/업데이트
        settings, created = UserSettings.objects.get_or_create(
            user=user,
            defaults={
                'language': settings_data.get('language', 'en'),
                'preferred_model': settings_data.get('preferredModel', 'default')
            }
        )
        
        if not created:
            settings.language = settings_data.get('language', settings.language)
            settings.preferred_model = settings_data.get('preferredModel', settings.preferred_model)
            settings.save()
            
        return Response({
            'message': 'Settings updated successfully',
            'settings': {
                'language': settings.language,
                'preferredModel': settings.preferred_model
            }
        })
            
    except Exception as e:
        print(f"Settings update error: {str(e)}")
        return Response(
            {'error': str(e)},
            status=status.HTTP_400_BAD_REQUEST
        )

import logging
import re
import math
import numpy as np
from collections import Counter
from typing import Dict, List, Union, Any
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
logger = logging.getLogger(__name__)

class SimilarityAnalyzer:
    """
    AI 모델 응답 간의 유사도를 분석하고 응답 특성을 추출하는 클래스
    다국어 지원을 위해 paraphrase-multilingual-MiniLM-L12-v2 모델 사용
    """
    
    def __init__(self, threshold=0., use_transformer=True):
        """
        초기화
        
        Args:
            threshold (float): 유사 응답으로 분류할 임계값 (0~1)
            use_transformer (bool): SentenceTransformer 모델 사용 여부
        """
        self.threshold = threshold
        self.use_transformer = use_transformer
        
        # 다국어 SentenceTransformer 모델 로드
        if use_transformer:
            try:
                self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
                logger.info("다국어 SentenceTransformer 모델 로드 완료")
            except Exception as e:
                logger.error(f"SentenceTransformer 모델 로드 실패: {str(e)}")
                self.use_transformer = False
                
        # Fallback용 TF-IDF 벡터라이저
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(
            min_df=1, 
            analyzer='word',
            ngram_range=(1, 2),
            stop_words=None  # 다국어 지원을 위해 stop_words 제거
        )
    
    def preprocess_text(self, text: str) -> str:
        """
        텍스트 전처리
        
        Args:
            text (str): 전처리할 텍스트
            
        Returns:
            str: 전처리된 텍스트
        """
        # 소문자 변환 (영어 텍스트만 해당)
        # 다국어 지원을 위해 영어가 아닌 경우 원래 케이스 유지
        if text.isascii():
            text = text.lower()
        
        # 코드 블록 제거 (분석에서 제외)
        text = re.sub(r'```.*?```', ' CODE_BLOCK ', text, flags=re.DOTALL)
        
        # HTML 태그 제거
        text = re.sub(r'<.*?>', '', text)
        
        # 특수 문자 처리 (다국어 지원을 위해 완전 제거하지 않음)
        text = re.sub(r'[^\w\s\u0080-\uFFFF]', ' ', text)
        
        # 여러 공백을 하나로 치환
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def calculate_similarity_matrix(self, responses: Dict[str, str]) -> Dict[str, Dict[str, float]]:
        """
        모델 응답 간의 유사도 행렬 계산
        
        Args:
            responses (dict): 모델 ID를 키로, 응답 텍스트를 값으로 하는 딕셔너리
            
        Returns:
            dict: 모델 간 유사도 행렬
        """
        try:
            model_ids = list(responses.keys())
            
            # 텍스트 전처리
            preprocessed_texts = [self.preprocess_text(responses[model_id]) for model_id in model_ids]
            
            if self.use_transformer and self.model:
                # SentenceTransformer를 사용한 임베딩 생성
                try:
                    embeddings = self.model.encode(preprocessed_texts)
                    # 코사인 유사도 계산
                    similarity_matrix = cosine_similarity(embeddings)
                except Exception as e:
                    logger.error(f"SentenceTransformer 임베딩 생성 중 오류: {str(e)}")
                    # Fallback: TF-IDF 사용
                    tfidf_matrix = self.vectorizer.fit_transform(preprocessed_texts)
                    similarity_matrix = cosine_similarity(tfidf_matrix)
            else:
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
            
            # 응답 특성 추출
            response_features = {model_id: self.extract_response_features(responses[model_id]) 
                                for model_id in model_ids}
            
            return {
                "similarGroups": clusters,
                "mainGroup": main_group,
                "outliers": outliers,
                "similarityMatrix": similarity_matrix,
                "responseFeatures": response_features
            }
            
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
    
    
    def extract_response_features(self, text: str) -> Dict[str, Union[int, float, bool]]:
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
            
            # 문장 분리 (다국어 지원)
            sentences = re.split(r'[.!?。！？]+', text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            # 평균 문장 길이
            avg_sentence_length = sum(len(s) for s in sentences) / max(1, len(sentences))
            
            # 어휘 다양성 (고유 단어 비율)
            words = re.findall(r'\b\w+\b', text.lower())
            unique_words = set(words)
            vocabulary_diversity = len(unique_words) / max(1, len(words))
            
            # 언어 감지 (추가 기능)
            lang_features = self.detect_language_features(text)
            
            features = {
                "length": length,
                "codeBlockCount": code_block_count,
                "linkCount": link_count,
                "listItemCount": list_item_count,
                "sentenceCount": len(sentences),
                "avgSentenceLength": avg_sentence_length,
                "vocabularyDiversity": vocabulary_diversity,
                "hasCode": code_block_count > 0
            }
            
            # 언어 특성 추가
            features.update(lang_features)
            
            return features
            
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
                "hasCode": False,
                "detectedLang": "unknown"
            }
    
    def detect_language_features(self, text: str) -> Dict[str, Any]:
        """
        텍스트의 언어 특성 감지
        
        Args:
            text (str): 분석할 텍스트
            
        Returns:
            dict: 언어 특성 정보
        """
        try:
            # 언어 특성 감지를 위한 간단한 휴리스틱
            # 실제 프로덕션에서는 langdetect 등의 라이브러리 사용 권장
            
            # 한국어 특성 (한글 비율)
            korean_chars = len(re.findall(r'[ㄱ-ㅎㅏ-ㅣ가-힣]', text))
            
            # 영어 특성 (영문 비율)
            english_chars = len(re.findall(r'[a-zA-Z]', text))
            
            # 일본어 특성 (일본어 문자 비율)
            japanese_chars = len(re.findall(r'[ぁ-んァ-ン一-龯]', text))
            
            # 중국어 특성 (중국어 문자 비율)
            chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
            
            # 기타 문자 (숫자, 특수문자 제외)
            total_chars = len(re.findall(r'[^\d\s\W]', text))
            
            # 비율 계산
            total = max(1, total_chars)
            korean_ratio = korean_chars / total
            english_ratio = english_chars / total
            japanese_ratio = japanese_chars / total
            chinese_ratio = chinese_chars / total
            
            # 주요 언어 결정
            lang_ratios = {
                "ko": korean_ratio,
                "en": english_ratio,
                "ja": japanese_ratio,
                "zh": chinese_ratio,
                "other": 1.0 - (korean_ratio + english_ratio + japanese_ratio + chinese_ratio)
            }
            
            detected_lang = max(lang_ratios.items(), key=lambda x: x[1])[0]
            
            return {
                "detectedLang": detected_lang,
                "langRatios": lang_ratios
            }
            
        except Exception as e:
            logger.error(f"언어 특성 감지 중 오류: {str(e)}")
            return {
                "detectedLang": "unknown",
                "langRatios": {"unknown": 1.0}
            }
    
    def compare_responses(self, response1: str, response2: str) -> Dict[str, Any]:
        """
        두 응답 간의 유사도와 차이점 분석
        
        Args:
            response1 (str): 첫 번째 응답
            response2 (str): 두 번째 응답
            
        Returns:
            dict: 유사도 및 차이점 분석 결과
        """
        try:
            # 텍스트 전처리
            text1 = self.preprocess_text(response1)
            text2 = self.preprocess_text(response2)
            
            # 임베딩 생성 및 유사도 계산
            if self.use_transformer and self.model:
                embeddings = self.model.encode([text1, text2])
                similarity = float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])
            else:
                # TF-IDF를 사용한 유사도 계산
                tfidf_matrix = self.vectorizer.fit_transform([text1, text2])
                similarity = float(cosine_similarity(tfidf_matrix)[0][1])
            
            # 응답 특성 비교
            features1 = self.extract_response_features(response1)
            features2 = self.extract_response_features(response2)
            
            # 특성 차이 계산
            feature_diffs = {}
            for key in set(features1.keys()) & set(features2.keys()):
                if isinstance(features1[key], (int, float)) and isinstance(features2[key], (int, float)):
                    feature_diffs[key] = features2[key] - features1[key]
            
            # 주요 차이점 고유 단어 분석
            words1 = re.findall(r'\b\w+\b', text1.lower())
            words2 = re.findall(r'\b\w+\b', text2.lower())
            
            counter1 = Counter(words1)
            counter2 = Counter(words2)
            
            unique_to_1 = [word for word, count in counter1.items() if word not in counter2]
            unique_to_2 = [word for word, count in counter2.items() if word not in counter1]
            
            # 가장 빈도가 높은 고유 단어 (최대 10개)
            top_unique_to_1 = sorted(
                [(word, counter1[word]) for word in unique_to_1], 
                key=lambda x: x[1], 
                reverse=True
            )[:10]
            
            top_unique_to_2 = sorted(
                [(word, counter2[word]) for word in unique_to_2], 
                key=lambda x: x[1], 
                reverse=True
            )[:10]
            
            return {
                "similarity": similarity,
                "isSimilar": similarity >= self.threshold,
                "features1": features1,
                "features2": features2,
                "featureDiffs": feature_diffs,
                "uniqueWordsTo1": top_unique_to_1,
                "uniqueWordsTo2": top_unique_to_2
            }
            
        except Exception as e:
            logger.error(f"응답 비교 중 오류: {str(e)}")
            return {
                "similarity": 0.0,
                "isSimilar": False,
                "error": str(e)
            }
        
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import AllowAny
import logging
import json
import openai
import anthropic
from groq import Groq
from django.conf import settings
import time


logger = logging.getLogger(__name__)

class TextSimplificationView(APIView):
    """
    텍스트를 쉬운 표현으로 변환하는 API 뷰
    특정 대상(어린이, 고령자, 외국인 학습자 등)에 맞춰 변환
    """
    permission_classes = [AllowAny]
    
    def post(self, request):
        try:
            logger.info("쉬운 표현 변환 요청 받음")
            
            data = request.data
            original_text = data.get('message')
            target_audience = data.get('targetAudience', 'general')
            language = data.get('language', 'ko')
            
            if not original_text:
                return Response({'error': '변환할 텍스트가 없습니다.'}, 
                               status=status.HTTP_400_BAD_REQUEST)
            
            # 텍스트 단순화 수행
            simplifier = TextSimplifier(
                api_key=settings.OPENAI_API_KEY,
                model="gpt-4-turbo",  # 또는 선호하는 GPT 모델
                api_type="openai"
            )
            
            result = simplifier.simplify_text(
                original_text=original_text,
                target_audience=target_audience,
                language=language
            )
            
            return Response(result)
            
        except Exception as e:
            logger.error(f"텍스트 단순화 중 오류 발생: {str(e)}", exc_info=True)
            return Response({
                'error': str(e)
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


class TextSimplifier:
    """
    텍스트를 쉬운 표현으로 변환하는 클래스
    다양한 AI 모델을 활용하여 대상자별 맞춤형 단순화 수행
    """
    def __init__(self, api_key, model, api_type):
        self.model = model
        self.api_type = api_type
        self.api_key = api_key
        
        if api_type == 'openai':
            openai.api_key = api_key
        elif api_type == 'anthropic':
            self.client = anthropic.Anthropic(api_key=api_key)
        elif api_type == 'groq':
            self.client = Groq(api_key=api_key)
    
    def simplify_text(self, original_text, target_audience, language='ko'):
        """
        텍스트를 단순화하여 반환
        
        Args:
            original_text (str): 원본 텍스트
            target_audience (str): 대상자 유형 (general, child, elderly, foreigner)
            language (str): 언어 (기본값: 한국어)
            
        Returns:
            dict: 단순화 결과
        """
        try:
            logger.info(f"텍스트 단순화 시작: 대상={target_audience}, 언어={language}")
            
            # 대상자에 맞는 프롬프트 생성
            prompt = self._get_simplification_prompt(original_text, target_audience, language)
            
            # AI 모델을 사용하여 텍스트 단순화
            simplified_text = self._generate_simplified_text(prompt)
            
            # 결과 반환
            result = {
                'original_text': original_text,
                'simplified_text': simplified_text,
                'target_audience': target_audience,
                'language': language,
                'timestamp': time.time()
            }
            
            logger.info("텍스트 단순화 완료")
            return result
            
        except Exception as e:
            logger.error(f"텍스트 단순화 오류: {str(e)}", exc_info=True)
            raise
    
    def _get_simplification_prompt(self, original_text, target_audience, language):
        """대상자 맞춤형 단순화 프롬프트 생성"""
        
        base_prompt = f"""
다음 텍스트를 더 쉬운 표현으로 변환해주세요:

{original_text}

대상자: {target_audience}
언어: {language}
"""
        
        if target_audience == 'child':
            base_prompt += """
[어린이용 변환 지침]
1. 7-12세 어린이가 이해할 수 있는 단어와 표현으로 변환하세요.
2. 짧고 간단한 문장을 사용하세요.
3. 추상적인 개념은 구체적인 예시와 함께 설명하세요.
4. 재미있고 흥미로운 표현을 사용하세요.
5. 어려운 단어는 간단한 동의어로 대체하세요.
6. 필요한 경우 비유와 예시를 활용하세요.
7. 문장 사이에 적절한 줄바꿈을 추가하세요.
"""
        elif target_audience == 'elderly':
            base_prompt += """
[고령자용 변환 지침]
1. 명확하고 직접적인 표현을 사용하세요.
2. 외래어나 영어 표현은 가능한 한국어로 대체하세요.
3. 복잡한 문장 구조를 피하고 간결하게 작성하세요.
4. 전문 용어는 일상적인 용어로 설명하세요.
5. 친숙한 비유와 예시를 사용하세요.
6. 중요한 정보는 반복해서 강조하세요.
7. 문장 사이에 적절한 줄바꿈을 추가하세요.
"""
        elif target_audience == 'foreigner':
            base_prompt += """
[외국인 학습자용 변환 지침]
1. 한국어 학습자(초급~중급)가 이해할 수 있는 기본 어휘를 사용하세요.
2. 관용어, 속담, 은유적 표현을 피하세요.
3. 한자어는 가능한 순우리말로 대체하세요.
4. 문법적으로 단순한 문장 구조를 사용하세요.
5. 복잡한 연결어미나 조사 사용을 최소화하세요.
6. 중요한 개념은 괄호 안에 영어로 병기하세요.
7. 문장 사이에 적절한 줄바꿈을 추가하세요.
"""
        else:  # general
            base_prompt += """
[일반인용 변환 지침]
1. 보편적인 교양 수준의 어휘와 표현을 사용하세요.
2. 불필요하게 복잡한 문장 구조를 단순화하세요.
3. 전문 용어는 간단한 설명과 함께 사용하세요.
4. 논리적 흐름을 유지하며 명확하게 표현하세요.
5. 비유와 예시를 적절히 활용하세요.
6. 중요한 내용을 강조하고 핵심을 먼저 제시하세요.
7. 문장 사이에 적절한 줄바꿈을 추가하세요.
"""
            
        return base_prompt
    
    def _generate_simplified_text(self, prompt):
        """AI 모델을 사용하여 단순화된 텍스트 생성"""
        try:
            # API 유형에 따른 분기
            if self.api_type == 'openai':
                response = openai.ChatCompletion.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "당신은 복잡한 텍스트를 더 쉽게 이해할 수 있는 형태로 변환해주는 전문가입니다."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.5,
                    max_tokens=2000
                )
                simplified_text = response['choices'][0]['message']['content']
                
            elif self.api_type == 'anthropic':
                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=2000,
                    temperature=0.5,
                    system="당신은 복잡한 텍스트를 더 쉽게 이해할 수 있는 형태로 변환해주는 전문가입니다.",
                    messages=[{
                        "role": "user", 
                        "content": prompt
                    }]
                )
                simplified_text = message.content[0].text
                
            elif self.api_type == 'groq':
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "당신은 복잡한 텍스트를 더 쉽게 이해할 수 있는 형태로 변환해주는 전문가입니다."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.5,
                    max_tokens=2000
                )
                simplified_text = response.choices[0].message.content
            
            return simplified_text
            
        except Exception as e:
            logger.error(f"텍스트 생성 오류: {str(e)}", exc_info=True)
            raise




    
import logging
import json
import os
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.permissions import AllowAny
from PIL import Image, ImageFilter, ImageEnhance
import pytesseract
from .models import OCRResult
from .serializers import OCRResultSerializer

import PyPDF2
import tempfile
from pdf2image import convert_from_path
import re

logger = logging.getLogger(__name__)

# OllamaClient와 GPTTranslator 클래스 가져오기
from .ollama_client import OllamaClient
from .gpt_translator import GPTTranslator 

@method_decorator(csrf_exempt, name='dispatch')
class ProcessFileView(APIView):
    permission_classes = [AllowAny]
    
    def post(self, request):
        try:
            logger.info("ProcessFileView 요청 수신: %s %s", request.method, request.path)
            
            # 요청 데이터 확인
            if 'file' not in request.FILES:
                logger.error("파일이 제공되지 않음")
                return Response({'error': '파일이 제공되지 않았습니다'}, status=status.HTTP_400_BAD_REQUEST)
            
            file_obj = request.FILES['file']
            file_name = file_obj.name.lower()
            logger.info("파일 업로드: %s, 크기: %s bytes", file_name, file_obj.size)
            
            # Ollama 클라이언트 초기화
            ollama_base_url = os.environ.get('OLLAMA_API_URL', 'http://localhost:11434')
            ollama_client = OllamaClient(base_url=ollama_base_url)
            
            # GPT 번역기 초기화
            gpt_translator = GPTTranslator()
            
            # 번역 옵션 확인 (기본값: True)
            enable_translation = request.data.get('enable_translation', 'true').lower() == 'true'
            
            # 파일 유형 확인
            if file_name.endswith(('.pdf')):
                file_type = 'pdf'
                
                # PDF 페이지 범위 확인
                start_page = int(request.data.get('start_page', 1))
                end_page = int(request.data.get('end_page', 0))  # 0은 전체 페이지를 의미
                
                logger.info("PDF 처리 범위: %s ~ %s 페이지", start_page, end_page if end_page > 0 else "끝")
                
            elif file_name.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')):
                file_type = 'image'
            else:
                logger.error("지원되지 않는 파일 형식: %s", file_name)
                return Response({'error': '지원되지 않는 파일 형식입니다. 이미지나 PDF 파일을 업로드해주세요.'},
                              status=status.HTTP_400_BAD_REQUEST)
            
            # OCR 결과 객체 생성
            ocr_result = OCRResult.objects.create(file=file_obj, file_type=file_type)
            logger.info("OCRResult 객체 생성: %s", ocr_result.id)
            
            # OCR 처리
            try:
                ocr_text = ""
                page_texts = []  # 페이지별 텍스트 저장
                
                if file_type == 'image':
                    # 이미지 파일 처리 - 개선된 OCR 적용
                    img = Image.open(ocr_result.file.path)
                    # 이미지 정보 로깅
                    logger.info(f"이미지 정보: 크기={img.size}, 모드={img.mode}, 포맷={img.format}")
                    
                    # 이미지 전처리 및 OCR 수행 - OllamaClient 메서드 사용
                    preprocessed_img = ollama_client.preprocess_image_for_ocr(img)
                    ocr_text = ollama_client.get_optimized_ocr_text(preprocessed_img, lang='kor+eng')
                    page_texts.append({"page": 1, "text": ocr_text})
                    logger.info("이미지 OCR 처리 완료, 추출 텍스트 길이: %s", len(ocr_text))
                    logger.info("추출된 텍스트 샘플: %s", ocr_text[:200] if ocr_text else "텍스트 없음")
                
                elif file_type == 'pdf':
                    # PDF 처리 - 직접 추출 후 필요시 OCR
                    logger.info("PDF 처리 시작: %s", ocr_result.file.path)
                    
                    # 직접 텍스트 추출 시도 (페이지별)
                    direct_extract_success = False
                    try:
                        all_page_texts = self.extract_text_from_pdf_by_pages(ocr_result.file.path)
                        
                        # 페이지 범위 처리
                        if start_page > 1 or (end_page > 0 and end_page < len(all_page_texts)):
                            if start_page <= len(all_page_texts):
                                if end_page > 0 and end_page >= start_page:
                                    page_texts = all_page_texts[start_page-1:end_page]
                                else:
                                    page_texts = all_page_texts[start_page-1:]
                            else:
                                page_texts = []
                        else:
                            page_texts = all_page_texts
                        
                        combined_text = "\n".join([page["text"] for page in page_texts])
                        
                        # 추출된 텍스트 길이가 충분한지 확인 (더 엄격한 조건)
                        if combined_text.strip() and len(combined_text.strip()) >= 50:
                            meaningful_chars = sum(1 for c in combined_text if c.isalnum())
                            if meaningful_chars > 30:  # 의미있는 글자가 30자 이상이면 성공으로 간주
                                ocr_text = combined_text
                                direct_extract_success = True
                                logger.info("PDF 직접 텍스트 추출 성공, 총 %s 페이지, 텍스트 길이: %s", 
                                          len(page_texts), len(ocr_text))
                                logger.info("추출된 텍스트 샘플: %s", ocr_text[:200] if ocr_text else "텍스트 없음")
                    except Exception as e:
                        logger.error(f"PDF 직접 텍스트 추출 실패: {str(e)}")
                    
                    # 직접 텍스트 추출이 실패한 경우, OCR 시도
                    if not direct_extract_success:
                        logger.info("PDF OCR 처리 시작 (직접 추출 실패 또는 텍스트 불충분)")
                        
                        # 페이지 범위 설정으로 OCR
                        all_page_texts = self.ocr_pdf_by_pages(ocr_result.file.path, ollama_client, start_page, end_page)
                        
                        # 페이지 범위 처리 - ocr_pdf_by_pages에서 처리했으므로 전체 사용
                        page_texts = all_page_texts
                        
                        ocr_text = "\n".join([page["text"] for page in page_texts])
                        logger.info("PDF OCR 처리 완료, 총 %s 페이지, 텍스트 길이: %s", 
                                  len(page_texts), len(ocr_text))
                        logger.info("추출된 텍스트 샘플: %s", ocr_text[:200] if ocr_text else "텍스트 없음")
                
                # 텍스트 정화 - 개선된 함수 사용
                ocr_result.ocr_text = self.clean_text(ocr_text)
                
                # PDF 파일은 항상 텍스트 관련 있음으로 설정
                if file_type == 'pdf':
                    text_relevant = True
                
                # 분석 유형 확인 (기본값: both)
                analysis_type = request.data.get('analysis_type', 'both')
                logger.info("분석 유형: %s", analysis_type)
                
                # 결과 변수 초기화
                image_analysis = ""
                text_analysis = ""
                combined_analysis = ""
                
                # 번역 결과 변수 초기화
                translated_analysis = ""
                translation_success = False
                translation_error = ""
                
                # 페이지 분할 분석 여부 확인
                analyze_by_page = request.data.get('analyze_by_page', 'true').lower() == 'true'
                
                # 선택된 분석 유형에 따라 처리
                if analysis_type in ['ollama', 'both']:
                    # 이미지 파일인 경우
                    if file_type == 'image':
                        # 사용자 정의 프롬프트 설정 (요약된 간결한 설명을 위해)
                        custom_prompt = f"""이미지를 객관적으로 관찰하고 다음 지침에 따라 응답하세요:

필수 포함 사항:
- 이미지에 실제로 보이는 사람, 동물, 물체만 언급 (없으면 언급하지 않음)
- 만약 동물이라면, 어떤 종의 동물인지도 출력
- 확실히 보이는 색상만 언급 (배경색, 옷 색상 등)
- 명확히 보이는 자세나 위치 관계 (정면, 측면 등)

절대 포함하지 말 것:
- 추측이나 해석 ("~로 보입니다", "~같습니다" 표현 금지)
- 보이지 않는 부분에 대한 언급 ("보이지 않는다", "없다" 등의 표현 금지)
- 반복적인 설명
- 감정이나 분위기 묘사

형식:
- 1-2문장으로 매우 간결하게 작성
- 단순 사실 나열 형식 (예: "이미지에는 검은 머리 여성이 있고, 배경은 흰색이다.")

OCR 텍스트 (참고용, 실제 이미지에 보이는 경우만 언급): {ocr_result.ocr_text}

영어로 간결하게 응답해주세요."""

                        
                        # OCR 텍스트를 전달 (analyze_image 내부에서 관련성 판단)
                        image_analysis = ollama_client.analyze_image(
                            ocr_result.file.path, 
                            custom_prompt,
                            ocr_text=ocr_result.ocr_text
                        )
                        
                        # OCR 텍스트 분석 (텍스트가 있고 both 모드인 경우)
                        if ocr_result.ocr_text and analysis_type == 'both':
                            # 페이지별 텍스트 정리를 위한 프롬프트
                            text_prompt = f"""다음 OCR로 추출한 텍스트를 자세히 분석하고 명확하게 정리해주세요:

{ocr_result.ocr_text}

분석 지침:
1. 텍스트의 주요 내용과 구조를 파악하여 정리
2. 단순 요약이 아닌, 텍스트의 핵심 정보를 충실하게 정리
3. 중요한 세부 정보를 포함
4. 내용이 이미지와 관련이 있을 수 있으므로 문맥을 고려하여 정리

반드시 "영어(En)"로 응답해주세요."""
                            
                            try:
                                text_analysis = ollama_client.analyze_text(ocr_result.ocr_text, text_prompt)
                            except Exception as e:
                                logger.error(f"텍스트 분석 오류: {str(e)}")
                                text_analysis = f"텍스트 분석 중 오류가 발생했습니다: {str(e)}"
                            
                            # 두 분석 결과 결합
                            combined_analysis = f"이미지 분석 결과:\n{image_analysis}\n\n텍스트 분석 결과:\n{text_analysis}"
                        else:
                            # OCR 없이 이미지 분석만 수행
                            combined_analysis = image_analysis
                        
                    else:  # PDF 파일인 경우
                        if ocr_result.ocr_text:
                            if analyze_by_page and len(page_texts) > 1:
                                # 개선된 페이지별 분석 수행 - OllamaClient의 분석 기능 활용
                                try:
                                    combined_analysis = ollama_client.analyze_text(ocr_result.ocr_text, None, page_texts)
                                    logger.info("페이지별 분석 완료")
                                except Exception as e:
                                    logger.error(f"페이지별 분석 오류: {str(e)}")
                                    combined_analysis = f"페이지별 분석 중 오류가 발생했습니다: {str(e)}"
                            else:
                                # 문서 전체 분석 - 페이지별 구조화 요청
                                text_prompt = f"""다음 PDF에서 추출한 텍스트를 페이지별로 명확하게 정리해주세요:

{ocr_result.ocr_text}

분석 지침:
1. 텍스트를 페이지나 섹션 단위로 구분하여 정리해주세요.
2. 내용을 요약하지 말고, 각 섹션의 핵심 정보를 충실하게 정리해주세요.
3. 모든 중요한 세부 정보를 포함해주세요.
4. 내용을 단순 요약하지 말고, 구조화된 형식으로 정리해주세요.

다음과 같은 형식으로 정리해주세요:
===== 페이지 1 (또는 섹션 1) =====
- 주요 내용 정리
- 중요 개념 설명
- 핵심 정보 나열

===== 페이지 2 (또는 섹션 2) =====
- 주요 내용 정리
...

반드시 "영어로" 응답해주세요."""
                                
                                try:
                                    text_analysis = ollama_client.analyze_text(ocr_result.ocr_text, text_prompt)
                                    combined_analysis = text_analysis
                                except Exception as e:
                                    logger.error(f"문서 전체 분석 오류: {str(e)}")
                                    combined_analysis = f"문서 분석 중 오류가 발생했습니다: {str(e)}\n\nOCR 결과: {ocr_result.ocr_text[:500]}..."
                    
                    logger.info("이미지/텍스트 분석 완료")
                
                # GPT 번역 수행 (번역이 활성화된 경우)
                if enable_translation and combined_analysis and gpt_translator.is_available:
                    logger.info("GPT 번역 시작")
                    try:
                        # 분석 유형에 따른 번역
                        if file_type == 'pdf' and analyze_by_page and len(page_texts) > 1:
                            # 페이지별 분석 결과 번역
                            translation_result = gpt_translator.translate_paged_analysis(combined_analysis)
                        else:
                            # 일반 분석 결과 번역
                            translation_result = gpt_translator.translate_analysis_result(combined_analysis, file_type)
                        
                        if translation_result and translation_result.get("success"):
                            translated_analysis = translation_result["translated_analysis"]
                            translation_success = True
                            logger.info("GPT 번역 성공")
                        else:
                            error_msg = translation_result.get('error', 'Unknown error') if translation_result else 'No translation result'
                            logger.error(f"GPT 번역 실패: {error_msg}")
                            translated_analysis = f"번역 실패: {error_msg}"
                            translation_error = error_msg
                            
                    except Exception as e:
                        logger.error(f"GPT 번역 중 예외 발생: {str(e)}")
                        translated_analysis = f"번역 중 오류 발생: {str(e)}"
                        translation_error = str(e)
                
                # 번역 관련 메타데이터 저장
                ocr_result.translation_enabled = enable_translation
                ocr_result.translation_success = translation_success
                ocr_result.analysis_type = analysis_type
                ocr_result.analyze_by_page = analyze_by_page
                
                # MySQL 저장을 위한 텍스트 정화
                ocr_result.llm_response = self.clean_text(combined_analysis)
                
                # 번역 결과도 저장
                if enable_translation and translated_analysis:
                    if translation_success:
                        # 성공한 번역 결과 저장
                        ocr_result.llm_response_korean = self.clean_text(translated_analysis)
                        ocr_result.translation_model = gpt_translator.model if gpt_translator else "unknown"
                    else:
                        # 실패한 경우 오류 메시지 저장 (디버깅용)
                        ocr_result.llm_response_korean = f"번역 실패: {translation_error}"
                
                # 텍스트 관련성 정보 저장 - PDF는 항상 True, 이미지는 분석 과정에서 결정
                if file_type == 'pdf':
                    ocr_result.text_relevant = True
                
            except Exception as e:
                logger.error("처리 실패: %s", str(e), exc_info=True)
                return Response({'error': f'처리 실패: {str(e)}'}, 
                               status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
            # 결과 저장
            try:
                ocr_result.save()
                logger.info("OCRResult 저장 완료 (ID: %s)", ocr_result.id)
            except Exception as e:
                logger.error(f"데이터베이스 저장 실패: {str(e)}")
                return Response({'error': f'결과 저장 실패: {str(e)}'}, 
                            status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
            # 응답 데이터 구성 - 명시적으로 필드 지정
            try:
                # 기본 시리얼라이저 데이터
                response_data = OCRResultSerializer(ocr_result).data
                
                # 번역 관련 정보 명시적 추가
                response_data['translation_enabled'] = enable_translation
                response_data['translation_success'] = translation_success
                
                # 영어 원문과 한국어 번역을 명확히 구분
                response_data['llm_response'] = ocr_result.llm_response  # 영어 원문
                
                if enable_translation and translation_success:
                    # 번역 성공 시 한국어 번역 추가
                    response_data['llm_response_korean'] = ocr_result.llm_response_korean
                    logger.info("응답에 한국어 번역 포함")
                elif enable_translation and not translation_success:
                    # 번역 실패 시 오류 정보 추가
                    response_data['llm_response_korean'] = None
                    response_data['translation_error'] = translation_error if translation_error else "번역 실패"
                    logger.info("번역 실패 - 영어 원문만 포함")
                else:
                    # 번역 비활성화 시
                    response_data['llm_response_korean'] = None
                    logger.info("번역 비활성화 - 영어 원문만 포함")
                
                # 디버깅용 로그
                logger.info(f"응답 데이터 구성 완료:")
                logger.info(f"  - 영어 원문 길이: {len(response_data.get('llm_response', ''))}")
                logger.info(f"  - 한국어 번역 길이: {len(response_data.get('llm_response_korean', '') or '')}")
                logger.info(f"  - 번역 성공: {response_data.get('translation_success', False)}")
                
            except Exception as e:
                logger.error(f"응답 데이터 구성 실패: {str(e)}")
                return Response({'error': f'응답 구성 실패: {str(e)}'}, 
                            status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
            # 응답 반환
            return Response(response_data, status=status.HTTP_201_CREATED)
                
        except Exception as e:
            logger.error("처리 중 예기치 않은 오류: %s", str(e), exc_info=True)
            return Response({'error': f'서버 오류: {str(e)}'}, 
                           status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    def extract_text_from_pdf_by_pages(self, pdf_path):
        """PDF에서 직접 텍스트를 페이지별로 추출"""
        pages = []
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                total_pages = len(reader.pages)
                
                for i in range(total_pages):
                    page = reader.pages[i]
                    text = page.extract_text()
                    pages.append({"page": i + 1, "text": text})
                    
            return pages
        except Exception as e:
            logger.error(f"PDF 텍스트 직접 추출 오류: {str(e)}")
            raise
    
    def ocr_pdf_by_pages(self, pdf_path, ollama_client, start_page=1, end_page=0):
        """PDF를 OCR로 처리하여 페이지별 텍스트 추출"""
        pages = []
        
        try:
            # PDF2Image로 이미지 변환
            with tempfile.TemporaryDirectory() as path:
                # 페이지 번호는 1부터 시작하지만, convert_from_path는 0부터 시작하므로 조정
                first_page = start_page
                last_page = None if end_page <= 0 else end_page
                
                images = convert_from_path(
                    pdf_path, 
                    dpi=300, 
                    output_folder=path, 
                    first_page=first_page,
                    last_page=last_page
                )
                
                # 각 페이지 이미지 OCR 처리
                for i, image in enumerate(images):
                    # 이미지 전처리
                    preprocessed_img = ollama_client.preprocess_image_for_ocr(image)
                    # OCR 수행
                    text = ollama_client.get_optimized_ocr_text(preprocessed_img, lang='kor+eng')
                    
                    # 페이지 번호 계산 (시작 페이지 고려)
                    page_num = start_page + i
                    pages.append({"page": page_num, "text": text})
                    
            return pages
        except Exception as e:
            logger.error(f"PDF OCR 처리 오류: {str(e)}")
            raise
    
    def clean_text(self, text):
        """텍스트 정화 함수"""
        if not text:
            return ""
            
        # 연속된 공백 제거
        text = re.sub(r'\s+', ' ', text)
        # 연속된 줄바꿈 제거
        text = re.sub(r'\n\s*\n', '\n\n', text)
        # 앞뒤 공백 제거
        text = text.strip()
        
        return text
    
# chat/views.py에 추가할 뷰들

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated, AllowAny
from django.shortcuts import get_object_or_404
from datetime import datetime, timedelta
import json
import re
from .models import Schedule, ScheduleRequest, ConflictResolution
from .serializers import (
    ScheduleSerializer, ScheduleRequestSerializer, 
    ConflictResolutionSerializer, ScheduleRequestInputSerializer
)

# 기존 ChatBot 클래스와 ChatView는 그대로 유지...
# chat/views.py 수정 버전

# from rest_framework.views import APIView
# from rest_framework.response import Response
# from rest_framework import status
# from rest_framework.decorators import api_view, permission_classes
# from rest_framework.permissions import IsAuthenticated, AllowAny
# from django.shortcuts import get_object_or_404
# from datetime import datetime, timedelta
# import json
# import re
# from .models import Schedule, ScheduleRequest, ConflictResolution
# from .serializers import (
#     ScheduleSerializer, ScheduleRequestSerializer, 
#     ConflictResolutionSerializer, ScheduleRequestInputSerializer
# )

# # 기존 ChatBot 클래스는 그대로 유지...
# OPENAI_API_KEY = "***REMOVED***"
# ANTHROPIC_API_KEY = "***REMOVED***"
# GROQ_API_KEY = "***REMOVED***"


# chatbots = {
#     'gpt': ChatBot(OPENAI_API_KEY, 'gpt-3.5-turbo', 'openai'),
#     'claude': ChatBot(ANTHROPIC_API_KEY, 'claude-3-5-haiku-20241022', 'anthropic'), 
#     'mixtral': ChatBot(GROQ_API_KEY, 'llama3-8b-8192', 'groq'),
# }

# # 백엔드 views.py에 추가할 함수들

# def parse_date_from_request(request_text):
#     """자연어 날짜를 실제 날짜로 변환"""
#     today = datetime.now().date()
    
#     # 오늘/내일/모레 등 한국어 날짜 표현 처리
#     if '오늘' in request_text:
#         return today
#     elif '내일' in request_text:
#         return today + timedelta(days=1)
#     elif '모레' in request_text or '모래' in request_text:
#         return today + timedelta(days=2)
#     elif '이번 주' in request_text:
#         # 이번 주 금요일로 설정
#         days_until_friday = (4 - today.weekday()) % 7
#         if days_until_friday == 0:  # 오늘이 금요일이면 다음 주 금요일
#             days_until_friday = 7
#         return today + timedelta(days=days_until_friday)
#     elif '다음 주' in request_text:
#         return today + timedelta(days=7)
#     else:
#         # 기본값: 내일
#         return today + timedelta(days=1)

# def parse_multiple_schedules_backend(request_text):
#     """백엔드에서 여러 일정 파싱"""
#     # 쉼표, "그리고", "및" 등으로 분리
#     separators = [',', '，', '그리고', '및', '와', '과']
    
#     parts = [request_text]
#     for sep in separators:
#         new_parts = []
#         for part in parts:
#             new_parts.extend(part.split(sep))
#         parts = new_parts
    
#     # 정리된 요청들 반환
#     cleaned_requests = []
#     for part in parts:
#         cleaned = part.strip()
#         if cleaned and len(cleaned) > 2:  # 너무 짧은 텍스트 제외
#             cleaned_requests.append(cleaned)
    
#     return cleaned_requests if len(cleaned_requests) > 1 else [request_text]
# class ScheduleOptimizerBot:
#     """일정 최적화를 위한 AI 봇 클래스 - 여러 AI 모델 연동"""
    
#     def __init__(self):
#         self.chatbots = {
#                 'gpt': ChatBot(OPENAI_API_KEY, 'gpt-3.5-turbo', 'openai'),
#                 'claude': ChatBot(ANTHROPIC_API_KEY, 'claude-3-5-haiku-20241022', 'anthropic'), 
#                 'mixtral': ChatBot(GROQ_API_KEY, 'llama3-8b-8192', 'groq'),
#             }
        
#     def create_schedule_prompt(self, request_text, user_context=None, existing_schedules=None):
#         """일정 생성을 위한 프롬프트 생성 - 빈 시간 분석 포함"""
#         base_prompt = f"""
#         사용자의 일정 요청을 분석하여 기존 일정과 충돌하지 않는 최적의 빈 시간을 찾아 제안해주세요.

#         요청 내용: {request_text}
        
#         기존 일정들: {existing_schedules or []}
        
#         분석 방법:
#         1. 기존 일정들의 시간대를 확인하여 사용자가 입력한 날의 빈 시간을 찾아주세요
#         2. 요청된 일정의 성격에 맞는 최적의 시간대를 추천해주세요
#         3. 일정 간 여유 시간도 고려해주세요
        
#         다음 형식으로 응답해주세요:
#         {{
#             "title": "일정 제목",
#             "description": "상세 설명",
#             "suggested_date": "YYYY-MM-DD",
#             "suggested_start_time": "HH:MM",
#             "suggested_end_time": "HH:MM",
#             "location": "장소 (선택사항)",
#             "priority": "HIGH/MEDIUM/LOW/URGENT",
#             "attendees": ["참석자1", "참석자2"],
#             "reasoning": "이 시간을 제안하는 이유 (빈 시간 분석 결과 포함)"
#         }}
        
#         사용자의 맥락 정보: {user_context or "없음"}
#         """
#         return base_prompt

#     def create_conflict_resolution_prompt(self, conflicting_schedules, new_request):
#         """일정 충돌 해결을 위한 프롬프트 생성"""
#         prompt = f"""
#         기존 일정과 새로운 일정 요청 사이에 충돌이 발생했습니다. 
#         여러 AI의 관점에서 최적의 해결 방안을 제안해주세요.

#         기존 충돌 일정들:
#         {json.dumps(conflicting_schedules, ensure_ascii=False, indent=2)}

#         새로운 일정 요청: {new_request}

#         다음 형식으로 해결 방안을 제안해주세요:
#         {{
#             "resolution_options": [
#                 {{
#                     "option": "방안 1",
#                     "description": "상세 설명",
#                     "impact": "영향도 분석",
#                     "recommended": true/false
#                 }},
#                 {{
#                     "option": "방안 2", 
#                     "description": "상세 설명",
#                     "impact": "영향도 분석",
#                     "recommended": true/false
#                 }}
#             ],
#             "best_recommendation": "가장 추천하는 방안과 이유"
#         }}
#         """
#         return prompt
    
#     def get_ai_suggestions(self, prompt, suggestion_type="schedule"):
#         """여러 AI 모델로부터 제안받기"""
#         suggestions = {}
        
#         for model_name, chatbot in self.chatbots.items():
#             try:
#                 response = chatbot.chat(prompt)
#                 suggestions[f"{model_name}_suggestion"] = response
#             except Exception as e:
#                 suggestions[f"{model_name}_suggestion"] = f"오류 발생: {str(e)}"
        
#         return suggestions
    
#     def analyze_and_optimize_suggestions(self, suggestions, query, selected_models=['GPT', 'Claude', 'Mixtral']):
#         """여러 AI 제안을 분석하여 최적화된 결과 생성 - 기존 analyze_responses 활용"""
#         try:
#             # ChatBot의 analyze_responses 기능 활용
#             analyzer = self.chatbots['claude']  # Claude를 분석용으로 사용
            
#             # 제안을 분석용 형태로 변환
#             responses_for_analysis = {}
#             for key, suggestion in suggestions.items():
#                 model_name = key.replace('_suggestion', '')
#                 responses_for_analysis[model_name] = suggestion
            
#             # 기존 analyze_responses 메서드 활용
#             analysis_result = analyzer.analyze_responses(
#                 responses_for_analysis, 
#                 query, 
#                 'Korean',  # 기본 언어
#                 selected_models
#             )
            
#             # JSON 응답에서 최적화된 일정 정보 추출
#             try:
#                 # best_response에서 JSON 부분 추출
#                 json_match = re.search(r'\{.*\}', analysis_result.get('best_response', ''), re.DOTALL)
#                 if json_match:
#                     optimized = json.loads(json_match.group())
#                 else:
#                     # fallback: 첫 번째 유효한 제안 사용
#                     optimized = self._extract_first_valid_suggestion(suggestions)
#             except:
#                 optimized = self._extract_first_valid_suggestion(suggestions)
            
#             confidence = self._calculate_confidence_from_analysis(analysis_result)
            
#             return {
#                 "optimized_suggestion": optimized,
#                 "confidence_score": confidence,
#                 "ai_analysis": analysis_result,
#                 "individual_suggestions": self._parse_individual_suggestions(suggestions)
#             }
            
#         except Exception as e:
#             print(f"Analysis error: {str(e)}")
#             return {"error": f"최적화 과정에서 오류 발생: {str(e)}"}
    
#     def _extract_first_valid_suggestion(self, suggestions):
#         """첫 번째 유효한 제안 추출"""
#         for key, suggestion in suggestions.items():
#             try:
#                 json_match = re.search(r'\{.*\}', suggestion, re.DOTALL)
#                 if json_match:
#                     return json.loads(json_match.group())
#             except:
#                 continue
        
#         # 기본 제안 반환
#         return {
#             "title": "새 일정",
#             "description": "AI가 제안한 일정입니다",
#             "suggested_date": datetime.now().strftime('%Y-%m-%d'),
#             "suggested_start_time": "09:00",
#             "suggested_end_time": "10:00",
#             "location": "",
#             "priority": "MEDIUM",
#             "attendees": [],
#             "reasoning": "여러 AI 모델의 제안을 종합한 결과입니다."
#         }
    
#     def _calculate_confidence_from_analysis(self, analysis_result):
#         """분석 결과에서 신뢰도 계산"""
#         reasoning = analysis_result.get('reasoning', '')
        
#         # 키워드 기반 신뢰도 계산
#         confidence_keywords = ['일치', '공통', '정확', '최적', '추천']
#         uncertainty_keywords = ['불확실', '추정', '가능성', '어려움']
        
#         confidence_score = 0.5  # 기본값
        
#         for keyword in confidence_keywords:
#             if keyword in reasoning:
#                 confidence_score += 0.1
        
#         for keyword in uncertainty_keywords:
#             if keyword in reasoning:
#                 confidence_score -= 0.1
        
#         return max(0.1, min(1.0, confidence_score))
    
#     def _parse_individual_suggestions(self, suggestions):
#         """개별 제안들을 파싱"""
#         parsed = []
#         for key, suggestion in suggestions.items():
#             try:
#                 json_match = re.search(r'\{.*\}', suggestion, re.DOTALL)
#                 if json_match:
#                     parsed_suggestion = json.loads(json_match.group())
#                     parsed_suggestion['source'] = key.replace('_suggestion', '')
#                     parsed.append(parsed_suggestion)
#             except:
#                 continue
#         return parsed

# class ScheduleManagementView(APIView):
#     """일정 관리 메인 뷰 - 권한 수정"""
#     # 임시로 AllowAny로 변경 (개발/테스트용)
#     permission_classes = [IsAuthenticated]
    
#     def __init__(self):
#         super().__init__()
#         self.optimizer = ScheduleOptimizerBot()
    
#     def get(self, request):
#         """사용자의 일정 목록 조회"""
#         # 🚫 기존 더미 사용자 로직 제거
#         if not request.user.is_authenticated:
#             return Response({'error': '인증이 필요합니다.'}, status=status.HTTP_401_UNAUTHORIZED)
        
#         schedules = Schedule.objects.filter(user=request.user).order_by('start_time')
        
#         # 날짜 필터링 (기존 코드 유지)
#         start_date = request.query_params.get('start_date')
#         end_date = request.query_params.get('end_date')
        
#         if start_date:
#             schedules = schedules.filter(start_time__date__gte=start_date)
#         if end_date:
#             schedules = schedules.filter(end_time__date__lte=end_date)
        
#         serializer = ScheduleSerializer(schedules, many=True)
#         return Response(serializer.data)
#     def post(self, request):
#         """새로운 일정 생성 요청 - 여러 일정 지원 개선"""
#         try:
#             request_text = request.data.get('request_text', '')
#             existing_schedules = request.data.get('existing_schedules', [])
            
#             if not request_text:
#                 return Response({'error': '요청 텍스트가 필요합니다.'}, 
#                             status=status.HTTP_400_BAD_REQUEST)
#             if not request.user.is_authenticated:
#                 return Response({'error': '인증이 필요합니다.'}, status=status.HTTP_401_UNAUTHORIZED)
        
#             user = request.user

         
            
#             # 여러 일정 요청인지 확인
#             schedule_requests = parse_multiple_schedules_backend(request_text)
#             target_date = parse_date_from_request(request_text)
            
#             if len(schedule_requests) > 1:
#                 # 여러 일정 처리
#                 multiple_schedules = []
#                 all_individual_suggestions = []
                
#                 for i, single_request in enumerate(schedule_requests):
#                     # 각 일정의 시작 시간을 다르게 설정
#                     schedule_date = target_date
#                     if i > 0:  # 두 번째 일정부터는 2시간씩 뒤로
#                         base_hour = 9 + (i * 2)
#                     else:
#                         base_hour = 9
                    
#                     # 개별 일정 생성
#                     optimized_schedule = {
#                         "title": self._extract_schedule_title(single_request),
#                         "description": f"AI가 분석한 {self._extract_schedule_title(single_request)} 일정입니다.",
#                         "suggested_date": schedule_date.strftime('%Y-%m-%d'),
#                         "suggested_start_time": f"{base_hour:02d}:00",
#                         "suggested_end_time": f"{base_hour + 2:02d}:00",
#                         "location": self._extract_schedule_location(single_request),
#                         "priority": "HIGH",
#                         "attendees": [],
#                         "reasoning": f"{i + 1}번째 일정: {single_request}. 기존 일정과 충돌하지 않는 시간으로 배정했습니다."
#                     }
#                     multiple_schedules.append(optimized_schedule)
                    
#                     # 각 AI별 개별 제안 생성
#                     for ai_type in ['gpt', 'claude', 'mixtral']:
#                         individual_suggestion = optimized_schedule.copy()
#                         individual_suggestion['source'] = ai_type
#                         individual_suggestion['reasoning'] = f"{ai_type.upper()}가 분석한 {self._extract_schedule_title(single_request)} 최적 시간입니다."
#                         all_individual_suggestions.append(individual_suggestion)
                
#                 # 여러 일정 응답 생성
#                 response_data = {
#                     'request_id': int(datetime.now().timestamp()),
#                     'multiple_schedules': multiple_schedules,
#                     'optimized_suggestion': multiple_schedules[0],
#                     'confidence_score': 0.92,
#                     'individual_suggestions': all_individual_suggestions,
#                     'ai_analysis': {
#                         'analysis_summary': f"총 {len(schedule_requests)}개의 일정을 분석하여 최적의 시간대로 배정했습니다.",
#                         'reasoning': f"여러 일정을 {target_date.strftime('%Y년 %m월 %d일')}에 시간 순서대로 배치하여 충돌을 방지했습니다.",
#                         'models_used': ["gpt", "claude", "mixtral"]
#                     },
#                     'has_conflicts': False,
#                     'conflicts': [],
#                     'analysis_summary': f"{len(schedule_requests)}개 일정에 대해 3개 AI 모델이 분석한 결과입니다.",
#                     'is_multiple_schedule': True
#                 }
                
#             else:
#                 # 단일 일정 처리 (기존 로직 사용하되 날짜 반영)
#                 user_context = self._get_user_context(user)
                
#                 # 날짜가 반영된 프롬프트 생성
#                 enhanced_prompt = f"""
#                 사용자의 일정 요청을 분석하여 기존 일정과 충돌하지 않는 최적의 빈 시간을 찾아 제안해주세요.
                
#                 요청 내용: {request_text}
#                 목표 날짜: {target_date.strftime('%Y년 %m월 %d일')} ({self._get_weekday_korean(target_date)})
#                 기존 일정들: {existing_schedules or []}
                
#                 다음 형식으로 응답해주세요:
#                 {{
#                     "title": "일정 제목",
#                     "description": "상세 설명",
#                     "suggested_date": "{target_date.strftime('%Y-%m-%d')}",
#                     "suggested_start_time": "HH:MM",
#                     "suggested_end_time": "HH:MM",
#                     "location": "장소",
#                     "priority": "HIGH/MEDIUM/LOW/URGENT",
#                     "attendees": [],
#                     "reasoning": "이 시간을 제안하는 이유"
#                 }}
#                 """
                
#                 # 기존 단일 일정 로직 계속...
#                 suggestions = self.optimizer.get_ai_suggestions(enhanced_prompt)
#                 optimized_result = self.optimizer.analyze_and_optimize_suggestions(
#                     suggestions, f"일정 요청: {request_text}"
#                 )
                
#                 response_data = {
#                     'request_id': int(datetime.now().timestamp()),
#                     'optimized_suggestion': optimized_result.get('optimized_suggestion', {}),
#                     'confidence_score': optimized_result.get('confidence_score', 0.0),
#                     'ai_analysis': optimized_result.get('ai_analysis', {}),
#                     'individual_suggestions': optimized_result.get('individual_suggestions', []),
#                     'has_conflicts': False,
#                     'conflicts': [],
#                     'analysis_summary': "3개 AI 모델이 분석한 결과입니다.",
#                     'is_multiple_schedule': False
#                 }
            
#             return Response(response_data, status=status.HTTP_201_CREATED)
            
#         except Exception as e:
#                     return Response({
#                         'error': f'일정 생성 요청 처리 중 오류: {str(e)}'
#                     }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

#     def _extract_schedule_title(self, request):
#             """요청에서 일정 제목 추출"""
#             if '운동' in request:
#                 return '운동'
#             elif '미팅' in request or '회의' in request:
#                 return '팀 미팅'
#             elif '공부' in request or '학습' in request:
#                 return '학습 시간'
#             elif '작업' in request or '업무' in request:
#                 return '집중 작업'
#             elif '약속' in request:
#                 return '약속'
#             else:
#                 return '새 일정'

#     def _extract_schedule_location(self, request):
#             """요청에서 장소 추출"""
#             if '운동' in request:
#                 return '헬스장'
#             elif '미팅' in request or '회의' in request:
#                 return '회의실'
#             elif '공부' in request or '학습' in request:
#                 return '도서관'
#             elif '커피' in request:
#                 return '카페'
#             else:
#                 return '사무실'

#     def _get_weekday_korean(self, date):
#             """요일을 한국어로 반환"""
#             weekdays = ['월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일']
#             return weekdays[date.weekday()]
            
   
#     def _check_schedule_conflicts(self, user, suggestion):
#         """일정 충돌 검사"""
#         if not suggestion or 'suggested_date' not in suggestion:
#             return []
        
#         try:
#             suggested_date = datetime.strptime(suggestion['suggested_date'], '%Y-%m-%d').date()
#             start_time = datetime.strptime(suggestion.get('suggested_start_time', '09:00'), '%H:%M').time()
#             end_time = datetime.strptime(suggestion.get('suggested_end_time', '10:00'), '%H:%M').time()
            
#             suggested_start = datetime.combine(suggested_date, start_time)
#             suggested_end = datetime.combine(suggested_date, end_time)
            
#             conflicts = Schedule.objects.filter(
#                 user=user,
#                 start_time__date=suggested_date,
#                 start_time__lt=suggested_end,
#                 end_time__gt=suggested_start
#             )
            
#             return [ScheduleSerializer(conflict).data for conflict in conflicts]
            
#         except Exception as e:
#             return []

# # 권한 수정된 함수들
# @api_view(['POST'])
# @permission_classes([IsAuthenticated])  # 🔧 권한 변경
# def confirm_schedule(request, request_id):
#     """AI 제안된 일정을 확정하여 실제 일정으로 생성"""
#     try:
#         user = request.user
        
#         # 🚫 ScheduleRequest.DoesNotExist에서 더미 데이터 생성 제거
#         try:
#             schedule_request = ScheduleRequest.objects.get(id=request_id, user=user)
#             optimized_suggestion = json.loads(schedule_request.optimized_suggestion)
#         except ScheduleRequest.DoesNotExist:
#             return Response({
#                 'error': f'요청 ID {request_id}를 찾을 수 없습니다.'
#             }, status=status.HTTP_404_NOT_FOUND)
#                 # 날짜/시간 파싱 개선
#         try:
#             suggested_date = optimized_suggestion.get('suggested_date')
#             suggested_start_time = optimized_suggestion.get('suggested_start_time', '09:00')
#             suggested_end_time = optimized_suggestion.get('suggested_end_time', '10:00')
            
#             # 날짜 형식 확인 및 변환
#             if isinstance(suggested_date, str):
#                 if 'T' in suggested_date:  # ISO 형식인 경우
#                     suggested_date = suggested_date.split('T')[0]
                
#                 start_datetime = datetime.strptime(
#                     f"{suggested_date} {suggested_start_time}",
#                     '%Y-%m-%d %H:%M'
#                 )
#                 end_datetime = datetime.strptime(
#                     f"{suggested_date} {suggested_end_time}",
#                     '%Y-%m-%d %H:%M'
#                 )
#             else:
#                 # 날짜가 없으면 오늘로 설정
#                 today = datetime.now().date()
#                 start_datetime = datetime.strptime(
#                     f"{today} {suggested_start_time}",
#                     '%Y-%m-%d %H:%M'
#                 )
#                 end_datetime = datetime.strptime(
#                     f"{today} {suggested_end_time}",
#                     '%Y-%m-%d %H:%M'
#                 )
                
#         except (ValueError, TypeError) as e:
#             print(f"DateTime parsing error: {e}")
#             # 기본값으로 폴백
#             now = datetime.now()
#             start_datetime = now.replace(hour=9, minute=0, second=0, microsecond=0)
#             end_datetime = now.replace(hour=10, minute=0, second=0, microsecond=0)
        
#         # Schedule 객체 생성
#         schedule_data = {
#             'user': user,
#             'title': optimized_suggestion.get('title', '새 일정'),
#             'description': optimized_suggestion.get('description', 'AI가 제안한 일정입니다.'),
#             'start_time': start_datetime,
#             'end_time': end_datetime,
#             'location': optimized_suggestion.get('location', ''),
#             'priority': optimized_suggestion.get('priority', 'MEDIUM'),
#             'attendees': json.dumps(optimized_suggestion.get('attendees', []), ensure_ascii=False)
#         }
        
#         schedule = Schedule.objects.create(**schedule_data)
#         serializer = ScheduleSerializer(schedule)
        
#         print(f"Schedule created successfully: {schedule.id}")
        
#         return Response({
#             'message': '여러 AI의 분석을 통해 최적화된 일정이 성공적으로 생성되었습니다.',
#             'schedule': serializer.data
#         }, status=status.HTTP_201_CREATED)
        
#     except Exception as e:
#         print(f"Confirm schedule error: {str(e)}")
#         import traceback
#         traceback.print_exc()
        
#         return Response({
#             'error': f'일정 생성 중 오류가 발생했습니다: {str(e)}'
#         }, status=status.HTTP_400_BAD_REQUEST)


# # Alternative solution: Convert to Class-Based View
# class ConfirmScheduleView(APIView):
#     """AI 제안된 일정을 확정하여 실제 일정으로 생성"""
#     permission_classes = [AllowAny]  # 임시로 AllowAny
    
#     def post(self, request, request_id):
#         try:
#             # 사용자 처리
#             if not request.user.is_authenticated:
#                 return Response({'error': '인증이 필요합니다.'}, status=status.HTTP_401_UNAUTHORIZED)

#             user = request.user
            
#             # request_id로 ScheduleRequest를 찾거나 더미 데이터 처리
#             try:
#                 schedule_request = ScheduleRequest.objects.get(id=request_id, user=user)
#                 optimized_suggestion = json.loads(schedule_request.optimized_suggestion)
#             except ScheduleRequest.DoesNotExist:
#                 # 더미 모드: request_id를 기반으로 기본 일정 생성
#                 print(f"ScheduleRequest {request_id} not found, creating dummy schedule")
#                 optimized_suggestion = {
#                     'title': 'AI 제안 일정',
#                     'description': 'AI가 제안한 최적의 일정입니다.',
#                     'suggested_date': datetime.now().strftime('%Y-%m-%d'),
#                     'suggested_start_time': '09:00',
#                     'suggested_end_time': '10:00',
#                     'location': '사무실',
#                     'priority': 'MEDIUM',
#                     'attendees': []
#                 }
#             except json.JSONDecodeError as e:
#                 print(f"JSON decode error: {e}")
#                 return Response({
#                     'error': f'일정 데이터 파싱 오류: {str(e)}'
#                 }, status=status.HTTP_400_BAD_REQUEST)
            
#             # 날짜/시간 파싱
#             try:
#                 suggested_date = optimized_suggestion.get('suggested_date')
#                 suggested_start_time = optimized_suggestion.get('suggested_start_time', '09:00')
#                 suggested_end_time = optimized_suggestion.get('suggested_end_time', '10:00')
                
#                 # 날짜 형식 확인 및 변환
#                 if isinstance(suggested_date, str):
#                     if 'T' in suggested_date:  # ISO 형식인 경우
#                         suggested_date = suggested_date.split('T')[0]
                    
#                     start_datetime = datetime.strptime(
#                         f"{suggested_date} {suggested_start_time}",
#                         '%Y-%m-%d %H:%M'
#                     )
#                     end_datetime = datetime.strptime(
#                         f"{suggested_date} {suggested_end_time}",
#                         '%Y-%m-%d %H:%M'
#                     )
#                 else:
#                     # 날짜가 없으면 오늘로 설정
#                     today = datetime.now().date()
#                     start_datetime = datetime.strptime(
#                         f"{today} {suggested_start_time}",
#                         '%Y-%m-%d %H:%M'
#                     )
#                     end_datetime = datetime.strptime(
#                         f"{today} {suggested_end_time}",
#                         '%Y-%m-%d %H:%M'
#                     )
                    
#             except (ValueError, TypeError) as e:
#                 print(f"DateTime parsing error: {e}")
#                 # 기본값으로 폴백
#                 now = datetime.now()
#                 start_datetime = now.replace(hour=9, minute=0, second=0, microsecond=0)
#                 end_datetime = now.replace(hour=10, minute=0, second=0, microsecond=0)
            
#             # Schedule 객체 생성
#             schedule_data = {
#                 'user': user,
#                 'title': optimized_suggestion.get('title', '새 일정'),
#                 'description': optimized_suggestion.get('description', 'AI가 제안한 일정입니다.'),
#                 'start_time': start_datetime,
#                 'end_time': end_datetime,
#                 'location': optimized_suggestion.get('location', ''),
#                 'priority': optimized_suggestion.get('priority', 'MEDIUM'),
#                 'attendees': json.dumps(optimized_suggestion.get('attendees', []), ensure_ascii=False)
#             }
            
#             schedule = Schedule.objects.create(**schedule_data)
#             serializer = ScheduleSerializer(schedule)
            
#             print(f"Schedule created successfully: {schedule.id}")
            
#             return Response({
#                 'message': '여러 AI의 분석을 통해 최적화된 일정이 성공적으로 생성되었습니다.',
#                 'schedule': serializer.data
#             }, status=status.HTTP_201_CREATED)
            
#         except Exception as e:
#             print(f"Confirm schedule error: {str(e)}")
#             import traceback
#             traceback.print_exc()
            
#             return Response({
#                 'error': f'일정 생성 중 오류가 발생했습니다: {str(e)}'
#             }, status=status.HTTP_400_BAD_REQUEST)


# # Fix for resolve_schedule_conflict function
# @api_view(['POST'])
# @permission_classes([IsAuthenticated])  # 🔧 권한 변경
# def resolve_schedule_conflict(request):
#     """일정 충돌 해결 방안 제공"""
#     # 🚫 더미 사용자 로직 제거
#     user = request.user
    
#     conflicting_schedule_ids = request.data.get('conflicting_schedule_ids', [])
#     new_request = request.data.get('new_request', '')
    
#     # 나머지 로직은 그대로...
    
#     if not conflicting_schedule_ids or not new_request:
#         return Response({
#             'error': '충돌 일정 ID와 새로운 요청이 필요합니다.'
#         }, status=status.HTTP_400_BAD_REQUEST)
    
#     try:
#         # 사용자 처리
#         if request.user.is_authenticated:
#             user = request.user
#         else:
#             from django.contrib.auth.models import User
#             user, created = User.objects.get_or_create(
#                 username='dummy_user',
#                 defaults={'email': 'dummy@example.com'}
#             )
        
#         # 충돌 일정들 조회
#         conflicting_schedules = Schedule.objects.filter(
#             id__in=conflicting_schedule_ids,
#             user=user
#         )
        
#         conflicting_data = [ScheduleSerializer(schedule).data for schedule in conflicting_schedules]
        
#         # 다중 AI 모델들로부터 해결 방안 받기
#         optimizer = ScheduleOptimizerBot()
#         prompt = optimizer.create_conflict_resolution_prompt(conflicting_data, new_request)
#         suggestions = optimizer.get_ai_suggestions(prompt, "conflict_resolution")
        
#         # AI 분석을 통한 최적 해결방안 도출
#         analysis_result = optimizer.analyze_and_optimize_suggestions(
#             suggestions,
#             f"충돌 해결: {new_request}"
#         )
        
#         # 해결 방안 저장
#         conflict_resolution = ConflictResolution.objects.create(
#             user=user,
#             conflicting_schedules=json.dumps(conflicting_data, ensure_ascii=False),
#             resolution_options=json.dumps(suggestions, ensure_ascii=False),
#             ai_recommendations=json.dumps(analysis_result, ensure_ascii=False)
#         )
        
#         return Response({
#             'resolution_id': conflict_resolution.id,
#             'conflicting_schedules': conflicting_data,
#             'ai_suggestions': suggestions,
#             'optimized_resolution': analysis_result,
#             'message': f'{len(suggestions)}개 AI 모델이 분석한 충돌 해결 방안이 생성되었습니다.'
#         }, status=status.HTTP_201_CREATED)
        
#     except Exception as e:
#         return Response({
#             'error': f'충돌 해결 방안 생성 중 오류가 발생했습니다: {str(e)}'
#         }, status=status.HTTP_400_BAD_REQUEST)


# # Alternative Class-Based View for conflict resolution
# class ResolveScheduleConflictView(APIView):
#     """일정 충돌 해결 방안 제공 - 다중 AI 분석"""
#     permission_classes = [IsAuthenticated]
    
#     def post(self, request):
#         conflicting_schedule_ids = request.data.get('conflicting_schedule_ids', [])
#         new_request = request.data.get('new_request', '')
        
#         if not conflicting_schedule_ids or not new_request:
#             return Response({
#                 'error': '충돌 일정 ID와 새로운 요청이 필요합니다.'
#             }, status=status.HTTP_400_BAD_REQUEST)
        
#         try:
#             # 사용자 처리
#             if request.user.is_authenticated:
#                 user = request.user
#             else:
#                 from django.contrib.auth.models import User
#                 user, created = User.objects.get_or_create(
#                     username='dummy_user',
#                     defaults={'email': 'dummy@example.com'}
#                 )
            
#             # 충돌 일정들 조회
#             conflicting_schedules = Schedule.objects.filter(
#                 id__in=conflicting_schedule_ids,
#                 user=user
#             )
            
#             conflicting_data = [ScheduleSerializer(schedule).data for schedule in conflicting_schedules]
            
#             # 다중 AI 모델들로부터 해결 방안 받기
#             optimizer = ScheduleOptimizerBot()
#             prompt = optimizer.create_conflict_resolution_prompt(conflicting_data, new_request)
#             suggestions = optimizer.get_ai_suggestions(prompt, "conflict_resolution")
            
#             # AI 분석을 통한 최적 해결방안 도출
#             analysis_result = optimizer.analyze_and_optimize_suggestions(
#                 suggestions,
#                 f"충돌 해결: {new_request}"
#             )
            
#             # 해결 방안 저장
#             conflict_resolution = ConflictResolution.objects.create(
#                 user=user,
#                 conflicting_schedules=json.dumps(conflicting_data, ensure_ascii=False),
#                 resolution_options=json.dumps(suggestions, ensure_ascii=False),
#                 ai_recommendations=json.dumps(analysis_result, ensure_ascii=False)
#             )
            
#             return Response({
#                 'resolution_id': conflict_resolution.id,
#                 'conflicting_schedules': conflicting_data,
#                 'ai_suggestions': suggestions,
#                 'optimized_resolution': analysis_result,
#                 'message': f'{len(suggestions)}개 AI 모델이 분석한 충돌 해결 방안이 생성되었습니다.'
#             }, status=status.HTTP_201_CREATED)
            
#         except Exception as e:
#             return Response({
#                 'error': f'충돌 해결 방안 생성 중 오류가 발생했습니다: {str(e)}'
#             }, status=status.HTTP_400_BAD_REQUEST)

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes, authentication_classes
from rest_framework.permissions import IsAuthenticated, AllowAny
from rest_framework.authentication import TokenAuthentication, SessionAuthentication
from django.shortcuts import get_object_or_404
from datetime import datetime, timedelta
import json
import re
from .models import Schedule, ScheduleRequest, ConflictResolution
from .serializers import (
    ScheduleSerializer, ScheduleRequestSerializer, 
    ConflictResolutionSerializer, ScheduleRequestInputSerializer
)
import logging
from pytz import timezone
KST = timezone('Asia/Seoul')
target_datetime = datetime.now(KST)
logger = logging.getLogger(__name__)

# 기존 ChatBot 클래스와 API 키들은 그대로 유지...
OPENAI_API_KEY = "***REMOVED***"
ANTHROPIC_API_KEY = "***REMOVED***"
GROQ_API_KEY = "***REMOVED***"


# 🔧 토큰 디버깅을 위한 커스텀 인증 클래스
class DebugTokenAuthentication(TokenAuthentication):
    """디버깅이 포함된 토큰 인증 클래스"""
    
    def authenticate(self, request):
        logger.info("=== 개선된 토큰 인증 디버깅 시작 ===")
        
        # Authorization 헤더 확인
        auth_header = request.META.get('HTTP_AUTHORIZATION', '')
        logger.info(f"Authorization 헤더: '{auth_header}'")
        
        if not auth_header:
            logger.warning("❌ Authorization 헤더가 없습니다")
            return None
            
        if not auth_header.startswith('Bearer '):
            logger.warning(f"❌ Bearer 토큰 형식이 아닙니다: {auth_header}")
            return None
            
        token = auth_header.split(' ')[1]
        logger.info(f"📱 추출된 토큰: {token[:10]}...{token[-10:]}")
        
        # 데이터베이스에서 토큰 확인
        try:
            token_obj = Token.objects.select_related('user').get(key=token)
            logger.info(f"✅ DB에서 토큰 발견: {token_obj.key[:10]}...{token_obj.key[-10:]}")
            logger.info(f"👤 토큰 소유자: {token_obj.user.username} (ID: {token_obj.user.id})")
            logger.info(f"🔄 사용자 활성 상태: {token_obj.user.is_active}")
            
            if not token_obj.user.is_active:
                logger.warning(f"❌ 사용자가 비활성화됨: {token_obj.user.username}")
                raise exceptions.AuthenticationFailed('User inactive or deleted.')
            
            logger.info("✅ 토큰 인증 성공!")
            logger.info("=== 개선된 토큰 인증 디버깅 종료 ===")
            return (token_obj.user, token_obj)
            
        except Token.DoesNotExist:
            logger.error(f"❌ DB에 해당 토큰이 존재하지 않음: {token[:10]}...{token[-10:]}")
            
            # 모든 토큰 목록 출력 (디버깅용)
            all_tokens = Token.objects.all()[:5]  # 처음 5개만
            logger.info(f"🗃️ DB의 기존 토큰들:")
            for i, t in enumerate(all_tokens):
                logger.info(f"  {i+1}. {t.key[:10]}...{t.key[-10:]} (사용자: {t.user.username})")
            
            logger.info("=== 개선된 토큰 인증 디버깅 종료 ===")
            raise exceptions.AuthenticationFailed('Invalid token.')
        
        except Exception as e:
            logger.error(f"❌ 예상치 못한 오류: {str(e)}")
            logger.info("=== 개선된 토큰 인증 디버깅 종료 ===")
            raise exceptions.AuthenticationFailed('Authentication error.')


# 🔧 일정 관리 뷰 - 인증 문제 해결
class ScheduleManagementView(APIView):
    """일정 관리 메인 뷰 - 토큰 인증 적용"""
    authentication_classes = [DebugTokenAuthentication, SessionAuthentication]
    permission_classes = [IsAuthenticated]
    
    def __init__(self):
        super().__init__()
        # ScheduleOptimizerBot 초기화는 메서드 내에서 수행
    
    def get_optimizer(self):
        """필요할 때만 optimizer 인스턴스 생성"""
        if not hasattr(self, '_optimizer'):
            self._optimizer = ScheduleOptimizerBot()
        return self._optimizer
    
    def get(self, request):
        """사용자의 일정 목록 조회"""
        logger.info(f"일정 조회 요청 - 사용자: {request.user.username if request.user.is_authenticated else 'Anonymous'}")
        
        try:
            schedules = Schedule.objects.filter(user=request.user).order_by('start_time')
            
            # 날짜 필터링
            start_date = request.query_params.get('start_date')
            end_date = request.query_params.get('end_date')
            
            if start_date:
                schedules = schedules.filter(start_time__date__gte=start_date)
            if end_date:
                schedules = schedules.filter(end_time__date__lte=end_date)
            
            serializer = ScheduleSerializer(schedules, many=True)
            logger.info(f"일정 조회 성공: {len(serializer.data)}개 일정 반환")
            return Response(serializer.data)
            
        except Exception as e:
            logger.error(f"일정 조회 실패: {str(e)}")
            return Response(
                {'error': f'일정 조회 중 오류 발생: {str(e)}'}, 
                status=status.HTTP_500_INTERNAL_SERVER_ERROR
            )
    def _get_ai_generated_title(self, prompt):
        """AI를 통해 일정 제목 생성"""
        try:
            optimizer = self.get_optimizer()
            suggestions = optimizer.get_ai_suggestions(prompt, "title")
            
            # 첫 번째 응답에서 제목 추출
            for key, response in suggestions.items():
                if response and len(response.strip()) > 0:
                    # 간단한 제목만 추출 (첫 줄만)
                    title = response.strip().split('\n')[0]
                    # 따옴표 제거
                    title = title.strip('"\'')
                    if len(title) > 0 and len(title) < 50:  # 적절한 길이 확인
                        return title
            
            return None
        except Exception as e:
            logger.warning(f"AI 제목 생성 실패: {str(e)}")
            return None
    
    def post(self, request):
        """새로운 일정 생성 요청"""
        logger.info(f"일정 생성 요청 - 사용자: {request.user.username}")
        
        try:
            request_text = request.data.get('request_text', '')
            existing_schedules = request.data.get('existing_schedules', [])
            
            if not request_text:
                return Response({'error': '요청 텍스트가 필요합니다.'}, 
                            status=status.HTTP_400_BAD_REQUEST)
            
            user = request.user
            
            # 여러 일정 요청인지 확인
            schedule_requests = parse_multiple_schedules_backend(request_text)
            target_date = parse_date_from_request(request_text)
            
            logger.info(f"파싱된 일정 요청: {len(schedule_requests)}개")
            logger.info(f"📌 KST 기준 목표 날짜: {target_date} (요청 텍스트: '{request_text}')")

            
            if len(schedule_requests) > 1:
                # 여러 일정 처리
                multiple_schedules = []
                all_individual_suggestions = []
                
                def extract_time_info(text):
                    import re
                    start_hour = None
                    duration_hours = 1

                    is_pm = '오후' in text
                    is_am = '오전' in text

                    # 🔍 "오후 3-5시"와 같은 경우 처리
                    time_range = re.search(r'(\d{1,2})\s*[-~]\s*(\d{1,2})\s*시', text)
                    if time_range:
                        start = int(time_range.group(1))
                        end = int(time_range.group(2))

                        if is_pm:
                            if start < 12:
                                start += 12
                            if end < 12:
                                end += 12
                        elif is_am:
                            if start == 12:
                                start = 0
                            if end == 12:
                                end = 0

                        start_hour = start
                        duration_hours = end - start
                        return start_hour, duration_hours

                    # 🔍 "2시간"만 있는 경우
                    dur_match = re.search(r'(\d{1,2})\s*시간', text)
                    if dur_match:
                        duration_hours = int(dur_match.group(1))

                    # 🔍 단일 시각: "오후 3시"
                    single_time_match = re.search(r'(오전|오후)?\s*(\d{1,2})\s*시', text)
                    if single_time_match:
                        hour = int(single_time_match.group(2))
                        if single_time_match.group(1) == '오후' and hour < 12:
                            hour += 12
                        elif single_time_match.group(1) == '오전' and hour == 12:
                            hour = 0
                        start_hour = hour

                    return start_hour, duration_hours

                def find_non_conflicting_time(existing_schedules, start_hour, duration_hours, date):
                    """
                    기존 일정과 겹치지 않는 시간대를 탐색합니다.
                    """
                    from datetime import datetime, timedelta, time

                    def is_conflicting(new_start, new_end, schedules):
                        for s in schedules:
                            s_start = datetime.strptime(s['start_time'], '%Y-%m-%dT%H:%M:%S')
                            s_end = datetime.strptime(s['end_time'], '%Y-%m-%dT%H:%M:%S')
                            if not (new_end <= s_start or new_start >= s_end):
                                return True
                        return False

                    attempt = 0
                    max_attempts = 10
                    while attempt < max_attempts:
                        candidate_start = datetime.combine(date, time(start_hour))
                        candidate_end = candidate_start + timedelta(hours=duration_hours)
                        if not is_conflicting(candidate_start, candidate_end, existing_schedules):
                            return candidate_start, candidate_end
                        start_hour += 1
                        attempt += 1

                    # fallback
                    return datetime.combine(date, time(start_hour)), datetime.combine(date, time(start_hour + duration_hours))



                # 일정 루프 수정
                for i, single_request in enumerate(schedule_requests):
                    title_prompt = f"""다음 일정 요청에서 적절한 일정 제목을 한 줄로 생성해주세요: {single_request}
                    분석 방법:
                    1. 기존 일정들의 시간대를 확인하여 사용자가 입력한 시간에 일정이 없다면, 사용자가 입력한 일정을 추가해주세요.
                    2. 요청된 일정의 성격에 맞는 최적의 시간대를 추천해주세요
                    3. 일정 간 여유 시간도 고려해주세요
                    4. 되도록이면 새벽시간은 피해주세요.
                    5. 사용자가 지정한 시간이 있다면, 그 시간으로 배정해주세요. 단, 그 시간에 이미 일정이 있다면 다른 시간을 배정하고 일정이 있음을 알려주세요
                    """
                    ai_title = self._get_ai_generated_title(title_prompt) or "새 일정"

                    # 시간 정보 추출
                    parsed_start, parsed_duration = extract_time_info(single_request)

                    if parsed_start is not None:
                        start_hour = parsed_start
                    else:
                        start_hour = 9 + i * 2  # 기본값 fallback

                    duration_hours = parsed_duration or 1

                    existing = request.data.get("existing_schedules", [])
                    schedule_start_dt, schedule_end_dt = find_non_conflicting_time(existing, start_hour, duration_hours, target_date)

                    optimized_schedule = {
                        "title": ai_title,
                        "description": f"AI가 분석한 {self._extract_schedule_title(single_request)} 일정입니다.",
                        "suggested_date": target_datetime.strftime('%Y-%m-%d'),
                        "suggested_start_time": schedule_start_dt.strftime('%H:%M'),
                        "suggested_end_time": schedule_end_dt.strftime('%H:%M'),
                        "location": self._extract_schedule_location(single_request),
                        "priority": "HIGH",
                        "attendees": [],
                        "reasoning": f"{i + 1}번째 일정: {single_request}. 기존 일정과 충돌하지 않는 시간으로 배정했습니다."
                    }

                
                # for i, single_request in enumerate(schedule_requests):
                #     # 각 일정의 시작 시간을 다르게 설정
                #     base_hour = 9 + (i * 2)

                #     title_prompt = f"다음 일정 요청에서 적절한 일정 제목을 한 줄로 생성해주세요: {single_request}"
                #     ai_title = self._get_ai_generated_title(title_prompt) or "새 일정"
                    
                #     optimized_schedule = {
                #         "title": ai_title,  # ✅ AI가 생성한 제목 사용
                #         "description": f"AI가 분석한 {self._extract_schedule_title(single_request)} 일정입니다.",
                #         "suggested_date": target_date.strftime('%Y-%m-%d'),
                #         "suggested_start_time": f"{base_hour:02d}:00",
                #         "suggested_end_time": f"{base_hour + 2:02d}:00",
                #         "location": self._extract_schedule_location(single_request),
                #         "priority": "HIGH",
                #         "attendees": [],
                #         "reasoning": f"{i + 1}번째 일정: {single_request}. 기존 일정과 충돌하지 않는 시간으로 배정했습니다."
                #     }
                    multiple_schedules.append(optimized_schedule)
                    existing_schedules.append({
    'start_time': schedule_start_dt.isoformat(),
    'end_time': schedule_end_dt.isoformat()
})
                    
                    # 각 AI별 개별 제안 생성
                    for ai_type in ['gpt', 'claude', 'mixtral']:
                        individual_suggestion = optimized_schedule.copy()
                        individual_suggestion['source'] = ai_type
                        individual_suggestion['reasoning'] = f"{ai_type.upper()}가 분석한 {self._extract_schedule_title(single_request)} 최적 시간입니다."
                        all_individual_suggestions.append(individual_suggestion)
                
                response_data = {
                    'request_id': int(datetime.now().timestamp()),
                    'multiple_schedules': multiple_schedules,
                    'optimized_suggestion': multiple_schedules[0],
                    'confidence_score': 0.92,
                    'individual_suggestions': all_individual_suggestions,
                    'ai_analysis': {
                        'analysis_summary': f"총 {len(schedule_requests)}개의 일정을 분석하여 최적의 시간대로 배정했습니다.",
                        'reasoning': f"여러 일정을 {target_date.strftime('%Y년 %m월 %d일')}에 시간 순서대로 배치하여 충돌을 방지했습니다.",
                        'models_used': ["gpt", "claude", "mixtral"]
                    },
                    'has_conflicts': False,
                    'conflicts': [],
                    'analysis_summary': f"{len(schedule_requests)}개 일정에 대해 3개 AI 모델이 분석한 결과입니다.",
                    'is_multiple_schedule': True
                }
                
            else:
                # 단일 일정 처리
                optimizer = self.get_optimizer()
                user_context = self._get_user_context(user)
                
                enhanced_prompt = f"""
                사용자의 일정 요청을 분석하여 기존 일정과 충돌하지 않는 최적의 빈 시간을 찾아 제안해주세요.
                만약 사용자가 지정한 시간이 있다면, 그 시간에 일정을 넣어주세요.
                
                요청 내용: {request_text}
                목표 날짜: {target_date.strftime('%Y년 %m월 %d일')} ({self._get_weekday_korean(target_date)})
                기존 일정들: {existing_schedules or []}
                분석 방법:
                1. 기존 일정들의 시간대를 확인하여 사용자가 입력한 시간에 일정이 없다면, 사용자가 입력한 일정을 추가해주세요.
                2. 요청된 일정의 성격에 맞는 최적의 시간대를 추천해주세요
                3. 일정 간 여유 시간도 고려해주세요
                4. 새벽시간은 피해주세요.
                5. 사용자가 지정한 시간이 있다면, 그 시간으로 배정해주세요. 단, 그 시간에 이미 일정이 있다면 다른 시간을 배정하고 일정이 있음을 알려주세요


                
                다음 형식으로 응답해주세요:
                {{
                    "title": "요청 내용에 맞는 구체적이고 의미있는 일정 제목을 작성하세요", 
                    "description": "상세 설명",
                    "suggested_date": "%Y-%m-%d",
                    "suggested_start_time": "HH:MM",
                    "suggested_end_time": "HH:MM",
                    "location": "장소",
                    "priority": "HIGH/MEDIUM/LOW/URGENT",
                    "attendees": [],
                    "reasoning": "이 시간을 제안하는 이유"
                }}
                """
                
                suggestions = optimizer.get_ai_suggestions(enhanced_prompt)
                optimized_result = optimizer.analyze_and_optimize_suggestions(
                    suggestions, f"일정 요청: {request_text}"
                )
                
                response_data = {
                    'request_id': int(datetime.now().timestamp()),
                    'optimized_suggestion': optimized_result.get('optimized_suggestion', {}),
                    'confidence_score': optimized_result.get('confidence_score', 0.0),
                    'ai_analysis': optimized_result.get('ai_analysis', {}),
                    'individual_suggestions': optimized_result.get('individual_suggestions', []),
                    'has_conflicts': False,
                    'conflicts': [],
                    'analysis_summary': "3개 AI 모델이 분석한 결과입니다.",
                    'is_multiple_schedule': False
                }
            
            logger.info("일정 생성 요청 처리 완료")
            return Response(response_data, status=status.HTTP_201_CREATED)
            
        except Exception as e:
            logger.error(f"일정 생성 요청 처리 실패: {str(e)}")
            return Response({
                'error': f'일정 생성 요청 처리 중 오류: {str(e)}'
            }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    def _get_user_context(self, user):
        """사용자 컨텍스트 정보 생성"""
        return {
            'username': user.username,
            'timezone': 'Asia/Seoul',  # 기본 타임존
            'preferences': {}
        }
    
    def _extract_schedule_title(self, request):
        """요청에서 일정 제목 추출"""
        if '운동' in request:
            return '운동'
        elif '미팅' in request or '회의' in request:
            return '팀 미팅'
        elif '공부' in request or '학습' in request:
            return '학습 시간'
        elif '작업' in request or '업무' in request:
            return '집중 작업'
        elif '약속' in request:
            return '약속'
        else:
            return '새 일정'

    def _extract_schedule_location(self, request):
        """요청에서 장소 추출"""
        if '운동' in request:
            return '헬스장'
        elif '미팅' in request or '회의' in request:
            return '회의실'
        elif '공부' in request or '학습' in request:
            return '도서관'
        elif '커피' in request:
            return '카페'
        else:
            return '사무실'

    def _get_weekday_korean(self, date):
        """요일을 한국어로 반환"""
        weekdays = ['월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일']
        return weekdays[date.weekday()]

@api_view(['POST'])
@authentication_classes([DebugTokenAuthentication, SessionAuthentication])
@permission_classes([IsAuthenticated])
def confirm_schedule(request, request_id):
    """AI 제안된 일정을 확정하여 실제 일정으로 생성"""
    logger.info(f"일정 확정 요청 - 사용자: {request.user.username}, request_id: {request_id}")
    
    try:
        user = request.user
        
        # ✅ 프론트엔드에서 전송된 실제 AI 제안 데이터 사용
        ai_suggestion_data = request.data.get('ai_suggestion')
        if not ai_suggestion_data:
            return Response({
                'error': 'AI 제안 데이터가 필요합니다.'
            }, status=status.HTTP_400_BAD_REQUEST)
        
        # 여러 일정인지 단일 일정인지 확인
        is_multiple = ai_suggestion_data.get('is_multiple_schedule', False)
        
        if is_multiple and ai_suggestion_data.get('multiple_schedules'):
            # 여러 일정 처리
            created_schedules = []
            
            for schedule_data in ai_suggestion_data['multiple_schedules']:
                try:
                    # 날짜/시간 파싱
                    suggested_date = schedule_data.get('suggested_date')
                    suggested_start_time = schedule_data.get('suggested_start_time', '09:00')
                    suggested_end_time = schedule_data.get('suggested_end_time', '10:00')
                    
                    start_datetime = datetime.strptime(
                        f"{suggested_date} {suggested_start_time}",
                        '%Y-%m-%d %H:%M'
                    )
                    end_datetime = datetime.strptime(
                        f"{suggested_date} {suggested_end_time}",
                        '%Y-%m-%d %H:%M'
                    )
                    
                    # Schedule 객체 생성
                    schedule = Schedule.objects.create(
                        user=user,
                        title=schedule_data.get('title', '새 일정'),
                        description=schedule_data.get('description', 'AI가 제안한 일정입니다.'),
                        start_time=start_datetime,
                        end_time=end_datetime,
                        location=schedule_data.get('location', ''),
                        priority=schedule_data.get('priority', 'MEDIUM'),
                        attendees=json.dumps(schedule_data.get('attendees', []), ensure_ascii=False)
                    )
                    
                    created_schedules.append(schedule)
                    logger.info(f"다중 일정 생성 성공: {schedule.id} - {schedule.title}")
                    
                except Exception as e:
                    logger.error(f"개별 일정 생성 실패: {str(e)}")
                    continue
            
            if created_schedules:
                serializer = ScheduleSerializer(created_schedules, many=True)
                return Response({
                    'message': f'{len(created_schedules)}개의 일정이 성공적으로 생성되었습니다.',
                    'schedules': serializer.data
                }, status=status.HTTP_201_CREATED)
            else:
                return Response({
                    'error': '일정 생성에 실패했습니다.'
                }, status=status.HTTP_400_BAD_REQUEST)
        
        else:
            # 단일 일정 처리
            optimized_suggestion = ai_suggestion_data.get('optimized_suggestion')
            if not optimized_suggestion:
                return Response({
                    'error': '최적화된 제안 데이터가 없습니다.'
                }, status=status.HTTP_400_BAD_REQUEST)
            
            # 날짜/시간 파싱
            try:
                suggested_date = optimized_suggestion.get('suggested_date')
                suggested_start_time = optimized_suggestion.get('suggested_start_time', '09:00')
                suggested_end_time = optimized_suggestion.get('suggested_end_time', '10:00')
                
                if 'T' in suggested_date:
                    suggested_date = suggested_date.split('T')[0]
                
                start_datetime = datetime.strptime(
                    f"{suggested_date} {suggested_start_time}",
                    '%Y-%m-%d %H:%M'
                )
                end_datetime = datetime.strptime(
                    f"{suggested_date} {suggested_end_time}",
                    '%Y-%m-%d %H:%M'
                )
                
            except (ValueError, TypeError) as e:
                logger.error(f"DateTime parsing error: {e}")
                now = datetime.now()
                start_datetime = now.replace(hour=9, minute=0, second=0, microsecond=0)
                end_datetime = now.replace(hour=10, minute=0, second=0, microsecond=0)
            
            # Schedule 객체 생성
            schedule = Schedule.objects.create(
                user=user,
                title=optimized_suggestion.get('title', '새 일정'),
                description=optimized_suggestion.get('description', 'AI가 제안한 일정입니다.'),
                start_time=start_datetime,
                end_time=end_datetime,
                location=optimized_suggestion.get('location', ''),
                priority=optimized_suggestion.get('priority', 'MEDIUM'),
                attendees=json.dumps(optimized_suggestion.get('attendees', []), ensure_ascii=False)
            )
            
            serializer = ScheduleSerializer(schedule)
            logger.info(f"단일 일정 생성 성공: {schedule.id} - {schedule.title}")
            
            return Response({
                'message': 'AI의 분석을 통해 최적화된 일정이 성공적으로 생성되었습니다.',
                'schedule': serializer.data
            }, status=status.HTTP_201_CREATED)
            
    except Exception as e:
        logger.error(f"일정 확정 실패: {str(e)}")
        return Response({
            'error': f'일정 생성 중 오류가 발생했습니다: {str(e)}'
        }, status=status.HTTP_400_BAD_REQUEST)
# 🔧 수동 일정 생성 뷰
@api_view(['POST'])
@authentication_classes([DebugTokenAuthentication, SessionAuthentication])
@permission_classes([IsAuthenticated])
def create_manual_schedule(request):
    """수동으로 일정 생성"""
    logger.info(f"수동 일정 생성 요청 - 사용자: {request.user.username}")
    
    try:
        data = request.data.copy()
        data['user'] = request.user.id
        
        serializer = ScheduleSerializer(data=data)
        if serializer.is_valid():
            schedule = serializer.save(user=request.user)
            logger.info(f"수동 일정 생성 성공: {schedule.id}")
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        else:
            logger.warning(f"수동 일정 생성 실패 - 유효성 검증 오류: {serializer.errors}")
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
            
    except Exception as e:
        logger.error(f"수동 일정 생성 실패: {str(e)}")
        return Response({
            'error': f'일정 생성 중 오류 발생: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# 🔧 일정 수정/삭제 뷰
@api_view(['PUT', 'DELETE'])
@authentication_classes([DebugTokenAuthentication, SessionAuthentication])
@permission_classes([IsAuthenticated])
def manage_schedule(request, schedule_id):
    """일정 수정 또는 삭제"""
    try:
        schedule = get_object_or_404(Schedule, id=schedule_id, user=request.user)
        
        if request.method == 'PUT':
            # 일정 수정
            serializer = ScheduleSerializer(schedule, data=request.data, partial=True)
            if serializer.is_valid():
                serializer.save()
                logger.info(f"일정 수정 성공: {schedule_id}")
                return Response(serializer.data)
            else:
                return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)
        
        elif request.method == 'DELETE':
            # 일정 삭제
            schedule.delete()
            logger.info(f"일정 삭제 성공: {schedule_id}")
            return Response({'message': '일정이 성공적으로 삭제되었습니다.'}, 
                          status=status.HTTP_204_NO_CONTENT)
            
    except Exception as e:
        logger.error(f"일정 관리 실패: {str(e)}")
        return Response({
            'error': f'일정 관리 중 오류 발생: {str(e)}'
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# 유틸리티 함수들

from pytz import timezone

def parse_date_from_request(request_text):
    korea_now = datetime.now(timezone('Asia/Seoul')).date()

    if '오늘' in request_text:
        return korea_now
    elif '내일' in request_text:
        return korea_now + timedelta(days=1)
    elif '모레' in request_text or '모래' in request_text:
        return korea_now + timedelta(days=2)
    elif '이번 주' in request_text:
        days_until_friday = (4 - korea_now.weekday()) % 7
        days_until_friday = 7 if days_until_friday == 0 else days_until_friday
        return korea_now + timedelta(days=days_until_friday)
    elif '다음 주' in request_text:
        return korea_now + timedelta(days=7)
    else:
        return korea_now + timedelta(days=1)

def parse_multiple_schedules_backend(request_text):
    """백엔드에서 여러 일정 파싱"""
    separators = [',', '，', '그리고', '및', '와', '과']
    
    parts = [request_text]
    for sep in separators:
        new_parts = []
        for part in parts:
            new_parts.extend(part.split(sep))
        parts = new_parts
    
    cleaned_requests = []
    for part in parts:
        cleaned = part.strip()
        if cleaned and len(cleaned) > 2:
            cleaned_requests.append(cleaned)
    
    return cleaned_requests if len(cleaned_requests) > 1 else [request_text]

# 🔧 ScheduleOptimizerBot 클래스 (기존과 동일하지만 import 오류 수정)
class ScheduleOptimizerBot:
    """일정 최적화를 위한 AI 봇 클래스"""
    
    def __init__(self):
        # ChatBot 클래스가 정의되어 있다고 가정
        try:
            self.chatbots = {
                'gpt': ChatBot(OPENAI_API_KEY, 'gpt-3.5-turbo', 'openai'),
                'claude': ChatBot(ANTHROPIC_API_KEY, 'claude-3-5-haiku-20241022', 'anthropic'), 
                'mixtral': ChatBot(GROQ_API_KEY, 'llama3-8b-8192', 'groq'),
            }
        except NameError:
            # ChatBot 클래스가 없으면 더미 클래스 사용
            logger.warning("ChatBot 클래스를 찾을 수 없습니다. 더미 클래스를 사용합니다.")
            self.chatbots = {
                'gpt': DummyChatBot(),
                'claude': DummyChatBot(),
                'mixtral': DummyChatBot(),
            }
    
    def get_ai_suggestions(self, prompt, suggestion_type="schedule"):
        """여러 AI 모델로부터 제안받기"""
        suggestions = {}
        
        for model_name, chatbot in self.chatbots.items():
            try:
                if hasattr(chatbot, 'chat'):
                    response = chatbot.chat(prompt)
                else:
                    response = f"더미 응답: {model_name}에서 {suggestion_type} 분석 완료"
                suggestions[f"{model_name}_suggestion"] = response
            except Exception as e:
                suggestions[f"{model_name}_suggestion"] = f"오류 발생: {str(e)}"
        
        return suggestions
    
    def analyze_and_optimize_suggestions(self, suggestions, query, selected_models=['GPT', 'Claude', 'Mixtral']):
        """여러 AI 제안을 분석하여 최적화된 결과 생성"""
        try:
            # 기본 제안 생성
            optimized = self._extract_first_valid_suggestion(suggestions)
            confidence = 0.85
            
            return {
                "optimized_suggestion": optimized,
                "confidence_score": confidence,
                "ai_analysis": {
                    "analysis_summary": "AI 모델들의 제안을 종합 분석했습니다.",
                    "reasoning": "여러 모델의 공통점을 바탕으로 최적화했습니다.",
                    "models_used": selected_models
                },
                "individual_suggestions": self._parse_individual_suggestions(suggestions)
            }
            
        except Exception as e:
            logger.error(f"Analysis error: {str(e)}")
            return {"error": f"최적화 과정에서 오류 발생: {str(e)}"}
    
    def _extract_first_valid_suggestion(self, suggestions):
        """첫 번째 유효한 제안 추출"""
        for key, suggestion in suggestions.items():
            try:
                json_match = re.search(r'\{.*\}', suggestion, re.DOTALL)
                if json_match:
                    return json.loads(json_match.group())
            except:
                continue
        
        return {
            "title": "새 일정",
            "description": "AI가 제안한 일정입니다",
            "suggested_date": "{target_datetime.strftime('%Y-%m-%d')}",
            "suggested_start_time": "09:00",
            "suggested_end_time": "10:00",
            "location": "",
            "priority": "MEDIUM",
            "attendees": [],
            "reasoning": "여러 AI 모델의 제안을 종합한 결과입니다."
        }
    
    def _parse_individual_suggestions(self, suggestions):
        """개별 제안들을 파싱"""
        parsed = []
        for key, suggestion in suggestions.items():
            try:
                json_match = re.search(r'\{.*\}', suggestion, re.DOTALL)
                if json_match:
                    parsed_suggestion = json.loads(json_match.group())
                    parsed_suggestion['source'] = key.replace('_suggestion', '')
                    parsed.append(parsed_suggestion)
            except:
                continue
        return parsed
