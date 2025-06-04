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
   
   def chat(self, user_input, image_file=None, analysis_mode=None, user_language=None):
       try:
           logger.info(f"Processing chat request for {self.api_type}")
           logger.info(f"User input: {user_input}")
           
           # 대화 기록에 사용자 입력 추가
           if image_file:
               # 예시로 system 메시지에 모드와 언어를 넣어줍니다
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

        #    self.conversation_history.append({"role": "user", "content": user_input})
           
           try:
               if self.api_type == 'openai':
                   response = openai.ChatCompletion.create(
                       model=self.model,
                       messages=self.conversation_history,
                       temperature=0.7,
                       max_tokens=1024
                   )
                   assistant_response = response['choices'][0]['message']['content']
                   # chat 메소드의 anthropic 부분 수정
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
                            system=system_message,  # 시스템 메시지를 system 파라미터로 전달
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

                # analyze_responses 메소드의 anthropic 부분 수정
               
               elif self.api_type == 'anthropic':
                   messages = []
                   for msg in self.conversation_history:
                       if msg["role"] != "system":
                           messages.append({
                               "role": msg["role"],
                               "content": msg["content"]
                           })
                   
                   message = self.client.messages.create(
                       model=self.model,
                       max_tokens=4096,
                       temperature=0,
                    #    messages=messages
                   )
                   assistant_response = message.content[0].text
                   
                   
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

   def analyze_responses(self, responses, query, user_language, selected_models):


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

                # The prompt remains the same
                analysis_prompt = f"""다음은 동일한 질문에 대한 {len(selected_models)}가지 AI의 응답을 분석하는 것입니다.
                        사용자가 선택한 언어는 '{user_language}'입니다.
                        반드시 이 언어({user_language})로 최적의 답을 작성해주세요.
                        반드시 이 언어({user_language})로 장점을 작성해주세요.
                        반드시 이 언어({user_language})로 단점을 작성해주세요.
                        반드시 이 언어({user_language})로 분석 근거를 작성해주세요.

                        질문: {query}
                        {responses_section}

                        [최적의 응답을 만들 때 고려할 사항]

                        모든 AI의 답변들을 종합하여 최적의 답변으로 반드시 재구성합니다.

                        즉, 기존 AI의 답변을 그대로 사용하면 안됩니다.

                        다수의 AI가 공통으로 제공한 정보는 가장 신뢰할 수 있는 올바른 정보로 간주합니다.

                        특정 AI가 다수의 AI와 다른 정보를 제공하면, 신뢰성이 낮은 정보로 판단하여 최적의 답변에서 제외하고, '단점' 항목에 별도로 명시합니다.

                        여러 AI의 답변에서 정확하고 관련성 높은 정보만 선택하여 반영합니다.

                        중복된 정보가 있을 경우 표현이 더 명확하고 상세한 내용을 우선 선택합니다.

                        논리적 흐름을 고려하여 자연스럽고 이해하기 쉬운 형태로 작성합니다.

                        코드를 묻는 질문일때는, AI의 답변 중 제일 좋은 답변을 선택해서 재구성해줘 

                        코드는 바로 복사해서 사용가능하도록 해줘

                        코드는 그대로 복사해서 실행 버튼만 누르면 실행 가능하도록 작성해야합니다.

                        코드와 코드가 아닌 부분을 구별되게 보여줘

                        반드시 JSON 형식으로 응답해주세요. 다른 설명은 포함하지 마세요.

                        뉴스 기사를 분석한 경우 육하원칙으로 답변해줘

                        [출력 형식]
                        {{
                            "preferredModel": "{self.api_type.upper()}",
                            "best_response": "최적의 답변 ({user_language}로 작성)",
                            "analysis": {{
                                {analysis_section}
                            }},
                            "reasoning": "반드시 이 언어({user_language})로 작성 최적의 응답을 선택한 이유"
                        }}"""

                # API 타입에 따른 분기 처리
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
                    # 시스템 메시지에서 언어 설정 추출
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
                
                # Use our new sanitize_and_parse_json function
                analysis_result = sanitize_and_parse_json(analysis_text, selected_models, responses)
                analysis_result['preferredModel'] = self.api_type.upper()
                
                return analysis_result
            
            except Exception as e:
                logger.error(f"❌ Analysis error: {str(e)}")
                # Fallback response in case of a major error
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
import logging
import json
from django.http import StreamingHttpResponse
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

# class ChatView(APIView):
#     permission_classes = [AllowAny]
    
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         # SimilarityAnalyzer 인스턴스 생성
#         self.similarity_analyzer = SimilarityAnalyzer(threshold=0.85)

#     def post(self, request, preferredModel):
#         try:
#             logger.info(f"Received chat request for {preferredModel}")
            
#             data = request.data
#             user_message = data.get('message')
#             compare_responses = data.get('compare', True)
            
#             # 선택된 모델들
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
                    
#                     # 응답이 2개 이상일 때만 유사도 분석 수행
#                     if len(responses) >= 2:
#                         try:
#                             # 유사도 분석 결과 계산
#                             similarity_result = self.similarity_analyzer.cluster_responses(responses)
                            
#                             # 결과를 직렬화 가능한 형태로 변환
#                             serializable_result = convert_to_serializable(similarity_result)
                            
#                             # 디버깅을 위한 유사도 분석 결과 로깅
#                             logger.info(f"Similarity analysis result: {serializable_result}")
                            
#                             # 유사도 분석 결과 전송
#                             yield json.dumps({
#                                 'type': 'similarity_analysis',
#                                 'result': serializable_result,
#                                 'requestId': request_id,
#                                 'timestamp': time.time(),
#                                 'userMessage': user_message  # 사용자 메시지 포함
#                             }) + '\n'
                            
#                         except Exception as e:
#                             logger.error(f"Error in similarity analysis: {str(e)}", exc_info=True)
#                             yield json.dumps({
#                                 'type': 'similarity_error',
#                                 'error': f"Similarity analysis error: {str(e)}",
#                                 'requestId': request_id
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
#                             'timestamp': time.time(),  # 타임스탬프 추가
#                             'userMessage': user_message  # 사용자 메시지 포함
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
class ChatView(APIView):
    permission_classes = [AllowAny]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.similarity_analyzer = SimilarityAnalyzer(threshold=0.85)

    def post(self, request, preferredModel):
        try:
            logger.info(f"Received chat request for {preferredModel}")
            data = request.data
            user_message = data.get('message')
            selected_models = data.get('selectedModels', ['gpt', 'claude', 'mixtral'])
            token = request.headers.get('Authorization')
            user_language = 'ko' if not token else data.get('language', 'ko')
            if not user_message:
                return Response({'error': 'No message provided'}, status=status.HTTP_400_BAD_REQUEST)

            # 링크만 있을 경우 페이지 내용으로 분석 요청
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

                    # 모델 응답 스트리밍
                    for bot_id, bot in selected_chatbots.items():
                        try:
                            bot.conversation_history = [system_message]
                            resp_text = bot.chat(user_message)
                            responses[bot_id] = resp_text
                            yield json.dumps({'type':'bot_response','botId':bot_id,'response':resp_text,'requestId':request_id}) + '\n'
                        except Exception as e:
                            yield json.dumps({'type':'bot_error','botId':bot_id,'error':str(e),'requestId':request_id}) + '\n'

                    # 유사도 분석
                    if len(responses) >= 2:
                        sim_res = self.similarity_analyzer.cluster_responses(responses)
                        serial = convert_to_serializable(sim_res)
                        yield json.dumps({'type':'similarity_analysis','result':serial,'requestId':request_id,'timestamp':time.time(),'userMessage':user_message}) + '\n'

                    # 최종 비교 및 분석
                    analyzer_bot = chatbots.get(preferredModel) or chatbots.get('gpt')
                    analyzer_bot.conversation_history = [system_message]
                    analysis = analyzer_bot.analyze_responses(responses, user_message, user_language, list(responses.keys()))
                    yield json.dumps({
                        'type':'analysis',
                        'preferredModel': analyzer_bot.api_type.upper(),
                        'best_response': analysis.get('best_response',''),
                        'analysis': analysis.get('analysis',{}),
                        'reasoning': analysis.get('reasoning',''),
                        'language': user_language,
                        'requestId': request_id,
                        'timestamp': time.time(),
                        'userMessage': user_message
                    }) + '\n'
                except Exception as e:
                    yield json.dumps({'type':'error','error':f"Stream error: {e}"}) + '\n'

            return StreamingHttpResponse(stream_responses(), content_type='text/event-stream')

        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            return Response({'error': str(e)}, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

# API 키 설정
OPENAI_API_KEY = "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"
ANTHROPIC_API_KEY = "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"
GROQ_API_KEY = "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"

chatbots = {
    'gpt': ChatBot(OPENAI_API_KEY, 'gpt-3.5-turbo', 'openai'),
    'claude': ChatBot(ANTHROPIC_API_KEY, 'claude-3-5-haiku-20241022', 'anthropic'), 
    'mixtral': ChatBot(GROQ_API_KEY, 'llama3-8b-8192', 'groq'),
}
# chatbots = {
#     'gpt': ChatBot(OPENAI_API_KEY, 'gpt-4-turbo', 'openai'),
#     'claude': ChatBot(ANTHROPIC_API_KEY, 'claude-3-opus-20240229', 'anthropic'), 
#     'mixtral': ChatBot(GROQ_API_KEY, 'llama3-8b-8192', 'groq'),
# }


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
# # similarity_analyzer.py
# import logging
# import re
# import math
# import numpy as np
# from collections import Counter
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity

# logger = logging.getLogger(__name__)

# class SimilarityAnalyzer:
#     """
#     AI 모델 응답 간의 유사도를 분석하고 응답 특성을 추출하는 클래스
#     """
    
#     def __init__(self, threshold=0.05):
#         """
#         초기화
        
#         Args:
#             threshold (float): 유사 응답으로 분류할 임계값 (0~1)
#         """
#         self.threshold = threshold
#         self.vectorizer = TfidfVectorizer(
#             min_df=1, 
#             analyzer='word',
#             ngram_range=(1, 2),
#             stop_words='english'
#         )
    
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
        
#         # 특수 문자 제거
#         text = re.sub(r'[^\w\s]', ' ', text)
        
#         # 여러 공백을 하나로 치환
#         text = re.sub(r'\s+', ' ', text).strip()
        
#         return text
    
#     def calculate_similarity_matrix(self, responses):
#         """
#         모델 응답 간의 유사도 행렬 계산
        
#         Args:
#             responses (dict): 모델 ID를 키로, 응답 텍스트를 값으로 하는 딕셔너리
            
#         Returns:🤖
#             dict: 모델 간 유사도 행렬
#         """
#         try:
#             model_ids = list(responses.keys())
            
#             # 텍스트 전처리
#             preprocessed_texts = [self.preprocess_text(responses[model_id]) for model_id in model_ids]
            
#             # TF-IDF 벡터화
#             tfidf_matrix = self.vectorizer.fit_transform(preprocessed_texts)
            
#             # 코사인 유사도 계산
#             similarity_matrix = cosine_similarity(tfidf_matrix)
            
#             # 결과를 딕셔너리로 변환
#             result = {}
#             for i, model1 in enumerate(model_ids):
#                 result[model1] = {}
#                 for j, model2 in enumerate(model_ids):
#                     result[model1][model2] = float(similarity_matrix[i][j])
            
#             return result
            
#         except Exception as e:
#             logger.error(f"유사도 행렬 계산 중 오류: {str(e)}")
#             # 오류 발생 시 빈 행렬 반환
#             return {model_id: {other_id: 0.0 for other_id in responses} for model_id in responses}
    
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
            
#             # 유사도 행렬 계산
#             similarity_matrix = self.calculate_similarity_matrix(responses)
            
#             # 계층적 클러스터링 수행
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
                
#                 # 임계값보다 유사도가 높으면 클러스터 병합
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
            
#             # 응답 특성 추출
#             response_features = {model_id: self.extract_response_features(responses[model_id]) 
#                                 for model_id in model_ids}
            
#             return {
#                 "similarGroups": clusters,
#                 "mainGroup": main_group,
#                 "outliers": outliers,
#                 "similarityMatrix": similarity_matrix,
#                 "responseFeatures": response_features
#             }
            
#         except Exception as e:
#             logger.error(f"응답 군집화 중 오류: {str(e)}")
#             # 오류 발생 시 모든 모델을 하나의 그룹으로 반환
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
            
#             return {
#                 "length": length,
#                 "codeBlockCount": code_block_count,
#                 "linkCount": link_count,
#                 "listItemCount": list_item_count,
#                 "sentenceCount": len(sentences),
#                 "avgSentenceLength": avg_sentence_length,
#                 "vocabularyDiversity": vocabulary_diversity,
#                 "hasCode": code_block_count > 0
#             }
            
#         except Exception as e:
#             logger.error(f"응답 특성 추출 중 오류: {str(e)}")
#             # 오류 발생 시 기본값 반환
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




# ---------


# views.py에 추가할 이미지 분석 API 코드

# from rest_framework.views import APIView
# from rest_framework.response import Response
# from rest_framework import status
# from rest_framework.permissions import AllowAny
# from django.http import StreamingHttpResponse
# import logging
# import json
# import base64
# import time
# import os
# from PIL import Image
# from io import BytesIO

# logger = logging.getLogger(__name__)

# class ImageAnalysisView(APIView):
#     permission_classes = [AllowAny]
    
#     def post(self, request):
#         try:
#             logger.info("이미지 분석 요청 수신")
            
#             # 요청 데이터 추출
#             image_file = request.FILES.get('image')
#             prompt = request.data.get('prompt', '이 이미지를 분석해주세요.')
#             analysis_mode = request.data.get('analysisMode', 'describe')
#             request_id = request.data.get('requestId')
#             selected_models = json.loads(request.data.get('selectedModels', '["gpt", "claude", "mixtral"]'))
#             user_language = request.data.get('language', 'ko')
            
#             if not image_file:
#                 return Response({'error': '이미지 파일이 없습니다.'}, status=status.HTTP_400_BAD_REQUEST)
            
#             # 이미지 처리 및 인코딩
#             image = Image.open(image_file)
            
#             # 이미지 크기 조정 (너무 큰 경우)
#             max_size = 1024
#             if max(image.width, image.height) > max_size:
#                 if image.width > image.height:
#                     new_width = max_size
#                     new_height = int(image.height * max_size / image.width)
#                 else:
#                     new_height = max_size
#                     new_width = int(image.width * max_size / image.height)
#                 image = image.resize((new_width, new_height), Image.LANCZOS)
#              # 투명도 채널(RGBA)이 있으면 RGB로 변환
#             if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
#                image = image.convert("RGB")
    
#             # API 토큰 및 모델 초기화
#             OPENAI_API_KEY = "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"
#             ANTHROPIC_API_KEY = "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"
#             GROQ_API_KEY = "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"
            
#             from .views import ChatBot  # ChatBot 클래스 임포트
            
#             chatbots = {
#                 'gpt': ChatBot(OPENAI_API_KEY, 'gpt-3.5-turbo', 'openai'),
#                 'claude': ChatBot(ANTHROPIC_API_KEY, 'claude-3-5-haiku-20241022', 'anthropic'), 
#                 'mixtral': ChatBot(GROQ_API_KEY, 'llama3-8b-8192', 'groq'),
#             }
            
#             # 이미지 분석 모드에 따른 시스템 메시지 구성
#             system_message = {
#                 "role": "system",
#                 "content": f"""사용자가 선택한 언어는 '{user_language}'입니다. 반드시 이 언어({user_language})로 응답하세요.
#                 당신은 이미지 분석 전문가입니다. 사용자가 업로드한 이미지를 상세히 분석하세요.
#                 분석 모드: {analysis_mode}
#                 - describe: 이미지의 모든 중요 요소를 자세히 설명
#                 - ocr: 이미지에서 텍스트 추출 및 구조화
#                 - objects: 이미지에서 모든 객체를 인식하고 목록화
#                 """
#             }
            
#             # 스트리밍 응답 함수
#             def stream_responses():
#                 try:
#                     responses = {}
                    
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
                            
#                             # 이미지 분석 프롬프트 생성
#                             response = bot.chat(
#                             prompt=prompt,
#                             image_file=image_file,           # ← JPEG로 리사이즈/convert 된 PIL 파일
#                             analysis_mode=analysis_mode,
#                             user_language=user_language
#                         )
                            
#                             # 이미지 분석 수행
#                             responses[bot_id] = response
                            
#                             # 각 봇 응답을 즉시 전송
#                             yield json.dumps({
#                                 'type': 'bot_response',
#                                 'botId': bot_id,
#                                 'response': response,
#                                 'requestId': request_id
#                             }) + '\n'
                            
#                         except Exception as e:
#                             logger.error(f"Error from {bot_id}: {str(e)}")
#                             responses[bot_id] = f"Error: {str(e)}"
                            
#                             # 에러도 즉시 전송
#                             yield json.dumps({
#                                 'type': 'bot_error',
#                                 'botId': bot_id,
#                                 'error': str(e),
#                                 'requestId': request_id
#                             }) + '\n'
                    
#                     # 유사도 분석 기능 - responses에 2개 이상의 응답이 있을 때
#                     if len(responses) >= 2:
#                         try:
#                             from .views import SimilarityAnalyzer
#                             similarity_analyzer = SimilarityAnalyzer(threshold=0.85)
#                             similarity_result = similarity_analyzer.cluster_responses(responses)
                            
#                             # 유사도 분석 결과 전송
#                             yield json.dumps({
#                                 'type': 'similarity_analysis',
#                                 'result': self.convert_to_serializable(similarity_result),
#                                 'requestId': request_id,
#                                 'timestamp': time.time(),
#                                 'userMessage': prompt
#                             }) + '\n'
                            
#                         except Exception as e:
#                             logger.error(f"유사도 분석 오류: {str(e)}")
#                             yield json.dumps({
#                                 'type': 'similarity_error',
#                                 'error': f"유사도 분석 오류: {str(e)}",
#                                 'requestId': request_id
#                             }) + '\n'
                    
#                     # 최종 분석 수행
#                     if responses:
#                         # 분석(비교)을 위해 선택된 모델 또는 기본 모델 사용
#                         analyzer_bot = chatbots.get(selected_models[0]) or chatbots.get('gpt')
                        
#                         # 분석용 봇도 새로운 대화 컨텍스트로 초기화
#                         analyzer_bot.conversation_history = [system_message]
                        
#                         # 분석 실행
#                         analysis = analyzer_bot.analyze_responses(
#                             responses, 
#                             prompt, 
#                             user_language, 
#                             list(responses.keys())
#                         )
                        
#                         # 분석 결과 전송
#                         yield json.dumps({
#                             'type': 'analysis',
#                             'preferredModel': analyzer_bot.api_type.upper(),
#                             'best_response': analysis.get('best_response', ''),
#                             'analysis': analysis.get('analysis', {}),
#                             'reasoning': analysis.get('reasoning', ''),
#                             'language': user_language,
#                             'requestId': request_id,
#                             'timestamp': time.time(),
#                             'userMessage': prompt
#                         }) + '\n'
#                     else:
#                         logger.warning("No responses to analyze")
                    
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
#             logger.error(f"이미지 분석 중 오류 발생: {str(e)}", exc_info=True)
#             return Response({
#                 'error': str(e)
#             }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
#     def convert_to_serializable(self, obj):
#         """모든 객체를 JSON 직렬화 가능한 형태로 변환"""
#         import numpy as np
        
#         if isinstance(obj, np.ndarray):
#             return obj.tolist()
#         elif isinstance(obj, np.integer):
#             return int(obj)
#         elif isinstance(obj, (np.float16, np.float32, np.float64)):
#             return float(obj)
#         elif isinstance(obj, (set, tuple)):
#             return list(obj)
#         elif hasattr(obj, 'isoformat'):
#             return obj.isoformat()
#         elif isinstance(obj, dict):
#             return {k: self.convert_to_serializable(v) for k, v in obj.items()}
#         elif isinstance(obj, list):
#             return [self.convert_to_serializable(i) for i in obj]
#         else:
#             try:
#                 return str(obj)
#             except:
#                 return repr(obj)
            

# import os
# import pytesseract
# from PIL import Image
# import requests
# import json
# import PyPDF2
# import tempfile
# from pdf2image import convert_from_path
# from django.conf import settings
# from rest_framework import status
# from rest_framework.response import Response
# from rest_framework.views import APIView
# from rest_framework.generics import RetrieveAPIView, ListAPIView
# from .models import OCRResult
# from .serializers import OCRResultSerializer

# from django.views.decorators.csrf import csrf_exempt
# from django.utils.decorators import method_decorator

# import logging
# @method_decorator(csrf_exempt, name='dispatch')
# class ProcessFileView(APIView):
#     permission_classes = [AllowAny]
    
#     def post(self, request):
#         try:
#             logger.info("ProcessFileView 요청 수신: %s %s", request.method, request.path)
#             logger.info("요청 헤더: %s", request.headers)
#             logger.info("요청 쿠키: %s", request.COOKIES)
#             logger.info("요청 META: %s, %s", request.META.get('CSRF_COOKIE', '없음'), request.META.get('HTTP_X_CSRFTOKEN', '없음'))
            
#             # 권한 체크 로깅
#             for permission in self.get_permissions():
#                 has_permission = permission.has_permission(request, self)
#                 logger.info("권한 체크 %s: %s", permission.__class__.__name__, has_permission)
            
#             # 1. 파일 확인
#             if 'file' not in request.FILES:
#                 logger.error("파일이 제공되지 않음")
#                 return Response({'error': '파일이 제공되지 않았습니다'}, status=status.HTTP_400_BAD_REQUEST)
            
#             file_obj = request.FILES['file']
#             file_name = file_obj.name.lower()
#             logger.info("파일 업로드: %s, 크기: %s bytes", file_name, file_obj.size)
            
#             # 2. 파일 유형 확인
#             if file_name.endswith(('.pdf')):
#                 file_type = 'pdf'
#             elif file_name.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')):
#                 file_type = 'image'
#             else:
#                 logger.error("지원되지 않는 파일 형식: %s", file_name)
#                 return Response({'error': '지원되지 않는 파일 형식입니다. 이미지나 PDF 파일을 업로드해주세요.'},
#                               status=status.HTTP_400_BAD_REQUEST)
            
#             logger.info("파일 유형: %s", file_type)
            
#             # 3. OCR 결과 모델 생성
#             try:
#                 ocr_result = OCRResult.objects.create(file=file_obj, file_type=file_type)
#                 logger.info("OCRResult 객체 생성: %s", ocr_result.id)
#             except Exception as e:
#                 logger.error("OCRResult 모델 생성 실패: %s", str(e), exc_info=True)
#                 return Response({'error': f'OCRResult 모델 생성 실패: {str(e)}'}, 
#                                status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
#             # 4. OCR 처리
#             try:
#                 ocr_text = ""
                
#                 if file_type == 'image':
#                     logger.info("이미지 OCR 처리 시작: %s", ocr_result.file.path)
#                     img = Image.open(ocr_result.file.path)
#                     ocr_text = pytesseract.image_to_string(img, lang='kor+eng')
#                     logger.info("이미지 OCR 처리 완료, 추출 텍스트 길이: %s", len(ocr_text))
                
#                 elif file_type == 'pdf':
#                     logger.info("PDF 처리 시작: %s", ocr_result.file.path)
#                     pdf_text = self.extract_text_from_pdf(ocr_result.file.path)
#                     logger.info("PDF 직접 텍스트 추출 완료, 텍스트 길이: %s", len(pdf_text))
                    
#                     if len(pdf_text.strip()) < 100:
#                         logger.info("PDF OCR 처리 시작 (텍스트가 충분하지 않음)")
#                         pdf_ocr_text = self.ocr_pdf(ocr_result.file.path)
#                         ocr_text = pdf_ocr_text if pdf_ocr_text else pdf_text
#                         logger.info("PDF OCR 처리 완료, 추출 텍스트 길이: %s", len(ocr_text))
#                     else:
#                         ocr_text = pdf_text
                
#                 ocr_result.ocr_text = ocr_text
#                 logger.info("OCR 텍스트 저장 (일부): %s...", ocr_text[:100])
            
#             except Exception as e:
#                 logger.error("OCR 처리 실패: %s", str(e), exc_info=True)
#                 return Response({'error': f'OCR 처리 실패: {str(e)}'}, 
#                                status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
#             # 5. LLM 처리
#             try:
#                 logger.info("LLM 처리 시작")
                
#                 # OpenAI API 키 설정 확인
#                 api_key = os.environ.get('OPENAI_API_KEY', "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")
#                 logger.info("API 키 설정: %s", api_key[:10] + "..." if api_key else "API 키 없음")
                
#                 if not api_key:
#                     logger.error("LLM API 키가 설정되지 않음")
#                     ocr_result.llm_response = "API 키가 설정되지 않아 LLM 처리를 수행할 수 없습니다."
#                 else:
#                     headers = {
#                         'Authorization': f'Bearer {api_key}',
#                         'Content-Type': 'application/json'
#                     }
                    
#                     data = {
#                         'model': 'gpt-3.5-turbo',
#                         'messages': [
#                             {'role': 'system', 'content': '당신은 OCR 텍스트를 처리하는 도우미입니다.'},
#                             {'role': 'user', 'content': f'다음 {"PDF에서" if file_type == "pdf" else "이미지에서"} 추출한 텍스트를 요약하고 분석해주세요:\n\n{ocr_text}'}
#                         ],
#                         'max_tokens': 500
#                     }
                    
#                     logger.info("OpenAI API 요청 시작")
#                     response = requests.post('https://api.openai.com/v1/chat/completions', 
#                                           headers=headers, data=json.dumps(data))
#                     logger.info("OpenAI API 응답 상태 코드: %s", response.status_code)
                    
#                     if response.status_code == 200:
#                         response_data = response.json()
#                         logger.info("OpenAI API 응답 데이터: %s", response_data)
                        
#                         if 'choices' in response_data and len(response_data['choices']) > 0:
#                             llm_response = response_data['choices'][0]['message']['content']
#                             ocr_result.llm_response = llm_response
#                             logger.info("LLM 응답 (일부): %s...", llm_response[:100])
#                         else:
#                             ocr_result.llm_response = "LLM 처리가 예상된 응답을 반환하지 않았습니다."
#                             logger.warning("LLM 응답에 'choices' 필드가 없거나 비어 있음")
#                     else:
#                         error_msg = f"OpenAI API 오류 (상태 코드: {response.status_code}): {response.text}"
#                         ocr_result.llm_response = error_msg
#                         logger.error(error_msg)
            
#             except Exception as e:
#                 error_msg = f"LLM 처리 오류: {str(e)}"
#                 ocr_result.llm_response = error_msg
#                 logger.error(error_msg, exc_info=True)
            
#             # 6. 최종 결과 저장
#             try:
#                 ocr_result.save()
#                 logger.info("OCRResult 저장 완료 (ID: %s)", ocr_result.id)
#             except Exception as e:
#                 logger.error("OCRResult 저장 실패: %s", str(e), exc_info=True)
#                 return Response({'error': f'결과 저장 실패: {str(e)}'}, 
#                                status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
#             # 7. 직렬화 및 응답
#             try:
#                 serializer = OCRResultSerializer(ocr_result)
#                 logger.info("처리 완료, 응답 반환")
#                 return Response(serializer.data, status=status.HTTP_201_CREATED)
#             except Exception as e:
#                 logger.error("응답 직렬화 실패: %s", str(e), exc_info=True)
#                 return Response({'error': f'응답 생성 실패: {str(e)}'}, 
#                                status=status.HTTP_500_INTERNAL_SERVER_ERROR)
                
#         except Exception as e:
#             logger.error("처리 중 예기치 않은 오류: %s", str(e), exc_info=True)
#             return Response({'error': f'서버 오류: {str(e)}'}, 
#                            status=status.HTTP_500_INTERNAL_SERVER_ERROR)

#     def extract_text_from_pdf(self, pdf_path):
#         """PyPDF2를 사용하여 PDF에서 직접 텍스트 추출"""
#         text = ""
#         with open(pdf_path, 'rb') as file:
#             reader = PyPDF2.PdfReader(file)
#             for page_num in range(len(reader.pages)):
#                 page = reader.pages[page_num]
#                 text += page.extract_text() + "\n"
#         return text
    
#     def ocr_pdf(self, pdf_path):
#         """PDF를 이미지로 변환하여 OCR 수행"""
#         all_text = ""
#         try:
#             # PDF를 이미지로 변환
#             with tempfile.TemporaryDirectory() as path:
#                 images = convert_from_path(pdf_path, output_folder=path)
                
#                 # 각 페이지 처리
#                 for i, img in enumerate(images):
#                     text = pytesseract.image_to_string(img, lang='kor+eng')
#                     all_text += f"\n--- 페이지 {i+1} ---\n{text}\n"
                    
#             return all_text
#         except Exception as e:
#             print(f"PDF OCR 오류: {e}")
#             return ""

# class OCRResultDetailView(RetrieveAPIView):
#     queryset = OCRResult.objects.all()
#     serializer_class = OCRResultSerializer

# class OCRResultListView(ListAPIView):
#     queryset = OCRResult.objects.all().order_by('-created_at')
#     serializer_class = OCRResultSerializer

# # chat/ollama_client.py
# import pytesseract
# import requests
# import json
# import logging
# import base64
# from io import BytesIO
# from PIL import Image
# from .models import OCRResult
# from .serializers import OCRResultSerializer  # OCRResultSerializer import 추가
# logger = logging.getLogger(__name__)

# class OllamaClient:
#     def __init__(self, base_url="http://localhost:11434"):
#         self.base_url = base_url
        
#     def analyze_image(self, image_path, prompt):
#         """이미지를 분석하는 Ollama API 호출"""
#         try:
#             # 이미지 파일 로드 및 base64 인코딩
#             with open(image_path, "rb") as image_file:
#                 image_data = image_file.read()
#                 base64_image = base64.b64encode(image_data).decode('utf-8')
            
#             # API 요청 데이터 준비
#             payload = {
#                 "model": "llava:latest",  # 또는 다른 멀티모달 모델
#                 "prompt": prompt,
#                 "images": [base64_image]
#             }
            
#             # API 호출
#             response = requests.post(
#                 f"{self.base_url}/api/generate",
#                 json=payload
#             )
            
#             if response.status_code != 200:
#                 logger.error(f"Ollama API 오류: {response.status_code}, {response.text}")
#                 return f"Ollama API 오류: {response.status_code}"
            
#             # 응답 처리
#             response_data = response.json()
#             return response_data.get("response", "응답을 받지 못했습니다.")
            
#         except Exception as e:
#             logger.error(f"Ollama 이미지 분석 오류: {str(e)}")
#             return f"이미지 분석 처리 중 오류가 발생했습니다: {str(e)}"
    
#     def analyze_text(self, text, prompt=None):
#         """텍스트를 분석하는 Ollama API 호출"""
#         try:
#             # 기본 프롬프트 설정
#             if not prompt:
#                   prompt = f"""다음 텍스트를 자세히 분석하고 페이지별로 명확하게 정리해주세요:

# {text}

# 분석 지침:
# 1. 각 페이지나 문단을 구분해서 정리해주세요.
# 2. 내용을 요약하지 말고, 각 페이지/섹션의 핵심 정보를 충실하게 정리해주세요.
# 3. 페이지 번호나 섹션이 명확하게 구분되어 있다면 그대로 표시해주세요.
# 4. 모든 중요한 세부 정보를 포함해주세요.
# 5. 내용을 단순 요약하지 말고, 구조화된 형식으로 정리해주세요.

# 다음과 같은 형식으로 정리해주세요:
# ===== 페이지 1 (또는 섹션 1) =====
# - 주요 내용 정리
# - 중요 개념 설명
# - 핵심 정보 나열

# ===== 페이지 2 (또는 섹션 2) =====
# - 주요 내용 정리
# ...

# 반드시 한국어로 응답해주세요."""
#             else:
#                 prompt = f"{prompt}\n\n{text}"
            
#             # API 요청 데이터 준비
#             payload = {
#                 "model": "llama3:8b",  # 텍스트 분석용 모델
#                 "prompt": prompt
#             }
            
#             # API 호출
#             response = requests.post(
#                 f"{self.base_url}/api/generate",
#                 json=payload
#             )
            
#             if response.status_code != 200:
#                 logger.error(f"Ollama API 오류: {response.status_code}, {response.text}")
#                 return f"Ollama API 오류: {response.status_code}"
            
#             # 응답 처리
#             response_data = response.json()
#             return response_data.get("response", "응답을 받지 못했습니다.")
            
#         except Exception as e:
#             logger.error(f"Ollama 텍스트 분석 오류: {str(e)}")
#             return f"텍스트 분석 처리 중 오류가 발생했습니다: {str(e)}"

# import logging
# import json
# import os
# from django.views.decorators.csrf import csrf_exempt
# from django.utils.decorators import method_decorator
# from rest_framework.views import APIView
# from rest_framework.response import Response
# from rest_framework import status
# from rest_framework.permissions import AllowAny
# from PIL import Image, ImageFilter, ImageEnhance
# import pytesseract
# from .models import OCRResult
# from .serializers import OCRResultSerializer

# import PyPDF2
# import tempfile
# from pdf2image import convert_from_path
# import re

# logger = logging.getLogger(__name__)

# # OllamaClient 클래스 가져오기
# from .ollama_client import OllamaClient
# @method_decorator(csrf_exempt, name='dispatch')
# class ProcessFileView(APIView):
#     permission_classes = [AllowAny]
    
#     def post(self, request):
#         try:
#             logger.info("ProcessFileView 요청 수신: %s %s", request.method, request.path)
            
#             # 요청 데이터 확인
#             if 'file' not in request.FILES:
#                 logger.error("파일이 제공되지 않음")
#                 return Response({'error': '파일이 제공되지 않았습니다'}, status=status.HTTP_400_BAD_REQUEST)
            
#             file_obj = request.FILES['file']
#             file_name = file_obj.name.lower()
#             logger.info("파일 업로드: %s, 크기: %s bytes", file_name, file_obj.size)
            
#             # 파일 유형 확인
#             if file_name.endswith(('.pdf')):
#                 file_type = 'pdf'
                
#                 # PDF 페이지 범위 확인
#                 start_page = int(request.data.get('start_page', 1))
#                 end_page = int(request.data.get('end_page', 0))  # 0은 전체 페이지를 의미
                
#                 logger.info("PDF 처리 범위: %s ~ %s 페이지", start_page, end_page if end_page > 0 else "끝")
                
#             elif file_name.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')):
#                 file_type = 'image'
#             else:
#                 logger.error("지원되지 않는 파일 형식: %s", file_name)
#                 return Response({'error': '지원되지 않는 파일 형식입니다. 이미지나 PDF 파일을 업로드해주세요.'},
#                               status=status.HTTP_400_BAD_REQUEST)
            
#             # OCR 결과 객체 생성
#             ocr_result = OCRResult.objects.create(file=file_obj, file_type=file_type)
#             logger.info("OCRResult 객체 생성: %s", ocr_result.id)
            
#             # OCR 처리
#             try:
#                 ocr_text = ""
#                 page_texts = []  # 페이지별 텍스트 저장
#                 text_relevant = False  # 텍스트 관련성 여부 (기본값: 관련 없음)
                
#                 if file_type == 'image':
#                     # 이미지 파일 처리 - 개선된 OCR 적용
#                     img = Image.open(ocr_result.file.path)
#                     # 이미지 정보 로깅
#                     logger.info(f"이미지 정보: 크기={img.size}, 모드={img.mode}, 포맷={img.format}")
                    
#                     # 이미지 전처리 및 OCR 수행
#                     preprocessed_img = self.preprocess_image_for_ocr(img)
#                     ocr_text = self.get_optimized_ocr_text(preprocessed_img, lang='kor+eng')
#                     page_texts.append({"page": 1, "text": ocr_text})
#                     logger.info("이미지 OCR 처리 완료, 추출 텍스트 길이: %s", len(ocr_text))
#                     logger.info("추출된 텍스트 샘플: %s", ocr_text[:200] if ocr_text else "텍스트 없음")
                
#                 elif file_type == 'pdf':
#                     # PDF 처리 - 직접 추출 후 필요시 OCR
#                     logger.info("PDF 처리 시작: %s", ocr_result.file.path)
                    
#                     # 직접 텍스트 추출 시도 (페이지별)
#                     direct_extract_success = False
#                     try:
#                         all_page_texts = self.extract_text_from_pdf_by_pages(ocr_result.file.path)
                        
#                         # 페이지 범위 처리
#                         if start_page > 1 or (end_page > 0 and end_page < len(all_page_texts)):
#                             if start_page <= len(all_page_texts):
#                                 if end_page > 0 and end_page >= start_page:
#                                     page_texts = all_page_texts[start_page-1:end_page]
#                                 else:
#                                     page_texts = all_page_texts[start_page-1:]
#                             else:
#                                 page_texts = []
#                         else:
#                             page_texts = all_page_texts
                        
#                         combined_text = "\n".join([page["text"] for page in page_texts])
                        
#                         # 추출된 텍스트 길이가 충분한지 확인 (더 엄격한 조건)
#                         if combined_text.strip() and len(combined_text.strip()) >= 50:
#                             meaningful_chars = sum(1 for c in combined_text if c.isalnum())
#                             if meaningful_chars > 30:  # 의미있는 글자가 30자 이상이면 성공으로 간주
#                                 ocr_text = combined_text
#                                 direct_extract_success = True
#                                 logger.info("PDF 직접 텍스트 추출 성공, 총 %s 페이지, 텍스트 길이: %s", 
#                                           len(page_texts), len(ocr_text))
#                                 logger.info("추출된 텍스트 샘플: %s", ocr_text[:200] if ocr_text else "텍스트 없음")
#                     except Exception as e:
#                         logger.error(f"PDF 직접 텍스트 추출 실패: {str(e)}")
                    
#                     # 직접 텍스트 추출이 실패한 경우, OCR 시도
#                     if not direct_extract_success:
#                         logger.info("PDF OCR 처리 시작 (직접 추출 실패 또는 텍스트 불충분)")
                        
#                         # 페이지 범위 설정으로 OCR
#                         all_page_texts = self.ocr_pdf_by_pages(ocr_result.file.path, start_page, end_page)
                        
#                         # 페이지 범위 처리 - ocr_pdf_by_pages에서 처리했으므로 전체 사용
#                         page_texts = all_page_texts
                        
#                         ocr_text = "\n".join([page["text"] for page in page_texts])
#                         logger.info("PDF OCR 처리 완료, 총 %s 페이지, 텍스트 길이: %s", 
#                                   len(page_texts), len(ocr_text))
#                         logger.info("추출된 텍스트 샘플: %s", ocr_text[:200] if ocr_text else "텍스트 없음")
                
#                 # 텍스트 정화 - 개선된 함수 사용
#                 ocr_result.ocr_text = self.clean_text(ocr_text)
                
#                 # PDF 파일은 항상 텍스트 관련 있음으로 설정
#                 if file_type == 'pdf':
#                     text_relevant = True
                
#                 # 분석 유형 확인 (기본값: both)
#                 analysis_type = request.data.get('analysis_type', 'both')
#                 logger.info("분석 유형: %s", analysis_type)
                
#                 # 결과 변수 초기화
#                 image_analysis = ""
#                 text_analysis = ""
#                 combined_analysis = ""
                
#                 # 페이지 분할 분석 여부 확인
#                 analyze_by_page = request.data.get('analyze_by_page', 'true').lower() == 'true'
                
#                 # 선택된 분석 유형에 따라 처리
#                 if analysis_type in ['ollama', 'both']:
#                     # Ollama 클라이언트 초기화
#                     ollama_base_url = os.environ.get('OLLAMA_API_URL', 'http://localhost:11434')
#                     ollama_client = OllamaClient(base_url=ollama_base_url)
                    
#                     # 이미지 파일인 경우 텍스트 관련성 확인
#                     if file_type == 'image' and ocr_result.ocr_text.strip():
#                         # 텍스트 관련성 확인 프롬프트 실행
#                         try:
#                             relevance_result = ollama_client.check_text_relevance(ocr_result.file.path, ocr_result.ocr_text)
#                             text_relevant = relevance_result
#                             logger.info(f"텍스트 관련성 확인 결과: {text_relevant}")
                            
#                             # 관련성 정보 저장
#                             ocr_result.text_relevant = text_relevant
#                         except Exception as e:
#                             logger.error(f"텍스트 관련성 확인 중 오류: {str(e)}")
#                             # 오류 발생 시 기본값(false) 유지
                    
#                     # 이미지 분석 (이미지 파일인 경우)
#                     if file_type == 'image':
#                         # 관련성에 따른 프롬프트 분기 - 개선된 함수 호출
#                         custom_prompt = None  # 사용자 정의 프롬프트가 있으면 여기서 처리
                        
#                         image_analysis = ollama_client.analyze_image(
#                             ocr_result.file.path, 
#                             custom_prompt, 
#                             ocr_text=ocr_result.ocr_text, 
#                             text_relevant=text_relevant
#                         )
                        
#                         # OCR 텍스트 분석 (텍스트가 있고 both 모드인 경우 & 관련 있는 경우만)
#                         if ocr_result.ocr_text and analysis_type == 'both' and text_relevant:
#                             # 페이지별 텍스트 정리를 위한 프롬프트
#                             text_prompt = f"""다음 OCR로 추출한 텍스트를 자세히 분석하고 명확하게 정리해주세요:

# {ocr_result.ocr_text}

# 분석 지침:
# 1. 텍스트의 주요 내용과 구조를 파악하여 정리
# 2. 단순 요약이 아닌, 텍스트의 핵심 정보를 충실하게 정리
# 3. 중요한 세부 정보를 포함
# 4. 내용이 이미지와 관련이 있으므로 문맥을 고려하여 정리

# 반드시 한국어로 응답해주세요."""
                            
#                             try:
#                                 text_analysis = ollama_client.analyze_text(ocr_result.ocr_text, text_prompt)
#                             except Exception as e:
#                                 logger.error(f"텍스트 분석 오류: {str(e)}")
#                                 text_analysis = f"텍스트 분석 중 오류가 발생했습니다: {str(e)}"
                            
#                             # 두 분석 결과 결합
#                             combined_analysis = f"이미지 분석 결과:\n{image_analysis}\n\n텍스트 분석 결과:\n{text_analysis}"
#                         else:
#                             # OCR 없이 이미지 분석만 수행
#                             combined_analysis = image_analysis
                        
#                     else:  # PDF 파일인 경우
#                         if ocr_result.ocr_text:
#                             if analyze_by_page and len(page_texts) > 1:
#                                 # 개선된 페이지별 분석 수행 - OllamaClient의 분석 기능 활용
#                                 try:
#                                     combined_analysis = ollama_client.analyze_text(ocr_result.ocr_text, None, page_texts)
#                                     logger.info("페이지별 분석 완료")
#                                 except Exception as e:
#                                     logger.error(f"페이지별 분석 오류: {str(e)}")
#                                     combined_analysis = f"페이지별 분석 중 오류가 발생했습니다: {str(e)}"
#                             else:
#                                 # 문서 전체 분석 - 페이지별 구조화 요청
#                                 text_prompt = f"""다음 PDF에서 추출한 텍스트를 페이지별로 명확하게 정리해주세요:

# {ocr_result.ocr_text}

# 분석 지침:
# 1. 텍스트를 페이지나 섹션 단위로 구분하여 정리해주세요.
# 2. 내용을 요약하지 말고, 각 섹션의 핵심 정보를 충실하게 정리해주세요.
# 3. 모든 중요한 세부 정보를 포함해주세요.
# 4. 내용을 단순 요약하지 말고, 구조화된 형식으로 정리해주세요.

# 다음과 같은 형식으로 정리해주세요:
# ===== 페이지 1 (또는 섹션 1) =====
# - 주요 내용 정리
# - 중요 개념 설명
# - 핵심 정보 나열

# ===== 페이지 2 (또는 섹션 2) =====
# - 주요 내용 정리
# ...

# 반드시 한국어로 응답해주세요."""
                                
#                                 try:
#                                     text_analysis = ollama_client.analyze_text(ocr_result.ocr_text, text_prompt)
#                                     combined_analysis = text_analysis
#                                 except Exception as e:
#                                     logger.error(f"문서 전체 분석 오류: {str(e)}")
#                                     combined_analysis = f"문서 분석 중 오류가 발생했습니다: {str(e)}\n\nOCR 결과: {ocr_result.ocr_text[:500]}..."
                    
#                     logger.info("이미지/텍스트 분석 완료")
                
#                 # MySQL 저장을 위한 텍스트 정화
#                 ocr_result.llm_response = self.clean_text(combined_analysis)
                
#             except Exception as e:
#                 logger.error("처리 실패: %s", str(e), exc_info=True)
#                 return Response({'error': f'처리 실패: {str(e)}'}, 
#                                status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
#             # 텍스트 관련성 정보 저장
#             ocr_result.text_relevant = text_relevant
            
#             # 결과 저장
#             ocr_result.save()
#             logger.info("OCRResult 저장 완료 (ID: %s)", ocr_result.id)
            
#             # 응답 반환
#             serializer = OCRResultSerializer(ocr_result)
#             return Response(serializer.data, status=status.HTTP_201_CREATED)
                
#         except Exception as e:
#             logger.error("처리 중 예기치 않은 오류: %s", str(e), exc_info=True)
#             return Response({'error': f'서버 오류: {str(e)}'}, 
#                            status=status.HTTP_500_INTERNAL_SERVER_ERROR)
                           


# import logging
# import json
# import os
# from django.views.decorators.csrf import csrf_exempt
# from django.utils.decorators import method_decorator
# from rest_framework.views import APIView
# from rest_framework.response import Response
# from rest_framework import status
# from rest_framework.permissions import AllowAny
# from PIL import Image, ImageFilter, ImageEnhance
# import pytesseract
# from .models import OCRResult
# from .serializers import OCRResultSerializer

# import PyPDF2
# import tempfile
# from pdf2image import convert_from_path
# import re

# logger = logging.getLogger(__name__)

# # OllamaClient 클래스 가져오기
# from .ollama_client import OllamaClient

# @method_decorator(csrf_exempt, name='dispatch')
# class ProcessFileView(APIView):
#     permission_classes = [AllowAny]
    
#     def post(self, request):
#         try:
#             logger.info("ProcessFileView 요청 수신: %s %s", request.method, request.path)
            
#             # 요청 데이터 확인
#             if 'file' not in request.FILES:
#                 logger.error("파일이 제공되지 않음")
#                 return Response({'error': '파일이 제공되지 않았습니다'}, status=status.HTTP_400_BAD_REQUEST)
            
#             file_obj = request.FILES['file']
#             file_name = file_obj.name.lower()
#             logger.info("파일 업로드: %s, 크기: %s bytes", file_name, file_obj.size)
            
#             # 파일 유형 확인
#             if file_name.endswith(('.pdf')):
#                 file_type = 'pdf'
                
#                 # PDF 페이지 범위 확인
#                 start_page = int(request.data.get('start_page', 1))
#                 end_page = int(request.data.get('end_page', 0))  # 0은 전체 페이지를 의미
                
#                 logger.info("PDF 처리 범위: %s ~ %s 페이지", start_page, end_page if end_page > 0 else "끝")
                
#             elif file_name.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')):
#                 file_type = 'image'
#             else:
#                 logger.error("지원되지 않는 파일 형식: %s", file_name)
#                 return Response({'error': '지원되지 않는 파일 형식입니다. 이미지나 PDF 파일을 업로드해주세요.'},
#                               status=status.HTTP_400_BAD_REQUEST)
            
#             # OCR 결과 객체 생성
#             ocr_result = OCRResult.objects.create(file=file_obj, file_type=file_type)
#             logger.info("OCRResult 객체 생성: %s", ocr_result.id)
            
#             # Ollama 클라이언트 초기화 - 여기서 객체 생성
#             ollama_base_url = os.environ.get('OLLAMA_API_URL', 'http://localhost:11434')
#             ollama_client = OllamaClient(base_url=ollama_base_url)
            
#             # OCR 처리
#             try:
#                 ocr_text = ""
#                 page_texts = []  # 페이지별 텍스트 저장
#                 text_relevant = False  # 텍스트 관련성 여부 (기본값: 관련 없음)
                
#                 if file_type == 'image':
#                     # 이미지 파일 처리 - 개선된 OCR 적용
#                     img = Image.open(ocr_result.file.path)
#                     # 이미지 정보 로깅
#                     logger.info(f"이미지 정보: 크기={img.size}, 모드={img.mode}, 포맷={img.format}")
                    
#                     # 이미지 전처리 및 OCR 수행 - OllamaClient 메서드 사용
#                     preprocessed_img = ollama_client.preprocess_image_for_ocr(img)
#                     ocr_text = ollama_client.get_optimized_ocr_text(preprocessed_img, lang='kor+eng')
#                     page_texts.append({"page": 1, "text": ocr_text})
#                     logger.info("이미지 OCR 처리 완료, 추출 텍스트 길이: %s", len(ocr_text))
#                     logger.info("추출된 텍스트 샘플: %s", ocr_text[:200] if ocr_text else "텍스트 없음")
                
#                 elif file_type == 'pdf':
#                     # PDF 처리 - 직접 추출 후 필요시 OCR
#                     logger.info("PDF 처리 시작: %s", ocr_result.file.path)
                    
#                     # 직접 텍스트 추출 시도 (페이지별)
#                     direct_extract_success = False
#                     try:
#                         all_page_texts = self.extract_text_from_pdf_by_pages(ocr_result.file.path)
                        
#                         # 페이지 범위 처리
#                         if start_page > 1 or (end_page > 0 and end_page < len(all_page_texts)):
#                             if start_page <= len(all_page_texts):
#                                 if end_page > 0 and end_page >= start_page:
#                                     page_texts = all_page_texts[start_page-1:end_page]
#                                 else:
#                                     page_texts = all_page_texts[start_page-1:]
#                             else:
#                                 page_texts = []
#                         else:
#                             page_texts = all_page_texts
                        
#                         combined_text = "\n".join([page["text"] for page in page_texts])
                        
#                         # 추출된 텍스트 길이가 충분한지 확인 (더 엄격한 조건)
#                         if combined_text.strip() and len(combined_text.strip()) >= 50:
#                             meaningful_chars = sum(1 for c in combined_text if c.isalnum())
#                             if meaningful_chars > 30:  # 의미있는 글자가 30자 이상이면 성공으로 간주
#                                 ocr_text = combined_text
#                                 direct_extract_success = True
#                                 logger.info("PDF 직접 텍스트 추출 성공, 총 %s 페이지, 텍스트 길이: %s", 
#                                           len(page_texts), len(ocr_text))
#                                 logger.info("추출된 텍스트 샘플: %s", ocr_text[:200] if ocr_text else "텍스트 없음")
#                     except Exception as e:
#                         logger.error(f"PDF 직접 텍스트 추출 실패: {str(e)}")
                    
#                     # 직접 텍스트 추출이 실패한 경우, OCR 시도
#                     if not direct_extract_success:
#                         logger.info("PDF OCR 처리 시작 (직접 추출 실패 또는 텍스트 불충분)")
                        
#                         # 페이지 범위 설정으로 OCR - 이 메서드도 정의해야 함
#                         all_page_texts = self.ocr_pdf_by_pages(ocr_result.file.path, start_page, end_page)
                        
#                         # 페이지 범위 처리 - ocr_pdf_by_pages에서 처리했으므로 전체 사용
#                         page_texts = all_page_texts
                        
#                         ocr_text = "\n".join([page["text"] for page in page_texts])
#                         logger.info("PDF OCR 처리 완료, 총 %s 페이지, 텍스트 길이: %s", 
#                                   len(page_texts), len(ocr_text))
#                         logger.info("추출된 텍스트 샘플: %s", ocr_text[:200] if ocr_text else "텍스트 없음")
                
#                 # 텍스트 정화 - 개선된 함수 사용
#                 ocr_result.ocr_text = self.clean_text(ocr_text)
                
#                 # PDF 파일은 항상 텍스트 관련 있음으로 설정
#                 if file_type == 'pdf':
#                     text_relevant = True
                
#                 # 분석 유형 확인 (기본값: both)
#                 analysis_type = request.data.get('analysis_type', 'both')
#                 logger.info("분석 유형: %s", analysis_type)
                
#                 # 결과 변수 초기화
#                 image_analysis = ""
#                 text_analysis = ""
#                 combined_analysis = ""
                
#                 # 페이지 분할 분석 여부 확인
#                 analyze_by_page = request.data.get('analyze_by_page', 'true').lower() == 'true'
                
#                 # 선택된 분석 유형에 따라 처리
#                 if analysis_type in ['ollama', 'both']:
#                     # 이미지 파일인 경우 텍스트 관련성 확인
#                     if file_type == 'image' and ocr_result.ocr_text.strip():
#                         # 텍스트 관련성 확인 프롬프트 실행
#                         try:
#                             relevance_result = ollama_client.check_text_relevance(ocr_result.file.path, ocr_result.ocr_text)
#                             text_relevant = relevance_result
#                             logger.info(f"텍스트 관련성 확인 결과: {text_relevant}")
                            
#                             # 관련성 정보 저장
#                             ocr_result.text_relevant = text_relevant
#                         except Exception as e:
#                             logger.error(f"텍스트 관련성 확인 중 오류: {str(e)}")
#                             # 오류 발생 시 기본값(false) 유지
                    
#                     # 이미지 분석 (이미지 파일인 경우)
#                     if file_type == 'image':
#                         # 관련성에 따른 프롬프트 분기 - 개선된 함수 호출
#                         custom_prompt = None  # 사용자 정의 프롬프트가 있으면 여기서 처리
                        
#                         image_analysis = ollama_client.analyze_image(
#                             ocr_result.file.path, 
#                             custom_prompt
#                         )
                        
#                         # OCR 텍스트 분석 (텍스트가 있고 both 모드인 경우 & 관련 있는 경우만)
#                         if ocr_result.ocr_text and analysis_type == 'both' and text_relevant:
#                             # 페이지별 텍스트 정리를 위한 프롬프트
#                             text_prompt = f"""다음 OCR로 추출한 텍스트를 자세히 분석하고 명확하게 정리해주세요:

# {ocr_result.ocr_text}

# 분석 지침:
# 1. 텍스트의 주요 내용과 구조를 파악하여 정리
# 2. 단순 요약이 아닌, 텍스트의 핵심 정보를 충실하게 정리
# 3. 중요한 세부 정보를 포함
# 4. 내용이 이미지와 관련이 있으므로 문맥을 고려하여 정리

# 반드시 한국어로 응답해주세요."""
                            
#                             try:
#                                 text_analysis = ollama_client.analyze_text(ocr_result.ocr_text, text_prompt)
#                             except Exception as e:
#                                 logger.error(f"텍스트 분석 오류: {str(e)}")
#                                 text_analysis = f"텍스트 분석 중 오류가 발생했습니다: {str(e)}"
                            
#                             # 두 분석 결과 결합
#                             combined_analysis = f"이미지 분석 결과:\n{image_analysis}\n\n텍스트 분석 결과:\n{text_analysis}"
#                         else:
#                             # OCR 없이 이미지 분석만 수행
#                             combined_analysis = image_analysis
                        
#                     else:  # PDF 파일인 경우
#                         if ocr_result.ocr_text:
#                             if analyze_by_page and len(page_texts) > 1:
#                                 # 개선된 페이지별 분석 수행 - OllamaClient의 분석 기능 활용
#                                 try:
#                                     combined_analysis = ollama_client.analyze_text(ocr_result.ocr_text, None, page_texts)
#                                     logger.info("페이지별 분석 완료")
#                                 except Exception as e:
#                                     logger.error(f"페이지별 분석 오류: {str(e)}")
#                                     combined_analysis = f"페이지별 분석 중 오류가 발생했습니다: {str(e)}"
#                             else:
#                                 # 문서 전체 분석 - 페이지별 구조화 요청
#                                 text_prompt = f"""다음 PDF에서 추출한 텍스트를 페이지별로 명확하게 정리해주세요:

# {ocr_result.ocr_text}

# 분석 지침:
# 1. 텍스트를 페이지나 섹션 단위로 구분하여 정리해주세요.
# 2. 내용을 요약하지 말고, 각 섹션의 핵심 정보를 충실하게 정리해주세요.
# 3. 모든 중요한 세부 정보를 포함해주세요.
# 4. 내용을 단순 요약하지 말고, 구조화된 형식으로 정리해주세요.

# 다음과 같은 형식으로 정리해주세요:
# ===== 페이지 1 (또는 섹션 1) =====
# - 주요 내용 정리
# - 중요 개념 설명
# - 핵심 정보 나열

# ===== 페이지 2 (또는 섹션 2) =====
# - 주요 내용 정리
# ...

# 반드시 한국어로 응답해주세요."""
                                
#                                 try:
#                                     text_analysis = ollama_client.analyze_text(ocr_result.ocr_text, text_prompt)
#                                     combined_analysis = text_analysis
#                                 except Exception as e:
#                                     logger.error(f"문서 전체 분석 오류: {str(e)}")
#                                     combined_analysis = f"문서 분석 중 오류가 발생했습니다: {str(e)}\n\nOCR 결과: {ocr_result.ocr_text[:500]}..."
                    
#                     logger.info("이미지/텍스트 분석 완료")
                
#                 # MySQL 저장을 위한 텍스트 정화
#                 ocr_result.llm_response = self.clean_text(combined_analysis)
                
#             except Exception as e:
#                 logger.error("처리 실패: %s", str(e), exc_info=True)
#                 return Response({'error': f'처리 실패: {str(e)}'}, 
#                                status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
#             # 텍스트 관련성 정보 저장
#             ocr_result.text_relevant = text_relevant
            
#             # 결과 저장
#             ocr_result.save()
#             logger.info("OCRResult 저장 완료 (ID: %s)", ocr_result.id)
            
#             # 응답 반환
#             serializer = OCRResultSerializer(ocr_result)
#             return Response(serializer.data, status=status.HTTP_201_CREATED)
                
#         except Exception as e:
#             logger.error("처리 중 예기치 않은 오류: %s", str(e), exc_info=True)
#             return Response({'error': f'서버 오류: {str(e)}'}, 
#                            status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
#     # 필요한 PDF 처리 메서드 추가
#     def extract_text_from_pdf_by_pages(self, pdf_path):
#         """PDF에서 직접 텍스트를 페이지별로 추출"""
#         pages = []
#         try:
#             with open(pdf_path, 'rb') as file:
#                 reader = PyPDF2.PdfReader(file)
#                 total_pages = len(reader.pages)
                
#                 for i in range(total_pages):
#                     page = reader.pages[i]
#                     text = page.extract_text()
#                     pages.append({"page": i + 1, "text": text})
                    
#             return pages
#         except Exception as e:
#             logger.error(f"PDF 텍스트 직접 추출 오류: {str(e)}")
#             raise
    
#     def ocr_pdf_by_pages(self, pdf_path, start_page=1, end_page=0):
#         """PDF를 OCR로 처리하여 페이지별 텍스트 추출"""
#         pages = []
        
#         try:
#             # PDF2Image로 이미지 변환
#             with tempfile.TemporaryDirectory() as path:
#                 # 페이지 번호는 1부터 시작하지만, convert_from_path는 0부터 시작하므로 조정
#                 first_page = start_page
#                 last_page = None if end_page <= 0 else end_page
                
#                 images = convert_from_path(
#                     pdf_path, 
#                     dpi=300, 
#                     output_folder=path, 
#                     first_page=first_page,
#                     last_page=last_page
#                 )
                
#                 # Ollama 클라이언트 초기화
#                 ollama_base_url = os.environ.get('OLLAMA_API_URL', 'http://localhost:11434')
#                 ollama_client = OllamaClient(base_url=ollama_base_url)
                
#                 # 각 페이지 이미지 OCR 처리
#                 for i, image in enumerate(images):
#                     # 이미지 전처리
#                     preprocessed_img = ollama_client.preprocess_image_for_ocr(image)
#                     # OCR 수행
#                     text = ollama_client.get_optimized_ocr_text(preprocessed_img, lang='kor+eng')
                    
#                     # 페이지 번호 계산 (시작 페이지 고려)
#                     page_num = start_page + i
#                     pages.append({"page": page_num, "text": text})
                    
#             return pages
#         except Exception as e:
#             logger.error(f"PDF OCR 처리 오류: {str(e)}")
#             raise
    
#     def clean_text(self, text):
#         """텍스트 정화 함수"""
#         if not text:
#             return ""
            
#         # 연속된 공백 제거
#         text = re.sub(r'\s+', ' ', text)
#         # 연속된 줄바꿈 제거
#         text = re.sub(r'\n\s*\n', '\n\n', text)
#         # 앞뒤 공백 제거
#         text = text.strip()
        
#         return text

# import logging
# import json
# import os
# from django.views.decorators.csrf import csrf_exempt
# from django.utils.decorators import method_decorator
# from rest_framework.views import APIView
# from rest_framework.response import Response
# from rest_framework import status
# from rest_framework.permissions import AllowAny
# from PIL import Image, ImageFilter, ImageEnhance
# import pytesseract
# from .models import OCRResult
# from .serializers import OCRResultSerializer

# import PyPDF2
# import tempfile
# from pdf2image import convert_from_path
# import re

# logger = logging.getLogger(__name__)

# # OllamaClient 클래스 가져오기
# from .ollama_client import OllamaClient

# @method_decorator(csrf_exempt, name='dispatch')
# class ProcessFileView(APIView):
#     permission_classes = [AllowAny]
    
#     def post(self, request):
#         try:
#             logger.info("ProcessFileView 요청 수신: %s %s", request.method, request.path)
            
#             # 요청 데이터 확인
#             if 'file' not in request.FILES:
#                 logger.error("파일이 제공되지 않음")
#                 return Response({'error': '파일이 제공되지 않았습니다'}, status=status.HTTP_400_BAD_REQUEST)
            
#             file_obj = request.FILES['file']
#             file_name = file_obj.name.lower()
#             logger.info("파일 업로드: %s, 크기: %s bytes", file_name, file_obj.size)
            
#             # Ollama 클라이언트 초기화 - 여기서 객체 생성
#             ollama_base_url = os.environ.get('OLLAMA_API_URL', 'http://localhost:11434')
#             ollama_client = OllamaClient(base_url=ollama_base_url)
            
#             # 파일 유형 확인
#             if file_name.endswith(('.pdf')):
#                 file_type = 'pdf'
                
#                 # PDF 페이지 범위 확인
#                 start_page = int(request.data.get('start_page', 1))
#                 end_page = int(request.data.get('end_page', 0))  # 0은 전체 페이지를 의미
                
#                 logger.info("PDF 처리 범위: %s ~ %s 페이지", start_page, end_page if end_page > 0 else "끝")
                
#             elif file_name.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')):
#                 file_type = 'image'
#             else:
#                 logger.error("지원되지 않는 파일 형식: %s", file_name)
#                 return Response({'error': '지원되지 않는 파일 형식입니다. 이미지나 PDF 파일을 업로드해주세요.'},
#                               status=status.HTTP_400_BAD_REQUEST)
            
#             # OCR 결과 객체 생성
#             ocr_result = OCRResult.objects.create(file=file_obj, file_type=file_type)
#             logger.info("OCRResult 객체 생성: %s", ocr_result.id)
            
#             # OCR 처리
#             try:
#                 ocr_text = ""
#                 page_texts = []  # 페이지별 텍스트 저장
#                 text_relevant = False  # 텍스트 관련성 여부 (기본값: 관련 없음)
                
#                 if file_type == 'image':
#                     # 이미지 파일 처리 - 개선된 OCR 적용
#                     img = Image.open(ocr_result.file.path)
#                     # 이미지 정보 로깅
#                     logger.info(f"이미지 정보: 크기={img.size}, 모드={img.mode}, 포맷={img.format}")
                    
#                     # 이미지 전처리 및 OCR 수행 - OllamaClient 메서드 사용
#                     preprocessed_img = ollama_client.preprocess_image_for_ocr(img)
#                     ocr_text = ollama_client.get_optimized_ocr_text(preprocessed_img, lang='kor+eng')
#                     page_texts.append({"page": 1, "text": ocr_text})
#                     logger.info("이미지 OCR 처리 완료, 추출 텍스트 길이: %s", len(ocr_text))
#                     logger.info("추출된 텍스트 샘플: %s", ocr_text[:200] if ocr_text else "텍스트 없음")
                
#                 elif file_type == 'pdf':
#                     # PDF 처리 - 직접 추출 후 필요시 OCR
#                     logger.info("PDF 처리 시작: %s", ocr_result.file.path)
                    
#                     # 직접 텍스트 추출 시도 (페이지별)
#                     direct_extract_success = False
#                     try:
#                         all_page_texts = self.extract_text_from_pdf_by_pages(ocr_result.file.path)
                        
#                         # 페이지 범위 처리
#                         if start_page > 1 or (end_page > 0 and end_page < len(all_page_texts)):
#                             if start_page <= len(all_page_texts):
#                                 if end_page > 0 and end_page >= start_page:
#                                     page_texts = all_page_texts[start_page-1:end_page]
#                                 else:
#                                     page_texts = all_page_texts[start_page-1:]
#                             else:
#                                 page_texts = []
#                         else:
#                             page_texts = all_page_texts
                        
#                         combined_text = "\n".join([page["text"] for page in page_texts])
                        
#                         # 추출된 텍스트 길이가 충분한지 확인 (더 엄격한 조건)
#                         if combined_text.strip() and len(combined_text.strip()) >= 50:
#                             meaningful_chars = sum(1 for c in combined_text if c.isalnum())
#                             if meaningful_chars > 30:  # 의미있는 글자가 30자 이상이면 성공으로 간주
#                                 ocr_text = combined_text
#                                 direct_extract_success = True
#                                 logger.info("PDF 직접 텍스트 추출 성공, 총 %s 페이지, 텍스트 길이: %s", 
#                                           len(page_texts), len(ocr_text))
#                                 logger.info("추출된 텍스트 샘플: %s", ocr_text[:200] if ocr_text else "텍스트 없음")
#                     except Exception as e:
#                         logger.error(f"PDF 직접 텍스트 추출 실패: {str(e)}")
                    
#                     # 직접 텍스트 추출이 실패한 경우, OCR 시도
#                     if not direct_extract_success:
#                         logger.info("PDF OCR 처리 시작 (직접 추출 실패 또는 텍스트 불충분)")
                        
#                         # 페이지 범위 설정으로 OCR
#                         all_page_texts = self.ocr_pdf_by_pages(ocr_result.file.path, ollama_client, start_page, end_page)
                        
#                         # 페이지 범위 처리 - ocr_pdf_by_pages에서 처리했으므로 전체 사용
#                         page_texts = all_page_texts
                        
#                         ocr_text = "\n".join([page["text"] for page in page_texts])
#                         logger.info("PDF OCR 처리 완료, 총 %s 페이지, 텍스트 길이: %s", 
#                                   len(page_texts), len(ocr_text))
#                         logger.info("추출된 텍스트 샘플: %s", ocr_text[:200] if ocr_text else "텍스트 없음")
                
#                 # 텍스트 정화 - 개선된 함수 사용
#                 ocr_result.ocr_text = self.clean_text(ocr_text)
                
#                 # PDF 파일은 항상 텍스트 관련 있음으로 설정
#                 if file_type == 'pdf':
#                     text_relevant = True
                
#                 # 분석 유형 확인 (기본값: both)
#                 analysis_type = request.data.get('analysis_type', 'both')
#                 logger.info("분석 유형: %s", analysis_type)
                
#                 # 결과 변수 초기화
#                 image_analysis = ""
#                 text_analysis = ""
#                 combined_analysis = ""
                
#                 # 페이지 분할 분석 여부 확인
#                 analyze_by_page = request.data.get('analyze_by_page', 'true').lower() == 'true'
                
#                 # 선택된 분석 유형에 따라 처리
#                 if analysis_type in ['ollama', 'both']:
#                     # 이미지 파일인 경우
#                     if file_type == 'image':
#                         # OCR을 먼저 수행하고 이미지 분석에 OCR 결과를 포함
#                         custom_prompt = None  # 사용자 정의 프롬프트가 있으면 여기서 처리
                        
#                         # 텍스트 관련성 확인은 하지 않고 OCR 텍스트를 항상 전달
#                         # analyze_image 함수는 내부적으로 텍스트 관련성을 결정
#                         image_analysis = ollama_client.analyze_image(
#                             ocr_result.file.path, 
#                             custom_prompt,
#                             ocr_text=ocr_result.ocr_text
#                         )
                        
#                         # OCR 텍스트 분석 (텍스트가 있고 both 모드인 경우)
#                         if ocr_result.ocr_text and analysis_type == 'both':
#                             # 페이지별 텍스트 정리를 위한 프롬프트
#                             text_prompt = f"""다음 OCR로 추출한 텍스트를 자세히 분석하고 명확하게 정리해주세요:

# {ocr_result.ocr_text}

# 분석 지침:
# 1. 텍스트의 주요 내용과 구조를 파악하여 정리
# 2. 단순 요약이 아닌, 텍스트의 핵심 정보를 충실하게 정리
# 3. 중요한 세부 정보를 포함
# 4. 내용이 이미지와 관련이 있을 수 있으므로 문맥을 고려하여 정리

# 반드시 한국어로 응답해주세요."""
                            
#                             try:
#                                 text_analysis = ollama_client.analyze_text(ocr_result.ocr_text, text_prompt)
#                             except Exception as e:
#                                 logger.error(f"텍스트 분석 오류: {str(e)}")
#                                 text_analysis = f"텍스트 분석 중 오류가 발생했습니다: {str(e)}"
                            
#                             # 두 분석 결과 결합
#                             combined_analysis = f"이미지 분석 결과:\n{image_analysis}\n\n텍스트 분석 결과:\n{text_analysis}"
#                         else:
#                             # OCR 없이 이미지 분석만 수행
#                             combined_analysis = image_analysis
                        
#                     else:  # PDF 파일인 경우
#                         if ocr_result.ocr_text:
#                             if analyze_by_page and len(page_texts) > 1:
#                                 # 개선된 페이지별 분석 수행 - OllamaClient의 분석 기능 활용
#                                 try:
#                                     combined_analysis = ollama_client.analyze_text(ocr_result.ocr_text, None, page_texts)
#                                     logger.info("페이지별 분석 완료")
#                                 except Exception as e:
#                                     logger.error(f"페이지별 분석 오류: {str(e)}")
#                                     combined_analysis = f"페이지별 분석 중 오류가 발생했습니다: {str(e)}"
#                             else:
#                                 # 문서 전체 분석 - 페이지별 구조화 요청
#                                 text_prompt = f"""다음 PDF에서 추출한 텍스트를 페이지별로 명확하게 정리해주세요:

# {ocr_result.ocr_text}

# 분석 지침:
# 1. 텍스트를 페이지나 섹션 단위로 구분하여 정리해주세요.
# 2. 내용을 요약하지 말고, 각 섹션의 핵심 정보를 충실하게 정리해주세요.
# 3. 모든 중요한 세부 정보를 포함해주세요.
# 4. 내용을 단순 요약하지 말고, 구조화된 형식으로 정리해주세요.

# 다음과 같은 형식으로 정리해주세요:
# ===== 페이지 1 (또는 섹션 1) =====
# - 주요 내용 정리
# - 중요 개념 설명
# - 핵심 정보 나열

# ===== 페이지 2 (또는 섹션 2) =====
# - 주요 내용 정리
# ...

# 반드시 한국어로 응답해주세요."""
                                
#                                 try:
#                                     text_analysis = ollama_client.analyze_text(ocr_result.ocr_text, text_prompt)
#                                     combined_analysis = text_analysis
#                                 except Exception as e:
#                                     logger.error(f"문서 전체 분석 오류: {str(e)}")
#                                     combined_analysis = f"문서 분석 중 오류가 발생했습니다: {str(e)}\n\nOCR 결과: {ocr_result.ocr_text[:500]}..."
                    
#                     logger.info("이미지/텍스트 분석 완료")
                
#                 # MySQL 저장을 위한 텍스트 정화
#                 ocr_result.llm_response = self.clean_text(combined_analysis)
#                 ocr_result.text_relevant = text_relevant
                
#             except Exception as e:
#                 logger.error("처리 실패: %s", str(e), exc_info=True)
#                 return Response({'error': f'처리 실패: {str(e)}'}, 
#                                status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
#             # 결과 저장
#             ocr_result.save()
#             logger.info("OCRResult 저장 완료 (ID: %s)", ocr_result.id)
            
#             # 응답 반환
#             serializer = OCRResultSerializer(ocr_result)
#             return Response(serializer.data, status=status.HTTP_201_CREATED)
                
#         except Exception as e:
#             logger.error("처리 중 예기치 않은 오류: %s", str(e), exc_info=True)
#             return Response({'error': f'서버 오류: {str(e)}'}, 
#                            status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
#     def extract_text_from_pdf_by_pages(self, pdf_path):
#         """PDF에서 직접 텍스트를 페이지별로 추출"""
#         pages = []
#         try:
#             with open(pdf_path, 'rb') as file:
#                 reader = PyPDF2.PdfReader(file)
#                 total_pages = len(reader.pages)
                
#                 for i in range(total_pages):
#                     page = reader.pages[i]
#                     text = page.extract_text()
#                     pages.append({"page": i + 1, "text": text})
                    
#             return pages
#         except Exception as e:
#             logger.error(f"PDF 텍스트 직접 추출 오류: {str(e)}")
#             raise
    
#     def ocr_pdf_by_pages(self, pdf_path, ollama_client, start_page=1, end_page=0):
#         """PDF를 OCR로 처리하여 페이지별 텍스트 추출"""
#         pages = []
        
#         try:
#             # PDF2Image로 이미지 변환
#             with tempfile.TemporaryDirectory() as path:
#                 # 페이지 번호는 1부터 시작하지만, convert_from_path는 0부터 시작하므로 조정
#                 first_page = start_page
#                 last_page = None if end_page <= 0 else end_page
                
#                 images = convert_from_path(
#                     pdf_path, 
#                     dpi=300, 
#                     output_folder=path, 
#                     first_page=first_page,
#                     last_page=last_page
#                 )
                
#                 # 각 페이지 이미지 OCR 처리
#                 for i, image in enumerate(images):
#                     # 이미지 전처리
#                     preprocessed_img = ollama_client.preprocess_image_for_ocr(image)
#                     # OCR 수행
#                     text = ollama_client.get_optimized_ocr_text(preprocessed_img, lang='kor+eng')
                    
#                     # 페이지 번호 계산 (시작 페이지 고려)
#                     page_num = start_page + i
#                     pages.append({"page": page_num, "text": text})
                    
#             return pages
#         except Exception as e:
#             logger.error(f"PDF OCR 처리 오류: {str(e)}")
#             raise
    
#     def clean_text(self, text):
#         """텍스트 정화 함수"""
#         if not text:
#             return ""
            
#         # 연속된 공백 제거
#         text = re.sub(r'\s+', ' ', text)
#         # 연속된 줄바꿈 제거
#         text = re.sub(r'\n\s*\n', '\n\n', text)
#         # 앞뒤 공백 제거
#         text = text.strip()
        
#         return text

# import logging
# import json
# import os
# from django.views.decorators.csrf import csrf_exempt
# from django.utils.decorators import method_decorator
# from rest_framework.views import APIView
# from rest_framework.response import Response
# from rest_framework import status
# from rest_framework.permissions import AllowAny
# from PIL import Image, ImageFilter, ImageEnhance
# import pytesseract
# from .models import OCRResult
# from .serializers import OCRResultSerializer

# import PyPDF2
# import tempfile
# from pdf2image import convert_from_path
# import re

# logger = logging.getLogger(__name__)

# # OllamaClient 클래스 가져오기
# from .ollama_client import OllamaClient

# @method_decorator(csrf_exempt, name='dispatch')
# class ProcessFileView(APIView):
#     permission_classes = [AllowAny]
    
#     def post(self, request):
#         try:
#             logger.info("ProcessFileView 요청 수신: %s %s", request.method, request.path)
            
#             # 요청 데이터 확인
#             if 'file' not in request.FILES:
#                 logger.error("파일이 제공되지 않음")
#                 return Response({'error': '파일이 제공되지 않았습니다'}, status=status.HTTP_400_BAD_REQUEST)
            
#             file_obj = request.FILES['file']
#             file_name = file_obj.name.lower()
#             logger.info("파일 업로드: %s, 크기: %s bytes", file_name, file_obj.size)
            
#             # Ollama 클라이언트 초기화 - 여기서 객체 생성
#             ollama_base_url = os.environ.get('OLLAMA_API_URL', 'http://localhost:11434')
#             ollama_client = OllamaClient(base_url=ollama_base_url)
            
#             # 파일 유형 확인
#             if file_name.endswith(('.pdf')):
#                 file_type = 'pdf'
                
#                 # PDF 페이지 범위 확인
#                 start_page = int(request.data.get('start_page', 1))
#                 end_page = int(request.data.get('end_page', 0))  # 0은 전체 페이지를 의미
                
#                 logger.info("PDF 처리 범위: %s ~ %s 페이지", start_page, end_page if end_page > 0 else "끝")
                
#             elif file_name.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')):
#                 file_type = 'image'
#             else:
#                 logger.error("지원되지 않는 파일 형식: %s", file_name)
#                 return Response({'error': '지원되지 않는 파일 형식입니다. 이미지나 PDF 파일을 업로드해주세요.'},
#                               status=status.HTTP_400_BAD_REQUEST)
            
#             # OCR 결과 객체 생성
#             ocr_result = OCRResult.objects.create(file=file_obj, file_type=file_type)
#             logger.info("OCRResult 객체 생성: %s", ocr_result.id)
            
#             # OCR 처리
#             try:
#                 ocr_text = ""
#                 page_texts = []  # 페이지별 텍스트 저장
                
#                 if file_type == 'image':
#                     # 이미지 파일 처리 - 개선된 OCR 적용
#                     img = Image.open(ocr_result.file.path)
#                     # 이미지 정보 로깅
#                     logger.info(f"이미지 정보: 크기={img.size}, 모드={img.mode}, 포맷={img.format}")
                    
#                     # 이미지 전처리 및 OCR 수행 - OllamaClient 메서드 사용
#                     preprocessed_img = ollama_client.preprocess_image_for_ocr(img)
#                     ocr_text = ollama_client.get_optimized_ocr_text(preprocessed_img, lang='kor+eng')
#                     page_texts.append({"page": 1, "text": ocr_text})
#                     logger.info("이미지 OCR 처리 완료, 추출 텍스트 길이: %s", len(ocr_text))
#                     logger.info("추출된 텍스트 샘플: %s", ocr_text[:200] if ocr_text else "텍스트 없음")
                
#                 elif file_type == 'pdf':
#                     # PDF 처리 - 직접 추출 후 필요시 OCR
#                     logger.info("PDF 처리 시작: %s", ocr_result.file.path)
                    
#                     # 직접 텍스트 추출 시도 (페이지별)
#                     direct_extract_success = False
#                     try:
#                         all_page_texts = self.extract_text_from_pdf_by_pages(ocr_result.file.path)
                        
#                         # 페이지 범위 처리
#                         if start_page > 1 or (end_page > 0 and end_page < len(all_page_texts)):
#                             if start_page <= len(all_page_texts):
#                                 if end_page > 0 and end_page >= start_page:
#                                     page_texts = all_page_texts[start_page-1:end_page]
#                                 else:
#                                     page_texts = all_page_texts[start_page-1:]
#                             else:
#                                 page_texts = []
#                         else:
#                             page_texts = all_page_texts
                        
#                         combined_text = "\n".join([page["text"] for page in page_texts])
                        
#                         # 추출된 텍스트 길이가 충분한지 확인 (더 엄격한 조건)
#                         if combined_text.strip() and len(combined_text.strip()) >= 50:
#                             meaningful_chars = sum(1 for c in combined_text if c.isalnum())
#                             if meaningful_chars > 30:  # 의미있는 글자가 30자 이상이면 성공으로 간주
#                                 ocr_text = combined_text
#                                 direct_extract_success = True
#                                 logger.info("PDF 직접 텍스트 추출 성공, 총 %s 페이지, 텍스트 길이: %s", 
#                                           len(page_texts), len(ocr_text))
#                                 logger.info("추출된 텍스트 샘플: %s", ocr_text[:200] if ocr_text else "텍스트 없음")
#                     except Exception as e:
#                         logger.error(f"PDF 직접 텍스트 추출 실패: {str(e)}")
                    
#                     # 직접 텍스트 추출이 실패한 경우, OCR 시도
#                     if not direct_extract_success:
#                         logger.info("PDF OCR 처리 시작 (직접 추출 실패 또는 텍스트 불충분)")
                        
#                         # 페이지 범위 설정으로 OCR
#                         all_page_texts = self.ocr_pdf_by_pages(ocr_result.file.path, ollama_client, start_page, end_page)
                        
#                         # 페이지 범위 처리 - ocr_pdf_by_pages에서 처리했으므로 전체 사용
#                         page_texts = all_page_texts
                        
#                         ocr_text = "\n".join([page["text"] for page in page_texts])
#                         logger.info("PDF OCR 처리 완료, 총 %s 페이지, 텍스트 길이: %s", 
#                                   len(page_texts), len(ocr_text))
#                         logger.info("추출된 텍스트 샘플: %s", ocr_text[:200] if ocr_text else "텍스트 없음")
                
#                 # 텍스트 정화 - 개선된 함수 사용
#                 ocr_result.ocr_text = self.clean_text(ocr_text)
                
#                 # PDF 파일은 항상 텍스트 관련 있음으로 설정
#                 if file_type == 'pdf':
#                     text_relevant = True
                
#                 # 분석 유형 확인 (기본값: both)
#                 analysis_type = request.data.get('analysis_type', 'both')
#                 logger.info("분석 유형: %s", analysis_type)
                
#                 # 결과 변수 초기화
#                 image_analysis = ""
#                 text_analysis = ""
#                 combined_analysis = ""
                
#                 # 페이지 분할 분석 여부 확인
#                 analyze_by_page = request.data.get('analyze_by_page', 'true').lower() == 'true'
                
#                 # 선택된 분석 유형에 따라 처리
#                 if analysis_type in ['ollama', 'both']:
#                     # 이미지 파일인 경우
#                     if file_type == 'image':
#                         # 사용자 정의 프롬프트 설정 (요약된 간결한 설명을 위해)
#                         custom_prompt = f"""이미지를 객관적으로 관찰하고 다음 지침에 따라 응답하세요:

# 필수 포함 사항:
# - 이미지에 실제로 보이는 사람, 동물, 물체만 언급 (없으면 언급하지 않음)
# - 만약 동물이라면, 어떤 종의 동물인지도 출력
# - 확실히 보이는 색상만 언급 (배경색, 옷 색상 등)
# - 명확히 보이는 자세나 위치 관계 (정면, 측면 등)

# 절대 포함하지 말 것:
# - 추측이나 해석 ("~로 보입니다", "~같습니다" 표현 금지)
# - 보이지 않는 부분에 대한 언급 ("보이지 않는다", "없다" 등의 표현 금지)
# - 반복적인 설명
# - 감정이나 분위기 묘사

# 형식:
# - 1-2문장으로 매우 간결하게 작성
# - 단순 사실 나열 형식 (예: "이미지에는 검은 머리 여성이 있고, 배경은 흰색이다.")

# OCR 텍스트 (참고용, 실제 이미지에 보이는 경우만 언급): {ocr_result.ocr_text}

# 영어로 간결하게 응답해주세요."""

                        
#                         # OCR 텍스트를 전달 (analyze_image 내부에서 관련성 판단)
#                         image_analysis = ollama_client.analyze_image(
#                             ocr_result.file.path, 
#                             custom_prompt,
#                             ocr_text=ocr_result.ocr_text
#                         )
                        
#                         # OCR 텍스트 분석 (텍스트가 있고 both 모드인 경우)
#                         if ocr_result.ocr_text and analysis_type == 'both':
#                             # 페이지별 텍스트 정리를 위한 프롬프트
#                             text_prompt = f"""다음 OCR로 추출한 텍스트를 자세히 분석하고 명확하게 정리해주세요:

# {ocr_result.ocr_text}

# 분석 지침:
# 1. 텍스트의 주요 내용과 구조를 파악하여 정리
# 2. 단순 요약이 아닌, 텍스트의 핵심 정보를 충실하게 정리
# 3. 중요한 세부 정보를 포함
# 4. 내용이 이미지와 관련이 있을 수 있으므로 문맥을 고려하여 정리

# 반드시 "영어(En)"로 응답해주세요."""
                            
#                             try:
#                                 text_analysis = ollama_client.analyze_text(ocr_result.ocr_text, text_prompt)
#                             except Exception as e:
#                                 logger.error(f"텍스트 분석 오류: {str(e)}")
#                                 text_analysis = f"텍스트 분석 중 오류가 발생했습니다: {str(e)}"
                            
#                             # 두 분석 결과 결합
#                             combined_analysis = f"이미지 분석 결과:\n{image_analysis}\n\n텍스트 분석 결과:\n{text_analysis}"
#                         else:
#                             # OCR 없이 이미지 분석만 수행
#                             combined_analysis = image_analysis
                        
#                     else:  # PDF 파일인 경우
#                         if ocr_result.ocr_text:
#                             if analyze_by_page and len(page_texts) > 1:
#                                 # 개선된 페이지별 분석 수행 - OllamaClient의 분석 기능 활용
#                                 try:
#                                     combined_analysis = ollama_client.analyze_text(ocr_result.ocr_text, None, page_texts)
#                                     logger.info("페이지별 분석 완료")
#                                 except Exception as e:
#                                     logger.error(f"페이지별 분석 오류: {str(e)}")
#                                     combined_analysis = f"페이지별 분석 중 오류가 발생했습니다: {str(e)}"
#                             else:
#                                 # 문서 전체 분석 - 페이지별 구조화 요청
#                                 text_prompt = f"""다음 PDF에서 추출한 텍스트를 페이지별로 명확하게 정리해주세요:

# {ocr_result.ocr_text}

# 분석 지침:
# 1. 텍스트를 페이지나 섹션 단위로 구분하여 정리해주세요.
# 2. 내용을 요약하지 말고, 각 섹션의 핵심 정보를 충실하게 정리해주세요.
# 3. 모든 중요한 세부 정보를 포함해주세요.
# 4. 내용을 단순 요약하지 말고, 구조화된 형식으로 정리해주세요.

# 다음과 같은 형식으로 정리해주세요:
# ===== 페이지 1 (또는 섹션 1) =====
# - 주요 내용 정리
# - 중요 개념 설명
# - 핵심 정보 나열

# ===== 페이지 2 (또는 섹션 2) =====
# - 주요 내용 정리
# ...

# 반드시 "영어(En)"로 응답해주세요."""
                                
#                                 try:
#                                     text_analysis = ollama_client.analyze_text(ocr_result.ocr_text, text_prompt)
#                                     combined_analysis = text_analysis
#                                 except Exception as e:
#                                     logger.error(f"문서 전체 분석 오류: {str(e)}")
#                                     combined_analysis = f"문서 분석 중 오류가 발생했습니다: {str(e)}\n\nOCR 결과: {ocr_result.ocr_text[:500]}..."
                    
#                     logger.info("이미지/텍스트 분석 완료")
                
#                 # MySQL 저장을 위한 텍스트 정화
#                 ocr_result.llm_response = self.clean_text(combined_analysis)
                
#                 # 텍스트 관련성 정보 저장 - PDF는 항상 True, 이미지는 분석 과정에서 결정
#                 if file_type == 'pdf':
#                     ocr_result.text_relevant = True
                
#             except Exception as e:
#                 logger.error("처리 실패: %s", str(e), exc_info=True)
#                 return Response({'error': f'처리 실패: {str(e)}'}, 
#                                status=status.HTTP_500_INTERNAL_SERVER_ERROR)
            
#             # 결과 저장
#             ocr_result.save()
#             logger.info("OCRResult 저장 완료 (ID: %s)", ocr_result.id)
            
#             # 응답 반환
#             serializer = OCRResultSerializer(ocr_result)
#             return Response(serializer.data, status=status.HTTP_201_CREATED)
                
#         except Exception as e:
#             logger.error("처리 중 예기치 않은 오류: %s", str(e), exc_info=True)
#             return Response({'error': f'서버 오류: {str(e)}'}, 
#                            status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
#     def extract_text_from_pdf_by_pages(self, pdf_path):
#         """PDF에서 직접 텍스트를 페이지별로 추출"""
#         pages = []
#         try:
#             with open(pdf_path, 'rb') as file:
#                 reader = PyPDF2.PdfReader(file)
#                 total_pages = len(reader.pages)
                
#                 for i in range(total_pages):
#                     page = reader.pages[i]
#                     text = page.extract_text()
#                     pages.append({"page": i + 1, "text": text})
                    
#             return pages
#         except Exception as e:
#             logger.error(f"PDF 텍스트 직접 추출 오류: {str(e)}")
#             raise
    
#     def ocr_pdf_by_pages(self, pdf_path, ollama_client, start_page=1, end_page=0):
#         """PDF를 OCR로 처리하여 페이지별 텍스트 추출"""
#         pages = []
        
#         try:
#             # PDF2Image로 이미지 변환
#             with tempfile.TemporaryDirectory() as path:
#                 # 페이지 번호는 1부터 시작하지만, convert_from_path는 0부터 시작하므로 조정
#                 first_page = start_page
#                 last_page = None if end_page <= 0 else end_page
                
#                 images = convert_from_path(
#                     pdf_path, 
#                     dpi=300, 
#                     output_folder=path, 
#                     first_page=first_page,
#                     last_page=last_page
#                 )
                
#                 # 각 페이지 이미지 OCR 처리
#                 for i, image in enumerate(images):
#                     # 이미지 전처리
#                     preprocessed_img = ollama_client.preprocess_image_for_ocr(image)
#                     # OCR 수행
#                     text = ollama_client.get_optimized_ocr_text(preprocessed_img, lang='kor+eng')
                    
#                     # 페이지 번호 계산 (시작 페이지 고려)
#                     page_num = start_page + i
#                     pages.append({"page": page_num, "text": text})
                    
#             return pages
#         except Exception as e:
#             logger.error(f"PDF OCR 처리 오류: {str(e)}")
#             raise
    
#     def clean_text(self, text):
#         """텍스트 정화 함수"""
#         if not text:
#             return ""
            
#         # 연속된 공백 제거
#         text = re.sub(r'\s+', ' ', text)
#         # 연속된 줄바꿈 제거
#         text = re.sub(r'\n\s*\n', '\n\n', text)
#         # 앞뒤 공백 제거
#         text = text.strip()
        
#         return text
    
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