# langchain_config.py - 새로 생성할 파일
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain, SequentialChain
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
import json
import logging
import anthropic
from groq import Groq

logger = logging.getLogger(__name__)

# Pydantic 모델 정의
class AIAnalysis(BaseModel):
    장점: str = Field(description="AI 모델의 장점")
    단점: str = Field(description="AI 모델의 단점")

class OptimalResponse(BaseModel):
    preferredModel: str = Field(description="선호되는 모델")
    best_response: str = Field(description="최적의 응답")
    analysis: Dict[str, AIAnalysis] = Field(description="각 AI 모델 분석")
    reasoning: str = Field(description="선택 이유")

# Custom Output Parser
class JSONOutputParser(BaseOutputParser):
    """JSON 형태의 응답을 파싱하는 커스텀 파서"""
    
    def parse(self, text: str) -> Dict[str, Any]:
        try:
            # 기존 sanitize_and_parse_json 로직 사용
            return self.sanitize_and_parse_json(text)
        except Exception as e:
            logger.error(f"JSON 파싱 실패: {e}")
            return {
                "preferredModel": "FALLBACK",
                "best_response": "파싱 오류가 발생했습니다.",
                "analysis": {},
                "reasoning": "응답 파싱 중 오류가 발생했습니다."
            }
    
    def sanitize_and_parse_json(self, text):
        # 기존 sanitize_and_parse_json 함수 내용 그대로 사용
        import re
        try:
            text = text.strip()
            if text.startswith('```json') and '```' in text:
                text = re.sub(r'```json(.*?)```', r'\1', text, flags=re.DOTALL).strip()
            elif text.startswith('```') and text.endswith('```'):
                text = text[3:-3].strip()
                
            json_pattern = r'({[\s\S]*})'
            json_matches = re.findall(json_pattern, text)
            if json_matches:
                text = json_matches[0]
                
            text = re.sub(r'\\([_"])', r'\1', text)
            result = json.loads(text)
            
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
            raise

# LangChain 설정 클래스
class LangChainManager:
    def __init__(self, openai_key, anthropic_key, groq_key):
        self.openai_key = openai_key
        self.anthropic_key = anthropic_key
        self.groq_key = groq_key
        
        # LLM 인스턴스 생성
        self.llms = {
            'gpt': ChatOpenAI(
                openai_api_key=openai_key,
                model_name='gpt-3.5-turbo',
                temperature=0.7,
                max_tokens=1024
            ),
            'claude': AnthropicLLM(
                api_key=anthropic_key,
                model='claude-3-5-haiku-20241022'
            ),
            'mixtral': GroqLLM(
                api_key=groq_key,
                model='llama-3.1-8b-instant'
            )
        }
        
        # 출력 파서
        self.output_parser = JSONOutputParser()
        
        # 프롬프트 템플릿 (모델별로 다르게 처리)
        self.chat_template_openai = ChatPromptTemplate.from_messages([
            ("system", "사용자가 선택한 언어는 '{user_language}'입니다. 반드시 이 언어({user_language})로 응답하세요."),
            ("user", "{user_input}")
        ])
        
        self.chat_template_others = PromptTemplate(
            input_variables=["user_input", "user_language"],
            template="사용자가 선택한 언어는 '{user_language}'입니다. 반드시 이 언어({user_language})로 응답하세요.\n\n{user_input}"
        )
        
        self.analysis_template = PromptTemplate(
            input_variables=["query", "responses", "user_language", "selected_models"],
            template="""다음은 동일한 질문에 대한 {num_models}가지 AI의 응답을 분석하는 것입니다.
            사용자가 선택한 언어는 '{user_language}'입니다.
            반드시 이 언어({user_language})로 최적의 답을 작성해주세요.

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
                "preferredModel": "ANALYZER",
                "best_response": "최적의 답변 ({user_language}로 작성)",
                "analysis": {{
                    {analysis_section}
                }},
                "reasoning": "반드시 이 언어({user_language})로 작성 최적의 응답을 선택한 이유"
            }}"""
        )
    
    def create_chat_chain(self, model_name: str):
        """개별 AI 모델과의 채팅 체인 생성"""
        if model_name not in self.llms:
            raise ValueError(f"지원하지 않는 모델: {model_name}")
        
        llm = self.llms[model_name]
        
        # OpenAI는 ChatPromptTemplate 사용, 나머지는 PromptTemplate 사용
        if model_name == 'gpt':
            template = self.chat_template_openai
        else:
            template = self.chat_template_others
        
        return LLMChain(
            llm=llm,
            prompt=template,
            verbose=True
        )
    
    def create_analysis_chain(self, analyzer_model='gpt'):
        """응답 분석 체인 생성"""
        analyzer_llm = self.llms.get(analyzer_model, self.llms['gpt'])
        
        return LLMChain(
            llm=analyzer_llm,
            prompt=self.analysis_template,
            output_parser=self.output_parser,
            verbose=True
        )
    
    def format_responses_for_analysis(self, responses: Dict[str, str], selected_models: List[str]) -> Dict[str, str]:
        """분석을 위한 응답 포맷팅"""
        responses_section = ""
        analysis_section = ""
        
        for model in selected_models:
            model_lower = model.lower()
            responses_section += f"\n{model.upper()} 응답: {responses.get(model_lower, '응답 없음')}"
            
            analysis_section += f"""
                "{model_lower}": {{
                    "장점": "{model.upper()} 답변의 장점",
                    "단점": "{model.upper()} 답변의 단점"
                }}{"," if model_lower != selected_models[-1].lower() else ""}"""
        
        return {
            "responses_section": responses_section,
            "analysis_section": analysis_section,
            "num_models": len(selected_models)
        }

# Anthropic용 커스텀 LLM 클래스
class AnthropicLLM(LLM):
    client: Optional[Any] = None
    model: str = "claude-3-5-haiku-20241022"
    max_tokens: int = 4096
    temperature: float = 0
    
    # Pydantic 모델 설정
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, api_key: str, model: str = "claude-3-5-haiku-20241022", **kwargs):
        # client를 kwargs에 추가하여 Pydantic 검증 통과
        kwargs['client'] = anthropic.Anthropic(api_key=api_key)
        super().__init__(model=model, **kwargs)
    
    @property
    def _llm_type(self) -> str:
        return "anthropic"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            return message.content[0].text
        except Exception as e:
            logger.error(f"Anthropic API 에러: {e}")
            raise

# Groq용 커스텀 LLM 클래스 
class GroqLLM(LLM):
    client: Optional[Any] = None
    model: str = 'llama-3.1-8b-instant'
    temperature: float = 0.7
    max_tokens: int = 1024
    
    # Pydantic 모델 설정
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, api_key: str, model: str = 'llama-3.1-8b-instant', **kwargs):
        # client를 kwargs에 추가하여 Pydantic 검증 통과
        kwargs['client'] = Groq(api_key=api_key)
        super().__init__(model=model, **kwargs)
    
    @property
    def _llm_type(self) -> str:
        return "groq"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Groq API 에러: {e}")
            raise