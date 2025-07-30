import time
from django.conf import settings
from groq import Groq
import openai
import anthropic

class LLMClient:
    """Enhanced LLM 클라이언트 - Django 버전"""
    
    def __init__(self):
        # Initialize clients
        self.groq_client = Groq(api_key=settings.GROQ_API_KEY)
        
        # OpenAI client
        try:
            self.openai_client = openai.OpenAI(api_key=settings.OPENAI_API_KEY)
            self.openai_available = True
        except Exception as e:
            print(f"⚠️ OpenAI 초기화 실패: {e}")
            self.openai_client = None
            self.openai_available = False
        
        # Anthropic client
        try:
            self.anthropic_client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
            self.anthropic_available = True
        except Exception as e:
            print(f"⚠️ Anthropic 초기화 실패: {e}")
            self.anthropic_client = None
            self.anthropic_available = False
    
    def call_groq_llm_enhanced(self, prompt, system_prompt="", model="llama3-70b-8192", max_retries=3):
        """Groq LLM 호출 with fallback"""
        
        # Groq 시도
        for attempt in range(max_retries):
            try:
                response = self.groq_client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1024,
                    top_p=0.9
                )
                content = response.choices[0].message.content.strip()
                # print(f"✅ Groq API 성공 (시도 {attempt + 1})")
                return content
                
            except Exception as e:
                print(f"⚠️ Groq API 실패 (시도 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)
                    continue
                else:
                    break
        
        # OpenAI fallback
        if self.openai_available and self.openai_client:
            try:
                response = self.openai_client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1024
                )
                content = response.choices[0].message.content.strip()
                print("✅ OpenAI API fallback 성공")
                return content
                
            except Exception as e:
                print(f"❌ OpenAI API도 실패: {e}")
        
        # Anthropic fallback
        if self.anthropic_available and self.anthropic_client:
            try:
                response = self.anthropic_client.messages.create(
                    model="claude-3-haiku-20240307",
                    max_tokens=1024,
                    temperature=0.7,
                    system=system_prompt,
                    messages=[{"role": "user", "content": prompt}]
                )
                content = response.content[0].text.strip()
                print("✅ Anthropic API fallback 성공")
                return content
                
            except Exception as e:
                print(f"❌ Anthropic API도 실패: {e}")
        
        # 모든 API 실패시 기본 응답
        return self._generate_fallback_response(prompt)
    
    def _generate_fallback_response(self, prompt):
        """기본 응답 생성"""
        if "검색" in prompt or "find" in prompt.lower():
            return "API 호출에 실패했지만, 검색 기능은 정상적으로 동작하고 있습니다. 다른 검색어를 시도해보세요."
        elif "하이라이트" in prompt or "highlight" in prompt.lower():
            return "하이라이트 감지 기능이 실행되었지만, AI 분석 서비스에 일시적인 문제가 있습니다."
        elif "요약" in prompt or "summary" in prompt.lower():
            return "비디오 요약 기능이 요청되었습니다. AI 서비스 복구 후 더 자세한 분석을 제공하겠습니다."
        else:
            return "죄송합니다. AI 서비스에 일시적인 문제가 있어 상세한 분석을 제공할 수 없습니다. 잠시 후 다시 시도해주세요."
    
    def call_multi_llm_enhanced(self, prompt, system_prompt=""):
        """다중 LLM 호출"""
        results = []
        
        # Groq 시도
        try:
            res_groq = self.call_groq_llm_enhanced(
                prompt, 
                system_prompt + "\n반드시 한국어로 답변하세요.", 
                model="llama3-70b-8192"
            )
            if "API 호출에 실패" not in res_groq:
                results.append("🤖 Groq (Llama3-70B): " + res_groq)
        except Exception as e:
            results.append(f"⚠️ Groq 오류: {e}")

        # OpenAI 시도
        if self.openai_available and self.openai_client:
            try:
                res_gpt = self.openai_client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": system_prompt + "\n반드시 한국어로 답변하세요."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1024
                )
                results.append("🧠 GPT-4o: " + res_gpt.choices[0].message.content.strip())
            except Exception as e:
                results.append(f"⚠️ GPT 오류: {e}")

        # Anthropic 시도
        if self.anthropic_available and self.anthropic_client:
            try:
                res_claude = self.anthropic_client.messages.create(
                    model="claude-3-opus-20240229",
                    max_tokens=1024,
                    temperature=0.7,
                    system=system_prompt + "\n반드시 한국어로 답변하세요.",
                    messages=[{"role": "user", "content": prompt}]
                )
                results.append("🎭 Claude Opus: " + res_claude.content[0].text.strip())
            except Exception as e:
                results.append(f"⚠️ Claude 오류: {e}")

        # 결과 통합
        if results:
            return "\n\n".join(results)
        else:
            return "모든 AI 서비스에 문제가 있습니다. 잠시 후 다시 시도해주세요."
    
    def generate_smart_response(self, user_query, search_results=None, video_info=None, use_multi_llm=False):
        """스마트한 LLM 응답 생성"""
        system_prompt = """당신은 비디오 분석 전문가입니다. 
        사용자의 질문에 대해 정확하고 친절하게 한국어로 답변하세요.
        검색 결과나 비디오 정보가 제공되면 이를 바탕으로 구체적이고 상세한 답변을 해주세요."""
        
        # 사용자 프롬프트 구성
        user_prompt = f"사용자 질문: {user_query}\n"
        
        if search_results:
            user_prompt += f"\n🔍 검색 결과 분석:\n"
            for i, result in enumerate(search_results[:3], 1):
                user_prompt += f"{i}. 프레임 {result.get('frame_id', 'N/A')} ({result.get('timestamp', 0):.1f}초)\n"
                user_prompt += f"   - 매칭 점수: {result.get('match_score', 0)}\n"
                user_prompt += f"   - 감지된 객체: {', '.join(result.get('detected_objects', []))}\n"
                
                # Enhanced 캡션 정보 활용
                if 'caption' in result:
                    user_prompt += f"   - 설명: {result['caption'][:100]}...\n"
        
        if video_info:
            user_prompt += f"📊 비디오 정보:\n{video_info}\n\n"
        
        user_prompt += """
        위 정보를 바탕으로 사용자의 질문에 대해 다음과 같이 답변해주세요:
        1. 검색 결과가 있다면 구체적인 시간대와 내용을 언급
        2. 발견된 객체나 장면에 대한 상세한 설명
        3. 필요하다면 추가 검색 방법이나 관련 기능 안내
        4. 친근하고 도움이 되는 톤으로 작성
        
        답변은 반드시 한국어로 해주세요.
        """
        
        # 멀티 LLM 또는 단일 LLM 선택
        if use_multi_llm:
            return self.call_multi_llm_enhanced(user_prompt, system_prompt)
        else:
            return self.call_groq_llm_enhanced(user_prompt, system_prompt)
    
    def get_api_status(self):
        """API 상태 확인"""
        status = {
            'groq': {'available': True, 'status': 'ok'},
            'openai': {'available': self.openai_available, 'status': 'ok' if self.openai_available else 'unavailable'},
            'anthropic': {'available': self.anthropic_available, 'status': 'ok' if self.anthropic_available else 'unavailable'}
        }
        
        # Groq API 테스트
        try:
            test_response = self.call_groq_llm_enhanced("테스트", "간단히 '정상'이라고 답하세요.", max_retries=1)
            if "정상" not in test_response and "모든 AI 서비스" in test_response:
                status['groq']['status'] = 'error'
        except:
            status['groq']['status'] = 'error'
        
        return status