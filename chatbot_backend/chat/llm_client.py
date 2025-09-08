# llm_client.py

import os
from typing import Optional

class LLMClient:
    """LLM API 클라이언트 (OpenAI GPT 또는 다른 LLM 서비스)"""
    
    def __init__(self):
        self.api_key = os.getenv('OPENAI_API_KEY')
        self.model = os.getenv('LLM_MODEL', 'gpt-3.5-turbo')
        self.enabled = bool(self.api_key)
        self.client = None
        
        if self.enabled:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=self.api_key)
                print(f"OpenAI 클라이언트 초기화 완료 (모델: {self.model})")
            except ImportError:
                print("OpenAI 라이브러리가 설치되지 않았습니다: pip install openai")
                self.enabled = False
            except Exception as e:
                print(f"OpenAI 클라이언트 초기화 실패: {e}")
                self.enabled = False
        else:
            print("OPENAI_API_KEY가 설정되지 않았습니다.")
    
    def generate_response(self, prompt: str, max_tokens: int = 500) -> Optional[str]:
        """프롬프트를 기반으로 LLM 응답 생성 (OpenAI 1.0+ 호환)"""
        if not self.enabled or not self.client:
            return None
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system", 
                        "content": "당신은 영상 분석 전문가입니다. 주어진 캡션 정보를 바탕으로 자연스럽고 읽기 쉬운 한국어 영상 설명을 작성해주세요."
                    },
                    {
                        "role": "user", 
                        "content": prompt
                    }
                ],
                max_tokens=max_tokens,
                temperature=0.7,
                top_p=0.9
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"OpenAI API 호출 실패: {e}")
            return None
    
    def is_available(self) -> bool:
        """LLM 서비스 사용 가능 여부"""
        return self.enabled and self.client is not None


# 대안: 로컬 LLM이나 다른 서비스를 위한 클라이언트
class LocalLLMClient:
    """로컬 LLM 또는 다른 API 서비스용 클라이언트"""
    
    def __init__(self):
        self.base_url = os.getenv('LOCAL_LLM_URL', 'http://localhost:11434')
        self.model = os.getenv('LOCAL_LLM_MODEL', 'llama2')
        self.enabled = False
        
        # 연결 테스트
        try:
            import requests
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            self.enabled = response.status_code == 200
            if self.enabled:
                print(f"로컬 LLM 연결 성공: {self.base_url}")
        except Exception as e:
            print(f"로컬 LLM 연결 실패: {e}")
    
    def generate_response(self, prompt: str, max_tokens: int = 500) -> Optional[str]:
        """로컬 LLM을 사용한 응답 생성"""
        if not self.enabled:
            return None
        
        try:
            import requests
            
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.7
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '').strip()
            
        except Exception as e:
            print(f"로컬 LLM API 호출 실패: {e}")
            
        return None
    
    def is_available(self) -> bool:
        return self.enabled


class MockLLMClient:
    """개발/테스트용 Mock LLM 클라이언트 - 더 자연스러운 설명 생성"""
    
    def __init__(self):
        self.enabled = True
        print("Mock LLM 클라이언트 초기화 완료 (개발 모드)")
    
    def generate_response(self, prompt: str, max_tokens: int = 500) -> str:
        """프롬프트 분석 기반 자연스러운 응답 생성"""
        
        prompt_lower = prompt.lower()
        
        # 영상 기본 정보 추출
        duration = "몇 초"
        if "길이:" in prompt:
            try:
                duration_match = prompt.split("길이:")[1].split("초")[0].strip()
                duration = f"{duration_match}초"
            except:
                pass
        
        # 장소 정보 추출
        location = ""
        location_keywords = {
            '쇼핑몰': '쇼핑몰',
            '실내': '실내 공간',
            '거리': '거리',
            '실외': '야외',
            '건물': '건물'
        }
        
        for keyword, description in location_keywords.items():
            if keyword in prompt_lower:
                location = description
                break
        
        # 시간대 정보 추출
        time_info = ""
        if '오후' in prompt_lower:
            time_info = "오후 시간대"
        elif '아침' in prompt_lower:
            time_info = "아침 시간대"
        elif '밤' in prompt_lower:
            time_info = "밤 시간대"
        elif '밝은' in prompt_lower:
            time_info = "밝은 환경"
        
        # 활동 정보 추출
        activities = []
        if '걷' in prompt_lower:
            activities.append('걷기')
        if '쇼핑' in prompt_lower:
            activities.append('쇼핑')
        if '서' in prompt_lower:
            activities.append('서있기')
        if '대화' in prompt_lower:
            activities.append('대화')
        
        # 사람 수 추출
        people_count = ""
        import re
        numbers = re.findall(r'(\d+)명', prompt)
        if numbers:
            people_count = f"{numbers[0]}명의 사람"
        elif '사람' in prompt_lower:
            people_count = "여러 사람"
        
        # 자연스러운 설명 구성
        description = f"이 영상은 {duration} 길이로, "
        
        if location and time_info:
            description += f"{location}에서 {time_info}에 촬영된 "
        elif location:
            description += f"{location}에서 촬영된 "
        elif time_info:
            description += f"{time_info}에 촬영된 "
        
        description += "일상적인 장면을 담고 있습니다.\n\n"
        
        # 주요 내용 설명
        if people_count:
            description += f"영상에는 {people_count}이 등장하여 "
            if activities:
                if len(activities) == 1:
                    description += f"{activities[0]}를 하고 있습니다. "
                else:
                    description += f"{', '.join(activities[:-1])} 및 {activities[-1]} 등의 활동을 보여줍니다. "
            else:
                description += "자연스러운 움직임을 보여줍니다. "
        
        # 전체적인 인상
        description += "\n"
        if location == '쇼핑몰':
            description += "쇼핑몰의 평범한 일상 풍경이 잘 담겨 있으며, "
        
        description += "자연스럽고 편안한 분위기의 영상입니다. "
        
        if activities:
            description += "사람들의 일상적인 행동과 움직임이 생동감 있게 기록되어 있어, "
            description += "실제 현장의 모습을 잘 보여주는 영상이라고 할 수 있습니다."
        
        return description
    
    def is_available(self) -> bool:
        return True


# 환경에 따라 적절한 클라이언트 선택
def get_llm_client():
    """환경에 맞는 LLM 클라이언트 반환"""
    
    # 1순위: OpenAI API
    if os.getenv('OPENAI_API_KEY'):
        client = LLMClient()
        if client.is_available():
            return client
    
    # 2순위: 로컬 LLM (Ollama 등)
    local_client = LocalLLMClient()
    if local_client.is_available():
        return local_client
    
    # 3순위: Mock 클라이언트 (개발/테스트용)
    return MockLLMClient()