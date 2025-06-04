# # # gpt_translator.py
# # import openai
# # import logging
# # import os
# # from typing import Dict, Any, Optional

# # logger = logging.getLogger(__name__)

# # class GPTTranslator:
# #     def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo"):
# #         """
# #         GPT 번역 클라이언트 초기화
        
# #         Args:
# #             api_key: OpenAI API 키
# #             model: 사용할 GPT 모델 (기본값: gpt-3.5-turbo)
# #         """
# #         # API 키 직접 설정
# #         self.api_key = api_key or "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"
# #         self.model = model
        
# #         if not self.api_key:
# #             logger.error("OpenAI API 키가 설정되지 않았습니다")
# #             self.is_available = False
# #             return
        
# #         try:
# #             # 구버전 OpenAI 패키지 설정 방식
# #             openai.api_key = self.api_key
            
# #             # API 키 유효성 테스트 (구버전 방식)
# #             test_response = openai.Model.list()
# #             self.is_available = True
# #             logger.info(f"GPT 번역 클라이언트 초기화 완료 - 모델: {self.model}")
# #             logger.info(f"API 키 상태: 정상 (키 시작: {self.api_key[:15]}...)")
            
# #         except Exception as e:
# #             logger.error(f"OpenAI 클라이언트 초기화 실패: {str(e)}")
# #             self.is_available = False
    
# #     def translate_to_korean(self, text: str, source_language: str = "English") -> Dict[str, Any]:
# #         """
# #         텍스트를 한국어로 번역
        
# #         Args:
# #             text: 번역할 텍스트
# #             source_language: 소스 언어 (기본값: English)
            
# #         Returns:
# #             dict: 번역 결과와 메타데이터
# #         """
# #         if not self.is_available:
# #             return {
# #                 "success": False,
# #                 "error": "OpenAI API 키가 설정되지 않았거나 클라이언트 초기화에 실패했습니다",
# #                 "original_text": text,
# #                 "translated_text": None
# #             }
        
# #         if not text or text.strip() == "":
# #             return {
# #                 "success": False,
# #                 "error": "번역할 텍스트가 없습니다",
# #                 "original_text": text,
# #                 "translated_text": None
# #             }
        
# #         try:
# #             # 번역 프롬프트 구성
# #             system_prompt = """당신은 전문 번역가입니다. 다음 지침을 따라 번역해주세요:

# # 1. 자연스럽고 정확한 한국어로 번역
# # 2. 원문의 의미와 뉘앙스를 정확히 전달
# # 3. 전문 용어는 적절한 한국어 용어로 번역
# # 4. 문맥과 톤을 유지
# # 5. 불필요한 설명이나 주석 없이 번역문만 제공"""

# #             user_prompt = f"""다음 {source_language} 텍스트를 한국어로 번역해주세요:

# # {text}"""

# #             # 구버전 OpenAI API 호출 방식
# #             response = openai.ChatCompletion.create(
# #                 model=self.model,
# #                 messages=[
# #                     {"role": "system", "content": system_prompt},
# #                     {"role": "user", "content": user_prompt}
# #                 ],
# #                 temperature=0.3,  # 일관성 있는 번역을 위해 낮은 temperature
# #                 max_tokens=4000   # 충분한 토큰 수
# #             )
            
# #             translated_text = response.choices[0].message.content.strip()
            
# #             logger.info(f"GPT 번역 완료 - 원문 길이: {len(text)}, 번역문 길이: {len(translated_text)}")
            
# #             return {
# #                 "success": True,
# #                 "original_text": text,
# #                 "translated_text": translated_text,
# #                 "source_language": source_language,
# #                 "target_language": "Korean",
# #                 "model_used": self.model,
# #                 "token_usage": {
# #                     "prompt_tokens": response.usage.prompt_tokens,
# #                     "completion_tokens": response.usage.completion_tokens,
# #                     "total_tokens": response.usage.total_tokens
# #                 }
# #             }
            
# #         except Exception as e:
# #             logger.error(f"GPT 번역 오류: {str(e)}")
# #             return {
# #                 "success": False,
# #                 "error": f"번역 중 오류가 발생했습니다: {str(e)}",
# #                 "original_text": text,
# #                 "translated_text": None
# #             }
    
# #     def translate_analysis_result(self, analysis_result: str, analysis_type: str = "image") -> Dict[str, Any]:
# #         """
# #         Ollama 분석 결과를 한국어로 번역 (특화된 번역)
        
# #         Args:
# #             analysis_result: Ollama 분석 결과
# #             analysis_type: 분석 유형 ("image", "text", "pdf")
            
# #         Returns:
# #             dict: 번역 결과
# #         """
# #         if not self.is_available:
# #             return {
# #                 "success": False,
# #                 "error": "OpenAI API 키가 설정되지 않았거나 클라이언트 초기화에 실패했습니다",
# #                 "original_analysis": analysis_result,
# #                 "translated_analysis": None
# #             }
        
# #         if not analysis_result or analysis_result.strip() == "":
# #             return {
# #                 "success": False,
# #                 "error": "번역할 분석 결과가 없습니다",
# #                 "original_analysis": analysis_result,
# #                 "translated_analysis": None
# #             }
        
# #         try:
# #             # 분석 유형에 따른 맞춤 프롬프트
# #             context_prompts = {
# #                 "image": "이미지 분석 결과",
# #                 "text": "텍스트 분석 결과", 
# #                 "pdf": "PDF 문서 분석 결과"
# #             }
            
# #             context = context_prompts.get(analysis_type, "AI 분석 결과")
            
# #             system_prompt = f"""당신은 AI 분석 결과를 한국어로 번역하는 전문가입니다. 다음 지침을 따라주세요:

# # 1. {context}를 자연스러운 한국어로 번역
# # 2. 기술적 용어는 적절한 한국어 표현으로 번역
# # 3. 분석의 구조와 형식을 유지
# # 4. 페이지 구분이나 섹션 제목도 한국어로 번역
# # 5. 원문의 분석 깊이와 세부사항을 그대로 유지"""

# #             user_prompt = f"""다음 {context}를 한국어로 번역해주세요:

# # {analysis_result}"""

# #             # 구버전 OpenAI API 호출
# #             response = openai.ChatCompletion.create(
# #                 model=self.model,
# #                 messages=[
# #                     {"role": "system", "content": system_prompt},
# #                     {"role": "user", "content": user_prompt}
# #                 ],
# #                 temperature=0.2,  # 정확한 번역을 위해 더 낮은 temperature
# #                 max_tokens=4000
# #             )
            
# #             translated_analysis = response.choices[0].message.content.strip()
            
# #             logger.info(f"분석 결과 번역 완료 - 유형: {analysis_type}")
            
# #             return {
# #                 "success": True,
# #                 "original_analysis": analysis_result,
# #                 "translated_analysis": translated_analysis,
# #                 "analysis_type": analysis_type,
# #                 "model_used": self.model,
# #                 "token_usage": {
# #                     "prompt_tokens": response.usage.prompt_tokens,
# #                     "completion_tokens": response.usage.completion_tokens,
# #                     "total_tokens": response.usage.total_tokens
# #                 }
# #             }
            
# #         except Exception as e:
# #             logger.error(f"분석 결과 번역 오류: {str(e)}")
# #             return {
# #                 "success": False,
# #                 "error": f"분석 결과 번역 중 오류가 발생했습니다: {str(e)}",
# #                 "original_analysis": analysis_result,
# #                 "translated_analysis": None
# #             }
    
# #     def translate_paged_analysis(self, paged_analysis: str) -> Dict[str, Any]:
# #         """
# #         페이지별 분석 결과를 한국어로 번역 (페이지 구조 유지)
        
# #         Args:
# #             paged_analysis: 페이지별 분석 결과
            
# #         Returns:
# #             dict: 번역 결과
# #         """
# #         if not self.is_available:
# #             return {
# #                 "success": False,
# #                 "error": "OpenAI API 키가 설정되지 않았거나 클라이언트 초기화에 실패했습니다",
# #                 "original_analysis": paged_analysis,
# #                 "translated_analysis": None
# #             }
        
# #         if not paged_analysis or paged_analysis.strip() == "":
# #             return {
# #                 "success": False,
# #                 "error": "번역할 페이지별 분석 결과가 없습니다",
# #                 "original_analysis": paged_analysis,
# #                 "translated_analysis": None
# #             }
        
# #         try:
# #             system_prompt = """당신은 페이지별 문서 분석 결과를 번역하는 전문가입니다. 다음 지침을 따라주세요:

# # 1. 페이지 구분 형식을 정확히 유지 (===== 페이지 X ===== 형식)
# # 2. 각 페이지의 분석 내용을 자연스러운 한국어로 번역
# # 3. 분석의 구조와 계층을 그대로 유지
# # 4. 전문 용어는 적절한 한국어로 번역
# # 5. "Page X"는 "페이지 X"로 번역
# # 6. 원문의 분석 깊이와 세부사항을 모두 유지"""

# #             user_prompt = f"""다음 페이지별 분석 결과를 한국어로 번역해주세요. 페이지 구분 형식과 구조를 정확히 유지해주세요:

# # {paged_analysis}"""

# #             # 구버전 OpenAI API 호출
# #             response = openai.ChatCompletion.create(
# #                 model=self.model,
# #                 messages=[
# #                     {"role": "system", "content": system_prompt},
# #                     {"role": "user", "content": user_prompt}
# #                 ],
# #                 temperature=0.2,
# #                 max_tokens=4000
# #             )
            
# #             translated_analysis = response.choices[0].message.content.strip()
            
# #             logger.info("페이지별 분석 결과 번역 완료")
            
# #             return {
# #                 "success": True,
# #                 "original_analysis": paged_analysis,
# #                 "translated_analysis": translated_analysis,
# #                 "analysis_type": "paged_analysis",
# #                 "model_used": self.model,
# #                 "token_usage": {
# #                     "prompt_tokens": response.usage.prompt_tokens,
# #                     "completion_tokens": response.usage.completion_tokens,
# #                     "total_tokens": response.usage.total_tokens
# #                 }
# #             }
            
# #         except Exception as e:
# #             logger.error(f"페이지별 분석 번역 오류: {str(e)}")
# #             return {
# #                 "success": False,
# #                 "error": f"페이지별 분석 번역 중 오류가 발생했습니다: {str(e)}",
# #                 "original_analysis": paged_analysis,
# #                 "translated_analysis": None
# #             }

# # gpt_translator.py
# import openai
# import logging
# import os
# from typing import Dict, Any, Optional

# logger = logging.getLogger(__name__)

# class GPTTranslator:
#     def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo"):
#         """
#         GPT 번역 클라이언트 초기화
        
#         Args:
#             api_key: OpenAI API 키
#             model: 사용할 GPT 모델 (기본값: gpt-3.5-turbo)
#         """
#         # API 키 직접 설정
#         self.api_key = api_key or "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"
#         self.model = model
        
#         if not self.api_key:
#             logger.error("OpenAI API 키가 설정되지 않았습니다")
#             self.is_available = False
#             return
        
#         try:
#             # 구버전 OpenAI 패키지 설정 방식
#             openai.api_key = self.api_key
            
#             # API 키 유효성 테스트 (구버전 방식)
#             test_response = openai.Model.list()
#             self.is_available = True
#             logger.info(f"GPT 번역 클라이언트 초기화 완료 - 모델: {self.model}")
#             logger.info(f"API 키 상태: 정상 (키 시작: {self.api_key[:15]}...)")
            
#         except Exception as e:
#             logger.error(f"OpenAI 클라이언트 초기화 실패: {str(e)}")
#             self.is_available = False
    
#     def translate_to_korean(self, text: str, source_language: str = "English") -> Dict[str, Any]:
#         """
#         텍스트를 한국어로 번역
        
#         Args:
#             text: 번역할 텍스트
#             source_language: 소스 언어 (기본값: English)
            
#         Returns:
#             dict: 번역 결과와 메타데이터
#         """
#         if not self.is_available:
#             return {
#                 "success": False,
#                 "error": "OpenAI API 키가 설정되지 않았거나 클라이언트 초기화에 실패했습니다",
#                 "original_text": text,
#                 "translated_text": None
#             }
        
#         if not text or text.strip() == "":
#             return {
#                 "success": False,
#                 "error": "번역할 텍스트가 없습니다",
#                 "original_text": text,
#                 "translated_text": None
#             }
        
#         try:
#             # 번역 프롬프트 구성
#             system_prompt = """당신은 전문 번역가입니다. 다음 지침을 따라 번역해주세요:

# 1. 자연스럽고 정확한 한국어로 번역
# 2. 원문의 의미와 뉘앙스를 정확히 전달
# 3. 전문 용어는 적절한 한국어 용어로 번역
# 4. 문맥과 톤을 유지
# 5. 불필요한 설명이나 주석 없이 번역문만 제공"""

#             user_prompt = f"""다음 {source_language} 텍스트를 한국어로 번역해주세요:

# {text}"""

#             # 구버전 OpenAI API 호출 방식
#             response = openai.ChatCompletion.create(
#                 model=self.model,
#                 messages=[
#                     {"role": "system", "content": system_prompt},
#                     {"role": "user", "content": user_prompt}
#                 ],
#                 temperature=0.3,  # 일관성 있는 번역을 위해 낮은 temperature
#                 max_tokens=4000   # 충분한 토큰 수
#             )
            
#             translated_text = response.choices[0].message.content.strip()
            
#             logger.info(f"GPT 번역 완료 - 원문 길이: {len(text)}, 번역문 길이: {len(translated_text)}")
            
#             return {
#                 "success": True,
#                 "original_text": text,
#                 "translated_text": translated_text,
#                 "source_language": source_language,
#                 "target_language": "Korean",
#                 "model_used": self.model,
#                 "token_usage": {
#                     "prompt_tokens": response.usage.prompt_tokens,
#                     "completion_tokens": response.usage.completion_tokens,
#                     "total_tokens": response.usage.total_tokens
#                 }
#             }
            
#         except Exception as e:
#             logger.error(f"GPT 번역 오류: {str(e)}")
#             return {
#                 "success": False,
#                 "error": f"번역 중 오류가 발생했습니다: {str(e)}",
#                 "original_text": text,
#                 "translated_text": None
#             }
    
#     def translate_analysis_result(self, analysis_result: str, analysis_type: str = "image") -> Dict[str, Any]:
#         """
#         Ollama 분석 결과를 한국어로 번역 (특화된 번역)
        
#         Args:
#             analysis_result: Ollama 분석 결과
#             analysis_type: 분석 유형 ("image", "text", "pdf")
            
#         Returns:
#             dict: 번역 결과
#         """
#         if not self.is_available:
#             return {
#                 "success": False,
#                 "error": "OpenAI API 키가 설정되지 않았거나 클라이언트 초기화에 실패했습니다",
#                 "original_analysis": analysis_result,
#                 "translated_analysis": None
#             }
        
#         if not analysis_result or analysis_result.strip() == "":
#             return {
#                 "success": False,
#                 "error": "번역할 분석 결과가 없습니다",
#                 "original_analysis": analysis_result,
#                 "translated_analysis": None
#             }
        
#         try:
#             # 분석 유형에 따른 맞춤 프롬프트
#             context_prompts = {
#                 "image": "이미지 분석 결과",
#                 "text": "텍스트 분석 결과", 
#                 "pdf": "PDF 문서 분석 결과"
#             }
            
#             context = context_prompts.get(analysis_type, "AI 분석 결과")
            
#             system_prompt = f"""당신은 AI 분석 결과를 한국어로 번역하고 정리하는 전문가입니다. 다음 지침을 따라주세요:

# 📋 번역 및 정리 지침:
# 1. {context}를 자연스럽고 이해하기 쉬운 한국어로 번역
# 2. 복잡한 기술 용어는 일반인도 이해할 수 있도록 쉽게 설명
# 3. 문장을 명확하고 읽기 쉽게 재구성
# 4. 중요한 정보는 강조하여 표현
# 5. 원문의 구조는 유지하되, 더 논리적으로 정리
# 6. 불필요한 반복이나 중복은 제거하고 핵심 내용만 간결하게 정리

# ❌ 절대 포함하지 말 것:
# - 문서의 완성도나 품질에 대한 판단 ("불완전하다", "단편적이다" 등)
# - 페이지 구성이나 구조에 대한 비평
# - 내용의 충분성에 대한 언급
# - 문서 자체에 대한 메타적 설명이나 평가

# ✨ 가독성 향상 요구사항:
# - 문장은 짧고 명확하게 작성
# - 전문 용어 뒤에 괄호로 쉬운 설명 추가 (예: "Classification(분류)" → "분류(Classification)")
# - 중요한 개념은 굵게 표시하거나 구분하여 작성
# - 리스트나 단계가 있으면 번호나 불릿으로 정리
# - 오직 실제 내용만 번역하고 정리"""

#             user_prompt = f"""다음 {context}를 한국어로 번역하고 이해하기 쉽게 정리해주세요:

# {analysis_result}

# 위 내용을 한국어로 번역하면서 다음 사항을 반영해주세요:
# - 일반인도 이해할 수 있도록 쉽게 설명
# - 문장을 명확하고 읽기 쉽게 재구성
# - 중요한 정보는 강조하여 표현
# - 논리적인 순서로 내용 정리
# - 문서 품질이나 완성도에 대한 언급은 절대 하지 마세요
# - 오직 실제 내용만 번역하고 정리하세요"""

#             # 구버전 OpenAI API 호출
#             response = openai.ChatCompletion.create(
#                 model=self.model,
#                 messages=[
#                     {"role": "system", "content": system_prompt},
#                     {"role": "user", "content": user_prompt}
#                 ],
#                 temperature=0.2,  # 정확한 번역을 위해 더 낮은 temperature
#                 max_tokens=4000
#             )
            
#             translated_analysis = response.choices[0].message.content.strip()
            
#             logger.info(f"분석 결과 번역 및 정리 완료 - 유형: {analysis_type}")
            
#             return {
#                 "success": True,
#                 "original_analysis": analysis_result,
#                 "translated_analysis": translated_analysis,
#                 "analysis_type": analysis_type,
#                 "model_used": self.model,
#                 "token_usage": {
#                     "prompt_tokens": response.usage.prompt_tokens,
#                     "completion_tokens": response.usage.completion_tokens,
#                     "total_tokens": response.usage.total_tokens
#                 }
#             }
            
#         except Exception as e:
#             logger.error(f"분석 결과 번역 오류: {str(e)}")
#             return {
#                 "success": False,
#                 "error": f"분석 결과 번역 중 오류가 발생했습니다: {str(e)}",
#                 "original_analysis": analysis_result,
#                 "translated_analysis": None
#             }
    
#     def translate_paged_analysis(self, paged_analysis: str, doc_type: str = "ppt") -> Dict[str, Any]:
#         """
#         페이지별 분석 결과를 한국어로 번역 (페이지 구조 유지)
        
#         Args:
#             paged_analysis: 페이지별 분석 결과
            
#         Returns:
#             dict: 번역 결과
#         """
        
#         if not self.is_available:
#             return {
#                 "success": False,
#                 "error": "OpenAI API 키가 설정되지 않았거나 클라이언트 초기화에 실패했습니다",
#                 "original_analysis": paged_analysis,
#                 "translated_analysis": None
#             }
        
#         if not paged_analysis or paged_analysis.strip() == "":
#             return {
#                 "success": False,
#                 "error": "번역할 페이지별 분석 결과가 없습니다",
#                 "original_analysis": paged_analysis,
#                 "translated_analysis": None
#             }
        
#         try:
            
# #             system_prompt = """당신은 페이지별 문서 분석 결과를 번역하고 정리하는 전문가입니다. 다음 지침을 따라주세요:

# # 📚 페이지별 번역 및 정리 지침:
# # 1. 페이지 구분 형식을 정확히 유지 (===== 페이지 X ===== 형식)
# # 2. 각 페이지의 내용을 이해하기 쉬운 한국어로 번역 및 정리
# # 3. 복잡한 기술 용어는 일반인도 이해할 수 있도록 쉽게 설명
# # 4. 전문 용어는 적절한 한국어로 번역하되, 필요시 영어 원문을 괄호로 병기
# # 5. "Page X"는 "페이지 X"로 번역
# # 6. 각 페이지의 핵심 내용을 명확하게 WJDFL

# # ❌ 절대 포함하지 말 것:
# # - 문서의 완성도나 품질에 대한 판단 ("불완전하다", "단편적이다" 등)
# # - 페이지 구성이나 구조에 대한 비평
# # - 내용의 충분성에 대한 언급 ("소개가 불완전하다" 등)
# # - 문서 자체에 대한 메타적 설명이나 평가
# # - 페이지의 완성도나 연결성에 대한 언급

# # ✨ 가독성 향상 요구사항:
# # - 각 페이지마다 핵심 내용을 먼저 간단히 요약
# # - 중요한 개념이나 용어는 굵게 표시하거나 강조
# # - 긴 문장은 짧고 명확한 문장으로 분리
# # - 리스트나 단계는 번호나 불릿으로 정리
# # - 페이지별 주요 키워드나 개념을 명확히 제시
# # - 오직 실제 내용만 번역하고 정리

# # 📖 구조화 요구사항:
# # - 각 페이지의 주제를 명확히 제시
# # - 주요 내용을 논리적 순서로 정리
# # - 페이지 간 연관성이 있으면 언급 (단, 품질 평가는 제외)"""

# #             user_prompt = f"""다음 페이지별 분석 결과를 한국어로 번역하고 이해하기 쉽게 정리해주세요. 페이지 구분 형식과 구조를 정확히 유지하면서, 각 페이지의 내용을 명확하고 읽기 쉽게 정리해주세요:

# # {paged_analysis}
#             system_prompt = """당신은 PPT 슬라이드 분석 내용을 한국어로 번역하고 정리하는 전문가입니다.
# 📚 페이지별 번역 및 정리 지침:
# 1. 페이지 구분 형식을 정확히 유지 (===== 페이지 X ===== 형식)
# 2. 각 페이지의 내용을 이해하기 쉬운 한국어로 번역 및 정리
# 3. 복잡한 기술 용어는 일반인도 이해할 수 있도록 쉽게 설명
# 4. 전문 용어는 적절한 한국어로 번역하되, 필요시 영어 원문을 괄호로 병기
# 5. "Page X"는 "페이지 X"로 번역

# 📝 작업 지침:
# 1. 각 슬라이드는 "📄 페이지 X" 형식으로 시작
# 2. 영어 원문의 번역을 "🔍 주요 내용:"에 작성
# 3. 슬라이드의 내용을 자연스럽고 논리적으로 풀어 "📋 상세 설명:"에 정리
# 4. 불완전하거나 단편적인 키워드라도 가능한 명확한 의미로 구성
# 5. 원문의 형식(표, 리스트 등)은 텍스트 기반 요약으로 풀어서 작성
# 6. 시각적 요소나 레이아웃 언급은 하지 않음

# ❌ 금지 사항:
# - "정보 부족", "불완전" 등의 판단
# - 슬라이드의 디자인, 완성도, 분량 등 언급
# - 문서 품질 평가나 메타 해석

# """



#             user_prompt = f"""다음은 PPT 문서의 슬라이드별 텍스트 추출 결과입니다. 각 슬라이드를 페이지로 간주하여 아래 형식에 맞춰 번역하고 정리해주세요.

# 입력 내용:
# {paged_analysis}


# 각 페이지별로 다음 형식을 따라 정리해주세요:
# ===== 페이지 X =====
# 📌 예시 형식:
# 📄 페이지 1  
# 🔍 주요 내용: 영어를 한국어로 번역해서 제공해주세요.
# 📋 **상세 설명**:  
# - **컴퓨터 그래픽스**는 디지털 환경에서 시각적 콘텐츠를 생성하는 기술입니다.  
# - 이 기술은 영화, 게임, 시뮬레이션 등 다양한 분야에 활용됩니다.



# 주의사항:
# 1. 텍스트를 페이지나 섹션 단위로 구분하여 정리해주세요.
# 2. 내용을 요약하지 말고, 각 섹션의 핵심 정보를 충실하게 정리해주세요.
# 3. 모든 중요한 세부 정보를 포함해주세요.
# 4. 내용을 단순 요약하지 말고, 구조화된 형식으로 정리해주세요.

# - ppt의 목차라고 생각되는 경우, "목차"라고 표시
# - 문서의 완성도나 품질에 대한 언급은 절대 하지 마세요
# - "불완전하다", "단편적이다", "소개가 부족하다" 등의 표현 금지
# 출력 형식을 정확히 지켜 주세요."""

#             # 구버전 OpenAI API 호출
#             response = openai.ChatCompletion.create(
#                 model=self.model,
#                 messages=[
#                     {"role": "system", "content": system_prompt},
#                     {"role": "user", "content": user_prompt}
#                 ],
#                 temperature=0.2,
#                 max_tokens=4000
#             )
            
#             translated_analysis = response.choices[0].message.content.strip()
            
#             logger.info("페이지별 분석 결과 번역 및 정리 완료")
            
#             return {
#                 "success": True,
#                 "original_analysis": paged_analysis,
#                 "translated_analysis": translated_analysis,
#                 "analysis_type": "paged_analysis",
#                 "model_used": self.model,
#                 "token_usage": {
#                     "prompt_tokens": response.usage.prompt_tokens,
#                     "completion_tokens": response.usage.completion_tokens,
#                     "total_tokens": response.usage.total_tokens
#                 }
#             }
            
#         except Exception as e:
#             logger.error(f"페이지별 분석 번역 오류: {str(e)}")
#             return {
#                 "success": False,
#                 "error": f"페이지별 분석 번역 중 오류가 발생했습니다: {str(e)}",
#                 "original_analysis": paged_analysis,
#                 "translated_analysis": None
#             }

import openai
import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

class GPTTranslator:
    def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo"):
        self.api_key = api_key or "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"
        self.model = model

        if not self.api_key:
            logger.error("OpenAI API 키가 설정되지 않았습니다")
            self.is_available = False
            return

        try:
            openai.api_key = self.api_key
            _ = openai.Model.list()
            self.is_available = True
            logger.info(f"GPT 번역 클라이언트 초기화 완료 - 모델: {self.model}")
        except Exception as e:
            logger.error(f"OpenAI 클라이언트 초기화 실패: {str(e)}")
            self.is_available = False

    def translate_to_korean(self, text: str, source_language: str = "English") -> Dict[str, Any]:
        if not self.is_available:
            return {"success": False, "error": "OpenAI API 키가 설정되지 않았거나 클라이언트 초기화에 실패했습니다", "original_text": text, "translated_text": None}

        if not text.strip():
            return {"success": False, "error": "번역할 텍스트가 없습니다", "original_text": text, "translated_text": None}

        try:
            system_prompt = """당신은 전문 번역가입니다. 다음 지침을 따라 번역해주세요:
1. 자연스럽고 정확한 한국어로 번역
2. 원문의 의미와 뉘앙스를 정확히 전달
3. 전문 용어는 적절한 한국어 용어로 번역
4. 문맥과 톤을 유지
5. 불필요한 설명이나 주석 없이 번역문만 제공"""

            user_prompt = f"""다음 {source_language} 텍스트를 한국어로 번역해주세요:

{text}"""

            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=4000
            )

            translated_text = response.choices[0].message.content.strip()
            return {
                "success": True,
                "original_text": text,
                "translated_text": translated_text,
                "source_language": source_language,
                "target_language": "Korean",
                "model_used": self.model,
                "token_usage": {
    "prompt_tokens": response.usage["prompt_tokens"],
    "completion_tokens": response.usage["completion_tokens"],
    "total_tokens": response.usage["total_tokens"]
}            }
        except Exception as e:
            logger.error(f"GPT 번역 오류: {str(e)}")
            return {"success": False, "error": f"번역 중 오류가 발생했습니다: {str(e)}", "original_text": text, "translated_text": None}

    def translate_analysis_result(self, analysis_result: str, analysis_type: str = "image") -> Dict[str, Any]:
        if not self.is_available:
            return {"success": False, "error": "OpenAI API 키가 설정되지 않았거나 클라이언트 초기화에 실패했습니다", "original_analysis": analysis_result, "translated_analysis": None}

        if not analysis_result.strip():
            return {"success": False, "error": "번역할 분석 결과가 없습니다", "original_analysis": analysis_result, "translated_analysis": None}

        try:
            context_prompts = {"image": "이미지 분석 결과", "text": "텍스트 분석 결과", "pdf": "PDF 문서 분석 결과"}
            context = context_prompts.get(analysis_type, "AI 분석 결과")

            system_prompt = f"""당신은 AI 분석 결과를 한국어로 번역하고 정리하는 전문가입니다. 다음 지침을 따라주세요:
📋 번역 및 정리 지침:
1. {context}를 자연스럽고 이해하기 쉬운 한국어로 번역
2. 복잡한 기술 용어는 일반인도 이해할 수 있도록 쉽게 설명
3. 문장을 명확하고 읽기 쉽게 재구성
4. 중요한 정보는 강조하여 표현
5. 원문의 구조는 유지하되, 더 논리적으로 정리
6. 불필요한 반복이나 중복은 제거하고 핵심 내용만 간결하게 정리
❌ 절대 포함하지 말 것:
- 문서의 품질에 대한 판단
- 페이지 구성이나 구조에 대한 비평
✨ 가독성 향상 요구사항:
- 문장은 짧고 명확하게 작성
- 전문 용어 뒤에 괄호로 쉬운 설명 추가
- 중요한 개념은 굵게 표시
- 리스트나 단계는 번호나 불릿으로 정리"""

            user_prompt = f"""다음 {context}를 한국어로 번역하고 이해하기 쉽게 정리해주세요:
{analysis_result}"""

            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=4000
            )

            translated_analysis = response.choices[0].message.content.strip()
            return {
                "success": True,
                "original_analysis": analysis_result,
                "translated_analysis": translated_analysis,
                "analysis_type": analysis_type,
                "model_used": self.model,
                "token_usage": {
    "prompt_tokens": response.usage["prompt_tokens"],
    "completion_tokens": response.usage["completion_tokens"],
    "total_tokens": response.usage["total_tokens"]
}            }
        except Exception as e:
            logger.error(f"분석 결과 번역 오류: {str(e)}")
            return {"success": False, "error": f"분석 결과 번역 중 오류가 발생했습니다: {str(e)}", "original_analysis": analysis_result, "translated_analysis": None}

    def translate_paged_analysis(self, paged_analysis: str, doc_type: str = "ppt") -> Dict[str, Any]:
        if not self.is_available:
            return {"success": False, "error": "OpenAI API 키가 설정되지 않았거나 클라이언트 초기화에 실패했습니다", "original_analysis": paged_analysis, "translated_analysis": None}

        if not paged_analysis.strip():
            return {"success": False, "error": "번역할 페이지별 분석 결과가 없습니다", "original_analysis": paged_analysis, "translated_analysis": None}

        try:
            if doc_type == "ppt":
                system_prompt = """당신은 PPT 슬라이드 분석 내용을 한국어로 번역하고 정리하는 전문가입니다.
📚 페이지별 번역 및 정리 지침:
1. 페이지 구분 형식을 정확히 유지 (===== 페이지 X ===== 형식)
2. 각 페이지의 내용을 이해하기 쉬운 한국어로 번역 및 정리
3. 복잡한 기술 용어는 일반인도 이해할 수 있도록 쉽게 설명
4. 전문 용어는 적절한 한국어로 번역하되, 필요시 영어 원문을 괄호로 병기
5. "Page X"는 "페이지 X"로 번역

📝 작업 지침:
1. 각 슬라이드는 "📄 페이지 X" 형식으로 시작
2. ollama로 출력한 영어 원문의 번역을 "🔍 주요 내용:"에 작성
3. 슬라이드의 내용을 자연스럽고 논리적으로 풀어 "📋 상세 설명:"에 정리
4. 불완전하거나 단편적인 키워드라도 가능한 명확한 의미로 구성
5. 원문의 형식(표, 리스트 등)은 텍스트 기반 요약으로 풀어서 작성
6. 시각적 요소나 레이아웃 언급은 하지 않음

❌ 금지 사항:
- "정보 부족", "불완전" 등의 판단
- 슬라이드의 디자인, 완성도, 분량 등 언급
- 문서 품질 평가나 메타 해석

"""

             
                user_prompt = f"""다음은 PPT 문서의 슬라이드별 텍스트 추출 결과입니다. 각 슬라이드를 페이지로 간주하여 아래 형식에 맞춰 번역하고 정리해주세요.

입력 내용:
{paged_analysis}


각 페이지별로 다음 형식을 따라 정리해주세요:
===== 페이지 X =====
📌 예시 형식:
📄 페이지 1  
🔍 주요 내용: ollama 로 출력한 영어를 한국어로 번역해서 제공해주세요.


📋 **상세 설명**:  
- **컴퓨터 그래픽스**는 디지털 환경에서 시각적 콘텐츠를 생성하는 기술입니다.  
- 이 기술은 영화, 게임, 시뮬레이션 등 다양한 분야에 활용됩니다.

주의사항:
1. 텍스트를 페이지나 섹션 단위로 구분하여 정리해주세요.
2. 내용을 요약하지 말고, 각 섹션의 핵심 정보를 충실하게 정리해주세요.
3. 모든 중요한 세부 정보를 포함해주세요.
4. 내용을 단순 요약하지 말고, 구조화된 형식으로 정리해주세요.

- ppt의 목차라고 생각되는 경우, "목차"라고 표시
- 문서의 완성도나 품질에 대한 언급은 절대 하지 마세요
- "불완전하다", "단편적이다", "소개가 부족하다" 등의 표현 금지
출력 형식을 정확히 지켜 주세요."""

            elif doc_type == "hwp":
                system_prompt = """당신은 한글 문서를 분석하고 번역하는 전문가입니다. 문단 또는 제목 단위로 아래 형식을 따라 정리하세요:
🧩 구역 X
🔍 주요 내용: 주제 요약
📋 상세 설명: 논리적 정리"""
                user_prompt = f"""다음은 한글 문서에서 추출한 텍스트입니다:
{paged_analysis}
각 문단 또는 제목 단위를 다음 형식으로 정리하세요:
🧩 구역 X
🔍 주요 내용: ...
📋 상세 설명: ..."""
            else:
                return {"success": False, "error": f"지원되지 않는 문서 유형입니다: {doc_type}", "original_analysis": paged_analysis, "translated_analysis": None}

            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.2,
                max_tokens=4000
            )

            translated_analysis = response.choices[0].message.content.strip()
            return {
                "success": True,
                "original_analysis": paged_analysis,
                "translated_analysis": translated_analysis,
                "analysis_type": f"{doc_type}_analysis",
                "model_used": self.model,
                "token_usage": {
    "prompt_tokens": response.usage["prompt_tokens"],
    "completion_tokens": response.usage["completion_tokens"],
    "total_tokens": response.usage["total_tokens"]
}            }
        except Exception as e:
            logger.error(f"페이지별 분석 번역 오류: {str(e)}")
            return {"success": False, "error": f"페이지별 분석 번역 중 오류가 발생했습니다: {str(e)}", "original_analysis": paged_analysis, "translated_analysis": None}
