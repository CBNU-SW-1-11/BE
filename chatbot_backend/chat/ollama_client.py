import asyncio
import concurrent.futures
import re
import os
import requests
import json
import logging
import base64
from io import BytesIO
from PIL import Image, ImageFilter, ImageEnhance
import pytesseract

logger = logging.getLogger(__name__)

class OllamaClient:
    def __init__(self, base_url=None):
        # 기본 주소 설정
        if base_url is None:
            # 먼저 환경 변수 확인
            base_url = os.environ.get('OLLAMA_API_URL', 'http://localhost:11434')
        self.base_url = base_url
        self.is_server_available = False
        
        # 서버 연결 테스트
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                self.is_server_available = True
                logger.info(f"Ollama 서버 연결 성공: {self.base_url}")
                # 사용 가능한 모델 확인
                models_data = response.json()
                available_models = models_data.get('models', [])
                if available_models:
                    model_names = [model.get('name') for model in available_models]
                    logger.info(f"사용 가능한 모델: {', '.join(model_names)}")
                    self.available_models = model_names
                else:
                    logger.warning("사용 가능한 모델이 없습니다.")
                    self.available_models = []
            else:
                logger.error(f"Ollama 서버 연결 실패: {response.status_code}")
        except Exception as e:
            logger.error(f"Ollama 서버 연결 시도 중 오류: {str(e)}")
            self.available_models = []
    
    def preprocess_image_for_ocr(self, image):
        """OCR 인식률 향상을 위한 이미지 전처리"""
        # 그레이스케일 변환
        if image.mode != 'L':
            image = image.convert('L')
        
        # 노이즈 제거 (가우시안 블러)
        image = image.filter(ImageFilter.GaussianBlur(radius=1))
        
        # 대비 향상
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)  # 대비 2배 증가
        
        # 이미지 크기 조정 (해상도 증가)
        width, height = image.size
        image = image.resize((width*2, height*2), Image.BICUBIC)
        
        return image
    
    def get_optimized_ocr_text(self, image, lang='kor+eng'):
        """최적화된 OCR 설정으로 텍스트 추출"""
        # 한국어 인식에 최적화된 Tesseract 설정
        custom_config = r'--oem 1 --psm 6 -c preserve_interword_spaces=1 -c textord_min_linesize=3'
        
        # OCR 실행
        text = pytesseract.image_to_string(image, lang=lang, config=custom_config)
        
        # 불필요한 공백 정리
        text = ' '.join(text.split())
        
        return text
    
    def _get_best_image_model(self):
        """사용 가능한 최적의 이미지 모델 선택"""
        # 우선순위에 따라 이미지 모델 선택
        image_models = [
            "llava",             # 기본 모델
            "llama3-vision",     # Llama 3 비전 모델
            "llava:13b",         # 더 큰 모델
            "bakllava",          # 대체 모델
        ]
        
        # 사용 가능한 모델 중 우선순위가 높은 것 선택
        for model in image_models:
            if model in self.available_models:
                return model
        
        # 이름에 "vision" 또는 "llava"가 포함된 모델 찾기
        for model in self.available_models:
            if "vision" in model.lower() or "llava" in model.lower():
                return model
        
        # 기본값 반환
        return "llava"
    
    def _get_best_text_model(self):
        """사용 가능한 최적의 텍스트 모델 선택"""
        # 우선순위에 따라 텍스트 모델 선택
        text_models = [
            "llama3",    # 최신 Llama 3 모델
            "phi3",      # MS의 Phi-3 모델  
            "gemma",     # Google의 Gemma 모델
            "llama2",    # Llama 2 모델
            "mistral",   # Mistral 모델
        ]
        
        # 사용 가능한 모델 중 우선순위가 높은 것 선택
        for model in text_models:
            if model in self.available_models:
                return model
        
        # 이름에 "llama"가 포함된 모델 찾기
        for model in self.available_models:
            if "llama" in model.lower():
                return model
        
        # 기본값 반환
        return "llama3"
    
    def check_text_relevance(self, image_path, ocr_text):
        """이미지에서 추출된 텍스트가 이미지 내용과 관련이 있는지 확인"""
        if not self.is_server_available:
            logger.warning("Ollama 서버가 사용 불가능하여 텍스트 관련성 확인 불가")
            return False
        
        if not ocr_text or ocr_text.strip() == "":
            logger.info("텍스트가 없어 관련성 확인 필요 없음")
            return False
        
        try:
            # 이미지 파일 로드 및 base64 인코딩
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
                base64_image = base64.b64encode(image_data).decode('utf-8')
            
            # 텍스트 관련성 판단을 위한 프롬프트
            relevance_prompt = f"""이 이미지에서 추출된 텍스트가 이미지의 내용과 관련이 있는지 판단해주세요.
    추출된 텍스트: 
    {ocr_text}

    이 텍스트가 이미지와 직접 관련이 있습니까? 
    - 이미지 속에 실제로 보이는 텍스트인가요?
    - 이미지 내용을 이해하는 데 필요한 텍스트인가요?


    "예" 또는 "아니오"로만 대답해주세요."""

            # 모델 선택 및 API 요청 데이터 준비
            model = self._get_best_image_model()
            relevance_payload = {
                "model": model,
                "prompt": relevance_prompt,
                "images": [base64_image],
                "stream": False,
                "temperature": 0.1
            }
            
            # 관련성 판단 API 호출
            logger.info(f"텍스트 관련성 확인 API 호출: 모델 {model}")
            relevance_response = requests.post(
                f"{self.base_url}/api/generate",
                json=relevance_payload,
                timeout=60
            )
            
            # 응답 처리
            if relevance_response.status_code == 200:
                relevance_data = relevance_response.json()
                relevance_answer = relevance_data.get("response", "").lower().strip()
                
                # '예' 또는 'yes'가 포함된 경우 관련 있음으로 판단
                is_relevant = '예' in relevance_answer or 'yes' in relevance_answer
                logger.info(f"텍스트 관련성 판단 결과: {'관련 있음' if is_relevant else '관련 없음'}")
                return is_relevant
            else:
                logger.error(f"텍스트 관련성 확인 API 오류: {relevance_response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"텍스트 관련성 확인 중 오류: {str(e)}")
            return False
    
    def analyze_image(self, image_path, prompt=None, ocr_text=None, text_relevant=False):
        """이미지를 분석하는 Ollama API 호출 - 텍스트 관련성 정보 활용"""
        # Ollama 서버가 사용 불가능한 경우 OCR만 수행
        if not self.is_server_available:
            try:
                if not ocr_text:
                    # OCR 텍스트가 제공되지 않은 경우 직접 추출
                    img = Image.open(image_path)
                    preprocessed_img = self.preprocess_image_for_ocr(img)
                    ocr_text = self.get_optimized_ocr_text(preprocessed_img, lang='kor+eng')
                
                return f"""이미지 분석 서비스에 연결할 수 없습니다.

    OCR로 추출한 텍스트 내용:
    {ocr_text if text_relevant else "[이미지와 관련 없는 텍스트가 추출되었습니다.]"}

    ※ Ollama 서버 연결을 확인해주세요. ({self.base_url})
    1. Ollama가 설치되어 있는지 확인: https://ollama.com/download
    2. 서버 실행 확인: 'ollama serve' 명령어 실행
    3. 모델 설치 확인: 'ollama pull llava'"""
            except Exception as e:
                logger.error(f"OCR 처리 오류: {str(e)}")
                return f"이미지 분석 및 OCR 처리 중 오류가 발생했습니다: Ollama 서버({self.base_url})에 연결할 수 없으며, OCR 처리도 실패했습니다."
        
        try:
            # 이미지 파일 로드 및 base64 인코딩
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()
                base64_image = base64.b64encode(image_data).decode('utf-8')
            
            # OCR 텍스트가 제공되지 않은 경우 직접 추출
            if ocr_text is None:
                try:
                    img = Image.open(image_path)
                    preprocessed_img = self.preprocess_image_for_ocr(img)
                    ocr_text = self.get_optimized_ocr_text(preprocessed_img, lang='kor+eng')
                    logger.info(f"이미지 OCR 추출 결과 (일부): {ocr_text[:100]}...")
                    
                    # 텍스트 관련성 확인 (텍스트가 있는 경우만)
                    if ocr_text and ocr_text.strip():
                        text_relevant = self.check_text_relevance(image_path, ocr_text)
                except Exception as e:
                    logger.error(f"이미지 OCR 추출 오류: {str(e)}")
                    ocr_text = ""
                    text_relevant = False
            
            # 기본 프롬프트 설정 - 텍스트 관련성에 따라 분기
            if not prompt:
                if text_relevant and ocr_text and ocr_text.strip():
                    # 텍스트가 이미지와 관련있는 경우
                     prompt = f"""이미지에서 눈으로 직접 확인 가능한 사실만 간결하게 서술하세요.

📌 작성 규칙:
- 1-2문장으로만 작성
- 사람/동물/사물 중 실제 보이는 것만 언급
- 색상은 확실히 보이는 것만 언급
- 위치와 구도는 명확히 보이는 것만 서술

❌ 금지 사항:
- 추측 표현 사용 금지 ("~같은", "~처럼 보이는" 등)
- 보이지 않는 것 언급 금지
- 감정이나 분위기 묘사 금지
- 문장 반복 금지

OCR 텍스트: {ocr_text}
(텍스트는 이미지와 관련이 있는 경우만 참고하세요)

영어로 매우 간결하게 응답하세요."""
                else:
                    # 텍스트가 이미지와 관련없거나 없는 경우
                    prompt = f"""이미지에서 눈으로 직접 확인 가능한 사실만 간결하게 서술하세요.

📌 작성 규칙:
- 1-2문장으로만 작성
- 사람/동물/사물 중 실제 보이는 것만 언급
- 색상은 확실히 보이는 것만 언급 
- 위치와 구도는 명확히 보이는 것만 서술

❌ 금지 사항:
- 추측 표현 사용 금지 ("~같은", "~처럼 보이는" 등)
- 보이지 않는 것 언급 금지
- 감정이나 분위기 묘사 금지
- 문장 반복 금지

영어로로 매우 간결하게 응답하세요."""
            
            # 모델 선택 - 사용 가능한 경우 더 나은 모델 선택
            model = self._get_best_image_model()
            
            # API 요청 데이터 준비
            payload = {
                "model": model,
                "prompt": prompt,
                "images": [base64_image],
                "stream": False,
                "temperature": 0.1  # 낮은 온도로 더 정확한 응답 유도
            }
            
            logger.info(f"Ollama API 요청: /api/generate (이미지 분석) 모델: {model}")
            # API 호출
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=180  # 타임아웃 3분으로 연장
            )
            
            # 응답 처리 및 로깅
            logger.info(f"Ollama API 응답 상태 코드: {response.status_code}")
            
            if response.status_code == 200:
                try:
                    response_data = response.json()
                    api_response = response_data.get("response", "응답을 받지 못했습니다.")
                    logger.info(f"Ollama API 응답 내용 (일부): {api_response[:100]}...")
                    
                    # 응답이 OCR 텍스트와 동일한지 확인 (오류 감지)
                    if ocr_text and len(ocr_text) > 20 and api_response.strip().startswith(ocr_text[:20].strip()):
                        if text_relevant:
                            return f"이미지 분석: 이미지에는 주로 텍스트가 포함되어 있으며, 텍스트는 다음과 같습니다: {ocr_text}"
                        else:
                            return "이미지를 제대로 분석하지 못했습니다. 이미지에 텍스트가 포함되어 있지만 이미지의 시각적 내용과 관련이 없습니다."
                    
                    return api_response
                except json.JSONDecodeError:
                    logger.error("JSON 파싱 오류")
                    return response.text
            elif response.status_code == 400 and "model not found" in response.text.lower():
                # 모델을 찾을 수 없는 경우
                return f"""요청한 이미지 분석 모델({model})을 찾을 수 없습니다.

    다음 명령어로 모델을 설치해주세요:
    'ollama pull {model}'

    또는 이미지 분석에 최적화된 최신 모델 설치:
    'ollama pull llava'

    OCR로 추출한 텍스트 내용:
    {ocr_text if text_relevant else "[이미지와 관련 없는 텍스트가 추출되었습니다.]"}"""
            else:
                # 404 오류 등 특별 처리
                if response.status_code == 404:
                    return f"""Ollama API에 접근할 수 없습니다 (404 오류).

    OCR로 추출한 텍스트 내용:
    {ocr_text if text_relevant else "[이미지와 관련 없는 텍스트가 추출되었습니다.]"}

    ※ 다음 사항을 확인해주세요:
    1. Ollama가 설치되어 있는지 확인: https://ollama.com/download
    2. 서버 실행 확인: 'ollama serve' 명령어 실행
    3. 모델 설치 확인: 'ollama pull llava'"""
                    
                raise Exception(f"Ollama API 오류: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Ollama 이미지 분석 오류: {str(e)}")
            # OCR 결과는 제공
            try:
                if not ocr_text:
                    img = Image.open(image_path)
                    preprocessed_img = self.preprocess_image_for_ocr(img)
                    ocr_text = self.get_optimized_ocr_text(preprocessed_img, lang='kor+eng')
                    
                    # 텍스트 관련성 확인
                    if ocr_text and ocr_text.strip():
                        text_relevant = self.check_text_relevance(image_path, ocr_text)
                
                return f"""이미지 분석 서비스에 연결할 수 없습니다.

    OCR로 추출한 텍스트 내용:
    {ocr_text if text_relevant else "[이미지와 관련 없는 텍스트가 추출되었습니다.]"}

    ※ 오류: {str(e)}"""
            except:
                return f"이미지 분석 처리 중 오류가 발생했습니다: {str(e)}"
   
    def _get_best_text_model(self):
        """사용 가능한 최적의 텍스트 모델 선택"""
        # 우선순위에 따라 텍스트 모델 선택
        text_models = [
            "llama3",    # 최신 Llama 3 모델
            "phi3",      # MS의 Phi-3 모델  
            "gemma",     # Google의 Gemma 모델
            "llama2",    # Llama 2 모델
            "mistral",   # Mistral 모델
        ]
        
        # 사용 가능한 모델 중 우선순위가 높은 것 선택
        for model in text_models:
            if model in self.available_models:
                return model
        
        # 이름에 "llama"가 포함된 모델 찾기
        for model in self.available_models:
            if "llama" in model.lower():
                return model
        
        # 기본값 반환
        return "llama3"
            
    def _detect_page_boundaries(self, text):
        """텍스트에서 페이지 경계 감지 (다양한 형식 지원)"""
        # 일반적인 페이지 구분자 패턴들
        page_patterns = [
            r'(?i)(?:^|\n)[\s-]*페이지\s*(\d+)[\s-]*(?:\n|$)',    # "페이지 1" 형식
            r'(?i)(?:^|\n)[\s-]*page\s*(\d+)[\s-]*(?:\n|$)',      # "Page 1" 형식
            r'(?i)(?:^|\n)[\s-]*p\.?\s*(\d+)[\s-]*(?:\n|$)',      # "p. 1" 형식
            r'(?i)(?:^|\n)[\s-]*\[(\d+)\][\s-]*(?:\n|$)',         # "[1]" 형식
            r'(?i)(?:^|\n)[\s-]*-\s*(\d+)\s*-[\s-]*(?:\n|$)',     # "- 1 -" 형식
        ]
        
        # 모든 페이지 구분자 찾기
        pages = []
        last_pos = 0
        current_page = 1
        
        # 먼저 모든 패턴으로 매치 검색
        all_matches = []
        for pattern in page_patterns:
            for match in re.finditer(pattern, text):
                try:
                    page_num = int(match.group(1))
                    all_matches.append((match.start(), page_num, match.end()))
                except:
                    pass
        
        # 매치가 없으면 빈 리스트 반환
        if not all_matches:
            return []
        
        # 위치로 정렬
        all_matches.sort()
        
        # 페이지별로 분할
        for i, (pos, page_num, end_pos) in enumerate(all_matches):
            # 첫 번째 매치가 아니라면 이전 페이지 내용 저장
            if i > 0:
                content = text[last_pos:pos].strip()
                if content:
                    pages.append({"page": current_page, "text": content})
            
            current_page = page_num
            last_pos = end_pos
        
        # 마지막 페이지 내용 추가
        if last_pos < len(text):
            content = text[last_pos:].strip()
            if content:
                pages.append({"page": current_page, "text": content})
        
        return pages

    def analyze_text(self, text, prompt=None, page_texts=None):
            """텍스트를 분석하는 Ollama API 호출 - 속도 개선 버전"""
            try:
                # 텍스트가 비어있는지 확인
                if not text or text.strip() == "":
                    return "분석할 텍스트가 없습니다."
                
                # 페이지별 분석 여부 판단
                analyze_by_page = False
                pages = []
                
                # 제공된 페이지 텍스트가 있는 경우
                if page_texts and isinstance(page_texts, list) and len(page_texts) > 1:
                    analyze_by_page = True
                    pages = page_texts
                else:
                    # 텍스트에서 페이지 경계를 자동 감지
                    detected_pages = self._detect_page_boundaries(text)
                    if detected_pages and len(detected_pages) > 1:
                        analyze_by_page = True
                        pages = detected_pages
                
                # 모델 선택 (한국어 처리에 최적화된 모델 사용)
                model = self._get_best_text_model()
                
                if analyze_by_page and len(pages) > 3:  # 3페이지 이상일 때만 최적화
                    # 방법 1: 전체 텍스트를 한 번에 처리하고 페이지별 구분 요청 (배치 처리)
                    formatted_pages = ""
                    for page in pages:
                        page_num = page.get("page", 1)
                        page_text = page.get("text", "").strip()
                        if page_text:
                            formatted_pages += f"===== 페이지 {page_num} =====\n{page_text}\n\n"
                    
                    batch_prompt = f"""다음은 문서의 여러 페이지 내용입니다. 각 페이지를 구분하여 분석해주세요:

        {formatted_pages}

        각 페이지의 내용을 다음 가이드라인에 따라 정리해주세요:
        각 페이지의 OCR 추출 결과를 넣어주세요.
   
        반드시 다음 형식으로 응답해주세요:
        ===== 페이지 1 분석 =====
        (page 1 OCR 추출 결과,분석 내용)

        ===== 페이지 2 분석 =====
        (page 2 OCR 추출 결과,분석 내용)

        ...

        반드시 "영어(En)"로 응답해주세요."""

                    # API 요청 데이터 준비 - 토큰 수 증가
                    payload = {
                        "model": model,
                        "prompt": batch_prompt,
                        "stream": False,
                        "temperature": 0.3,  # 정확성 중시
                        "max_tokens": 4096   # 토큰 수 증가
                    }
                    
                    # API 호출
                    response = requests.post(
                        f"{self.base_url}/api/generate",
                        json=payload,
                        timeout=300  # 시간 증가
                    )
                    
                    if response.status_code == 200:
                        response_data = response.json()
                        return response_data.get("response", "응답을 받지 못했습니다.")
                    else:
                        raise Exception(f"Ollama API 오류: {response.status_code}")
                        
                elif analyze_by_page:
                    # 방법 2: 비동기 처리로 여러 페이지 동시에 처리
                    async def process_pages():
                        # 최대 동시 작업 수 제한
                        max_workers = min(5, len(pages))  # 최대 5개 페이지 동시 처리
                        
                        async def analyze_page(page_data):
                            page_num = page_data.get("page", 1)
                            page_text = page_data.get("text", "")
                            
                            if not page_text.strip():
                                return f"===== 페이지 {page_num} =====\n페이지 내용이 없습니다.\n"
                            
                            page_prompt = f"""다음은 문서의 {page_num}페이지에서 추출한 텍스트입니다:

        {page_text}


        이 페이지의 내용을 자세히 분석하고 정리해주세요.
        단순 요약이 아닌 페이지의 핵심 정보를 충실하게 정리해주세요.
        "불완전하다", "단편적이다", "소개가 부족하다" 등의 표현 금지
        단, 문장이나 단어가 아니라고 생각될 경우 분석하지 않아도 됩니다. 또는 정보 없음이라고 출력
        반드시 "영어(En)"로 응답해주세요."""
                            
                            # ThreadPoolExecutor를 사용하여 비동기 HTTP 요청
                            with concurrent.futures.ThreadPoolExecutor() as executor:
                                page_payload = {
                                    "model": model,
                                    "prompt": page_prompt,
                                    "stream": False,
                                    "temperature": 0.3,
                                    "max_tokens": 1024
                                }
                                
                                def make_request():
                                    try:
                                        r = requests.post(
                                            f"{self.base_url}/api/generate",
                                            json=page_payload,
                                            timeout=180
                                        )
                                        if r.status_code == 200:
                                            data = r.json()
                                            return f"===== 페이지 {page_num} =====\n{data.get('response', '분석 실패')}"
                                        return f"===== 페이지 {page_num} =====\n페이지 분석 중 오류 발생: {r.status_code}"
                                    except Exception as e:
                                        return f"===== 페이지 {page_num} =====\n페이지 분석 중 오류 발생: {str(e)}"
                                
                                return await asyncio.get_event_loop().run_in_executor(executor, make_request)
                        
                        # 페이지 그룹화 - 2페이지씩 묶기
                        grouped_pages = []
                        for i in range(0, len(pages), 2):  # 2페이지씩 그룹화
                            group = pages[i:i+2]
                            if group:
                                grouped_text = "\n\n".join([p.get("text", "") for p in group])
                                start_page = group[0].get("page", 1)
                                end_page = group[-1].get("page", start_page)
                                grouped_pages.append({
                                    "page": f"{start_page}-{end_page}",
                                    "text": grouped_text
                                })
                        
                        # 동시에 처리할 페이지 그룹 분석
                        tasks = [analyze_page(page) for page in grouped_pages]
                        results = await asyncio.gather(*tasks)
                        return "\n\n".join(results)
                    
                    # 비동기 실행 결과 반환
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    try:
                        return loop.run_until_complete(process_pages())
                    finally:
                        loop.close()
                else:
                    # 기존 방식 - 단일 문서 또는 적은 페이지 수
                    prompt = f"""다음 텍스트를 자세히 분석하고 내용을 명확하게 정리해주세요:

        {text}

        분석 지침:
        1. 텍스트의 주요 내용과 구조를 명확하게 파악하여 정리
        2. 단순 요약이 아닌, 텍스트의 핵심 정보를 충실하게 정리
        3. 중요한 섹션이나 주제별로 구분하여 정리
        4. 모든 중요한 세부 정보를 포함
        단, 문장이나 단어가 아니라고 생각될 경우 분석하지 않아도 됩니다. 또는 정보 없음이라고 출력
        반드시 "영어(En)"(En)로 응답해주세요."""
                    
                    if prompt:
                        # 기존 프롬프트에 텍스트 추가
                        if "{text}" in prompt:
                            prompt = prompt.format(text=text)
                        else:
                            prompt = f"{prompt}\n\n{text}"
                    
                    # API 요청 데이터 준비
                    payload = {
                        "model": model,
                        "prompt": prompt,
                        "stream": False,
                        "temperature": 0.7,
                        "top_p": 0.9,
                        "max_tokens": 2048
                    }
                    
                    # API 호출
                    response = requests.post(
                        f"{self.base_url}/api/generate",
                        json=payload,
                        timeout=180
                    )
                    
                    if response.status_code == 200:
                        response_data = response.json()
                        return response_data.get("response", "응답을 받지 못했습니다.")
                    else:
                        raise Exception(f"Ollama API 오류: {response.status_code}")
            
            except Exception as e:
                logger.error(f"Ollama 텍스트 분석 오류: {str(e)}")
                return f"텍스트 분석 처리 중 오류가 발생했습니다: {str(e)}"

