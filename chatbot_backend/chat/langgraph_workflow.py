# langgraph_workflow.py - 새로 생성할 파일
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Any, Annotated
from langchain.schema import BaseMessage
import asyncio
import logging
import time
import json
from .langchain_config import LangChainManager, GroqLLM

logger = logging.getLogger(__name__)

# State 정의
class AIResponseState(TypedDict):
    user_message: str
    user_language: str
    selected_models: List[str]
    request_id: str
    
    # 각 단계별 결과
    individual_responses: Dict[str, str]
    similarity_analysis: Dict[str, Any]
    final_analysis: Dict[str, Any]
    
    # 에러 처리
    errors: List[str]
    current_step: str

# 워크플로우 클래스
class AIComparisonWorkflow:
    def __init__(self, langchain_manager: LangChainManager, similarity_analyzer):
        self.langchain_manager = langchain_manager
        self.similarity_analyzer = similarity_analyzer
        self.groq_llm = GroqLLM(langchain_manager.groq_key)
        
        # 워크플로우 그래프 생성
        self.workflow = self.create_workflow()
    
    def create_workflow(self) -> StateGraph:
        """LangGraph 워크플로우 생성"""
        workflow = StateGraph(AIResponseState)
        
        # 노드 추가
        workflow.add_node("collect_responses", self.collect_ai_responses)
        workflow.add_node("analyze_similarity", self.analyze_response_similarity)
        workflow.add_node("generate_optimal", self.generate_optimal_response)
        workflow.add_node("handle_error", self.handle_workflow_error)
        
        # 엣지 추가 (워크플로우 순서)
        workflow.set_entry_point("collect_responses")
        
        # 조건부 라우팅
        workflow.add_conditional_edges(
            "collect_responses",
            self.should_continue_after_collection,
            {
                "continue": "analyze_similarity",
                "error": "handle_error"
            }
        )
        
        workflow.add_conditional_edges(
            "analyze_similarity", 
            self.should_continue_after_similarity,
            {
                "continue": "generate_optimal",
                "error": "handle_error"
            }
        )
        
        workflow.add_edge("generate_optimal", END)
        workflow.add_edge("handle_error", END)
        
        return workflow.compile()
    
    async def collect_ai_responses(self, state: AIResponseState) -> AIResponseState:
        """1단계: 각 AI 모델로부터 응답 수집"""
        logger.info("🤖 1단계: AI 응답 수집 시작")
        state["current_step"] = "collecting_responses"
        
        responses = {}
        errors = []
        
        # selectedModels가 올바른 리스트인지 확인
        selected_models = state["selected_models"]
        if isinstance(selected_models, str):
            logger.error(f"❌ selected_models가 문자열입니다: {selected_models}")
            try:
                import json
                selected_models = json.loads(selected_models)
                logger.info(f"✅ JSON 파싱 성공: {selected_models}")
            except:
                logger.error(f"❌ JSON 파싱 실패, 기본값 사용")
                selected_models = ['gpt', 'claude', 'mixtral']
        
        logger.info(f"🎯 처리할 모델 목록: {selected_models} (타입: {type(selected_models)})")
        
        # 병렬로 각 모델에서 응답 수집
        tasks = []
        for model in selected_models:
            logger.info(f"📡 {model} 모델 응답 요청 준비 중...")
            
            if model in ['gpt', 'claude', 'mixtral']:  # 모든 모델이 LangChain 통합됨
                task = self.get_langchain_response(
                    model, 
                    state["user_message"], 
                    state["user_language"]
                )
            else:
                # 지원하지 않는 모델의 경우 에러 발생
                error_msg = f"지원하지 않는 모델: {model}"
                errors.append(error_msg)
                logger.error(f"❌ {error_msg}")
                continue
            tasks.append((model, task))
        
        logger.info(f"🚀 {len(tasks)}개 모델 병렬 실행 시작")
        
        # 비동기 실행
        for model, task in tasks:
            try:
                logger.info(f"⏳ {model} 응답 대기 중...")
                response = await task
                responses[model] = response
                logger.info(f"✅ {model} 응답 완료: {response[:50]}...")
            except Exception as e:
                error_msg = f"{model} 응답 실패: {str(e)}"
                errors.append(error_msg)
                logger.error(error_msg)
        
        state["individual_responses"] = responses
        state["errors"].extend(errors)
        
        logger.info(f"📊 응답 수집 완료: {len(responses)}개 모델, {len(errors)}개 오류")
        return state
    
    async def get_langchain_response(self, model: str, message: str, language: str) -> str:
        """LangChain을 통한 응답 획득"""
        try:
            chain = self.langchain_manager.create_chat_chain(model)
            if model == 'gpt':
                # ChatOpenAI는 비동기 지원
                result = await chain.arun(
                    user_input=message,
                    user_language=language
                )
            else:
                # 커스텀 LLM들은 동기 방식 사용
                result = chain.run(
                    user_input=message,
                    user_language=language
                )
            return result
        except Exception as e:
            logger.error(f"LangChain {model} 에러: {e}")
            raise
    
    async def get_groq_response(self, message: str, language: str) -> str:
        """Groq을 통한 응답 획득 (이제 LangChain 통합됨)"""
        try:
            return await self.get_langchain_response('mixtral', message, language)
        except Exception as e:
            logger.error(f"Groq 에러: {e}")
            raise
    
    async def analyze_response_similarity(self, state: AIResponseState) -> AIResponseState:
        """2단계: 응답 유사도 분석"""
        logger.info("🔍 2단계: 유사도 분석 시작")
        state["current_step"] = "analyzing_similarity"
        
        try:
            responses = state["individual_responses"]
            if len(responses) >= 2:
                similarity_result = self.similarity_analyzer.cluster_responses(responses)
                state["similarity_analysis"] = self.convert_to_serializable(similarity_result)
                logger.info("✅ 유사도 분석 완료")
            else:
                state["similarity_analysis"] = {}
                logger.warning("⚠️ 유사도 분석을 위한 충분한 응답이 없음")
        except Exception as e:
            error_msg = f"유사도 분석 실패: {str(e)}"
            state["errors"].append(error_msg)
            logger.error(error_msg)
        
        return state
    
    async def generate_optimal_response(self, state: AIResponseState) -> AIResponseState:
        """3단계: 최적 응답 생성"""
        logger.info("✨ 3단계: 최적 응답 생성 시작")
        state["current_step"] = "generating_optimal"
        
        try:
            responses = state["individual_responses"]
            selected_models = list(responses.keys())
            
            # 분석에 사용할 모델 결정 (가능한 모델 중 첫 번째 사용)
            analyzer_model = 'gpt'  # 기본값
            if 'gpt' in selected_models:
                analyzer_model = 'gpt'
            elif 'claude' in selected_models:
                analyzer_model = 'claude'
            elif 'mixtral' in selected_models:
                analyzer_model = 'mixtral'
            
            # LangChain 분석 체인 사용
            analysis_chain = self.langchain_manager.create_analysis_chain(analyzer_model)
            
            # 응답 포맷팅
            formatted = self.langchain_manager.format_responses_for_analysis(
                responses, selected_models
            )
            
            # 최종 분석 실행
            if analyzer_model == 'gpt':
                # ChatOpenAI는 비동기 지원
                analysis_result = await analysis_chain.arun(
                    query=state["user_message"],
                    user_language=state["user_language"],
                    selected_models=selected_models,
                    **formatted
                )
            else:
                # 커스텀 LLM들은 동기 방식
                analysis_result = analysis_chain.run(
                    query=state["user_message"],
                    user_language=state["user_language"],
                    selected_models=selected_models,
                    **formatted
                )
            
            state["final_analysis"] = analysis_result
            logger.info("✅ 최적 응답 생성 완료")
            
        except Exception as e:
            error_msg = f"최적 응답 생성 실패: {str(e)}"
            state["errors"].append(error_msg)
            logger.error(error_msg)
            
            # 폴백 응답 생성
            state["final_analysis"] = self.create_fallback_response(
                state["individual_responses"], 
                selected_models
            )
        
        return state
    
    async def handle_workflow_error(self, state: AIResponseState) -> AIResponseState:
        """에러 처리"""
        logger.error(f"🚨 워크플로우 에러 처리: {state['errors']}")
        state["current_step"] = "error_handled"
        
        # 가능한 부분적 결과라도 반환
        if not state.get("final_analysis"):
            state["final_analysis"] = self.create_fallback_response(
                state.get("individual_responses", {}),
                state["selected_models"]
            )
        
        return state
    
    def should_continue_after_collection(self, state: AIResponseState) -> str:
        """응답 수집 후 계속할지 결정"""
        if len(state["individual_responses"]) == 0:
            return "error"
        return "continue"
    
    def should_continue_after_similarity(self, state: AIResponseState) -> str:
        """유사도 분석 후 계속할지 결정"""
        # 유사도 분석이 실패해도 최적 응답 생성은 진행
        return "continue"
    
    def convert_to_serializable(self, obj):
        """객체를 직렬화 가능한 형태로 변환"""
        if hasattr(obj, '__dict__'):
            return {k: self.convert_to_serializable(v) for k, v in obj.__dict__.items()}
        elif isinstance(obj, dict):
            return {k: self.convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    def create_fallback_response(self, responses: Dict[str, str], selected_models: List[str]) -> Dict[str, Any]:
        """폴백 응답 생성"""
        best_response = ""
        if responses:
            best_response = max(responses.values(), key=len)
        
        error_analysis = {}
        for model in selected_models:
            model_lower = model.lower()
            error_analysis[model_lower] = {
                "장점": "분석 실패", 
                "단점": "분석 실패"
            }
        
        return {
            "preferredModel": "FALLBACK",
            "best_response": best_response,
            "analysis": error_analysis,
            "reasoning": "워크플로우 실행 중 오류가 발생하여 폴백 응답을 생성했습니다."
        }
    
    async def run_workflow(self, user_message: str, selected_models: List[str], 
                          user_language: str = 'ko', request_id: str = None) -> Dict[str, Any]:
        """워크플로우 실행"""
        if not request_id:
            request_id = str(time.time())
        
        # 초기 상태 설정
        initial_state = AIResponseState(
            user_message=user_message,
            user_language=user_language,
            selected_models=selected_models,
            request_id=request_id,
            individual_responses={},
            similarity_analysis={},
            final_analysis={},
            errors=[],
            current_step="initialized"
        )
        
        logger.info(f"🚀 워크플로우 시작 - Request ID: {request_id}")
        
        try:
            # 워크플로우 실행
            final_state = await self.workflow.ainvoke(initial_state)
            
            logger.info(f"🎉 워크플로우 완료 - Request ID: {request_id}")
            
            return {
                "request_id": request_id,
                "individual_responses": final_state["individual_responses"],
                "similarity_analysis": final_state["similarity_analysis"],
                "final_analysis": final_state["final_analysis"],
                "errors": final_state["errors"],
                "final_step": final_state["current_step"]
            }
            
        except Exception as e:
            logger.error(f"🚨 워크플로우 실행 실패: {e}")
            return {
                "request_id": request_id,
                "individual_responses": {},
                "similarity_analysis": {},
                "final_analysis": self.create_fallback_response({}, selected_models),
                "errors": [str(e)],
                "final_step": "workflow_failed"
            }