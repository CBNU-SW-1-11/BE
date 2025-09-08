# Enhanced Video QA System - 개선된 비디오 질의응답 시스템

import json
import os
from typing import List, Dict, Any, Optional
from datetime import datetime
import re

class EnhancedVideoQASystem:
    """비디오 분석 결과를 바탕으로 한 개선된 질의응답 시스템"""
    
    def __init__(self, rag_system, llm_client):
        self.rag_system = rag_system
        self.llm_client = llm_client
        self.question_patterns = self._init_question_patterns()
        self.context_memory = {}  # 대화 컨텍스트 유지
    
    def _init_question_patterns(self):
        """질문 패턴별 처리 방식 정의"""
        return {
            'object_detection': {
                'patterns': ['무엇이', '뭐가', '객체', '사물', '나오는', '보이는'],
                'handler': self._handle_object_questions
            },
            'people_analysis': {
                'patterns': ['사람', '인물', '얼굴', '성별', '나이', '옷'],
                'handler': self._handle_people_questions
            },
            'scene_analysis': {
                'patterns': ['장면', '배경', '환경', '장소', '위치', '시간'],
                'handler': self._handle_scene_questions
            },
            'action_analysis': {
                'patterns': ['행동', '동작', '하고있', '움직임', '활동'],
                'handler': self._handle_action_questions
            },
            'summary': {
                'patterns': ['요약', '정리', '전체', '내용', '줄거리'],
                'handler': self._handle_summary_questions
            },
            'specific_search': {
                'patterns': ['찾아', '검색', '언제', '어디서', '몇 번째'],
                'handler': self._handle_search_questions
            }
        }
    
    def answer_question(self, video_id: str, question: str, context: Dict = None) -> Dict:
        """질문에 대한 답변 생성 - 메인 엔트리 포인트"""
        try:
            # 대화 컨텍스트 저장
            if video_id not in self.context_memory:
                self.context_memory[video_id] = []
            
            self.context_memory[video_id].append({
                'question': question,
                'timestamp': datetime.now().isoformat()
            })
            
            # 질문 분석 및 카테고리 분류
            question_category = self._classify_question(question)
            
            # 관련 데이터 검색
            search_results = self.rag_system.search_video_content(video_id, question, top_k=5)
            
            # 질문 유형별 처리
            if question_category in self.question_patterns:
                handler = self.question_patterns[question_category]['handler']
                response = handler(video_id, question, search_results, context)
            else:
                response = self._handle_general_questions(video_id, question, search_results, context)
            
            # 응답에 컨텍스트 정보 추가
            response['question_category'] = question_category
            response['search_results_count'] = len(search_results)
            response['timestamp'] = datetime.now().isoformat()
            
            # 컨텍스트에 응답 저장
            self.context_memory[video_id][-1]['response'] = response['answer']
            self.context_memory[video_id][-1]['category'] = question_category
            
            return response
            
        except Exception as e:
            return {
                'answer': f'질문 처리 중 오류가 발생했습니다: {str(e)}',
                'success': False,
                'error': str(e)
            }
    
    def _classify_question(self, question: str) -> str:
        """질문을 카테고리별로 분류"""
        question_lower = question.lower()
        
        for category, config in self.question_patterns.items():
            if any(pattern in question_lower for pattern in config['patterns']):
                return category
        
        return 'general'
    
    def _handle_object_questions(self, video_id: str, question: str, search_results: List, context: Dict) -> Dict:
        """객체 관련 질문 처리"""
        try:
            objects_info = self._extract_objects_from_results(search_results)
            
            if not objects_info:
                return {
                    'answer': '비디오에서 객체 정보를 찾을 수 없습니다.',
                    'success': False
                }
            
            # 객체별 통계 생성
            object_stats = {}
            total_frames = len(search_results)
            
            for result in search_results:
                objects = result.get('objects', [])
                for obj in objects:
                    if obj not in object_stats:
                        object_stats[obj] = 0
                    object_stats[obj] += 1
            
            # 가장 자주 등장하는 객체들
            sorted_objects = sorted(object_stats.items(), key=lambda x: x[1], reverse=True)
            
            answer = f"비디오에서 발견된 주요 객체들:\n"
            for obj, count in sorted_objects[:10]:
                percentage = (count / total_frames) * 100
                answer += f"• {obj}: {count}회 등장 ({percentage:.1f}%)\n"
            
            return {
                'answer': answer,
                'success': True,
                'objects': objects_info,
                'statistics': object_stats,
                'most_common': sorted_objects[:5]
            }
            
        except Exception as e:
            return {
                'answer': f'객체 분석 중 오류 발생: {str(e)}',
                'success': False,
                'error': str(e)
            }
    
    def _handle_people_questions(self, video_id: str, question: str, search_results: List, context: Dict) -> Dict:
        """인물 관련 질문 처리"""
        try:
            people_info = []
            gender_count = {'male': 0, 'female': 0, 'unknown': 0}
            
            for result in search_results:
                content = result.get('content', '')
                objects = result.get('objects', [])
                
                # 사람 관련 객체 찾기
                people_objects = [obj for obj in objects if 'person' in obj.lower() or 'people' in obj.lower()]
                
                if people_objects or '사람' in content or 'person' in content.lower():
                    people_info.append({
                        'frame_id': result.get('frame_id'),
                        'timestamp': result.get('timestamp'),
                        'content': content,
                        'people_objects': people_objects
                    })
            
            if not people_info:
                return {
                    'answer': '비디오에서 사람을 찾을 수 없습니다.',
                    'success': False
                }
            
            # 성비 관련 질문인지 확인
            if '성비' in question or '남여' in question or '성별' in question:
                # 컨텐츠에서 성별 정보 추출 시도
                gender_analysis = self._analyze_gender_from_content(people_info)
                
                answer = f"비디오 인물 분석 결과:\n"
                answer += f"• 사람이 등장하는 장면: {len(people_info)}개\n"
                
                if gender_analysis['has_gender_info']:
                    answer += f"• 성별 분포: 남성 {gender_analysis['male']}명, 여성 {gender_analysis['female']}명\n"
                else:
                    answer += "• 정확한 성별 구분이 어려운 상황입니다.\n"
                
                return {
                    'answer': answer,
                    'success': True,
                    'people_scenes': len(people_info),
                    'gender_analysis': gender_analysis
                }
            else:
                answer = f"비디오에서 총 {len(people_info)}개 장면에서 사람이 등장합니다.\n"
                for info in people_info[:5]:  # 상위 5개만 표시
                    answer += f"• {info['timestamp']:.1f}초: {info['content'][:100]}...\n"
                
                return {
                    'answer': answer,
                    'success': True,
                    'people_info': people_info
                }
            
        except Exception as e:
            return {
                'answer': f'인물 분석 중 오류 발생: {str(e)}',
                'success': False,
                'error': str(e)
            }
    
    def _handle_scene_questions(self, video_id: str, question: str, search_results: List, context: Dict) -> Dict:
        """장면 관련 질문 처리"""
        try:
            scene_info = []
            locations = set()
            times_of_day = set()
            
            for result in search_results:
                content = result.get('content', '')
                frame_id = result.get('frame_id')
                timestamp = result.get('timestamp')
                
                # 장소/시간 키워드 추출
                location_keywords = self._extract_location_keywords(content)
                time_keywords = self._extract_time_keywords(content)
                
                locations.update(location_keywords)
                times_of_day.update(time_keywords)
                
                scene_info.append({
                    'frame_id': frame_id,
                    'timestamp': timestamp,
                    'content': content,
                    'locations': location_keywords,
                    'times': time_keywords
                })
            
            answer = f"비디오 장면 분석 결과:\n"
            answer += f"• 총 분석된 장면: {len(scene_info)}개\n"
            
            if locations:
                answer += f"• 감지된 장소: {', '.join(list(locations)[:5])}\n"
            
            if times_of_day:
                answer += f"• 시간대: {', '.join(list(times_of_day))}\n"
            
            # 장면별 상세 정보 (상위 3개)
            answer += "\n주요 장면들:\n"
            for i, info in enumerate(scene_info[:3], 1):
                answer += f"{i}. {info['timestamp']:.1f}초: {info['content'][:150]}...\n"
            
            return {
                'answer': answer,
                'success': True,
                'scene_count': len(scene_info),
                'locations': list(locations),
                'times_of_day': list(times_of_day),
                'scene_details': scene_info
            }
            
        except Exception as e:
            return {
                'answer': f'장면 분석 중 오류 발생: {str(e)}',
                'success': False,
                'error': str(e)
            }
    
    def _handle_action_questions(self, video_id: str, question: str, search_results: List, context: Dict) -> Dict:
        """행동/동작 관련 질문 처리"""
        try:
            actions = []
            action_keywords = ['걷기', '뛰기', '앉기', '서기', '말하기', '웃기', '운전', '요리', '춤추기']
            
            for result in search_results:
                content = result.get('content', '')
                frame_id = result.get('frame_id')
                timestamp = result.get('timestamp')
                
                detected_actions = [action for action in action_keywords if action in content]
                
                if detected_actions or any(verb in content for verb in ['하고', '중이', '하는']):
                    actions.append({
                        'frame_id': frame_id,
                        'timestamp': timestamp,
                        'content': content,
                        'detected_actions': detected_actions
                    })
            
            if not actions:
                return {
                    'answer': '특별한 행동이나 동작을 찾을 수 없습니다.',
                    'success': False
                }
            
            answer = f"비디오에서 감지된 행동/동작:\n"
            for i, action in enumerate(actions[:5], 1):
                answer += f"{i}. {action['timestamp']:.1f}초: {action['content'][:100]}...\n"
            
            return {
                'answer': answer,
                'success': True,
                'actions': actions
            }
            
        except Exception as e:
            return {
                'answer': f'행동 분석 중 오류 발생: {str(e)}',
                'success': False,
                'error': str(e)
            }
    
    def _handle_summary_questions(self, video_id: str, question: str, search_results: List, context: Dict) -> Dict:
        """요약 관련 질문 처리"""
        try:
            # 전체 컨텐츠 수집
            all_content = []
            for result in search_results:
                all_content.append({
                    'timestamp': result.get('timestamp', 0),
                    'content': result.get('content', '')
                })
            
            # 시간순 정렬
            all_content.sort(key=lambda x: x['timestamp'])
            
            # 요약 생성 (중요한 내용 우선)
            summary_prompt = f"""
            다음은 비디오의 프레임별 분석 결과입니다:
            
            {chr(10).join([f"{content['timestamp']:.1f}초: {content['content']}" for content in all_content[:10]])}
            
            이 비디오의 주요 내용을 3-5문장으로 요약해주세요.
            """
            
            summary_response = self.llm_client.generate_smart_response(
                user_query=summary_prompt,
                search_results=None,
                video_info=f"비디오 {video_id}",
                use_multi_llm=False
            )
            
            return {
                'answer': summary_response,
                'success': True,
                'total_scenes': len(search_results),
                'summary_type': 'comprehensive'
            }
            
        except Exception as e:
            return {
                'answer': f'요약 생성 중 오류 발생: {str(e)}',
                'success': False,
                'error': str(e)
            }
    
    def _handle_search_questions(self, video_id: str, question: str, search_results: List, context: Dict) -> Dict:
        """검색 관련 질문 처리"""
        try:
            # 키워드 추출
            keywords = self._extract_search_keywords(question)
            
            if not keywords:
                return {
                    'answer': '검색할 키워드를 찾을 수 없습니다.',
                    'success': False
                }
            
            # 키워드별 매칭 결과
            matched_results = []
            for result in search_results:
                content = result.get('content', '').lower()
                match_score = sum(1 for keyword in keywords if keyword.lower() in content)
                
                if match_score > 0:
                    matched_results.append({
                        'frame_id': result.get('frame_id'),
                        'timestamp': result.get('timestamp'),
                        'content': result.get('content'),
                        'match_score': match_score,
                        'matched_keywords': [k for k in keywords if k.lower() in content]
                    })
            
            # 매칭 점수순 정렬
            matched_results.sort(key=lambda x: x['match_score'], reverse=True)
            
            if not matched_results:
                return {
                    'answer': f"'{', '.join(keywords)}'와 관련된 장면을 찾을 수 없습니다.",
                    'success': False
                }
            
            answer = f"'{', '.join(keywords)}' 검색 결과 ({len(matched_results)}개 발견):\n\n"
            
            for i, result in enumerate(matched_results[:5], 1):
                answer += f"{i}. {result['timestamp']:.1f}초 (매칭: {', '.join(result['matched_keywords'])})\n"
                answer += f"   {result['content'][:150]}...\n\n"
            
            return {
                'answer': answer,
                'success': True,
                'search_keywords': keywords,
                'matched_results': matched_results,
                'total_matches': len(matched_results)
            }
            
        except Exception as e:
            return {
                'answer': f'검색 처리 중 오류 발생: {str(e)}',
                'success': False,
                'error': str(e)
            }
    
    def _handle_general_questions(self, video_id: str, question: str, search_results: List, context: Dict) -> Dict:
        """일반 질문 처리"""
        try:
            # RAG 시스템을 통한 답변 생성
            rag_answer = self.rag_system.answer_question(video_id, question)
            
            return {
                'answer': rag_answer,
                'success': True,
                'method': 'rag_general'
            }
            
        except Exception as e:
            return {
                'answer': f'질문 처리 중 오류 발생: {str(e)}',
                'success': False,
                'error': str(e)
            }
    
    # 헬퍼 메서드들
    def _extract_objects_from_results(self, search_results: List) -> List:
        """검색 결과에서 객체 정보 추출"""
        objects = []
        for result in search_results:
            if 'objects' in result:
                objects.extend(result['objects'])
        return list(set(objects))  # 중복 제거
    
    def _analyze_gender_from_content(self, people_info: List) -> Dict:
        """컨텐츠에서 성별 정보 분석"""
        male_indicators = ['남자', '남성', 'man', 'male', '아저씨', '청년']
        female_indicators = ['여자', '여성', 'woman', 'female', '아줌마', '소녀']
        
        male_count = 0
        female_count = 0
        
        for info in people_info:
            content = info['content'].lower()
            if any(indicator in content for indicator in male_indicators):
                male_count += 1
            elif any(indicator in content for indicator in female_indicators):
                female_count += 1
        
        return {
            'male': male_count,
            'female': female_count,
            'has_gender_info': male_count > 0 or female_count > 0
        }
    
    def _extract_location_keywords(self, content: str) -> List[str]:
        """위치 관련 키워드 추출"""
        location_patterns = [
            '실내', '실외', '거실', '주방', '침실', '사무실', '학교', '공원', 
            '거리', '카페', '식당', '병원', '상점', '차안', '집', '건물'
        ]
        return [loc for loc in location_patterns if loc in content]
    
    def _extract_time_keywords(self, content: str) -> List[str]:
        """시간 관련 키워드 추출"""
        time_patterns = ['아침', '낮', '저녁', '밤', '새벽', '오전', '오후']
        return [time for time in time_patterns if time in content]
    
    def _extract_search_keywords(self, question: str) -> List[str]:
        """질문에서 검색 키워드 추출"""
        # 불용어 제거
        stop_words = {'은', '는', '이', '가', '을', '를', '에', '에서', '로', '으로', '와', '과'}
        
        # 간단한 형태소 분석 (실제로는 더 정교한 방법 사용)
        words = re.findall(r'[가-힣a-zA-Z]+', question)
        keywords = [word for word in words if len(word) > 1 and word not in stop_words]
        
        return keywords[:5]  # 상위 5개만
    
    def get_conversation_context(self, video_id: str) -> List[Dict]:
        """대화 컨텍스트 조회"""
        return self.context_memory.get(video_id, [])
    
    def clear_context(self, video_id: str = None):
        """컨텍스트 초기화"""
        if video_id:
            self.context_memory.pop(video_id, None)
        else:
            self.context_memory.clear()


# 사용 예시 및 통합 함수
def integrate_enhanced_qa_system(existing_rag_system, existing_llm_client):
    """기존 시스템과 통합"""
    enhanced_qa = EnhancedVideoQASystem(existing_rag_system, existing_llm_client)
    
    def enhanced_answer_function(video_id: str, question: str, context: Dict = None):
        """향상된 질의응답 함수"""
        return enhanced_qa.answer_question(video_id, question, context)
    
    return enhanced_answer_function, enhanced_qa

# Django 뷰에서 사용할 수 있도록 수정된 메서드
def create_enhanced_chat_response(enhanced_qa: EnhancedVideoQASystem, video_id: str, user_query: str, video: object) -> Dict:
    """Django 뷰에서 사용할 향상된 채팅 응답 생성"""
    try:
        # 질의응답 수행
        qa_result = enhanced_qa.answer_question(video_id, user_query)
        
        # 검색 결과가 있는 경우 프레임 정보도 함께 제공
        frame_references = []
        if 'matched_results' in qa_result:
            frame_references = [
                {
                    'frame_id': result['frame_id'],
                    'timestamp': result['timestamp']
                }
                for result in qa_result['matched_results'][:3]
            ]
        elif 'scene_details' in qa_result:
            frame_references = [
                {
                    'frame_id': scene['frame_id'],
                    'timestamp': scene['timestamp']
                }
                for scene in qa_result['scene_details'][:3]
            ]
        
        return {
            'response_type': 'enhanced_qa_response',
            'response': qa_result['answer'],
            'query': user_query,
            'video_info': {'id': video.id, 'name': video.original_name},
            'question_category': qa_result.get('question_category', 'general'),
            'success': qa_result.get('success', True),
            'frame_references': frame_references,
            'additional_info': {
                k: v for k, v in qa_result.items() 
                if k not in ['answer', 'success', 'question_category']
            }
        }
        
    except Exception as e:
        return {
            'response_type': 'enhanced_qa_error',
            'response': f'질의응답 처리 중 오류가 발생했습니다: {str(e)}',
            'query': user_query,
            'video_info': {'id': video.id, 'name': video.original_name},
            'success': False,
            'error': str(e)
        }