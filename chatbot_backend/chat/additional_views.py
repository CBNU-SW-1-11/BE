# # additional_views.py - 추가 고급 분석 뷰들

# import os
# import json
# import time
# import cv2
# import numpy as np
# from django.conf import settings
# from django.http import JsonResponse, HttpResponse, FileResponse
# from django.views.decorators.csrf import csrf_exempt
# from django.views.decorators.http import require_http_methods
# from django.utils.decorators import method_decorator
# from django.views import View
# from django.core.files.storage import default_storage
# from django.core.files.base import ContentFile
# from rest_framework.views import APIView
# from rest_framework.response import Response
# from rest_framework import status
# from rest_framework.parsers import MultiPartParser, FormParser
# from rest_framework.permissions import AllowAny
# from datetime import datetime, timedelta
# from collections import Counter
# import threading
# import queue

# from .models import Video, VideoAnalysis, Scene, Frame, SearchHistory
# from .llm_client import LLMClient
# from .video_analyzer import VideoAnalyzer

# class AnalysisFeaturesView(APIView):
#     """분석 기능별 상세 정보 제공"""
#     permission_classes = [AllowAny]
    
#     def get(self, request):
#         try:
#             analyzer = VideoAnalyzer()
            
#             features = {
#                 'object_detection': {
#                     'name': '객체 감지',
#                     'description': 'YOLO 기반 실시간 객체 감지 및 분류',
#                     'available': True,
#                     'processing_time_factor': 1.0,
#                     'icon': '🎯',
#                     'details': '비디오 내 사람, 차량, 동물 등 다양한 객체를 정확하게 감지합니다.'
#                 },
#                 'clip_analysis': {
#                     'name': 'CLIP 씬 분석',
#                     'description': 'OpenAI CLIP 모델을 활용한 고급 씬 이해',
#                     'available': analyzer.clip_available,
#                     'processing_time_factor': 1.5,
#                     'icon': '🖼️',
#                     'details': '이미지의 의미적 컨텍스트를 이해하여 씬 분류 및 분석을 수행합니다.'
#                 },
#                 'ocr': {
#                     'name': 'OCR 텍스트 추출',
#                     'description': 'EasyOCR을 사용한 다국어 텍스트 인식',
#                     'available': analyzer.ocr_available,
#                     'processing_time_factor': 1.2,
#                     'icon': '📝',
#                     'details': '비디오 내 한글, 영문 텍스트를 정확하게 인식하고 추출합니다.'
#                 },
#                 'vqa': {
#                     'name': 'VQA 질문답변',
#                     'description': 'BLIP 모델 기반 시각적 질문 답변',
#                     'available': analyzer.vqa_available,
#                     'processing_time_factor': 2.0,
#                     'icon': '❓',
#                     'details': '이미지에 대한 질문을 생성하고 답변하여 깊이 있는 분석을 제공합니다.'
#                 },
#                 'scene_graph': {
#                     'name': 'Scene Graph',
#                     'description': '객체간 관계 및 상호작용 분석',
#                     'available': analyzer.scene_graph_available,
#                     'processing_time_factor': 3.0,
#                     'icon': '🕸️',
#                     'details': '객체들 사이의 관계와 상호작용을 분석하여 복잡한 씬을 이해합니다.'
#                 },
#                 'enhanced_caption': {
#                     'name': '고급 캡션 생성',
#                     'description': '모든 분석 결과를 통합한 상세 캡션',
#                     'available': True,
#                     'processing_time_factor': 1.1,
#                     'icon': '💬',
#                     'details': '여러 AI 모델의 결과를 종합하여 상세하고 정확한 캡션을 생성합니다.'
#                 }
#             }
            
#             return Response({
#                 'features': features,
#                 'device': analyzer.device,
#                 'total_available': sum(1 for f in features.values() if f['available']),
#                 'recommended_configs': {
#                     'basic': ['object_detection', 'enhanced_caption'],
#                     'enhanced': ['object_detection', 'clip_analysis', 'ocr', 'enhanced_caption'],
#                     'comprehensive': list(features.keys())
#                 }
#             })
            
#         except Exception as e:
#             return Response({
#                 'error': f'분석 기능 정보 조회 실패: {str(e)}'
#             }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# class AdvancedVideoSearchView(APIView):
#     """고급 비디오 검색 API"""
#     permission_classes = [AllowAny]
    
#     def __init__(self):
#         super().__init__()
#         self.video_analyzer = VideoAnalyzer()
#         self.llm_client = LLMClient()
    
#     def post(self, request):
#         try:
#             video_id = request.data.get('video_id')
#             query = request.data.get('query', '').strip()
#             search_options = request.data.get('search_options', {})
            
#             if not query:
#                 return Response({
#                     'error': '검색어를 입력해주세요.'
#                 }, status=status.HTTP_400_BAD_REQUEST)
            
#             video = Video.objects.get(id=video_id)
            
#             # 고급 검색 수행
#             search_results = self.video_analyzer.search_comprehensive(video, query)
            
#             # 고급 분석 결과가 포함된 프레임들에 대해 추가 정보 수집
#             enhanced_results = []
#             for result in search_results[:10]:
#                 frame_id = result.get('frame_id')
#                 try:
#                     frame = Frame.objects.get(video=video, image_id=frame_id)
#                     enhanced_result = dict(result)
                    
#                     # 고급 분석 결과 추가
#                     comprehensive_features = frame.comprehensive_features or {}
                    
#                     if search_options.get('include_clip_analysis') and 'clip_features' in comprehensive_features:
#                         enhanced_result['clip_analysis'] = comprehensive_features['clip_features']
                    
#                     if search_options.get('include_ocr_text') and 'ocr_text' in comprehensive_features:
#                         enhanced_result['ocr_text'] = comprehensive_features['ocr_text']
                    
#                     if search_options.get('include_vqa_results') and 'vqa_results' in comprehensive_features:
#                         enhanced_result['vqa_insights'] = comprehensive_features['vqa_results']
                    
#                     if search_options.get('include_scene_graph') and 'scene_graph' in comprehensive_features:
#                         enhanced_result['scene_graph'] = comprehensive_features['scene_graph']
                    
#                     enhanced_results.append(enhanced_result)
                    
#                 except Frame.DoesNotExist:
#                     enhanced_results.append(result)
            
#             # AI 기반 검색 인사이트 생성
#             search_insights = self._generate_search_insights(query, enhanced_results, video)
            
#             return Response({
#                 'search_results': enhanced_results,
#                 'query': query,
#                 'insights': search_insights,
#                 'total_matches': len(search_results),
#                 'search_type': 'advanced',
#                 'video_info': {
#                     'id': video.id,
#                     'name': video.original_name,
#                     'analysis_type': getattr(video, 'analysis_type', 'basic')
#                 }
#             })
            
#         except Video.DoesNotExist:
#             return Response({
#                 'error': '비디오를 찾을 수 없습니다.'
#             }, status=status.HTTP_404_NOT_FOUND)
#         except Exception as e:
#             return Response({
#                 'error': f'고급 검색 실패: {str(e)}'
#             }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
#     def _generate_search_insights(self, query, results, video):
#         """검색 결과에 대한 AI 인사이트 생성"""
#         try:
#             if not results:
#                 return "검색 결과가 없습니다. 다른 검색어를 시도해보세요."
            
#             # 검색 결과 요약
#             insights_prompt = f"""
#             검색어: "{query}"
#             비디오: {video.original_name}
#             검색 결과: {len(results)}개 매칭
            
#             주요 발견사항:
#             {json.dumps(results[:3], ensure_ascii=False, indent=2)}
            
#             이 검색 결과에 대한 간단하고 유용한 인사이트를 한국어로 제공해주세요.
#             """
            
#             insights = self.llm_client.generate_smart_response(
#                 user_query=insights_prompt,
#                 search_results=results[:5],
#                 video_info=f"비디오: {video.original_name}",
#                 use_multi_llm=False
#             )
            
#             return insights
            
#         except Exception as e:
#             return f"인사이트 생성 중 오류 발생: {str(e)}"


# class EnhancedFrameView(APIView):
#     """고급 분석 정보가 포함된 프레임 데이터 제공"""
#     permission_classes = [AllowAny]
    
#     def get(self, request, video_id, frame_number):
#         try:
#             video = Video.objects.get(id=video_id)
            
#             # 프레임 데이터 조회
#             try:
#                 frame = Frame.objects.get(video=video, image_id=frame_number)
                
#                 frame_data = {
#                     'frame_id': frame.image_id,
#                     'timestamp': frame.timestamp,
#                     'caption': frame.caption,
#                     'enhanced_caption': frame.enhanced_caption,
#                     'final_caption': frame.final_caption,
#                     'detected_objects': frame.detected_objects,
#                     'comprehensive_features': frame.comprehensive_features,
#                     'analysis_quality': frame.comprehensive_features.get('caption_quality', 'basic')
#                 }
                
#                 # 고급 분석 결과 분해
#                 if frame.comprehensive_features:
#                     features = frame.comprehensive_features
                    
#                     frame_data['advanced_analysis'] = {
#                         'clip_analysis': features.get('clip_features', {}),
#                         'ocr_text': features.get('ocr_text', {}),
#                         'vqa_results': features.get('vqa_results', {}),
#                         'scene_graph': features.get('scene_graph', {}),
#                         'scene_complexity': features.get('scene_complexity', 0)
#                     }
                
#                 return Response(frame_data)
                
#             except Frame.DoesNotExist:
#                 # 프레임 데이터가 없으면 기본 이미지만 반환
#                 return Response({
#                     'frame_id': frame_number,
#                     'message': '프레임 데이터는 없지만 이미지는 사용 가능합니다.',
#                     'image_url': f'/frame/{video_id}/{frame_number}/'
#                 })
            
#         except Video.DoesNotExist:
#             return Response({
#                 'error': '비디오를 찾을 수 없습니다.'
#             }, status=status.HTTP_404_NOT_FOUND)
#         except Exception as e:
#             return Response({
#                 'error': f'프레임 정보 조회 실패: {str(e)}'
#             }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# class EnhancedScenesView(APIView):
#     """고급 분석 정보가 포함된 씬 데이터 제공"""
#     permission_classes = [AllowAny]
    
#     def get(self, request, video_id):
#         try:
#             video = Video.objects.get(id=video_id)
#             scenes = Scene.objects.filter(video=video).order_by('scene_id')
            
#             enhanced_scenes = []
#             for scene in scenes:
#                 scene_data = {
#                     'scene_id': scene.scene_id,
#                     'start_time': scene.start_time,
#                     'end_time': scene.end_time,
#                     'duration': scene.duration,
#                     'frame_count': scene.frame_count,
#                     'dominant_objects': scene.dominant_objects,
#                     'enhanced_captions_count': scene.enhanced_captions_count,
#                     'caption_type': 'enhanced' if scene.enhanced_captions_count > 0 else 'basic'
#                 }
                
#                 # 씬 내 프레임들의 고급 분석 결과 집계
#                 scene_frames = Frame.objects.filter(
#                     video=video,
#                     timestamp__gte=scene.start_time,
#                     timestamp__lte=scene.end_time
#                 )
                
#                 if scene_frames.exists():
#                     # 고급 기능 사용 통계
#                     clip_count = sum(1 for f in scene_frames if f.comprehensive_features.get('clip_features'))
#                     ocr_count = sum(1 for f in scene_frames if f.comprehensive_features.get('ocr_text', {}).get('texts'))
#                     vqa_count = sum(1 for f in scene_frames if f.comprehensive_features.get('vqa_results'))
                    
#                     scene_data['advanced_features'] = {
#                         'clip_analysis_frames': clip_count,
#                         'ocr_text_frames': ocr_count,
#                         'vqa_analysis_frames': vqa_count,
#                         'total_frames': scene_frames.count()
#                     }
                    
#                     # 씬 복잡도 평균
#                     complexities = [f.comprehensive_features.get('scene_complexity', 0) for f in scene_frames]
#                     scene_data['average_complexity'] = sum(complexities) / len(complexities) if complexities else 0
                
#                 enhanced_scenes.append(scene_data)
            
#             return Response({
#                 'scenes': enhanced_scenes,
#                 'total_scenes': len(enhanced_scenes),
#                 'analysis_type': 'enhanced' if any(s.get('advanced_features') for s in enhanced_scenes) else 'basic',
#                 'video_info': {
#                     'id': video.id,
#                     'name': video.original_name
#                 }
#             })
            
#         except Video.DoesNotExist:
#             return Response({
#                 'error': '비디오를 찾을 수 없습니다.'
#             }, status=status.HTTP_404_NOT_FOUND)
#         except Exception as e:
#             return Response({
#                 'error': f'고급 씬 정보 조회 실패: {str(e)}'
#             }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# class AnalysisResultsView(APIView):
#     """종합 분석 결과 제공"""
#     permission_classes = [AllowAny]
    
#     def get(self, request, video_id):
#         try:
#             video = Video.objects.get(id=video_id)
            
#             if not video.is_analyzed:
#                 return Response({
#                     'error': '아직 분석이 완료되지 않았습니다.'
#                 }, status=status.HTTP_400_BAD_REQUEST)
            
#             analysis = video.analysis
#             scenes = Scene.objects.filter(video=video)
#             frames = Frame.objects.filter(video=video)
            
#             # 종합 분석 결과
#             results = {
#                 'video_info': {
#                     'id': video.id,
#                     'name': video.original_name,
#                     'duration': video.duration,
#                     'analysis_type': analysis.analysis_statistics.get('analysis_type', 'basic'),
#                     'processing_time': analysis.processing_time_seconds,
#                     'success_rate': analysis.success_rate
#                 },
#                 'analysis_summary': {
#                     'total_scenes': scenes.count(),
#                     'total_frames_analyzed': frames.count(),
#                     'unique_objects': analysis.analysis_statistics.get('unique_objects', 0),
#                     'features_used': analysis.analysis_statistics.get('features_used', []),
#                     'scene_types': analysis.analysis_statistics.get('scene_types', [])
#                 },
#                 'advanced_features': {
#                     'clip_analysis': analysis.analysis_statistics.get('clip_analysis', False),
#                     'ocr_text_extracted': analysis.analysis_statistics.get('text_extracted', False),
#                     'vqa_analysis': analysis.analysis_statistics.get('vqa_analysis', False),
#                     'scene_graph_analysis': analysis.analysis_statistics.get('scene_graph_analysis', False)
#                 },
#                 'content_insights': {
#                     'dominant_objects': analysis.analysis_statistics.get('dominant_objects', []),
#                     'text_content_length': analysis.caption_statistics.get('text_content_length', 0),
#                     'enhanced_captions_count': analysis.caption_statistics.get('enhanced_captions', 0),
#                     'average_confidence': analysis.caption_statistics.get('average_confidence', 0)
#                 }
#             }
            
#             # 프레임별 고급 분석 통계
#             if frames.exists():
#                 clip_frames = sum(1 for f in frames if f.comprehensive_features.get('clip_features'))
#                 ocr_frames = sum(1 for f in frames if f.comprehensive_features.get('ocr_text', {}).get('texts'))
#                 vqa_frames = sum(1 for f in frames if f.comprehensive_features.get('vqa_results'))
                
#                 results['frame_statistics'] = {
#                     'total_frames': frames.count(),
#                     'clip_analyzed_frames': clip_frames,
#                     'ocr_processed_frames': ocr_frames,
#                     'vqa_analyzed_frames': vqa_frames,
#                     'coverage': {
#                         'clip': (clip_frames / frames.count()) * 100 if frames.count() > 0 else 0,
#                         'ocr': (ocr_frames / frames.count()) * 100 if frames.count() > 0 else 0,
#                         'vqa': (vqa_frames / frames.count()) * 100 if frames.count() > 0 else 0
#                     }
#                 }
            
#             return Response(results)
            
#         except Video.DoesNotExist:
#             return Response({
#                 'error': '비디오를 찾을 수 없습니다.'
#             }, status=status.HTTP_404_NOT_FOUND)
#         except Exception as e:
#             return Response({
#                 'error': f'분석 결과 조회 실패: {str(e)}'
#             }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# class AnalysisSummaryView(APIView):
#     """분석 결과 요약 제공"""
#     permission_classes = [AllowAny]
    
#     def __init__(self):
#         super().__init__()
#         self.llm_client = LLMClient()
    
#     def get(self, request, video_id):
#         try:
#             video = Video.objects.get(id=video_id)
            
#             if not video.is_analyzed:
#                 return Response({
#                     'error': '아직 분석이 완료되지 않았습니다.'
#                 }, status=status.HTTP_400_BAD_REQUEST)
            
#             # 분석 결과 데이터 수집
#             analysis = video.analysis
#             frames = Frame.objects.filter(video=video)[:10]  # 상위 10개 프레임
            
#             # AI 기반 요약 생성
#             summary_data = {
#                 'video_name': video.original_name,
#                 'analysis_type': analysis.analysis_statistics.get('analysis_type', 'basic'),
#                 'features_used': analysis.analysis_statistics.get('features_used', []),
#                 'dominant_objects': analysis.analysis_statistics.get('dominant_objects', []),
#                 'scene_types': analysis.analysis_statistics.get('scene_types', []),
#                 'processing_time': analysis.processing_time_seconds
#             }
            
#             # 대표 프레임들의 캡션 수집
#             sample_captions = []
#             for frame in frames:
#                 if frame.final_caption:
#                     sample_captions.append(frame.final_caption)
            
#             summary_prompt = f"""
#             다음 비디오 분석 결과를 바탕으로 상세하고 유용한 요약을 작성해주세요:
            
#             비디오: {video.original_name}
#             분석 유형: {summary_data['analysis_type']}
#             사용된 기능: {', '.join(summary_data['features_used'])}
#             주요 객체: {', '.join(summary_data['dominant_objects'][:5])}
#             씬 유형: {', '.join(summary_data['scene_types'][:3])}
            
#             대표 캡션들:
#             {chr(10).join(sample_captions[:5])}
            
#             이 비디오의 주요 내용, 특징, 활용 방안을 포함하여 한국어로 요약해주세요.
#             """
            
#             ai_summary = self.llm_client.generate_smart_response(
#                 user_query=summary_prompt,
#                 search_results=None,
#                 video_info=f"비디오: {video.original_name}",
#                 use_multi_llm=True  # 고품질 요약을 위해 다중 LLM 사용
#             )
            
#             return Response({
#                 'video_id': video.id,
#                 'video_name': video.original_name,
#                 'ai_summary': ai_summary,
#                 'analysis_data': summary_data,
#                 'key_insights': {
#                     'total_objects': len(summary_data['dominant_objects']),
#                     'scene_variety': len(summary_data['scene_types']),
#                     'analysis_depth': len(summary_data['features_used']),
#                     'processing_efficiency': f"{summary_data['processing_time']}초"
#                 },
#                 'generated_at': datetime.now().isoformat()
#             })
            
#         except Video.DoesNotExist:
#             return Response({
#                 'error': '비디오를 찾을 수 없습니다.'
#             }, status=status.HTTP_404_NOT_FOUND)
#         except Exception as e:
#             return Response({
#                 'error': f'요약 생성 실패: {str(e)}'
#             }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# class AnalysisExportView(APIView):
#     """분석 결과 내보내기"""
#     permission_classes = [AllowAny]
    
#     def get(self, request, video_id):
#         try:
#             video = Video.objects.get(id=video_id)
            
#             if not video.is_analyzed:
#                 return Response({
#                     'error': '아직 분석이 완료되지 않았습니다.'
#                 }, status=status.HTTP_400_BAD_REQUEST)
            
#             export_format = request.GET.get('format', 'json')
            
#             # 전체 분석 데이터 수집
#             analysis = video.analysis
#             scenes = Scene.objects.filter(video=video)
#             frames = Frame.objects.filter(video=video)
            
#             export_data = {
#                 'export_info': {
#                     'video_id': video.id,
#                     'video_name': video.original_name,
#                     'export_date': datetime.now().isoformat(),
#                     'export_format': export_format
#                 },
#                 'video_metadata': {
#                     'filename': video.filename,
#                     'duration': video.duration,
#                     'file_size': video.file_size,
#                     'uploaded_at': video.uploaded_at.isoformat()
#                 },
#                 'analysis_metadata': {
#                     'analysis_type': analysis.analysis_statistics.get('analysis_type', 'basic'),
#                     'enhanced_analysis': analysis.enhanced_analysis,
#                     'success_rate': analysis.success_rate,
#                     'processing_time_seconds': analysis.processing_time_seconds,
#                     'features_used': analysis.analysis_statistics.get('features_used', [])
#                 },
#                 'scenes': [
#                     {
#                         'scene_id': scene.scene_id,
#                         'start_time': scene.start_time,
#                         'end_time': scene.end_time,
#                         'duration': scene.duration,
#                         'frame_count': scene.frame_count,
#                         'dominant_objects': scene.dominant_objects
#                     }
#                     for scene in scenes
#                 ],
#                 'frames': [
#                     {
#                         'frame_id': frame.image_id,
#                         'timestamp': frame.timestamp,
#                         'caption': frame.caption,
#                         'enhanced_caption': frame.enhanced_caption,
#                         'final_caption': frame.final_caption,
#                         'detected_objects': frame.detected_objects,
#                         'comprehensive_features': frame.comprehensive_features
#                     }
#                     for frame in frames
#                 ],
#                 'statistics': {
#                     'total_scenes': scenes.count(),
#                     'total_frames': frames.count(),
#                     'unique_objects': analysis.analysis_statistics.get('unique_objects', 0),
#                     'scene_types': analysis.analysis_statistics.get('scene_types', []),
#                     'dominant_objects': analysis.analysis_statistics.get('dominant_objects', [])
#                 }
#             }
            
#             if export_format == 'json':
#                 response = JsonResponse(export_data, json_dumps_params={'ensure_ascii': False, 'indent': 2})
#                 response['Content-Disposition'] = f'attachment; filename="{video.original_name}_analysis.json"'
#                 return response
            
#             elif export_format == 'csv':
#                 # CSV 형태로 프레임 데이터 내보내기
#                 import csv
#                 from io import StringIO
                
#                 output = StringIO()
#                 writer = csv.writer(output)
                
#                 # 헤더
#                 writer.writerow(['frame_id', 'timestamp', 'caption', 'enhanced_caption', 'objects_count', 'scene_complexity'])
                
#                 # 데이터
#                 for frame_data in export_data['frames']:
#                     writer.writerow([
#                         frame_data['frame_id'],
#                         frame_data['timestamp'],
#                         frame_data.get('caption', ''),
#                         frame_data.get('enhanced_caption', ''),
#                         len(frame_data.get('detected_objects', [])),
#                         frame_data.get('comprehensive_features', {}).get('scene_complexity', 0)
#                     ])
                
#                 response = HttpResponse(output.getvalue(), content_type='text/csv')
#                 response['Content-Disposition'] = f'attachment; filename="{video.original_name}_analysis.csv"'
#                 return response
            
#             else:
#                 return Response({
#                     'error': '지원하지 않는 내보내기 형식입니다. json 또는 csv를 사용해주세요.'
#                 }, status=status.HTTP_400_BAD_REQUEST)
            
#         except Video.DoesNotExist:
#             return Response({
#                 'error': '비디오를 찾을 수 없습니다.'
#             }, status=status.HTTP_404_NOT_FOUND)
#         except Exception as e:
#             return Response({
#                 'error': f'내보내기 실패: {str(e)}'
#             }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# # 검색 관련 뷰들
# class ObjectSearchView(APIView):
#     """객체별 검색"""
#     permission_classes = [AllowAny]
    
#     def get(self, request):
#         try:
#             object_type = request.GET.get('object', '')
#             video_id = request.GET.get('video_id')
            
#             if not object_type:
#                 return Response({
#                     'error': '검색할 객체 타입을 입력해주세요.'
#                 }, status=status.HTTP_400_BAD_REQUEST)
            
#             # 특정 비디오 또는 전체 비디오에서 검색
#             if video_id:
#                 videos = Video.objects.filter(id=video_id, is_analyzed=True)
#             else:
#                 videos = Video.objects.filter(is_analyzed=True)
            
#             results = []
#             for video in videos:
#                 frames = Frame.objects.filter(video=video)
                
#                 for frame in frames:
#                     for obj in frame.detected_objects:
#                         if object_type.lower() in obj.get('class', '').lower():
#                             results.append({
#                                 'video_id': video.id,
#                                 'video_name': video.original_name,
#                                 'frame_id': frame.image_id,
#                                 'timestamp': frame.timestamp,
#                                 'object_class': obj.get('class'),
#                                 'confidence': obj.get('confidence'),
#                                 'caption': frame.final_caption or frame.caption
#                             })
            
#             return Response({
#                 'search_query': object_type,
#                 'results': results[:50],  # 최대 50개 결과
#                 'total_matches': len(results)
#             })
            
#         except Exception as e:
#             return Response({
#                 'error': f'객체 검색 실패: {str(e)}'
#             }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# class TextSearchView(APIView):
#     """텍스트 검색 (OCR 결과 기반)"""
#     permission_classes = [AllowAny]
    
#     def get(self, request):
#         try:
#             search_text = request.GET.get('text', '')
#             video_id = request.GET.get('video_id')
            
#             if not search_text:
#                 return Response({
#                     'error': '검색할 텍스트를 입력해주세요.'
#                 }, status=status.HTTP_400_BAD_REQUEST)
            
#             # 특정 비디오 또는 전체 비디오에서 검색
#             if video_id:
#                 videos = Video.objects.filter(id=video_id, is_analyzed=True)
#             else:
#                 videos = Video.objects.filter(is_analyzed=True)
            
#             results = []
#             for video in videos:
#                 frames = Frame.objects.filter(video=video)
                
#                 for frame in frames:
#                     ocr_data = frame.comprehensive_features.get('ocr_text', {})
#                     if 'full_text' in ocr_data and search_text.lower() in ocr_data['full_text'].lower():
#                         results.append({
#                             'video_id': video.id,
#                             'video_name': video.original_name,
#                             'frame_id': frame.image_id,
#                             'timestamp': frame.timestamp,
#                             'extracted_text': ocr_data['full_text'],
#                             'text_details': ocr_data.get('texts', []),
#                             'caption': frame.final_caption or frame.caption
#                         })
            
#             return Response({
#                 'search_query': search_text,
#                 'results': results[:50],
#                 'total_matches': len(results)
#             })
            
#         except Exception as e:
#             return Response({
#                 'error': f'텍스트 검색 실패: {str(e)}'
#             }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)


# class SceneSearchView(APIView):
#     """씬 타입별 검색"""
#     permission_classes = [AllowAny]
    
#     def get(self, request):
#         try:
#             scene_type = request.GET.get('scene', '')
#             video_id = request.GET.get('video_id')
            
#             if not scene_type:
#                 return Response({
#                     'error': '검색할 씬 타입을 입력해주세요.'
#                 }, status=status.HTTP_400_BAD_REQUEST)
            
#             # 특정 비디오 또는 전체 비디오에서 검색
#             if video_id:
#                 videos = Video.objects.filter(id=video_id, is_analyzed=True)
#             else:
#                 videos = Video.objects.filter(is_analyzed=True)
            
#             results = []
#             for video in videos:
#                 if hasattr(video, 'analysis'):
#                     scene_types = video.analysis.analysis_statistics.get('scene_types', [])
#                     if any(scene_type.lower() in st.lower() for st in scene_types):
#                         results.append({
#                             'video_id': video.id,
#                             'video_name': video.original_name,
#                             'scene_types': scene_types,
#                             'analysis_type': video.analysis.analysis_statistics.get('analysis_type', 'basic'),
#                             'dominant_objects': video.analysis.analysis_statistics.get('dominant_objects', [])
#                         })
            
#             return Response({
#                 'search_query': scene_type,
#                 'results': results,
#                 'total_matches': len(results)
#             })
            
#         except Exception as e:
#             return Response({
#                 'error': f'씬 검색 실패: {str(e)}'
#             }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)