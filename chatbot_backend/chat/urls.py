# # # from django import views
# # # from django.urls import path
# # # from .views import ChatView, ProcessFileView, ScheduleManagementView, confirm_schedule
# # # from django.urls import path, include

# # # from django.urls import path



# # # # from chatbot_backend.chat.similarity_analyzer import ChatView
# # # from . import views




# # # urlpatterns = [
# # #     path('api/auth/kakao/callback/', views.kakao_callback, name='kakao_callback'),
# # #     path('chat/<str:preferredModel>/', ChatView.as_view(), name='chat'),
# # #     path('auth/naver/callback/', views.naver_callback, name='naver_callback'),
# # #     # path('api/analyze_responses/', views.analyze_responses, name='analyze_responses'),
# # #     path('', views.ScheduleManagementView.as_view(), name='schedule-management'),
        
# # #         # 🔧 일정 확정
# # #     path('confirm/<int:request_id>/', views.confirm_schedule, name='confirm-schedule'),
        
# # #         # 🔧 수동 일정 생성
# # #     path('create/', views.create_manual_schedule, name='create-manual-schedule'),
        
# # #     # 🔧 일정 수정/삭제
# # #     path('<int:schedule_id>/', views.manage_schedule, name='manage-schedule'),
# # #     # path('analyze-image/', ImageAnalysisView.as_view(), name='analyze_image'),

# # #     path('api/auth/', include('dj_rest_auth.urls')),
# # #     path('ocr/process-file/', ProcessFileView.as_view(), name='process-file'),
# # #     # path('ocr/results/<int:pk>/', OCRResultDetailView.as_view(), name='ocr-result-detail'),
# # #     # path('ocr/results/', OCRResultListView.as_view(), name='ocr-result-list'),
    
# # #     path('api/auth/registration/', include('dj_rest_auth.registration.urls')),
    
# # #     # path('api/simplify/', TextSimplificationView.as_view(), name='simplify-text'),

# # #     path('accounts/google/callback/', views.google_callback, name='google_callback'),
# # #     path('api/auth/google/callback', views.google_callback, name='google_callback'),

# # #     # path('api/user-settings/', UserSettingsView.as_view(), name='user_settings'),
# # #     path('auth/google/callback/', views.google_callback, name='google_callback'),
# # #     # path('api/auth/google/callback/', google_callback, name='google_callback'),

# # #     path('api/user/settings/', views.update_user_settings, name='update_user_settings'),
# # #     # 사용자 정보 조회 엔드포인트

# # #     # dj-rest-auth의 URL 연결
# # #     path('auth/', include('dj_rest_auth.urls')),
# # #     path('auth/registration/', include('dj_rest_auth.registration.urls')),
# # #     path('api/schedule/<int:schedule_id>/', views.manage_schedule),
# # #     path('api/schedule/', ScheduleManagementView.as_view(), name='schedule_management'),
# # #     path('api/schedule/confirm/<int:request_id>/', confirm_schedule, name='confirm_schedule'),
# # #     # path('api/schedule/resolve-conflict/', resolve_schedule_conflict, name='resolve_schedule_conflict'),
    
# # # ]

# # from django import views
# # from django.urls import path
# # from .views import ChatView, ProcessFileView, ScheduleManagementView, confirm_schedule
# # from django.urls import path, include

# # from django.urls import path

# # # 비디오 관련 뷰 import (새로 추가)
# # from .views import (
# #     VideoListView, 
# #     VideoUploadView, 
# #     VideoChatView, 
# #     FrameView, 
# #     ScenesView, 
# #     APIStatusView, 
# #     AnalyzeVideoView, 
# #     AnalysisStatusView
# # )

# # from . import views

# # urlpatterns = [
# #     # 기존 인증 및 채팅 관련 URL들
# #     path('api/auth/kakao/callback/', views.kakao_callback, name='kakao_callback'),
# #     path('chat/<str:preferredModel>/', ChatView.as_view(), name='chat'),  # 기존 채팅 기능
# #     path('auth/naver/callback/', views.naver_callback, name='naver_callback'),
    
# #     # 스케줄 관리 관련 URL들
# #     path('', views.ScheduleManagementView.as_view(), name='schedule-management'),
# #     path('confirm/<int:request_id>/', views.confirm_schedule, name='confirm-schedule'),
# #     path('create/', views.create_manual_schedule, name='create-manual-schedule'),
# #     path('<int:schedule_id>/', views.manage_schedule, name='manage-schedule'),
    
# #     # 인증 관련 URL들
# #     path('api/auth/', include('dj_rest_auth.urls')),
# #     path('ocr/process-file/', ProcessFileView.as_view(), name='process-file'),
# #     path('api/auth/registration/', include('dj_rest_auth.registration.urls')),
    
# #     # Google 인증 관련
# #     path('accounts/google/callback/', views.google_callback, name='google_callback'),
# #     path('api/auth/google/callback', views.google_callback, name='google_callback'),
# #     path('auth/google/callback/', views.google_callback, name='google_callback'),
    
# #     # 사용자 설정 관련
# #     path('api/user/settings/', views.update_user_settings, name='update_user_settings'),
    
# #     # dj-rest-auth URL들
# #     path('auth/', include('dj_rest_auth.urls')),
# #     path('auth/registration/', include('dj_rest_auth.registration.urls')),
    
# #     # 스케줄 API 관련
# #     path('api/schedule/<int:schedule_id>/', views.manage_schedule),
# #     path('api/schedule/', ScheduleManagementView.as_view(), name='schedule_management'),
# #     path('api/schedule/confirm/<int:request_id>/', confirm_schedule, name='confirm_schedule'),
    
# #     # ========== 비디오 관련 URL들 (새로 추가) ==========
# #     path('videos/', VideoListView.as_view(), name='video_list'),
# #     path('upload_video/', VideoUploadView.as_view(), name='upload_video'),
# #     path('video/chat/', VideoChatView.as_view(), name='video_chat'),
# #     path('analyze_video/', AnalyzeVideoView.as_view(), name='analyze_video'),
# #     path('analysis_status/<int:video_id>/', AnalysisStatusView.as_view(), name='analysis_status'),
# #     path('frame/<int:video_id>/<int:frame_number>/', FrameView.as_view(), name='frame_normal'),
# #     path('frame/<int:video_id>/<int:frame_number>/<str:frame_type>/', FrameView.as_view(), name='frame_with_type'),
# #     path('scenes/<int:video_id>/', ScenesView.as_view(), name='scenes'),
# #     path('api_status/', APIStatusView.as_view(), name='api_status'),
# # ]

# # urls.py - 수정된 URL 패턴

# from django import views
# from django.urls import path, include
# from .views import ChatView, ProcessFileView, ScheduleManagementView, confirm_schedule

# # 비디오 관련 뷰 import
# from .views import (
#     VideoListView, 
#     VideoUploadView, 
#     VideoChatView, 
#     FrameView, 
#     ScenesView, 
#     APIStatusView, 
#     AnalyzeVideoView, 
#     AnalysisStatusView
# )

# from . import views

# urlpatterns = [
#     # ========== 비디오 관련 URL들 (맨 위로 이동) ==========
#     path('videos/', VideoListView.as_view(), name='video_list'),
#     path('upload_video/', VideoUploadView.as_view(), name='upload_video'),
#     path('video/chat/', VideoChatView.as_view(), name='video_chat'),
#     path('analyze_video/', AnalyzeVideoView.as_view(), name='analyze_video'),
#     path('analysis_status/<int:video_id>/', AnalysisStatusView.as_view(), name='analysis_status'),
#     path('frame/<int:video_id>/<int:frame_number>/', FrameView.as_view(), name='frame_normal'),
#     path('frame/<int:video_id>/<int:frame_number>/<str:frame_type>/', FrameView.as_view(), name='frame_with_type'),
#     path('scenes/<int:video_id>/', ScenesView.as_view(), name='scenes'),
#     path('api_status/', APIStatusView.as_view(), name='api_status'),  # 🔧 이 URL이 맞는지 확인
#     path('analysis_progress/<int:video_id>/', views.AnalysisProgressView.as_view(), name='analysis_progress'),
#     # 기존 인증 및 채팅 관련 URL들
#     path('api/auth/kakao/callback/', views.kakao_callback, name='kakao_callback'),
#     path('chat/<str:preferredModel>/', ChatView.as_view(), name='chat'),  # 기존 채팅 기능
#     path('auth/naver/callback/', views.naver_callback, name='naver_callback'),
    
#     # 스케줄 관리 관련 URL들
#     path('', views.ScheduleManagementView.as_view(), name='schedule-management'),
#     path('confirm/<int:request_id>/', views.confirm_schedule, name='confirm-schedule'),
#     path('create/', views.create_manual_schedule, name='create-manual-schedule'),
#     path('<int:schedule_id>/', views.manage_schedule, name='manage-schedule'),
    
#     # 인증 관련 URL들
#     path('api/auth/', include('dj_rest_auth.urls')),
#     path('ocr/process-file/', ProcessFileView.as_view(), name='process-file'),
#     path('api/auth/registration/', include('dj_rest_auth.registration.urls')),
    
#     # Google 인증 관련
#     path('accounts/google/callback/', views.google_callback, name='google_callback'),
#     path('api/auth/google/callback', views.google_callback, name='google_callback'),
#     path('auth/google/callback/', views.google_callback, name='google_callback'),
    
#     # 사용자 설정 관련
#     path('api/user/settings/', views.update_user_settings, name='update_user_settings'),
    
#     # dj-rest-auth URL들
#     path('auth/', include('dj_rest_auth.urls')),
#     path('auth/registration/', include('dj_rest_auth.registration.urls')),
    
#     # 스케줄 API 관련
#     path('api/schedule/<int:schedule_id>/', views.manage_schedule),
#     path('api/schedule/', ScheduleManagementView.as_view(), name='schedule_management'),
#     path('api/schedule/confirm/<int:request_id>/', confirm_schedule, name='confirm_schedule'),
# ]
# urls.py - 고급 분석 엔드포인트 추가

from django import views
from django.urls import path, include
from .views import ChatView, ProcessFileView, ScheduleManagementView, confirm_schedule
# from .additional_views import AnalysisFeaturesView
# 비디오 관련 뷰 import (기존 + 새로운 고급 분석 뷰들)
from .views import (
    VideoListView, 
    VideoUploadView, 
    VideoChatView, 
    FrameView, 
    ScenesView, 
    APIStatusView, 
    AnalyzeVideoView, 
    AnalysisStatusView,
    AnalysisFeaturesView,
    # 새로 추가된 고급 분석 뷰들
    EnhancedAnalyzeVideoView,
    AnalysisCapabilitiesView,
    EnhancedVideoChatView,
    AnalysisProgressView,
    CrossVideoSearchView,
    IntraVideoTrackingView,
    TimeBasedAnalysisView,
    AdvancedSearchAutoView,
)

from . import views

urlpatterns = [
    # ========== 고급 비디오 분석 관련 URL들 (우선순위) ==========
    
    # 기본 비디오 관련 엔드포인트들
    path('search/cross-video/', CrossVideoSearchView.as_view(), name='cross_video_search'),
    path('search/object-tracking/', IntraVideoTrackingView.as_view(), name='object_tracking'),
    path('analysis/time-based/', TimeBasedAnalysisView.as_view(), name='time_based_analysis'),
    path('search/advanced/', AdvancedSearchAutoView.as_view(), name='advanced_search_auto'),
    path('videos/<int:video_id>/delete/', views.delete_video, name='delete_video'),
    path('videos/<int:video_id>/', views.video_detail, name='video_detail'),
    path('videos/<int:video_id>/exists/', views.check_video_exists, name='check_video_exists'),

    path('videos/', VideoListView.as_view(), name='video_list'),
    path('upload_video/', VideoUploadView.as_view(), name='upload_video'),
    path('api_status/', APIStatusView.as_view(), name='api_status'),
    
    # 고급 분석 관련 새로운 엔드포인트들
    path('analyze_video_enhanced/', EnhancedAnalyzeVideoView.as_view(), name='analyze_video_enhanced'),
    path('analysis_capabilities/', AnalysisCapabilitiesView.as_view(), name='analysis_capabilities'),
    path('analysis_features/', AnalysisFeaturesView.as_view(), name='analysis_features'),
    
    # 분석 상태 및 진행률 관련
    path('analysis_status/<int:video_id>/', AnalysisStatusView.as_view(), name='analysis_status'),
    path('analysis_progress/<int:video_id>/', AnalysisProgressView.as_view(), name='analysis_progress'),
    
    # 채팅 관련 (기존 + 고급)
    path('video/chat/', VideoChatView.as_view(), name='video_chat'),
    path('video/chat/enhanced/', EnhancedVideoChatView.as_view(), name='video_chat_enhanced'),
    path('video/search/advanced/', views.AdvancedVideoSearchView.as_view(), name='video_search_advanced'),
    
    # 프레임 및 씬 관련
    path('frame/<int:video_id>/<int:frame_number>/', FrameView.as_view(), name='frame_normal'),
    path('frame/<int:video_id>/<int:frame_number>/<str:frame_type>/', FrameView.as_view(), name='frame_with_type'),
    path('frame/<int:video_id>/<int:frame_number>/enhanced/', views.EnhancedFrameView.as_view(), name='frame_enhanced'),
    path('scenes/<int:video_id>/', ScenesView.as_view(), name='scenes'),
    path('scenes/<int:video_id>/enhanced/', views.EnhancedScenesView.as_view(), name='scenes_enhanced'),
    
    # 분석 결과 관련
    path('analysis_results/<int:video_id>/', views.AnalysisResultsView.as_view(), name='analysis_results'),
    path('analysis_summary/<int:video_id>/', views.AnalysisSummaryView.as_view(), name='analysis_summary'),
    path('analysis_export/<int:video_id>/', views.AnalysisExportView.as_view(), name='analysis_export'),
    
    # 고급 검색 및 필터링
    path('search/objects/', views.ObjectSearchView.as_view(), name='object_search'),
    path('search/text/', views.TextSearchView.as_view(), name='text_search'),
    path('search/scenes/', views.SceneSearchView.as_view(), name='scene_search'),
    
    # 통계 및 인사이트
    # path('analytics/overview/', views.AnalyticsOverviewView.as_view(), name='analytics_overview'),
    # path('analytics/comparison/', views.AnalyticsComparisonView.as_view(), name='analytics_comparison'),
    
    # 기존 분석 시작 (호환성 유지)
    path('analyze_video/', AnalyzeVideoView.as_view(), name='analyze_video'),
    
    # ========== 기존 인증 및 채팅 관련 URL들 ==========
    path('api/auth/kakao/callback/', views.kakao_callback, name='kakao_callback'),
    path('chat/<str:preferredModel>/', ChatView.as_view(), name='chat'),  # 기존 채팅 기능
    path('auth/naver/callback/', views.naver_callback, name='naver_callback'),
    
    # 스케줄 관리 관련 URL들
    path('', views.ScheduleManagementView.as_view(), name='schedule-management'),
    path('confirm/<int:request_id>/', views.confirm_schedule, name='confirm-schedule'),
    path('create/', views.create_manual_schedule, name='create-manual-schedule'),
    path('<int:schedule_id>/', views.manage_schedule, name='manage-schedule'),
    
    # 인증 관련 URL들
    path('api/auth/', include('dj_rest_auth.urls')),
    path('ocr/process-file/', ProcessFileView.as_view(), name='process-file'),
    path('api/auth/registration/', include('dj_rest_auth.registration.urls')),
    
    # Google 인증 관련
    path('accounts/google/callback/', views.google_callback, name='google_callback'),
    path('api/auth/google/callback', views.google_callback, name='google_callback'),
    path('auth/google/callback/', views.google_callback, name='google_callback'),
    
    # 사용자 설정 관련
    path('api/user/settings/', views.update_user_settings, name='update_user_settings'),
    
    # dj-rest-auth URL들
    path('auth/', include('dj_rest_auth.urls')),
    path('auth/registration/', include('dj_rest_auth.registration.urls')),
    
    # 스케줄 API 관련
    path('api/schedule/<int:schedule_id>/', views.manage_schedule),
    path('api/schedule/', ScheduleManagementView.as_view(), name='schedule_management'),
    path('api/schedule/confirm/<int:request_id>/', confirm_schedule, name='confirm_schedule'),

    path('videos/search/advanced/', views.AdvancedVideoSearchView.as_view(), name='advanced-video-search'),
    
    # 특화된 검색 엔드포인트들
    path('search/objects/', views.ObjectSearchView.as_view(), name='object-search'),
    path('search/text/', views.TextSearchView.as_view(), name='text-search'),
    path('search/scenes/', views.SceneSearchView.as_view(), name='scene-search'),
    
    # 고급 프레임 분석 정보
    path('frame/<int:video_id>/<int:frame_number>/enhanced/', views.EnhancedFrameView.as_view(), name='enhanced-frame'),
    path('frame/<int:video_id>/<int:frame_number>/bbox/', views.FrameWithBboxView.as_view(), name='frame-with-bbox'),
    
    # 고급 씬 정보
    path('scenes/<int:video_id>/enhanced/', views.EnhancedScenesView.as_view(), name='enhanced-scenes'),
    
    # 분석 결과 관련
    path('analysis_results/<int:video_id>/', views.AnalysisResultsView.as_view(), name='analysis-results'),
    path('analysis_summary/<int:video_id>/', views.AnalysisSummaryView.as_view(), name='analysis-summary'),
    path('analysis_export/<int:video_id>/', views.AnalysisExportView.as_view(), name='analysis-export'),
    
    # 분석 기능 정보
    path('analysis_features/', views.AnalysisFeaturesView.as_view(), name='analysis-features'),

    
]
