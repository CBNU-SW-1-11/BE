from django import views
from django.urls import path
from .views import ChatView, ScheduleManagementView, confirm_schedule
from django.urls import path, include

from django.urls import path



# from chatbot_backend.chat.similarity_analyzer import ChatView
from . import views




urlpatterns = [
    path('api/auth/kakao/callback/', views.kakao_callback, name='kakao_callback'),
    path('chat/<str:preferredModel>/', ChatView.as_view(), name='chat'),
    path('auth/naver/callback/', views.naver_callback, name='naver_callback'),
    # path('api/analyze_responses/', views.analyze_responses, name='analyze_responses'),
    path('', views.ScheduleManagementView.as_view(), name='schedule-management'),
        
        # 🔧 일정 확정
    path('confirm/<int:request_id>/', views.confirm_schedule, name='confirm-schedule'),
        
        # 🔧 수동 일정 생성
    path('create/', views.create_manual_schedule, name='create-manual-schedule'),
        
    # 🔧 일정 수정/삭제
    path('<int:schedule_id>/', views.manage_schedule, name='manage-schedule'),
    # path('analyze-image/', ImageAnalysisView.as_view(), name='analyze_image'),

    path('api/auth/', include('dj_rest_auth.urls')),
    # path('ocr/process-file/', ProcessFileView.as_view(), name='process-file'),
    # path('ocr/results/<int:pk>/', OCRResultDetailView.as_view(), name='ocr-result-detail'),
    # path('ocr/results/', OCRResultListView.as_view(), name='ocr-result-list'),
    
    path('api/auth/registration/', include('dj_rest_auth.registration.urls')),
    
    # path('api/simplify/', TextSimplificationView.as_view(), name='simplify-text'),

    path('accounts/google/callback/', views.google_callback, name='google_callback'),
    path('api/auth/google/callback', views.google_callback, name='google_callback'),

    # path('api/user-settings/', UserSettingsView.as_view(), name='user_settings'),
    path('auth/google/callback/', views.google_callback, name='google_callback'),
    # path('api/auth/google/callback/', google_callback, name='google_callback'),

    path('api/user/settings/', views.update_user_settings, name='update_user_settings'),
    # 사용자 정보 조회 엔드포인트

    # dj-rest-auth의 URL 연결
    path('auth/', include('dj_rest_auth.urls')),
    path('auth/registration/', include('dj_rest_auth.registration.urls')),
    path('api/schedule/<int:schedule_id>/', views.manage_schedule),
    path('api/schedule/', ScheduleManagementView.as_view(), name='schedule_management'),
    path('api/schedule/confirm/<int:request_id>/', confirm_schedule, name='confirm_schedule'),
    # path('api/schedule/resolve-conflict/', resolve_schedule_conflict, name='resolve_schedule_conflict'),
    
]