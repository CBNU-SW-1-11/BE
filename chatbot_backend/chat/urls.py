from django import views
from django.urls import path
from .views import ChatView, ProcessFileView, TextSimplificationView, UserSettingsView, google_callback, naver_callback
from django.urls import path, include

from django.urls import path
from . import views




urlpatterns = [
    path('api/auth/kakao/callback/', views.kakao_callback, name='kakao_callback'),
    path('chat/<str:preferredModel>/', ChatView.as_view(), name='chat'),
    path('auth/naver/callback/', naver_callback, name='naver_callback'),
    # path('api/analyze_responses/', views.analyze_responses, name='analyze_responses'),

    # path('analyze-image/', ImageAnalysisView.as_view(), name='analyze_image'),

    path('api/auth/', include('dj_rest_auth.urls')),
    path('ocr/process-file/', ProcessFileView.as_view(), name='process-file'),
    # path('ocr/results/<int:pk>/', OCRResultDetailView.as_view(), name='ocr-result-detail'),
    # path('ocr/results/', OCRResultListView.as_view(), name='ocr-result-list'),
    
    path('api/auth/registration/', include('dj_rest_auth.registration.urls')),
    
    path('api/simplify/', TextSimplificationView.as_view(), name='simplify-text'),

    path('accounts/google/callback/', views.google_callback, name='google_callback'),
    path('api/auth/google/callback', views.google_callback, name='google_callback'),

    path('api/user-settings/', UserSettingsView.as_view(), name='user_settings'),
    path('auth/google/callback/', views.google_callback, name='google_callback'),
    path('api/auth/google/callback/', google_callback, name='google_callback'),

    path('api/user/settings/', views.update_user_settings, name='update_user_settings'),
    # 사용자 정보 조회 엔드포인트

    # dj-rest-auth의 URL 연결
    path('auth/', include('dj_rest_auth.urls')),
    path('auth/registration/', include('dj_rest_auth.registration.urls')),
    
]