# chat/google_auth.py - Google Calendar 인증 처리

from django.shortcuts import redirect
from django.conf import settings
from django.contrib.auth.decorators import login_required
from django.http import JsonResponse
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from google_auth_oauthlib.flow import Flow
from googleapiclient.discovery import build
import json
import os

# Google OAuth 설정
GOOGLE_OAUTH_SCOPES = ['https://www.googleapis.com/auth/calendar']
GOOGLE_CLIENT_SECRETS_FILE = os.path.join(settings.BASE_DIR, 'google_credentials.json')

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def google_calendar_auth(request):
    """Google Calendar 인증 시작"""
    try:
        # OAuth 2.0 flow 생성
        flow = Flow.from_client_secrets_file(
            GOOGLE_CLIENT_SECRETS_FILE,
            scopes=GOOGLE_OAUTH_SCOPES,
            redirect_uri=request.build_absolute_uri('/api/calendar/auth/google/callback/')
        )
        
        # 인증 URL 생성
        auth_url, state = flow.authorization_url(
            access_type='offline',
            include_granted_scopes='true',
            prompt='consent'  # 매번 동의 화면 표시 (refresh_token 확보)
        )
        
        # state를 세션에 저장
        request.session['oauth_state'] = state
        
        return JsonResponse({
            'auth_url': auth_url,
            'message': '이 URL로 이동하여 Google Calendar 접근을 허용해주세요.'
        })
        
    except Exception as e:
        return JsonResponse({'error': f'인증 URL 생성 실패: {str(e)}'}, status=500)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def google_calendar_callback(request):
    """Google Calendar 인증 콜백"""
    try:
        # state 검증
        session_state = request.session.get('oauth_state')
        callback_state = request.GET.get('state')
        
        if not session_state or session_state != callback_state:
            return JsonResponse({'error': 'Invalid state parameter'}, status=400)
        
        # Authorization code 받기
        code = request.GET.get('code')
        if not code:
            return JsonResponse({'error': 'Authorization code not found'}, status=400)
        
        # OAuth flow 재생성
        flow = Flow.from_client_secrets_file(
            GOOGLE_CLIENT_SECRETS_FILE,
            scopes=GOOGLE_OAUTH_SCOPES,
            redirect_uri=request.build_absolute_uri('/api/calendar/auth/google/callback/')
        )
        flow.state = session_state
        
        # 토큰 교환
        flow.fetch_token(code=code)
        credentials = flow.credentials
        
        # 사용자 프로필에 토큰 저장
        user = request.user
        if not hasattr(user, 'profile'):
            from .models import UserProfile
            UserProfile.objects.create(user=user)
        
        user.profile.google_calendar_token = credentials.to_json()
        user.profile.save()
        
        # 캘린더 접근 테스트
        service = build('calendar', 'v3', credentials=credentials)
        calendar_list = service.calendarList().list().execute()
        
        return JsonResponse({
            'message': 'Google Calendar 연동이 완료되었습니다!',
            'calendars_count': len(calendar_list.get('items', [])),
            'redirect_url': '/schedule-management'  # 프론트엔드에서 리다이렉트할 URL
        })
        
    except Exception as e:
        return JsonResponse({'error': f'인증 콜백 처리 실패: {str(e)}'}, status=500)

@api_view(['GET'])
@permission_classes([IsAuthenticated])
def check_calendar_connection(request):
    """Google Calendar 연동 상태 확인"""
    try:
        user = request.user
        
        if not hasattr(user, 'profile') or not user.profile.google_calendar_token:
            return JsonResponse({
                'connected': False,
                'message': 'Google Calendar가 연동되지 않았습니다.'
            })
        
        # 토큰 유효성 검증
        from google.oauth2.credentials import Credentials
        from google.auth.transport.requests import Request
        
        token_info = json.loads(user.profile.google_calendar_token)
        credentials = Credentials.from_authorized_user_info(token_info, GOOGLE_OAUTH_SCOPES)
        
        if not credentials.valid:
            if credentials.expired and credentials.refresh_token:
                credentials.refresh(Request())
                user.profile.google_calendar_token = credentials.to_json()
                user.profile.save()
            else:
                return JsonResponse({
                    'connected': False,
                    'message': '토큰이 만료되었습니다. 재인증이 필요합니다.'
                })
        
        # 캘린더 정보 조회
        service = build('calendar', 'v3', credentials=credentials)
        calendar_list = service.calendarList().list().execute()
        primary_calendar = next(
            (cal for cal in calendar_list['items'] if cal.get('primary')), 
            None
        )
        
        return JsonResponse({
            'connected': True,
            'calendar_name': primary_calendar.get('summary', 'Primary Calendar') if primary_calendar else 'Unknown',
            'calendar_email': primary_calendar.get('id', '') if primary_calendar else '',
            'total_calendars': len(calendar_list.get('items', []))
        })
        
    except Exception as e:
        return JsonResponse({
            'connected': False,
            'error': f'연동 상태 확인 실패: {str(e)}'
        }, status=500)

@api_view(['POST'])
@permission_classes([IsAuthenticated])
def disconnect_calendar(request):
    """Google Calendar 연동 해제"""
    try:
        user = request.user
        
        if hasattr(user, 'profile'):
            user.profile.google_calendar_token = None
            user.profile.save()
        
        return JsonResponse({
            'message': 'Google Calendar 연동이 해제되었습니다.'
        })
        
    except Exception as e:
        return JsonResponse({'error': f'연동 해제 실패: {str(e)}'}, status=500)

