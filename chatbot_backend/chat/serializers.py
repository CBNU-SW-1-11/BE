# chat/serializers.py에 추가할 내용

from rest_framework import serializers
from django.contrib.auth.models import User
from .models import OCRResult  # 기존 import

# 기존 OCRResultSerializer는 그대로 두고 추가
# serializers.py
# serializers.py
from rest_framework import serializers
from .models import OCRResult

class OCRResultSerializer(serializers.ModelSerializer):
    class Meta:
        model = OCRResult
        fields = [
            'id', 
            'file', 
            'file_type', 
            'ocr_text', 
            'llm_response', 
            'llm_response_korean',
            'translation_enabled', 
            'translation_success', 
            'translation_model',
            'analysis_type', 
            'analyze_by_page', 
            'text_relevant',
            'created_at', 
            'updated_at'
        ]
        read_only_fields = ['created_at', 'updated_at']
    def to_representation(self, instance):
        """응답 데이터 커스터마이징"""
        data = super().to_representation(instance)
        
        # 번역 관련 필드 명시적 처리
        data['translation_enabled'] = getattr(instance, 'translation_enabled', False)
        data['translation_success'] = getattr(instance, 'translation_success', False)
        
        # 영어 원문
        data['llm_response'] = getattr(instance, 'llm_response', None)
        
        # 한국어 번역 (번역 성공 시만 포함)
        if data['translation_success']:
            data['llm_response_korean'] = getattr(instance, 'llm_response_korean', None)
        else:
            data['llm_response_korean'] = None
        
        return data

# 누락된 UserSettingsSerializer 추가
class UserSerializer(serializers.ModelSerializer):
    """사용자 설정 시리얼라이저"""
    
    class Meta:
        model = User
        fields = [
            'id',
            'username', 
            'email',
            'first_name',
            'last_name',
            'date_joined',
            'last_login'
        ]
        read_only_fields = ['id', 'username', 'date_joined', 'last_login']
    
    def update(self, instance, validated_data):
        """사용자 정보 업데이트"""
        instance.email = validated_data.get('email', instance.email)
        instance.first_name = validated_data.get('first_name', instance.first_name)
        instance.last_name = validated_data.get('last_name', instance.last_name)
        instance.save()
        return instance

# 추가적으로 필요할 수 있는 다른 시리얼라이저들
class UserProfileSerializer(serializers.Serializer):
    """사용자 프로필 정보 시리얼라이저"""
    username = serializers.CharField(read_only=True)
    email = serializers.EmailField()
    first_name = serializers.CharField(max_length=30, required=False)
    last_name = serializers.CharField(max_length=30, required=False)
    date_joined = serializers.DateTimeField(read_only=True)
    
class ChatSettingsSerializer(serializers.Serializer):
    """채팅 설정 시리얼라이저"""
    model_preference = serializers.CharField(max_length=50, required=False)
    language_preference = serializers.CharField(max_length=10, default='ko')
    theme = serializers.CharField(max_length=20, default='light')
    auto_translate = serializers.BooleanField(default=True)


# chat/serializers.py에 추가할 시리얼라이저들

from rest_framework import serializers
from .models import User, SocialAccount, Schedule, ScheduleRequest, ConflictResolution
import json
from datetime import datetime, timedelta

# 기존 UserSerializer, SocialAccountSerializer는 그대로 유지...

class ScheduleSerializer(serializers.ModelSerializer):
    attendees_list = serializers.SerializerMethodField()
    
    class Meta:
        model = Schedule
        fields = [
            'id', 'title', 'description', 'start_time', 'end_time', 
            'location', 'priority', 'status', 'attendees', 'attendees_list',
            'is_recurring', 'recurring_pattern', 'created_at', 'updated_at'
        ]
        read_only_fields = ('id', 'created_at', 'updated_at')
    
    def get_attendees_list(self, obj):
        try:
            return json.loads(obj.attendees) if obj.attendees else []
        except:
            return []
    
    def validate(self, data):
        if data['start_time'] >= data['end_time']:
            raise serializers.ValidationError("종료 시간은 시작 시간보다 늦어야 합니다.")
        return data

class ScheduleRequestSerializer(serializers.ModelSerializer):
    class Meta:
        model = ScheduleRequest
        fields = [
            'id', 'original_request', 'gpt_suggestion', 'claude_suggestion',
            'mixtral_suggestion', 'optimized_suggestion', 'confidence_score',
            'created_at'
        ]
        read_only_fields = ('id', 'created_at')

class ConflictResolutionSerializer(serializers.ModelSerializer):
    conflicting_schedules_data = serializers.SerializerMethodField()
    resolution_options_data = serializers.SerializerMethodField()
    ai_recommendations_data = serializers.SerializerMethodField()
    
    class Meta:
        model = ConflictResolution
        fields = [
            'id', 'conflicting_schedules', 'conflicting_schedules_data',
            'resolution_options', 'resolution_options_data',
            'selected_option', 'ai_recommendations', 'ai_recommendations_data',
            'created_at'
        ]
        read_only_fields = ('id', 'created_at')
    
    def get_conflicting_schedules_data(self, obj):
        try:
            return json.loads(obj.conflicting_schedules)
        except:
            return []
    
    def get_resolution_options_data(self, obj):
        try:
            return json.loads(obj.resolution_options)
        except:
            return []
    
    def get_ai_recommendations_data(self, obj):
        try:
            return json.loads(obj.ai_recommendations)
        except:
            return []

class ScheduleRequestInputSerializer(serializers.Serializer):
    """일정 생성 요청을 위한 입력 시리얼라이저"""
    request_text = serializers.CharField(max_length=1000)
    preferred_date = serializers.DateField(required=False)
    preferred_time = serializers.TimeField(required=False)
    duration_hours = serializers.FloatField(required=False, min_value=0.5, max_value=24)
    attendees = serializers.ListField(
        child=serializers.CharField(max_length=100),
        required=False,
        allow_empty=True
    )
    priority = serializers.ChoiceField(
        choices=Schedule.PRIORITY_CHOICES,
        required=False,
        default='MEDIUM'
    )