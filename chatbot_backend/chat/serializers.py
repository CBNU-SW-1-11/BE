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