# auth/models.py
from django.db import models
from django.contrib.auth.models import AbstractUser
from django.utils.translation import gettext_lazy as _
import logging
from django.conf import settings

from django.contrib.auth.models import AbstractUser
from django.db import models

from django.db import models
from django.contrib.auth.models import AbstractUser
from django.utils.translation import gettext_lazy as _

class User(AbstractUser):
    email = models.EmailField(_('이메일 주소'), unique=True)
    username = models.CharField(  # username 필드 재정의
        _('username'),
        max_length=150,
        unique=False,  # unique 제약 제거
    )

    groups = models.ManyToManyField(
        'auth.Group', related_name='chat_user_set', blank=True
    )
    user_permissions = models.ManyToManyField(
        'auth.Permission', related_name='chat_user_permissions', blank=True
    )

    class Meta:
        verbose_name = _('사용자')
        verbose_name_plural = _('사용자들')

    def __str__(self):
        return self.email


from django.db import models
from django.conf import settings

class SocialAccount(models.Model):
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='social_accounts'
    )
    provider = models.CharField(max_length=20)  # 'google', 'kakao', 'naver' 등
    email = models.EmailField(unique=True)
    nickname = models.CharField(max_length=100, null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        unique_together = ['provider', 'email']
        verbose_name = '소셜 계정'
        verbose_name_plural = '소셜 계정들'

    def __str__(self):
        return f"{self.provider} - {self.email}"



from django.db import models
from django.contrib.auth.models import User

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    language = models.CharField(max_length=50, default='English (United States)')
    preferred_model = models.CharField(max_length=50, null=True, blank=True)
    
    def __str__(self):
        return f"{self.user.username}'s profile"
    
from django.db import models
from django.contrib.auth.models import User
from django.db.models.signals import post_save
from django.dispatch import receiver

class UserSettings(models.Model):
    user = models.OneToOneField(
        User,
        on_delete=models.CASCADE,
        related_name='user_settings',  # 명시적 related_name 추가
        db_constraint=True
    )
    language = models.CharField(max_length=50, default='en')
    preferred_model = models.CharField(max_length=50, default='default')

    class Meta:
        db_table = 'chat_user_settings' 
        
         # 명시적 테이블 이름 지정
@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    """User 생성 시 Profile과 Settings를 생성하는 시그널 핸들러"""
    try:
        # get_or_create를 사용하여 중복 생성 방지
        UserProfile.objects.get_or_create(user=instance)
        UserSettings.objects.get_or_create(user=instance)
    except Exception as e:
        logger.error(f"Error creating user profile/settings for user {instance.id}: {str(e)}")

# save_user_profile 핸들러는 제거

from django.db import models

from django.db import models

# 기존 모델들...

class OCRResult(models.Model):
    FILE_TYPE_CHOICES = [
        ('image', '이미지'),
        ('pdf', 'PDF'),
    ]
    
    file = models.FileField(upload_to='uploads/')
    file_type = models.CharField(max_length=10, choices=FILE_TYPE_CHOICES, default='image')
    ocr_text = models.TextField(blank=True)
    llm_response = models.TextField(blank=True)
    text_relevant = models.BooleanField(default=False)  # 추가된 필드
    created_at = models.DateTimeField(auto_now_add=True)
    llm_response = models.TextField(blank=True, null=True)
    llm_response_korean = models.TextField(blank=True, null=True)  # 이 필드 추가
    translation_enabled = models.BooleanField(default=False)
    translation_success = models.BooleanField(default=False)
    translation_model = models.CharField(max_length=50, blank=True, null=True)
    analysis_type = models.CharField(max_length=20, default='both')
    analyze_by_page = models.BooleanField(default=True)
    file = models.FileField(upload_to='ocr_files/')
    file_type = models.CharField(max_length=20)
    ocr_text = models.TextField(blank=True, null=True)
    llm_response = models.TextField(blank=True, null=True)
    llm_response_korean = models.TextField(blank=True, null=True)  # 한국어 번역
    
    # 번역 관련 필드
    translation_enabled = models.BooleanField(default=False)
    translation_success = models.BooleanField(default=False)
    translation_model = models.CharField(max_length=50, blank=True, null=True)
    
    # 분석 관련 필드
    analysis_type = models.CharField(max_length=20, default='both')
    analyze_by_page = models.BooleanField(default=True)
    text_relevant = models.BooleanField(default=False)
    
    # 타임스탬프 필드 추가
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        db_table = 'ocr_result'  # 기존 테이블명이 있다면 유지
        
    def __str__(self):
        return f"OCRResult {self.id} - {self.file_type}"


# chat/models.py에 추가할 모델들

from django.db import models
from django.contrib.auth.models import AbstractUser
from django.utils.translation import gettext_lazy as _
from django.conf import settings
import json

# 기존 User, SocialAccount 모델은 그대로 유지...

class Schedule(models.Model):
    PRIORITY_CHOICES = [
        ('LOW', '낮음'),
        ('MEDIUM', '보통'),
        ('HIGH', '높음'),
        ('URGENT', '긴급'),
    ]
    
    STATUS_CHOICES = [
        ('SCHEDULED', '예정'),
        ('IN_PROGRESS', '진행중'),
        ('COMPLETED', '완료'),
        ('CANCELLED', '취소'),
    ]
    
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='schedules'
    )
    title = models.CharField(max_length=200, verbose_name='제목')
    description = models.TextField(blank=True, verbose_name='설명')
    start_time = models.DateTimeField(verbose_name='시작 시간')
    end_time = models.DateTimeField(verbose_name='종료 시간')
    location = models.CharField(max_length=200, blank=True, verbose_name='장소')
    priority = models.CharField(
        max_length=10, 
        choices=PRIORITY_CHOICES, 
        default='MEDIUM',
        verbose_name='우선순위'
    )
    status = models.CharField(
        max_length=15,
        choices=STATUS_CHOICES,
        default='SCHEDULED',
        verbose_name='상태'
    )
    attendees = models.TextField(blank=True, verbose_name='참석자')  # JSON 형태로 저장
    is_recurring = models.BooleanField(default=False, verbose_name='반복 일정')
    recurring_pattern = models.CharField(max_length=50, blank=True, verbose_name='반복 패턴')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    

    class Meta:
        db_table = 'Schedule'  # 기존 테이블명이 있다면 유지
        verbose_name = _('일정')
        verbose_name_plural = _('일정들')
        ordering = ['start_time']
    
    def __str__(self):
        return f"{self.title} - {self.start_time.strftime('%Y-%m-%d %H:%M')}"

class ScheduleRequest(models.Model):
    """AI 모델들의 일정 제안을 저장하는 모델"""
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='schedule_requests'
    )
    original_request = models.TextField(verbose_name='원본 요청')
    gpt_suggestion = models.TextField(blank=True, verbose_name='GPT 제안')
    claude_suggestion = models.TextField(blank=True, verbose_name='Claude 제안')
    mixtral_suggestion = models.TextField(blank=True, verbose_name='Mixtral 제안')
    optimized_suggestion = models.TextField(blank=True, verbose_name='최적화된 제안')
    confidence_score = models.FloatField(default=0.0, verbose_name='신뢰도 점수')
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'ScheduleRequest'  # 기존 테이블명이 있다면 유지
        verbose_name = _('일정 요청')
        verbose_name_plural = _('일정 요청들')
        ordering = ['-created_at']

class ConflictResolution(models.Model):
    """일정 충돌 해결 방안을 저장하는 모델"""
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name='conflict_resolutions'
    )
    conflicting_schedules = models.TextField(verbose_name='충돌 일정들')  # JSON 형태
    resolution_options = models.TextField(verbose_name='해결 방안들')  # JSON 형태
    selected_option = models.TextField(blank=True, verbose_name='선택된 방안')
    ai_recommendations = models.TextField(verbose_name='AI 추천 사항')  # JSON 형태
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        db_table = 'ConflictResolution'  # 기존 테이블명이 있다면 유지
        verbose_name = _('충돌 해결')
        verbose_name_plural = _('충돌 해결들')
        ordering = ['-created_at']

# models.py - 고급 분석 기능을 위한 모델 확장
# chat/models.py - 비용 절약을 위한 모델 확장

from django.db import models
from django.contrib.auth.models import User
import json
import os
from datetime import datetime

# 기존 Video 모델에 추가 필드
class Video(models.Model):
    filename = models.CharField(max_length=255)
    original_name = models.CharField(max_length=255)
    file_path = models.CharField(max_length=500)
    file_size = models.BigIntegerField(default=0)
    duration = models.FloatField(default=0.0)
    uploaded_at = models.DateTimeField(auto_now_add=True)
    is_analyzed = models.BooleanField(default=False)
    analysis_status = models.CharField(
        max_length=20,
        choices=[
            ('pending', 'Pending'),
            ('processing', 'Processing'),
            ('completed', 'Completed'),
            ('failed', 'Failed')
        ],
        default='pending'
    )
    
    # 🔥 비용 절약을 위한 새로운 필드들
    image_analysis_completed = models.BooleanField(default=False)  # 이미지 분석 완료 여부
    image_analysis_date = models.DateTimeField(null=True, blank=True)  # 이미지 분석 완료 일시
    chat_analysis_json_path = models.CharField(max_length=500, blank=True)  # 채팅 분석 JSON 경로
    total_chat_count = models.IntegerField(default=0)  # 총 채팅 횟수
    
    # API 비용 추적
    api_cost_tracking = models.JSONField(default=dict, blank=True)
    # 예: {
    #   "total_api_calls": 5,
    #   "image_analysis_calls": 1,
    #   "text_only_calls": 4,
    #   "estimated_cost_usd": 0.15,
    #   "models_used": ["gpt-4o-mini", "claude-3.5"],
    #   "last_analysis_cost": 0.05
    # }

    def __str__(self):
        return f"{self.original_name} ({'분석완료' if self.image_analysis_completed else '분석대기'})"

    def get_analysis_cost_summary(self):
        """분석 비용 요약 반환"""
        costs = self.api_cost_tracking
        return {
            'total_calls': costs.get('total_api_calls', 0),
            'image_calls': costs.get('image_analysis_calls', 0),
            'text_calls': costs.get('text_only_calls', 0),
            'estimated_cost': costs.get('estimated_cost_usd', 0.0),
            'cost_per_chat': costs.get('estimated_cost_usd', 0.0) / max(self.total_chat_count, 1)
        }
    
    def increment_chat_count(self, is_image_analysis=False, estimated_cost=0.0):
        """채팅 카운트 및 비용 추적 증가"""
        self.total_chat_count += 1
        
        if not self.api_cost_tracking:
            self.api_cost_tracking = {}
        
        self.api_cost_tracking['total_api_calls'] = self.api_cost_tracking.get('total_api_calls', 0) + 1
        self.api_cost_tracking['estimated_cost_usd'] = self.api_cost_tracking.get('estimated_cost_usd', 0.0) + estimated_cost
        
        if is_image_analysis:
            self.api_cost_tracking['image_analysis_calls'] = self.api_cost_tracking.get('image_analysis_calls', 0) + 1
            self.image_analysis_completed = True
            self.image_analysis_date = datetime.now()
        else:
            self.api_cost_tracking['text_only_calls'] = self.api_cost_tracking.get('text_only_calls', 0) + 1
        
        self.save()

    class Meta:
        ordering = ['-uploaded_at']


class ChatSession(models.Model):
    """채팅 세션 추적 - 비용 분석용"""
    video = models.ForeignKey(Video, on_delete=models.CASCADE, related_name='chat_sessions')
    session_id = models.CharField(max_length=100)  # 브라우저 세션 ID
    
    # 세션 정보
    started_at = models.DateTimeField(auto_now_add=True)
    last_activity = models.DateTimeField(auto_now=True)
    total_messages = models.IntegerField(default=0)
    
    # 첫 번째 메시지 (이미지 분석)
    first_message = models.TextField(blank=True)
    first_message_cost = models.FloatField(default=0.0)
    image_analysis_performed = models.BooleanField(default=False)
    
    # 후속 메시지들 (텍스트 기반)
    subsequent_messages_cost = models.FloatField(default=0.0)
    
    # 사용된 모델들
    models_used = models.JSONField(default=list)
    
    def __str__(self):
        return f"Session {self.session_id} for {self.video.original_name}"
    
    @property
    def total_session_cost(self):
        return self.first_message_cost + self.subsequent_messages_cost
    
    @property
    def cost_per_message(self):
        if self.total_messages == 0:
            return 0
        return self.total_session_cost / self.total_messages

    class Meta:
        db_table = 'chat_sessions'
        indexes = [
            models.Index(fields=['session_id']),
            models.Index(fields=['started_at']),
        ]


class CostAnalysis(models.Model):
    """API 비용 분석 전용 모델"""
    
    # 일별/월별 집계
    date = models.DateField()
    period_type = models.CharField(
        max_length=10,
        choices=[('daily', '일별'), ('monthly', '월별'), ('yearly', '연별')],
        default='daily'
    )
    
    # 비용 통계
    total_api_calls = models.IntegerField(default=0)
    image_analysis_calls = models.IntegerField(default=0)  # 비싼 호출
    text_only_calls = models.IntegerField(default=0)       # 저렴한 호출
    
    # 모델별 사용량
    model_usage = models.JSONField(default=dict)
    # 예: {
    #   "gpt-4o-mini": {"calls": 10, "cost": 0.05},
    #   "claude-3.5": {"calls": 5, "cost": 0.12},
    #   "groq-llama": {"calls": 20, "cost": 0.01}
    # }
    
    # 비용 정보
    estimated_total_cost = models.FloatField(default=0.0)
    cost_by_type = models.JSONField(default=dict)
    # 예: {
    #   "image_analysis": 0.15,
    #   "text_generation": 0.03,
    #   "embedding": 0.01
    # }
    
    # 최적화 메트릭
    cost_efficiency_score = models.FloatField(default=0.0)  # 비용 대비 효율성
    savings_from_caching = models.FloatField(default=0.0)   # 캐싱으로 절약된 비용
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Cost Analysis {self.date} ({self.period_type}): ${self.estimated_total_cost:.3f}"
    
    @classmethod
    def get_daily_summary(cls, date):
        """특정 날짜의 비용 요약"""
        try:
            return cls.objects.get(date=date, period_type='daily')
        except cls.DoesNotExist:
            return None
    
    @classmethod
    def calculate_monthly_savings(cls, year, month):
        """월별 절약 효과 계산"""
        from django.db.models import Sum
        monthly_data = cls.objects.filter(
            date__year=year,
            date__month=month,
            period_type='daily'
        )
        
        total_cost = monthly_data.aggregate(Sum('estimated_total_cost'))['estimated_total_cost__sum'] or 0
        total_savings = monthly_data.aggregate(Sum('savings_from_caching'))['savings_from_caching__sum'] or 0
        
        return {
            'total_cost': total_cost,
            'total_savings': total_savings,
            'savings_percentage': (total_savings / (total_cost + total_savings)) * 100 if (total_cost + total_savings) > 0 else 0
        }

    class Meta:
        db_table = 'cost_analysis'
        unique_together = ['date', 'period_type']
        indexes = [
            models.Index(fields=['date', 'period_type']),
            models.Index(fields=['estimated_total_cost']),
        ]


class VideoAnalysis(models.Model):
    """비디오 분석 결과"""
    video = models.OneToOneField(Video, on_delete=models.CASCADE, related_name='analysis')
    enhanced_analysis = models.BooleanField(default=False)
    success_rate = models.FloatField(default=0.0)
    processing_time_seconds = models.IntegerField(default=0)
    
    # 기존 통계 정보
    analysis_statistics = models.JSONField(default=dict)
    caption_statistics = models.JSONField(default=dict)
    
    # 고급 분석 통계 추가
    advanced_statistics = models.JSONField(default=dict, blank=True)
    # 예: {
    #   "clip_frames_analyzed": 50,
    #   "ocr_text_found": 12,
    #   "vqa_questions_answered": 100,
    #   "scene_graph_complexity": 8.5,
    #   "total_processing_time_by_feature": {
    #     "clip": 120, "ocr": 80, "vqa": 200, "scene_graph": 300
    #   }
    # }
    
    # 분석 품질 메트릭
    quality_metrics = models.JSONField(default=dict, blank=True)
    # 예: {
    #   "caption_quality_score": 0.95,
    #   "object_detection_confidence": 0.87,
    #   "feature_coverage": 0.92,
    #   "overall_quality": "excellent"
    # }
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Analysis for {self.video.original_name}"


class Scene(models.Model):
    """비디오 씬 정보"""
    video = models.ForeignKey(Video, on_delete=models.CASCADE, related_name='scenes')
    scene_id = models.IntegerField()
    start_time = models.FloatField()
    end_time = models.FloatField()
    duration = models.FloatField()
    frame_count = models.IntegerField(default=0)
    dominant_objects = models.JSONField(default=list)
    enhanced_captions_count = models.IntegerField(default=0)
    
    # 고급 분석 정보 추가
    scene_type = models.CharField(max_length=100, blank=True)  # CLIP으로 분류된 씬 타입
    complexity_score = models.FloatField(default=0.0)  # 씬 복잡도
    
    # 씬별 고급 분석 통계
    advanced_features = models.JSONField(default=dict, blank=True)
    # 예: {
    #   "clip_scene_confidence": 0.92,
    #   "ocr_text_density": 0.15,
    #   "vqa_insights": ["indoor scene", "multiple people"],
    #   "object_relationships": 8,
    #   "temporal_consistency": 0.88
    # }

    class Meta:
        unique_together = ['video', 'scene_id']
        ordering = ['scene_id']

    def __str__(self):
        return f"Scene {self.scene_id} of {self.video.original_name}"


class Frame(models.Model):
    """개별 프레임 정보"""
    video = models.ForeignKey(Video, on_delete=models.CASCADE, related_name='frames')
    image_id = models.IntegerField()
    timestamp = models.FloatField()
    
    # 기본 캡션들
    caption = models.TextField(blank=True)
    enhanced_caption = models.TextField(blank=True)
    final_caption = models.TextField(blank=True)
    
    # 새로 추가: 고급 분석별 캡션
    clip_caption = models.TextField(blank=True)  # CLIP 기반 캡션
    vqa_caption = models.TextField(blank=True)   # VQA 기반 캡션
    
    # 감지된 객체들
    detected_objects = models.JSONField(default=list)
    
    # 고급 분석 결과를 위한 확장된 필드
    comprehensive_features = models.JSONField(default=dict, blank=True)
    # 예: {
    #   "clip_features": {
    #     "scene_type": "outdoor",
    #     "confidence": 0.94,
    #     "top_matches": [{"description": "a photo of people", "confidence": 0.89}]
    #   },
    #   "ocr_text": {
    #     "texts": [{"text": "STOP", "confidence": 0.98, "bbox": [...]}],
    #     "full_text": "STOP SIGN",
    #     "language": "영어"
    #   },
    #   "vqa_results": {
    #     "qa_pairs": [{"question": "What is the main subject?", "answer": "traffic sign"}],
    #     "summary": {"main_subject": "traffic sign", "location": "street"}
    #   },
    #   "scene_graph": {
    #     "objects": [...],
    #     "relationships": [...],
    #     "complexity_score": 6
    #   },
    #   "caption_quality": "enhanced",
    #   "scene_complexity": 7,
    #   "processing_time": {"clip": 0.5, "ocr": 0.3, "vqa": 2.1}
    # }
    
    # 프레임 품질 메트릭
    quality_score = models.FloatField(default=0.0)  # 전체적인 분석 품질 점수
    
    # BLIP 캡션 (기존 유지)
    blip_caption = models.TextField(blank=True)

    class Meta:
        unique_together = ['video', 'image_id']
        ordering = ['image_id']

    def __str__(self):
        return f"Frame {self.image_id} of {self.video.original_name}"
    
    def get_best_caption(self):
        """가장 좋은 품질의 캡션 반환"""
        if self.final_caption:
            return self.final_caption
        elif self.enhanced_caption:
            return self.enhanced_caption
        elif self.vqa_caption:
            return self.vqa_caption
        elif self.clip_caption:
            return self.clip_caption
        elif self.caption:
            return self.caption
        else:
            return self.blip_caption
    
    def get_analysis_features_used(self):
        """이 프레임에서 사용된 고급 분석 기능들 반환"""
        features = []
        if self.comprehensive_features.get('clip_features'):
            features.append('CLIP')
        if self.comprehensive_features.get('ocr_text', {}).get('texts'):
            features.append('OCR')
        if self.comprehensive_features.get('vqa_results'):
            features.append('VQA')
        if self.comprehensive_features.get('scene_graph'):
            features.append('Scene Graph')
        return features


class SearchHistory(models.Model):
    """검색 기록 (고급 검색 지원)"""
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    video = models.ForeignKey(Video, on_delete=models.CASCADE)
    query = models.TextField()
    search_type = models.CharField(
        max_length=20,
        choices=[
            ('basic', 'Basic Search'),
            ('advanced', 'Advanced Search'),
            ('object', 'Object Search'),
            ('text', 'Text Search'),
            ('scene', 'Scene Search'),
            ('semantic', 'Semantic Search')
        ],
        default='basic'
    )
    
    # 검색 옵션 및 필터
    search_options = models.JSONField(default=dict, blank=True)
    # 예: {
    #   "include_clip_analysis": True,
    #   "include_ocr_text": True,
    #   "time_range": {"start": 0, "end": 120},
    #   "confidence_threshold": 0.8
    # }
    
    # 검색 결과 요약
    results_summary = models.JSONField(default=dict, blank=True)
    # 예: {
    #   "total_matches": 15,
    #   "best_match_score": 0.94,
    #   "search_quality": "excellent",
    #   "processing_time": 1.2
    # }
    
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Search: {self.query[:50]}... ({self.search_type})"


class AnalysisProgress(models.Model):
    """분석 진행률 추적 (데이터베이스 기반)"""
    video = models.OneToOneField(Video, on_delete=models.CASCADE, related_name='progress')
    
    # 기본 진행률 정보
    progress_percentage = models.FloatField(default=0.0)
    current_step = models.CharField(max_length=200, blank=True)
    estimated_time_remaining = models.FloatField(null=True, blank=True)
    
    # 고급 분석 진행률
    analysis_type = models.CharField(max_length=20, default='enhanced')
    current_feature = models.CharField(max_length=50, blank=True)
    completed_features = models.JSONField(default=list)
    total_features = models.IntegerField(default=4)
    
    # 프레임 처리 진행률
    processed_frames = models.IntegerField(default=0)
    total_frames = models.IntegerField(default=0)
    
    # 각 기능별 처리 시간
    feature_processing_times = models.JSONField(default=dict, blank=True)
    # 예: {"clip": 120.5, "ocr": 45.2, "vqa": 210.1}
    
    # 진행률 상세 로그
    progress_log = models.JSONField(default=list, blank=True)
    # 예: [
    #   {"timestamp": "2024-01-01T10:00:00", "step": "초기화 완료", "progress": 5},
    #   {"timestamp": "2024-01-01T10:01:00", "step": "CLIP 분석 시작", "progress": 10}
    # ]
    
    started_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    completed_at = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return f"Progress for {self.video.original_name}: {self.progress_percentage}%"
    
    def add_progress_log(self, step, progress):
        """진행률 로그 추가"""
        from datetime import datetime
        if not self.progress_log:
            self.progress_log = []
        
        self.progress_log.append({
            "timestamp": datetime.now().isoformat(),
            "step": step,
            "progress": progress
        })
        self.save()
    
    def get_feature_progress(self):
        """기능별 진행률 계산"""
        if self.total_features == 0:
            return 0
        return (len(self.completed_features) / self.total_features) * 100


class AnalysisTemplate(models.Model):
    """분석 템플릿 (사용자 정의 분석 설정 저장)"""
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True)
    user = models.ForeignKey(User, on_delete=models.CASCADE, null=True, blank=True)
    
    # 분석 설정
    analysis_config = models.JSONField(default=dict)
    # 예: {
    #   "object_detection": True,
    #   "clip_analysis": True,
    #   "ocr": True,
    #   "vqa": False,
    #   "scene_graph": False,
    #   "enhanced_caption": True,
    #   "quality_threshold": 0.8,
    #   "frame_sampling_rate": 1.0
    # }
    
    # 사용 통계
    usage_count = models.IntegerField(default=0)
    
    is_public = models.BooleanField(default=False)  # 다른 사용자들과 공유 가능
    is_system_template = models.BooleanField(default=False)  # 시스템 기본 템플릿
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-usage_count', 'name']

    def __str__(self):
        return f"Template: {self.name}"
    
    def get_enabled_features(self):
        """활성화된 기능들 반환"""
        return [key for key, value in self.analysis_config.items() if value is True]


class VideoInsight(models.Model):
    """AI 생성 비디오 인사이트"""
    video = models.ForeignKey(Video, on_delete=models.CASCADE, related_name='insights')
    
    insight_type = models.CharField(
        max_length=30,
        choices=[
            ('summary', 'Summary'),
            ('highlights', 'Highlights'),
            ('objects', 'Object Analysis'),
            ('scenes', 'Scene Analysis'),
            ('text', 'Text Analysis'),
            ('temporal', 'Temporal Analysis'),
            ('comparative', 'Comparative Analysis')
        ]
    )
    
    # AI 생성 인사이트 내용
    title = models.CharField(max_length=200)
    content = models.TextField()
    
    # 인사이트 메타데이터
    confidence_score = models.FloatField(default=0.0)
    relevance_score = models.FloatField(default=0.0)
    
    # 인사이트 생성에 사용된 데이터
    source_data = models.JSONField(default=dict, blank=True)
    # 예: {
    #   "frames_analyzed": [1, 5, 10, 20],
    #   "features_used": ["clip", "vqa"],
    #   "llm_model": "gpt-4",
    #   "prompt_version": "v2.1"
    # }
    
    # 사용자 피드백
    user_rating = models.IntegerField(null=True, blank=True)  # 1-5 점
    user_feedback = models.TextField(blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-relevance_score', '-created_at']

    def __str__(self):
        return f"{self.insight_type} insight for {self.video.original_name}: {self.title}"
    

# models.py - 기존 모델에 추가할 새로운 모델들

from django.db import models
import json

# 기존 Video, VideoAnalysis, Scene, Frame 모델은 그대로 유지하고 다음 모델들을 추가

class VideoMetadata(models.Model):
    """비디오 메타데이터 - 날씨, 시간대, 장소 등"""
    video = models.OneToOneField('Video', on_delete=models.CASCADE, related_name='metadata')
    
    # 날씨/환경 정보
    dominant_weather = models.CharField(max_length=50, default='unknown')
    weather_confidence = models.FloatField(default=0.0)
    weather_analysis = models.JSONField(default=dict)
    
    # 시간대 정보
    dominant_time_period = models.CharField(max_length=50, default='unknown')
    time_confidence = models.FloatField(default=0.0)
    estimated_hour_range = models.CharField(max_length=50, default='unknown')
    
    # 전반적인 비디오 특성
    overall_brightness = models.FloatField(default=0.0)
    overall_contrast = models.FloatField(default=0.0)
    scene_complexity = models.FloatField(default=0.0)
    
    # 주요 색상 테마
    dominant_colors = models.JSONField(default=list)
    color_diversity = models.FloatField(default=0.0)
    
    # 장소/환경 추정
    location_type = models.CharField(max_length=100, default='unknown')  # indoor, outdoor, street, park 등
    location_confidence = models.FloatField(default=0.0)
    
    # 분석 메타데이터
    analysis_timestamp = models.DateTimeField(auto_now_add=True)
    analysis_version = models.CharField(max_length=20, default='1.0')
    
    def __str__(self):
        return f"Metadata for {self.video.original_name}"
    
    class Meta:
        db_table = 'video_metadata'


class PersonDetection(models.Model):
    """사람 감지 및 속성 정보"""
    frame = models.ForeignKey('Frame', on_delete=models.CASCADE, related_name='person_detections')
    
    # 기본 감지 정보
    person_id = models.IntegerField()  # 프레임 내 사람 ID
    track_id = models.IntegerField(null=True, blank=True)  # 추적 ID (같은 사람)
    
    # 바운딩 박스 정보
    bbox_x1 = models.FloatField()
    bbox_y1 = models.FloatField()
    bbox_x2 = models.FloatField()
    bbox_y2 = models.FloatField()
    confidence = models.FloatField()
    
    # 외형 속성
    gender_estimation = models.CharField(max_length=20, default='unknown')  # male, female, unknown
    gender_confidence = models.FloatField(default=0.0)
    
    age_group = models.CharField(max_length=30, default='unknown')  # child, young_adult, adult, elderly
    age_confidence = models.FloatField(default=0.0)
    
    # 의상 정보
    upper_body_color = models.CharField(max_length=30, default='unknown')
    upper_color_confidence = models.FloatField(default=0.0)
    lower_body_color = models.CharField(max_length=30, default='unknown')
    lower_color_confidence = models.FloatField(default=0.0)
    
    # 자세/행동 정보
    posture = models.CharField(max_length=50, default='unknown')  # standing, sitting, walking 등
    posture_confidence = models.FloatField(default=0.0)
    
    # 상세 분석 데이터 (JSON)
    detailed_attributes = models.JSONField(default=dict)
    
    # 추가 메타데이터
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Person {self.person_id} in Frame {self.frame.image_id}"
    
    class Meta:
        db_table = 'person_detections'
        indexes = [
            models.Index(fields=['track_id']),
            models.Index(fields=['gender_estimation', 'gender_confidence']),
            models.Index(fields=['upper_body_color']),
        ]


class TemporalStatistics(models.Model):
    """시간별 통계 정보"""
    video = models.ForeignKey('Video', on_delete=models.CASCADE, related_name='temporal_stats')
    
    # 시간 구간 정보
    start_timestamp = models.FloatField()  # 시작 시간 (초)
    end_timestamp = models.FloatField()    # 종료 시간 (초)
    duration = models.FloatField()         # 구간 길이 (초)
    
    # 사람 관련 통계
    total_persons_detected = models.IntegerField(default=0)
    unique_persons_estimated = models.IntegerField(default=0)
    
    # 성별 분포
    male_count = models.IntegerField(default=0)
    female_count = models.IntegerField(default=0)
    unknown_gender_count = models.IntegerField(default=0)
    gender_analysis_confidence = models.FloatField(default=0.0)
    
    # 나이 분포
    child_count = models.IntegerField(default=0)
    adult_count = models.IntegerField(default=0)
    elderly_count = models.IntegerField(default=0)
    
    # 활동 밀도
    activity_density = models.FloatField(default=0.0)  # 단위 시간당 사람 수
    movement_intensity = models.FloatField(default=0.0)  # 움직임 강도
    
    # 의상 색상 분포 (상위 3개)
    top_colors = models.JSONField(default=list)
    
    # 상세 통계 데이터
    detailed_statistics = models.JSONField(default=dict)
    
    # 분석 메타데이터
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Stats {self.start_timestamp:.1f}-{self.end_timestamp:.1f}s for {self.video.original_name}"
    
    @property
    def gender_ratio(self):
        """성비 계산"""
        total = self.male_count + self.female_count
        if total == 0:
            return {'male_percentage': 0, 'female_percentage': 0}
        return {
            'male_percentage': (self.male_count / total) * 100,
            'female_percentage': (self.female_count / total) * 100
        }
    
    class Meta:
        db_table = 'temporal_statistics'
        indexes = [
            models.Index(fields=['start_timestamp', 'end_timestamp']),
            models.Index(fields=['activity_density']),
        ]


class ObjectTracking(models.Model):
    """객체 추적 정보"""
    video = models.ForeignKey('Video', on_delete=models.CASCADE, related_name='object_tracks')
    
    # 추적 기본 정보
    track_id = models.IntegerField()
    object_class = models.CharField(max_length=50)
    
    # 시간 범위
    first_appearance = models.FloatField()  # 첫 등장 시간
    last_appearance = models.FloatField()   # 마지막 등장 시간
    total_duration = models.FloatField()    # 총 등장 시간
    
    # 추적 품질
    tracking_confidence = models.FloatField(default=0.0)
    total_detections = models.IntegerField(default=0)
    tracking_quality = models.CharField(max_length=20, default='medium')  # high, medium, low
    
    # 이동 정보
    movement_path = models.JSONField(default=list)  # 중심점 좌표들
    movement_distance = models.FloatField(default=0.0)
    average_speed = models.FloatField(default=0.0)
    
    # 객체 속성 (사람인 경우)
    person_attributes = models.JSONField(null=True, blank=True)
    
    # 상호작용 정보  
    interacted_objects = models.JSONField(default=list)  # 상호작용한 다른 객체들
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Track {self.track_id}: {self.object_class} in {self.video.original_name}"
    
    class Meta:
        db_table = 'object_tracking'
        unique_together = ['video', 'track_id', 'object_class']
        indexes = [
            models.Index(fields=['track_id']),
            models.Index(fields=['object_class']),
            models.Index(fields=['first_appearance', 'last_appearance']),
        ]


class VideoSearchIndex(models.Model):
    """비디오 검색 인덱스 - 빠른 검색을 위한 전처리된 데이터"""
    video = models.OneToOneField('Video', on_delete=models.CASCADE, related_name='search_index')
    
    # 텍스트 검색용
    searchable_text = models.TextField(default='')  # 모든 캡션, OCR 텍스트 통합
    keywords = models.JSONField(default=list)       # 추출된 키워드들
    
    # 객체 검색용
    all_objects = models.JSONField(default=list)    # 감지된 모든 객체들
    dominant_objects = models.JSONField(default=list)  # 주요 객체들
    object_counts = models.JSONField(default=dict)  # 객체별 등장 횟수
    
    # 사람 검색용
    person_attributes_summary = models.JSONField(default=dict)  # 사람 속성 요약
    clothing_colors = models.JSONField(default=list)           # 의상 색상들
    gender_distribution = models.JSONField(default=dict)       # 성별 분포
    
    # 시간/환경 검색용
    weather_tags = models.JSONField(default=list)     # 날씨 태그들
    time_tags = models.JSONField(default=list)        # 시간대 태그들  
    location_tags = models.JSONField(default=list)    # 장소 태그들
    
    # 시각적 특성
    visual_features = models.JSONField(default=dict)  # 색상, 밝기, 대비 등
    scene_complexity_avg = models.FloatField(default=0.0)
    
    # 검색 가중치 (검색 품질 향상용)
    search_weights = models.JSONField(default=dict)
    
    # 인덱스 메타데이터
    last_updated = models.DateTimeField(auto_now=True)
    index_version = models.CharField(max_length=20, default='1.0')
    
    def __str__(self):
        return f"Search Index for {self.video.original_name}"
    
    class Meta:
        db_table = 'video_search_index'


class SearchQuery(models.Model):
    """검색 쿼리 로그 및 결과 캐싱"""
    query_text = models.TextField()
    query_hash = models.CharField(max_length=64, unique=True)  # 쿼리 해시값
    
    # 검색 조건
    search_conditions = models.JSONField(default=dict)
    
    # 검색 결과
    matching_videos = models.JSONField(default=list)
    result_count = models.IntegerField(default=0)
    
    # 성능 메트릭
    search_duration_ms = models.IntegerField(default=0)
    cache_hit = models.BooleanField(default=False)
    
    # 사용 통계
    usage_count = models.IntegerField(default=1)
    last_used = models.DateTimeField(auto_now=True)
    created_at = models.DateTimeField(auto_now_add=True)
    
    def __str__(self):
        return f"Query: {self.query_text[:50]}..."
    
    class Meta:
        db_table = 'search_queries'
        indexes = [
            models.Index(fields=['query_hash']),
            models.Index(fields=['last_used']),
        ]