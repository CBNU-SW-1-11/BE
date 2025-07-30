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

from django.db import models
from django.contrib.auth.models import User
import json

class Video(models.Model):
    """비디오 파일 정보"""
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
    
    # 고급 분석 관련 추가 필드
    analysis_type = models.CharField(
        max_length=20,
        choices=[
            ('basic', 'Basic'),
            ('enhanced', 'Enhanced'),
            ('comprehensive', 'Comprehensive'),
            ('custom', 'Custom')
        ],
        default='enhanced'
    )
    
    # 사용된 고급 기능들
    features_used = models.JSONField(default=dict, blank=True)
    # 예: {"clip_analysis": True, "ocr": True, "vqa": False, "scene_graph": False}
    
    # 빠른 액세스를 위한 분석 요약 정보
    analysis_summary = models.JSONField(default=dict, blank=True)
    # 예: {"dominant_objects": ["person", "car"], "scene_types": ["outdoor", "urban"], "total_objects": 25}

    def __str__(self):
        return f"{self.original_name} ({self.analysis_status})"

    class Meta:
        ordering = ['-uploaded_at']


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