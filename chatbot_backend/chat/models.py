# auth/models.py
from django.db import models
from django.contrib.auth.models import AbstractUser
from django.utils.translation import gettext_lazy as _

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