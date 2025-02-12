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
from django.contrib.auth import get_user_model
from django.utils.translation import gettext_lazy as _

User = get_user_model()

class SocialAccount(models.Model):
    user = models.ForeignKey(
        User,
        on_delete=models.CASCADE,
        related_name='social_accounts'
    )
    provider = models.CharField(max_length=30)
    email = models.EmailField()
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    nickname = models.CharField(max_length=255, blank=True, null=True)  # Add this line

    class Meta:
        unique_together = ('provider', 'email')
        verbose_name = _('소셜 계정')
        verbose_name_plural = _('소셜 계정들')

    def __str__(self):
        return f'{self.provider} - {self.email}'
