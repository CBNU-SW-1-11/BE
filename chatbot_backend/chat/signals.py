from django.db.models.signals import post_save
from django.dispatch import receiver
from django.contrib.auth.models import User
from .models import UserSettings
import logging

logger = logging.getLogger(__name__)

@receiver(post_save, sender=User)
def handle_user_settings(sender, instance, created, **kwargs):
    """
    User 모델이 저장될 때 UserSettings를 처리하는 단일 시그널 핸들러
    """
    try:
        # get_or_create를 사용하여 중복 생성 방지
        settings, created = UserSettings.objects.get_or_create(user=instance)
        if created:
            logger.info(f"Created new UserSettings for user {instance.id}")
    except Exception as e:
        logger.error(f"Error handling UserSettings for user {instance.id}: {str(e)}")

# save_user_settings 시그널 제거 (불필요)

