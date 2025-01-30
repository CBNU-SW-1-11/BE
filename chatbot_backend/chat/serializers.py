# chat/serializers.py
from rest_framework import serializers
from django.contrib.auth import get_user_model

User = get_user_model()

class UserSerializer(serializers.ModelSerializer):
    nickname = serializers.SerializerMethodField()

    class Meta:
        model = User
        fields = ('id', 'username', 'email', 'nickname')

    def get_nickname(self, obj):
        # 소셜 계정의 닉네임을 우선적으로 반환
        social_account = obj.social_accounts.filter(provider='kakao').first()
        if social_account and hasattr(social_account, 'nickname') and social_account.nickname:
            return social_account.nickname
        return obj.username

    def to_representation(self, instance):
        # 직렬화 과정 로깅
        data = super().to_representation(instance)
        social_account = instance.social_accounts.filter(provider='kakao').first()
        if social_account:
            print(f"Social Account found: {social_account.nickname}")
        return data