# auth/admin.py
from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import User, SocialAccount


@admin.register(SocialAccount)
class SocialAccountAdmin(admin.ModelAdmin):
    list_display = ('user', 'provider', 'email', 'created_at')
    list_filter = ('provider',)
    search_fields = ('user__email', 'email')
from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import User

# Unregister the default User model
from django.contrib.auth.models import User as DefaultUser
admin.site.unregister(DefaultUser)

# Register your custom User model
@admin.register(User)
class CustomUserAdmin(UserAdmin):
    # You can customize the UserAdmin options here
    list_display = ('username', 'email', 'first_name', 'last_name', 'is_active')
    search_fields = ('email', 'username',)
    ordering = ('email',)
from django.contrib import admin
from .models import OCRResult

@admin.register(OCRResult)
class OCRResultAdmin(admin.ModelAdmin):
    list_display = ('id', 'file_type', 'created_at', 'short_ocr_text')
    list_filter = ('file_type', 'created_at')
    search_fields = ('ocr_text', 'llm_response')
    readonly_fields = ('ocr_text', 'llm_response', 'created_at')
    
    def short_ocr_text(self, obj):
        """OCR 텍스트의 짧은 미리보기를 반환합니다"""
        if len(obj.ocr_text) > 50:
            return obj.ocr_text[:50] + "..."
        return obj.ocr_text
    short_ocr_text.short_description = "추출된 텍스트"