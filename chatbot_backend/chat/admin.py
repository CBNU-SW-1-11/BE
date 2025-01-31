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
