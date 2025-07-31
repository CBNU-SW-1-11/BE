import os
from pathlib import Path
import os
from dotenv import load_dotenv
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")


BASE_DIR = Path(__file__).resolve().parent.parent

DEBUG = True
APPEND_SLASH = True  # (기본값: True)


ALLOWED_HOSTS = ['*']  # 필요한 도메인 추가

ROOT_URLCONF = 'chatbot_backend.urls'
STATIC_URL = '/static/'
AUTHENTICATION_BACKENDS = (
    'django.contrib.auth.backends.ModelBackend',
    'allauth.account.auth_backends.AuthenticationBackend',
)

AUTHENTICATION_BACKENDS = (
    'allauth.account.auth_backends.AuthenticationBackend',
    'django.contrib.auth.backends.ModelBackend',
    
)
SITE_ID = 1
LOGIN_REDIRECT_URL = '/'

# settings.py

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
        },
    },
    'root': {
        'handlers': ['console'],
        'level': 'INFO',
    },
    'loggers': {
        'django': {
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': True,
        },
    },

}
SOCIAL_AUTH_PIPELINE = (
    'social_core.pipeline.social_auth.social_details',
    'social_core.pipeline.social_auth.social_uid',
    'social_core.pipeline.social_auth.auth_allowed',
    'social_core.pipeline.social_auth.social_user',
    'social_core.pipeline.user.get_username',
    'social_core.pipeline.user.create_user',  # 여기서 DB에 저장
    'social_core.pipeline.social_auth.associate_user',
    'social_core.pipeline.social_auth.load_extra_data',
    'social_core.pipeline.user.user_details',
)
# settings.py
REST_FRAMEWORK = {
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.AllowAny',
        'rest_framework.authentication.TokenAuthentication',
        'rest_framework.authentication.SessionAuthentication',
    ],
}
from datetime import timedelta
SIMPLE_JWT = {
    'ACCESS_TOKEN_LIFETIME': timedelta(hours=1),
    'REFRESH_TOKEN_LIFETIME': timedelta(days=1),
}
GOOGLE_OAUTH2_SCOPES = [
    'https://www.googleapis.com/auth/userinfo.email',
    'https://www.googleapis.com/auth/userinfo.profile',
    'openid'
]
SOCIAL_AUTH = {
    'REACT_APP_KAKAO_CLIENT_ID': '8bfca9df8364fead1243d41c773ec5a2',
    'REACT_APP_KAKAO_REDIRECT_URI': 'http://localhost:3000/auth/kakao/callback',
    'GOOGLE_CLIENT_ID': '94821981810-32iorb0jccvsdi4jq3pp3mc6rvmb0big.apps.googleusercontent.com',
    'GOOGLE_SECRET_KEY': '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',
    'GOOGLE_REDIRECT_URI': 'http://localhost:8000/api/auth/google/callback'
}
# settings.py
NAVER_CLIENT_ID = 'ZjUi3kZpQGpMuGCeQStJ'
NAVER_CLIENT_SECRET = 'SXK_WdyPuU'
NAVER_REDIRECT_URI = 'http://localhost:3000/auth/naver/callback'

AUTHENTICATION_BACKENDS = (
    'social_core.backends.google.GoogleOAuth2',
    'django.contrib.auth.backends.ModelBackend',  # Keep the default backend
)


# Add necessary REST Framework and OAuth2 settings
REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework_social_oauth2.authentication.SocialAuthentication',
    ],
}

AUTHENTICATION_BACKENDS = [
    'django.contrib.auth.backends.ModelBackend',
    'allauth.account.auth_backends.AuthenticationBackend'
]

SITE_ID = 1


# Social Auth 설정
SOCIALACCOUNT_PROVIDERS = {
    'google': {
        'APP': {
            'client_id': 'your-google-client-id',
            'secret': 'your-google-client-secret',
            'key': ''
        },
        'SCOPE': [
            'profile',
            'email',
        ],
    },
    'kakao': {
        'APP': {
            'client_id': 'your-kakao-client-id',
            'secret': 'your-kakao-client-secret',
            'key': ''
        }
    }
}



# CORS 설정
CORS_ALLOW_ALL_ORIGINS = True  # 개발 환경에서만 사용
CORS_ALLOW_CREDENTIALS = True
# 소셜 로그인 설정
KAKAO_CLIENT_ID = 'b577d53567193b28d0b393c91c043123'
KAKAO_REDIRECT_URI = os.getenv('KAKAO_REDIRECT_URI')

GOOGLE_CLIENT_ID = os.getenv('GOOGLE_CLIENT_ID')
GOOGLE_SECRET_KEY = os.getenv('GOOGLE_SECRET_KEY')
GOOGLE_REDIRECT_URI = os.getenv('GOOGLE_REDIRECT_URI')

# CORS 설정


REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework.authentication.SessionAuthentication',
        'rest_framework.authentication.TokenAuthentication',
    ],
}
ANALYZER_BOT = 'claude'  # 기본값을 'claude'로 설정

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.mysql',
        'NAME': 'AIofAI',  # 원하는 데이터베이스 이름
        'USER': 'root',                # MySQL 사용자명
        'PASSWORD': 'k13976376',   # MySQL 비밀번호
        'HOST': 'localhost',
        'PORT': '3306',
        'OPTIONS': {
            'charset': 'utf8mb4',
            'init_command': "SET NAMES 'utf8mb4'"
    }
    }
}


REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': (
        'rest_framework.authentication.SessionAuthentication',  # 기본 세션 인증
        'rest_framework_simplejwt.authentication.JWTAuthentication',  # JWT 인증
    ),
    'DEFAULT_PERMISSION_CLASSES': (
        'rest_framework.permissions.IsAuthenticated',  # 인증된 사용자만 접근 가능
    ),
}

# REST API 기본 설정
REST_USE_JWT = True
ACCOUNT_EMAIL_VERIFICATION = 'mandatory'
ACCOUNT_EMAIL_REQUIRED = True

INSTALLED_APPS = [
    
    'django.contrib.admin',
    
    'django.contrib.auth',
    'django.contrib.contenttypes',
    'django.contrib.sessions',
    'django.contrib.messages',
    'django.contrib.staticfiles',
    'rest_framework',
    'rest_framework.authtoken',
    'corsheaders',
    'django.contrib.sites',

 # Token 인증 사용 시
    
    'rest_framework_social_oauth2',
    'rest_framework_simplejwt',

    'allauth',
    'allauth.account',
    'allauth.socialaccount',
    'allauth.socialaccount.providers.google',
    'allauth.socialaccount.providers.kakao',
    
    

    'dj_rest_auth',


    'chat.apps.ChatConfig',
]
SIMPLE_JWT = {
    "AUTH_HEADER_TYPES": ("Bearer",),
}
REST_USE_JWT = True
AUTH_USER_MODEL = 'auth.User' 
SOCIAL_AUTH_GOOGLE_OAUTH2_KEY = '94821981810-ftg2njljaurf7p50vpgs24bimkc7mfcg.apps.googleusercontent.com'
SOCIAL_AUTH_GOOGLE_OAUTH2_SECRET = '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'
MIDDLEWARE = [
    'corsheaders.middleware.CorsMiddleware',  # 최상단에 위치
    'django.middleware.security.SecurityMiddleware',
    'django.contrib.sessions.middleware.SessionMiddleware',
    'django.middleware.common.CommonMiddleware',
    'django.middleware.csrf.CsrfViewMiddleware',
    'django.contrib.auth.middleware.AuthenticationMiddleware',
    'django.contrib.messages.middleware.MessageMiddleware',
    'django.middleware.clickjacking.XFrameOptionsMiddleware',

    'allauth.account.middleware.AccountMiddleware', 
]
CORS_ALLOW_ALL_ORIGINS = True

# CORS 설정
CORS_ALLOWED_ORIGINS = [
    "http://localhost:3000",
]


# settings.py
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
        },
    },
    'loggers': {
        'django.request': {
            'handlers': ['console'],
            'level': 'DEBUG',
            'propagate': True,
        },
    },
}

# settings.py


# 로깅 설정 추가
LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
        },
    },
    'root': {
        'handlers': ['console'],
        'level': 'INFO',
    },
    'loggers': {
        'chat': {
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': True,
        },
    },
}
# 추가 CORS 설정
CORS_ALLOW_METHODS = [
    'DELETE',
    'GET',
    'OPTIONS',
    'PATCH',
    'POST',
    'PUT',
]

CORS_ALLOW_HEADERS = [
    'accept',
    'accept-encoding',
    'authorization',
    'content-type',
    'dnt',
    'origin',
    'user-agent',
    'x-csrftoken',
    'x-requested-with',
]

# Session 설정
SESSION_COOKIE_SAMESITE = 'Lax'  # 또는 'None'
SESSION_COOKIE_SECURE = False  # 개발 환경에서는 False, 프로덕션에서는 True
CSRF_COOKIE_SECURE = False    # 개발 환경에서는 False, 프로덕션에서는 True
CSRF_COOKIE_SAMESITE = 'Lax'  # 또는 'None'
CSRF_TRUSTED_ORIGINS = ['http://localhost:3000']

# CORS 설정

SECRET_KEY = 'fl)a4kismb2m2=vhr+g2u!yn#q#z51=!4t1ftu)^-6lvm!_%bg'



TEMPLATES = [
    {
        'BACKEND': 'django.template.backends.django.DjangoTemplates',
        'DIRS': [],  # 템플릿 디렉터리를 지정할 수 있습니다.
        'APP_DIRS': True,
        'OPTIONS': {
            'context_processors': [
                'django.template.context_processors.debug',
                'django.template.context_processors.request',
                'django.contrib.auth.context_processors.auth',
                'django.contrib.messages.context_processors.messages',
            ],
        },
    },
]
# settings.py
REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework_simplejwt.authentication.JWTAuthentication',
    ],
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.IsAuthenticated',  # 인증된 사용자만 접근 가능
    ]
}


REST_FRAMEWORK = {
    'DEFAULT_AUTHENTICATION_CLASSES': [
        'rest_framework.authentication.SessionAuthentication',
        'rest_framework.authentication.TokenAuthentication', # 토큰 인증 추가
        # JWT 사용 시: 'rest_framework_simplejwt.authentication.JWTAuthentication',
    ],
    'DEFAULT_PERMISSION_CLASSES': [
        'rest_framework.permissions.IsAuthenticated',
        'rest_framework_simplejwt.authentication.JWTAuthentication',
    ]
}

# CORS 설정 확인
CORS_ALLOW_CREDENTIALS = True
CORS_ALLOWED_ORIGINS = [
    "http://localhost:3000",  # 프론트엔드 개발 서버 주소
]
import os
MEDIA_URL = '/media/'
MEDIA_ROOT = os.path.join(BASE_DIR, 'media')

# 정적 파일 설정
STATIC_URL = '/static/'
STATIC_ROOT = os.path.join(BASE_DIR, 'staticfiles')

# 비디오 관련 설정 추가
VIDEO_FOLDER = os.path.join(MEDIA_ROOT, 'videos')
UPLOAD_FOLDER = os.path.join(MEDIA_ROOT, 'uploads')  
IMAGE_FOLDER = os.path.join(MEDIA_ROOT, 'images')

# 폴더가 없으면 생성
os.makedirs(VIDEO_FOLDER, exist_ok=True)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(IMAGE_FOLDER, exist_ok=True)
