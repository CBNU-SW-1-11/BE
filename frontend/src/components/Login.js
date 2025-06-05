// src/components/Login.js
import { useState } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import { useNavigate } from 'react-router-dom';
import { useGoogleLogin } from '@react-oauth/google';
import { loginSuccess, loginFailure } from '../store/authSlice';

const Login = () => {
  const dispatch = useDispatch();
  const navigate = useNavigate();
  const { error, user } = useSelector((state) => state.auth);
  const [loading, setLoading] = useState(false);

 // Login.js의 handleKakaoLogin 함수 수정
const handleKakaoLogin = () => {
  const kakaoAuthUrl = `https://kauth.kakao.com/oauth/authorize?response_type=code&client_id=${process.env.REACT_APP_KAKAO_CLIENT_ID}&redirect_uri=${process.env.REACT_APP_KAKAO_REDIRECT_URI}&scope=profile_nickname,account_email&prompt=login`;
  window.location.href = kakaoAuthUrl;
};
const googleLogin = useGoogleLogin({
  onSuccess: async (codeResponse) => {
    setLoading(true);
    try {
      const backendResponse = await fetch('http://localhost:8000/api/auth/google/callback/', {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${codeResponse.access_token}`,
          'Content-Type': 'application/json',
        },
        credentials: 'include',
      });

      if (!backendResponse.ok) {
        const errorData = await backendResponse.json();
        throw new Error(errorData.error || '로그인 실패');
      }

      const data = await backendResponse.json();
      
      // 🔧 토큰 저장 추가
      if (data.token) {
        localStorage.setItem('access_token', data.token);
        localStorage.setItem('authToken', data.token);
      }
      
      dispatch(loginSuccess(data.user));
      
      // 🔧 로그인 성공 후 리다이렉트
      navigate('/schedule'); // 또는 원하는 페이지로
      
    } catch (error) {
      console.error('로그인 에러:', error);
      dispatch(loginFailure(error.message));
    } finally {
      setLoading(false);
    }
  },
  onError: (error) => {
    console.error('로그인 실패:', error);
    dispatch(loginFailure('구글 로그인 실패'));
  },
});
  // const googleLogin = useGoogleLogin({
  //   onSuccess: async (codeResponse) => {
  //     setLoading(true);
  //     try {
  //       const backendResponse = await fetch('http://localhost:8000/api/auth/google/callback/', {
  //         method: 'GET',
  //         headers: {
  //           'Authorization': `Bearer ${codeResponse.access_token}`,
  //           'Content-Type': 'application/json',
  //         },
  //         credentials: 'include',
  //       });

  //       if (!backendResponse.ok) {
  //         const errorData = await backendResponse.json();
  //         throw new Error(errorData.error || '로그인 실패');
  //       }

  //       const data = await backendResponse.json();
  //       dispatch(loginSuccess(data.user));
  //     } catch (error) {
  //       console.error('로그인 에러:', error);
  //       dispatch(loginFailure(error.message));
  //     } finally {
  //       setLoading(false);
  //     }
  //   },
  //   onError: (error) => {
  //     console.error('로그인 실패:', error);
  //     dispatch(loginFailure('구글 로그인 실패'));
  //   },
  // });

  const handleChatClick = () => {
    navigate('/chat');
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 py-12 px-4">
      <div className="max-w-md w-full space-y-8">
        <div>
          <h2 className="mt-6 text-center text-3xl font-extrabold text-gray-900">
            계정에 로그인하세요
          </h2>
        </div>
        {error && (
          <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded relative">
            {error}
          </div>
        )}
        {user ? (
          <div className="space-y-4">
            <div className="bg-green-100 border border-green-400 text-green-700 px-4 py-3 rounded relative">
              <p>환영합니다, {user.nickname || user.username}님!</p>
              <p>이메일: {user.email}</p>
            </div>
            <button
              onClick={handleChatClick}
              className="w-full flex justify-center py-3 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
            >
              채팅 시작하기
            </button>
          </div>
        ) : (
          <div className="mt-8 space-y-4">
            <button
              onClick={() => googleLogin()}
              disabled={loading}
              className="w-full flex justify-center py-3 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white bg-red-600 hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500 disabled:opacity-50"
            >
              {loading ? '로그인 중...' : '구글로 로그인'}
            </button>
            <button
              onClick={handleKakaoLogin}
              disabled={loading}
              className="w-full flex justify-center py-3 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-black bg-yellow-300 hover:bg-yellow-400 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-yellow-500 disabled:opacity-50"
            >
              카카오로 로그인
            </button>
          </div>
        )}
      </div>
    </div>
  );
};

export default Login;