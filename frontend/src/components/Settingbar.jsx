
// Settingbar.js
import React, { useState, useEffect } from 'react';
import { X } from 'lucide-react';
import { useDispatch, useSelector } from 'react-redux';
import { loginSuccess, loginFailure } from '../store/authSlice';
import { useGoogleLogin } from '@react-oauth/google';

const Settingbar = ({ isOpen, onClose }) => {
  const [isLoginModalOpen, setIsLoginModalOpen] = useState(false);
  const [isSignupModalOpen, setIsSignupModalOpen] = useState(false);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [isLoading, setLoading] = useState(false);

  const dispatch = useDispatch();
  const { user } = useSelector((state) => state.auth);

  const handleNaverLogin = async () => {
    localStorage.removeItem('naverState');
    localStorage.removeItem('naverAccessToken');
    
    const state = Math.random().toString(36).substr(2, 11);
    const naverAuthUrl = `https://nid.naver.com/oauth2.0/authorize?` +
        `response_type=code` +
        `&client_id=${process.env.REACT_APP_NAVER_CLIENT_ID}` +
        `&redirect_uri=${encodeURIComponent(process.env.REACT_APP_NAVER_REDIRECT_URI)}` +
        `&state=${state}` +
        `&auth_type=reauthenticate` +
        `&prompt=consent` +
        `&service_provider=NAVER` +
        `&access_type=offline` +
        `&include_granted_scopes=true`;
    
    localStorage.setItem('naverState', state);
    window.location.href = naverAuthUrl;
  };

  const handleNaverCallback = async (code, state) => {
    const savedState = localStorage.getItem('naverState');
    if (state !== savedState) {
      console.error('State mismatch');
      return;
    }

    setLoading(true);
    try {
      const backendResponse = await fetch(`http://localhost:8000/auth/naver/callback/?code=${code}&state=${state}`, {
        method: 'GET',
        credentials: 'include',
      });

      if (!backendResponse.ok) {
        const errorData = await backendResponse.json();
        throw new Error(errorData.error || '네이버 로그인 실패');
      }

      const data = await backendResponse.json();
      localStorage.setItem('naverAccessToken', data.access_token);
      dispatch(loginSuccess(data.user));
      setIsLoginModalOpen(false);
      window.history.pushState({}, null, window.location.pathname);
    } catch (error) {
      console.error('로그인 에러:', error);
      dispatch(loginFailure(error.message));
    } finally {
      setLoading(false);
      localStorage.removeItem('naverState');
    }
  };

  const handleKakaoLogin = async () => {
    const kakaoAuthUrl = `https://kauth.kakao.com/oauth/authorize?response_type=code&client_id=${process.env.REACT_APP_KAKAO_CLIENT_ID}&redirect_uri=${process.env.REACT_APP_KAKAO_REDIRECT_URI}&scope=profile_nickname,account_email&prompt=login`;
    window.location.href = kakaoAuthUrl;
  };
  
  const handleKakaoCallback = async (code) => {
    setLoading(true);
    try {
      const backendResponse = await fetch(`http://localhost:8000/api/auth/kakao/callback/?code=${code}`, {
        method: 'GET',
        credentials: 'include',
      });
  
      if (!backendResponse.ok) {
        const errorData = await backendResponse.json();
        throw new Error(errorData.error || '카카오 로그인 실패');
      }
  
      const data = await backendResponse.json();
      dispatch(loginSuccess(data.user));
      setIsLoginModalOpen(false);
    } catch (error) {
      console.error('로그인 에러:', error);
      dispatch(loginFailure(error.message));
    } finally {
      setLoading(false);
    }
  };
  
  useEffect(() => {
    const queryParams = new URLSearchParams(window.location.search);
    const code = queryParams.get('code');
    const state = queryParams.get('state');
    
    if (code) {
      if (state) {
        handleNaverCallback(code, state);
      } else {
        handleKakaoCallback(code);
      }
    }
  }, []);
  
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
          throw new Error(errorData.error || '구글 로그인 실패');
        }

        const data = await backendResponse.json();
        dispatch(loginSuccess(data.user));
        setIsLoginModalOpen(false);
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
      setLoading(false);
    },
  });


  if (!isOpen) return null;

  return (
    <div>
      {/* 설정 모달 */}
      {!isLoginModalOpen && !isSignupModalOpen && (
        <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50">
          <div className="bg-white p-6 rounded-lg shadow-lg w-96 relative flex flex-col items-center">
            <X
              className="absolute top-3 right-3 w-6 h-6 cursor-pointer"
              onClick={onClose}
            />
            <h2 className="text-2xl font-bold">AI OF AI</h2>
            <p className="text-sm text-gray-600 mb-4">
              AI 통합 기반 답변 최적화 플랫폼
            </p>

            {user ? (
              // 로그인된 상태
              <div className="w-full space-y-4">
                <div className="bg-green-100 border border-green-400 text-green-700 p-4 rounded">
                  <p>환영합니다, {user.nickname || user.username}님!</p>
                  <p>이메일: {user.email}</p>
                </div>
                <button
                  onClick={onClose}
                  className="w-full bg-indigo-600 text-white p-2 rounded hover:bg-indigo-700"
                >
                  닫기
                </button>
              </div>
            ) : (
              // 비로그인 상태
              <>
                <button 
                  className="w-full bg-gray-300 p-2 rounded mb-2" 
                  onClick={() => setIsLoginModalOpen(true)}
                >
                  로그인
                </button>
                <button 
                  className="w-full bg-gray-300 p-2 rounded mb-2" 
                  onClick={() => setIsSignupModalOpen(true)}
                >
                  회원가입
                </button>
              </>
            )}
          </div>
        </div>
      )}

      {/* 로그인 모달 */}
      {isLoginModalOpen && (
        <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50">
          <div className="bg-white p-6 rounded-lg shadow-lg w-96 relative flex flex-col items-center">
            <X
              className="absolute top-3 right-3 w-6 h-6 cursor-pointer"
              onClick={() => setIsLoginModalOpen(false)}
            />
            <h2 className="text-2xl font-bold">AI OF AI</h2>
            <p className="text-sm text-gray-600 mb-4">
              AI 통합 기반 답변 최적화 플랫폼
            </p>
            <button className="w-full bg-gray-300 p-2 rounded mb-2" onClick={googleLogin}>
              Google로 로그인
            </button>
            <button className="w-full bg-gray-300 p-2 rounded mb-2" onClick={handleKakaoLogin}>
              Kakao로 로그인
            </button>
            <button className="w-full bg-gray-300 p-2 rounded mb-4" onClick={handleNaverLogin}>
              Naver로 로그인
            </button>
            
            <hr className="w-full border-gray-400 mb-4" />
            <input
              type="email"
              placeholder="이메일"
              className="w-full p-2 border rounded mb-2"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
            />
            <input
              type="password"
              placeholder="비밀번호"
              className="w-full p-2 border rounded mb-2"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
            <div className="text-xs text-gray-600 flex justify-between w-full">
              <span>비밀번호를 잊으셨나요?</span>
              <span className="text-blue-500 cursor-pointer">비밀번호 찾기</span>
            </div>
            <button
              className="w-full bg-gray-800 text-white p-2 rounded mt-4"
              onClick={() => alert('로그인 시도')}
            >
              로그인
            </button>
            <div className="text-xs text-gray-600 mt-2">
              아직 계정이 없나요?{' '}
              <span className="text-blue-500 cursor-pointer" onClick={() => {
                setIsLoginModalOpen(false);
                setIsSignupModalOpen(true);
              }}>
                회원가입
              </span>
            </div>
          </div>
        </div>
      )}

      {/* 회원가입 모달 */}
      {isSignupModalOpen && (
        <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50">
          <div className="bg-white p-6 rounded-lg shadow-lg w-96 relative flex flex-col items-center">
            <X
              className="absolute top-3 right-3 w-6 h-6 cursor-pointer"
              onClick={() => setIsSignupModalOpen(false)}
            />
            <h2 className="text-2xl font-bold">AI OF AI</h2>
            <p className="text-sm text-gray-600 mb-4">
              AI 통합 기반 답변 최적화 플랫폼
            </p>
           
            <button 
              className="w-full bg-gray-300 p-2 rounded mb-2" 
              onClick={() => {
                if(window.confirm('Google 계정으로 회원가입을 진행하시겠습니까?')) {
                  googleLogin();
                }
              }}
            >
              Google로 회원가입
            </button>
            <button 
              className="w-full bg-gray-300 p-2 rounded mb-2"
              onClick={() => {
                if(window.confirm('Kakao 계정으로 회원가입을 진행하시겠습니까?')) {
                  handleKakaoLogin();
                }
              }}
            >
              Kakao로 회원가입
            </button>
            <button 
              className="w-full bg-gray-300 p-2 rounded mb-2"
              onClick={() => {
                if(window.confirm('Naver 계정으로 회원가입을 진행하시겠습니까?')) {
                  handleNaverLogin();
                }
              }}
            >
              Naver로 회원가입
            </button>
          
            <div className="text-xs text-gray-600 mt-2">
              계정이 없으신가요?{' '}
              <span className="text-blue-500 cursor-pointer" onClick={() => {
                setIsSignupModalOpen(false);
                setIsLoginModalOpen(true);
              }}>
                회원가입
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Settingbar;