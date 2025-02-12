
import React, { useState, useEffect } from 'react';

import { Settings, X } from 'lucide-react';
import { useDispatch, useSelector } from 'react-redux';
import { loginSuccess, loginFailure } from '../store/authSlice';
import { useGoogleLogin } from '@react-oauth/google';

const Settingbar = () => {
  const [isLoginModalOpen, setIsLoginModalOpen] = useState(false);
  const [isSignupModalOpen, setIsSignupModalOpen] = useState(false);
  const [isSettingModalOpen, setIsSettingModalOpen] = useState(false);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [nickname, setNickname] = useState('');
  const [isLoading, setLoading] = useState(false);

  const dispatch = useDispatch();
  const { user } = useSelector((state) => state.auth);
  const handleKakaoLogin = async () => {
        const kakaoAuthUrl = `https://kauth.kakao.com/oauth/authorize?response_type=code&client_id=${process.env.REACT_APP_KAKAO_CLIENT_ID}&redirect_uri=${process.env.REACT_APP_KAKAO_REDIRECT_URI}&scope=profile_nickname,account_email&prompt=login`;
        window.location.href = kakaoAuthUrl; // 카카오 로그인 페이지로 리디렉션
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
          setIsLoginModalOpen(false); // 로그인 모달 닫기
        } catch (error) {
          console.error('로그인 에러:', error);
          dispatch(loginFailure(error.message)); // 실패 시 상태 업데이트
        } finally {
          setLoading(false);
        }
      };
      
      useEffect(() => {
        const queryParams = new URLSearchParams(window.location.search);
        const code = queryParams.get('code');
        
        if (code) {
          handleKakaoCallback(code); // 카카오 로그인 후 콜백 처리
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

  const handleSignUp = async () => {
    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/api/auth/signup', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email, password, nickname }),
      });
      const data = await response.json();
      if (response.ok) {
        dispatch(loginSuccess(data.user));
        setIsLoginModalOpen(false);
      } else {
        throw new Error(data.error || '회원가입 실패');
      }
    } catch (error) {
      console.error('회원가입 에러:', error);
      dispatch(loginFailure(error.message));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      {/* 설정 버튼 */}
      <Settings
        className="w-5 h-5 text-gray-600 cursor-pointer"
        onClick={() => setIsSettingModalOpen(true)}
      />

      {/* 설정 모달 */}
      {/* {isSettingModalOpen && !isLoginModalOpen && !isSignupModalOpen && (
        <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50">
          <div className="bg-white p-6 rounded-lg shadow-lg w-96 relative flex flex-col items-center">
            <X
              className="absolute top-3 right-3 w-6 h-6 cursor-pointer"
              onClick={() => setIsSettingModalOpen(false)}
            />
            <h2 className="text-2xl font-bold">AI OF AI</h2>
            <p className="text-sm text-gray-600 mb-4">
              AI 통합 기반 답변 최적화 플랫폼
            </p>

            {/* 로그인 & 회원가입 버튼 */}
            {/* <button 
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
          </div>
        </div>
      )}  */}

{/* 설정 모달 */}
{isSettingModalOpen && !isLoginModalOpen && !isSignupModalOpen && (
        <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50">
          <div className="bg-white p-6 rounded-lg shadow-lg w-96 relative flex flex-col items-center">
            <X
              className="absolute top-3 right-3 w-6 h-6 cursor-pointer"
              onClick={() => setIsSettingModalOpen(false)}
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
                  onClick={() => setIsSettingModalOpen(false)}
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
            <button className="w-full bg-gray-300 p-2 rounded mb-4" onClick={() => alert('Naver login')}>
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
            <button className="w-full bg-gray-300 p-2 rounded mb-4">Naver로 회원가입</button>
            <div className="text-xs text-gray-600 mt-2">
              이미 계정이 있으신가요?{' '}
              <span className="text-blue-500 cursor-pointer" onClick={() => {
                setIsSignupModalOpen(false);
                setIsLoginModalOpen(true);
              }}>
                로그인
              </span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};


export default Settingbar;

