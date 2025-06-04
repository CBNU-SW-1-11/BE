// // import React, { useState } from 'react';
// // import { X } from 'lucide-react';

// // const Loginbar = ({ onClose }) => {
// //   const [isLoginModalOpen, setIsLoginModalOpen] = useState(false);
// //   const [isSignupModalOpen, setIsSignupModalOpen] = useState(false);

// //   return (
// //     <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50 z-40">
// //       <div className="bg-white p-6 rounded-lg shadow-lg w-96 relative flex flex-col items-center">
// //         <X
// //           className="absolute top-3 right-3 w-6 h-6 cursor-pointer"
// //           onClick={onClose} // `onClose` 사용하여 Settingbar 닫기
// //         />
// //         <h2 className="text-2xl font-bold">AI OF AI</h2>
// //         <p className="text-sm text-gray-600 mb-4">
// //           AI 통합 기반 답변 최적화 플랫폼
// //         </p>

// //         {/* 로그인 & 회원가입 버튼 */}
// //         {!isLoginModalOpen && !isSignupModalOpen && (
// //           <div className="w-full flex flex-col items-center space-y-2">
// //             <button 
// //               className="w-full p-2 border rounded-lg bg-white hover:bg-blue-50 transition-colors" 
// //               onClick={() => setIsLoginModalOpen(true)}
// //             >
// //               로그인
// //             </button>
// //             <button 
// //               className="w-full p-2 border rounded-lg bg-white hover:bg-blue-50 transition-colors" 
// //               onClick={() => setIsSignupModalOpen(true)}
// //             >
// //               회원가입
// //             </button>
// //           </div>
// //         )}

// //         {/* 로그인 모달 */}
// //         {isLoginModalOpen && (
// //           <div className="w-full flex flex-col items-center space-y-2">
// //             <button className="w-full p-2 border rounded-lg bg-white hover:bg-blue-50 transition-colors">Google로 로그인</button>
// //             <button className="w-full p-2 border rounded-lg bg-white hover:bg-blue-50 transition-colors">Kakao로 로그인</button>
// //             <button className="w-full p-2 border rounded-lg bg-white hover:bg-blue-50 transition-colors">Naver로 로그인</button>
            
// //             <hr className="w-full border-gray-400 mb-4" />
// //             <input type="email" placeholder="이메일" className="w-full p-2 border rounded mb-2" />
// //             <input type="password" placeholder="비밀번호" className="w-full p-2 border rounded mb-2" />
// //             <div className="text-xs text-gray-600 flex justify-between w-full">
// //               <span>비밀번호를 잊으셨나요?</span>{' '}
// //               <span className="text-blue-500 cursor-pointer">비밀번호 찾기</span>
// //             </div>
// //             <button className="w-full bg-gray-800 text-white p-2 rounded mt-4">로그인</button>
// //             <div className="text-xs text-gray-600 mt-2">
// //               계정이 없으신가요?{' '}
// //               <span className="text-blue-500 cursor-pointer" onClick={() => {
// //                 setIsLoginModalOpen(false);
// //                 setIsSignupModalOpen(true);
// //               }}>
// //                 회원가입
// //               </span>
// //             </div>
// //           </div>
// //         )}

// //         {/* 회원가입 모달 */}
// //         {isSignupModalOpen && (
// //           <div className="w-full flex flex-col items-center space-y-2">
// //             <button className="w-full p-2 border rounded-lg bg-white hover:bg-blue-50 transition-colors">Google로 회원가입</button>
// //             <button className="w-full p-2 border rounded-lg bg-white hover:bg-blue-50 transition-colors">Kakao로 회원가입</button>
// //             <button className="w-full p-2 border rounded-lg bg-white hover:bg-blue-50 transition-colors">Naver로 회원가입</button>
// //             <div className="text-xs text-gray-600 mt-2">
// //               이미 계정이 있으신가요?{' '}
// //               <span className="text-blue-500 cursor-pointer" onClick={() => {
// //                 setIsSignupModalOpen(false);
// //                 setIsLoginModalOpen(true);
// //               }}>
// //                 로그인
// //               </span>
// //             </div>
// //           </div>
// //         )}
// //       </div>
// //     </div>
// //   );
// // };

// // export default Loginbar;


// import { useHistory } from 'react-router-dom';
// // Settingbar.js
// import React, { useState, useEffect } from 'react';
// import { X } from 'lucide-react';
// import { useDispatch, useSelector } from 'react-redux';
// import { loginSuccess, loginFailure } from '../store/authSlice';
// import { useGoogleLogin } from '@react-oauth/google';
// import {  useRef } from 'react';

// import { useNavigate, useLocation } from 'react-router-dom';
// const Loginbar = ({ isOpen, onClose }) => {
//   const [isLoginModalOpen, setIsLoginModalOpen] = useState(false);
//   const [isSignupModalOpen, setIsSignupModalOpen] = useState(false);
//   const [email, setEmail] = useState('');
//   const [password, setPassword] = useState('');
//   const [isLoading, setLoading] = useState(false);
//   const navigate = useNavigate();
//   const dispatch = useDispatch();
//   const location = useLocation();
//   const [loading] = useState(false);
//   const { user } = useSelector((state) => state.auth);
//   const processedRef = useRef(false);
//   const handleNaverLogin = async () => {
//     localStorage.removeItem('naverState');
//     localStorage.removeItem('naverAccessToken');
    
//     const state = Math.random().toString(36).substr(2, 11);
//     const naverAuthUrl = `https://nid.naver.com/oauth2.0/authorize?` +
//         `response_type=code` +
//         `&client_id=${process.env.REACT_APP_NAVER_CLIENT_ID}` +
//         `&redirect_uri=${encodeURIComponent(process.env.REACT_APP_NAVER_REDIRECT_URI)}` +
//         `&state=${state}` +
//         `&auth_type=reauthenticate` +
//         `&prompt=consent` +
//         `&service_provider=NAVER` +
//         `&access_type=offline` +
//         `&include_granted_scopes=true`;
    
//     localStorage.setItem('naverState', state);
//     window.location.href = naverAuthUrl;
//   };

//   const handleNaverCallback = async (code, state) => {
//     const savedState = localStorage.getItem('naverState');
//     if (state !== savedState) {
//       console.error('State mismatch');
//       return;
//     }

//     setLoading(true);
//     try {
//       const backendResponse = await fetch(`http://localhost:8000/auth/naver/callback/?code=${code}&state=${state}`, {
//         method: 'GET',
//         headers: {
//           'Authorization': `Bearer ${code.access_token}`,
//           'Content-Type': 'application/json',
//         },
//         credentials: 'include',
//       });

//       if (!backendResponse.ok) {
//         const errorData = await backendResponse.json();
//         throw new Error(errorData.error || '네이버 로그인 실패');
//       }

//       const data = await backendResponse.json();
//       localStorage.setItem('accessToken', data.token); // Save the Django token
//       console.log('Login response data:', data); // 응답 데이터 확인

//       // dispatch(loginSuccess(data.user));
//       dispatch(loginSuccess({
//         user: data.user,
//         token: data.access_token
//       }));

//       localStorage.setItem('accessToken', data.access_token);

//       setIsLoginModalOpen(false);
//       window.history.pushState({}, null, window.location.pathname);
//     } catch (error) {
//       console.error('로그인 에러:', error);
//       dispatch(loginFailure(error.message));
//     } finally {
//       setLoading(false);
//       localStorage.removeItem('naverState');
//     }
//   };

//   // const handleKakaoLogin = async () => {
//   //   const kakaoAuthUrl = `https://kauth.kakao.com/oauth/authorize?response_type=code&client_id=${process.env.REACT_APP_KAKAO_CLIENT_ID}&redirect_uri=${process.env.REACT_APP_KAKAO_REDIRECT_URI}&scope=profile_nickname,account_email&prompt=login`;
//   //   window.location.href = kakaoAuthUrl;
//   // };
  
//   // const handleKakaoCallback = async (code) => {
//   //   setLoading(true);
//   //   try {
//   //     const backendResponse = await fetch(`http://localhost:8000/api/auth/kakao/callback/?code=${code}`, {
//   //       method: 'GET',
//   //       credentials: 'include',
//   //     });
  
//   //     if (!backendResponse.ok) {
//   //       const errorData = await backendResponse.json();
//   //       throw new Error(errorData.error || '카카오 로그인 실패');
//   //     }
  
//   //     const data = await backendResponse.json();
//   //     dispatch(loginSuccess(data.user));
//   //     setIsLoginModalOpen(false);
//   //   } catch (error) {
//   //     console.error('로그인 에러:', error);
//   //     dispatch(loginFailure(error.message));
//   //   } finally {
//   //     setLoading(false);
//   //   }
//   // };
  
//   // useEffect(() => {
//   //   const queryParams = new URLSearchParams(window.location.search);
//   //   const code = queryParams.get('code');
//   //   const state = queryParams.get('state');
    
//   //   if (code) {
//   //     if (state) {
//   //       handleNaverCallback(code, state);
//   //     } else {
//   //       handleKakaoCallback(code);
//   //     }
//   //   }
//   // }, []);
//   const googleLogin = useGoogleLogin({
//     onSuccess: async (codeResponse) => {
//       try {
//         const backendResponse = await fetch('http://localhost:8000/api/auth/google/callback/', {
//           method: 'GET',
//           headers: {
//             'Authorization': `Bearer ${codeResponse.access_token}`,
//             'Content-Type': 'application/json',
//           },
//           credentials: 'include',
//         });
        
//         const data = await backendResponse.json();
//         console.log('Login response data:', data); // 응답 데이터 확인
  
//         // loginSuccess에 전달하는 데이터 형식 수정
//         dispatch(loginSuccess({
//           user: data.user,
//           token: data.access_token
//         }));
        
//         localStorage.setItem('accessToken', data.access_token);
//         setIsLoginModalOpen(false);
//       } catch (error) {
//         console.error('로그인 에러:', error);
//         dispatch(loginFailure(error.message));
//       }
//     }
//   });
  
//   const effectRan = useRef(false);
// useEffect(() => {
//     if (effectRan.current) return;
//     const queryParams = new URLSearchParams(window.location.search);
//     const code = queryParams.get('code');
//     const state = queryParams.get('state');

//     if (code) {
//         if (state) {
//             handleNaverCallback(code, state);
//         } else {
//             handleKakaoCallback(code);
//         }
//         window.history.replaceState({}, document.title, window.location.pathname);
//     }
// }, []);


//   const handleKakaoLogin = async () => {
//     const kakaoAuthUrl = `https://kauth.kakao.com/oauth/authorize?response_type=code&client_id=${process.env.REACT_APP_KAKAO_CLIENT_ID}&redirect_uri=${process.env.REACT_APP_KAKAO_REDIRECT_URI}&scope=profile_nickname,account_email&prompt=login`;
//     window.location.href = kakaoAuthUrl;
//     const token = new URL(window.location.href).searchParams.get("code");
// };
// // 예: 카카오 콜백 처리
// const handleKakaoCallback = async () => {
//   if (processedRef.current) return;
//   setLoading(true);
  
//   const code = new URLSearchParams(location.search).get('code');
//   console.log('Received code:', code);
//   if (!code) {
//     setLoading(false);
//     return;
//   }
  
//   processedRef.current = true;
//   try {
//     const response = await fetch(`http://localhost:8000/api/auth/kakao/callback/?code=${code}`);
//     if (!response.ok) {
//       const errorData = await response.json();
//       throw new Error(errorData.error || '카카오 로그인 실패');
//     }
    
//     const data = await response.json();
//     dispatch(loginSuccess({
//       user: data.user,
//       token: data.access_token,
//     }));
//     localStorage.setItem('accessToken', data.access_token);
//     setIsLoginModalOpen(false);
//     window.history.replaceState({}, document.title, '/');
//     navigate('/'); // 리로드 없이 이동
//   } catch (error) {
//     console.error('Kakao login error:', error);
//     dispatch(loginFailure(error.message));
//     navigate('/');
//   } finally {
//     setLoading(false);
//   }
// };

// // const handleKakaoCallback = async () => {
// //  setLoading(true);
// //   const token = new URL(window.location.href).searchParams.get("code");
// //   if (processedRef.current) return;
      
// //   const code = new URLSearchParams(location.search).get('code');
// //   console.log('Received code:', code);  // 코드 로깅
// //   if (!code) return;

// //   processedRef.current = true;
// //   try {
// //     const response = await fetch(`http://localhost:8000/api/auth/kakao/callback/?code=${code}`);
// //     console.log('Response status:', response.status);  // 응답 상태 로깅
// //     const data = await response.json();
// //     console.log('Response data:', data);  // 응답 데이터 로깅
    
// //     //   // 서버에 인증 코드 전송 및 액세스 토큰 요청
// //     //   const backendResponse = await fetch(`http://localhost:8000/api/auth/kakao/callback/?code=${code}`, {
// //     //       method: 'GET',
// //     //       headers: {
// //     //         Authorization: `Bearer ${code.access_token}`,
// //     //         'Content-Type': 'application/json',
// //     //       },
// //     //       credentials: 'include',
// //     //   }
// //     // );
   

// //       if (!response.ok) {
// //           const errorData = await response.json();
// //           throw new Error(errorData.error || '카카오 로그인 실패');
// //       }

// //       // const data = await backendResponse.json();
// //       // console.log('Login response data:', data);

// //       // 로그인 성공 처리
// //       dispatch(loginSuccess({
// //           user: data.user,
// //           token: data.access_token
// //       }));

// //       navigate('/');
// //       localStorage.setItem('accessToken', data.access_token);
      
// //       setIsLoginModalOpen(false);
// //   } catch (error) {
// //     console.error('Kakao login error:', error);
// //     dispatch(loginFailure(error.message));
// //     navigate('/');
// //   } finally {
// //       setLoading(false);
// //   }
// // };
// useEffect(() => {
//   if (user) {
//     setIsLoginModalOpen(false); // 사용자가 로그인되면 로그인 모달을 자동으로 닫음
//     navigate('/'); // 홈으로 리디렉션
//   }
// }, [user, navigate]);


// if (loading) {
//   return (
//     <div className="min-h-screen flex items-center justify-center bg-gray-100">
//       <div className="bg-white p-8 rounded-lg shadow-md">
//         <div className="flex flex-col items-center">
//           <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-gray-900"></div>
//           <h2 className="mt-4 text-xl font-semibold text-gray-700">
//             카카오 로그인 처리중...
//           </h2>
//         </div>
//       </div>
//     </div>
//   );
// }


//   // const googleLogin = useGoogleLogin({
//   //   onSuccess: async (codeResponse) => {
//   //     console.log('Access token:', codeResponse.access_token); // 구글 OAuth 토큰
      
//   //     setLoading(true);
//   //     try {
//   //       // 구글 액세스 토큰을 백엔드에 전송
//   //       const backendResponse = await fetch('http://localhost:8000/api/auth/google/callback/', {
//   //         method: 'GET',
//   //         headers: {
//   //           'Authorization': `Bearer ${codeResponse.access_token}`,
//   //           'Content-Type': 'application/json',
//   //         },
//   //         credentials: 'include',
//   //       });
    
//   //       if (!backendResponse.ok) {
//   //         const errorData = await backendResponse.json();
//   //         throw new Error(errorData.error || 'Google 로그인 실패');
//   //       }
//   //       // 백엔드 응답 처리
//   //       const data = await backendResponse.json();
//   //       // 통합된 키로 저장
//   //       localStorage.setItem('accessToken', data.token);
//   //       dispatch(loginSuccess(data.user));
    
//   //       setIsLoginModalOpen(false);
//   //     } catch (error) {
//   //       console.error('로그인 에러:', error);
//   //       dispatch(loginFailure(error.message));
//   //     } finally {
//   //       setLoading(false);
//   //     }
//   //   }
//   // });

//   return (
//     <div>
//       {/* 설정 모달 */}
//       {!isLoginModalOpen && !isSignupModalOpen && (
//         <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50">
//           <div className="bg-white p-6 rounded-lg shadow-lg w-96 relative flex flex-col items-center">
//             <X
//               className="absolute top-3 right-3 w-6 h-6 cursor-pointer"
//               onClick={onClose}
//             />
//             <h2 className="text-2xl font-bold">AI OF AI</h2>
//             <p className="text-sm text-gray-600 mb-4">
//               AI 통합 기반 답변 최적화 플랫폼
//             </p>

//             {user ? (
//   // 로그인된 상태
//   <div className="w-full space-y-4">
//     <div className="bg-green-100 border border-green-400 text-green-700 p-4 rounded">
//       <p>환영합니다, {user.nickname || user.username}님!</p>
//       <p>이메일: {user.email}</p>
//     </div>
//     <button
//       onClick={onClose}
//       className="w-full bg-indigo-600 text-white p-2 rounded hover:bg-indigo-700"
//     >
//       닫기
//     </button>
//   </div>
// ) : (
//   // 비로그인 상태
//   <>
//     <button 
//       className="w-full bg-gray-300 p-2 rounded mb-2" 
//       onClick={() => setIsLoginModalOpen(true)}
//     >
//       로그인
//     </button>
//     <button 
//       className="w-full bg-gray-300 p-2 rounded mb-2" 
//       onClick={() => setIsSignupModalOpen(true)}
//     >
//       회원가입
//     </button>
//   </>
// )}

    
//           </div>
//         </div>
//       )}

//       {/* 로그인 모달 */}
//       {isLoginModalOpen && (
//         <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50">
//           <div className="bg-white p-6 rounded-lg shadow-lg w-96 relative flex flex-col items-center">
//             <X
//               className="absolute top-3 right-3 w-6 h-6 cursor-pointer"
//               onClick={() => setIsLoginModalOpen(false)}
//             />
//             <h2 className="text-2xl font-bold">AI OF AI</h2>
//             <p className="text-sm text-gray-600 mb-4">
//               AI 통합 기반 답변 최적화 플랫폼
//             </p>
//             <button className="w-full bg-gray-300 p-2 rounded mb-2" onClick={googleLogin}>
//               Google로 로그인
//             </button>
//             <button className="w-full bg-gray-300 p-2 rounded mb-2" onClick={handleKakaoLogin}>
//               Kakao로 로그인
//             </button>
//             <button className="w-full bg-gray-300 p-2 rounded mb-4" onClick={handleNaverLogin}>
//               Naver로 로그인
//             </button>
            
//             <hr className="w-full border-gray-400 mb-4" />
//             <input
//               type="email"
//               placeholder="이메일"
//               className="w-full p-2 border rounded mb-2"
//               value={email}
//               onChange={(e) => setEmail(e.target.value)}
//             />
//             <input
//               type="password"
//               placeholder="비밀번호"
//               className="w-full p-2 border rounded mb-2"
//               value={password}
//               onChange={(e) => setPassword(e.target.value)}
//             />
//             <div className="text-xs text-gray-600 flex justify-between w-full">
//               <span>비밀번호를 잊으셨나요?</span>
//               <span className="text-blue-500 cursor-pointer">비밀번호 찾기</span>
//             </div>
//             <button
//               className="w-full bg-gray-800 text-white p-2 rounded mt-4"
//               onClick={() => alert('로그인 시도')}
//             >
//               로그인
//             </button>
//             <div className="text-xs text-gray-600 mt-2">
//               아직 계정이 없나요?{' '}
//               <span className="text-blue-500 cursor-pointer" onClick={() => {
//                 setIsLoginModalOpen(false);
//                 setIsSignupModalOpen(true);
//               }}>
//                 회원가입
//               </span>
//             </div>
//           </div>
//         </div>
//       )}

//       {/* 회원가입 모달 */}
//       {isSignupModalOpen && (
//         <div className="fixed inset-0 flex items-center justify-center bg-black bg-opacity-50">
//           <div className="bg-white p-6 rounded-lg shadow-lg w-96 relative flex flex-col items-center">
//             <X
//               className="absolute top-3 right-3 w-6 h-6 cursor-pointer"
//               onClick={() => setIsSignupModalOpen(false)}
//             />
//             <h2 className="text-2xl font-bold">AI OF AI</h2>
//             <p className="text-sm text-gray-600 mb-4">
//               AI 통합 기반 답변 최적화 플랫폼
//             </p>
           
//             <button 
//               className="w-full bg-gray-300 p-2 rounded mb-2" 
//               onClick={() => {
//                 if(window.confirm('Google 계정으로 회원가입을 진행하시겠습니까?')) {
//                   googleLogin();
//                 }
//               }}
//             >
//               Google로 회원가입
//             </button>
//             <button 
//               className="w-full bg-gray-300 p-2 rounded mb-2"
//               onClick={() => {
//                 if(window.confirm('Kakao 계정으로 회원가입을 진행하시겠습니까?')) {
//                   handleKakaoLogin();
//                 }
//               }}
//             >
//               Kakao로 회원가입
//             </button>
//             <button 
//               className="w-full bg-gray-300 p-2 rounded mb-2"
//               onClick={() => {
//                 if(window.confirm('Naver 계정으로 회원가입을 진행하시겠습니까?')) {
//                   handleNaverLogin();
//                 }
//               }}
//             >
//               Naver로 회원가입
//             </button>
          
//             <div className="text-xs text-gray-600 mt-2">
//               계정이 없으신가요?{' '}
//               <span className="text-blue-500 cursor-pointer" onClick={() => {
//                 setIsSignupModalOpen(false);
//                 setIsLoginModalOpen(true);
//               }}>
//                 회원가입
//               </span>
//             </div>
//           </div>
//         </div>
//       )}
//     </div>
//   );
// };

// export default Loginbar;
// src/components/Loginbar.js
import React, { useState, useEffect, useRef } from "react";
import { X } from "lucide-react";
import { useDispatch, useSelector } from "react-redux";
import { loginSuccess, loginFailure } from "../store/authSlice";
import { useGoogleLogin } from "@react-oauth/google";
import { useNavigate, useLocation } from "react-router-dom";

const Loginbar = ({ isOpen, onClose }) => {
  const [isLoginModalOpen, setIsLoginModalOpen] = useState(false);
  const [isSignupModalOpen, setIsSignupModalOpen] = useState(false);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  // 단일 로딩 상태
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();
  const dispatch = useDispatch();
  const location = useLocation();
  const { user } = useSelector((state) => state.auth);
  const processedRef = useRef(false);

  /* ===================== 네이버 로그인 ===================== */
  const handleNaverLogin = async () => {
    localStorage.removeItem("naverState");
    localStorage.removeItem("naverAccessToken");

    const state = Math.random().toString(36).substr(2, 11);
    const naverAuthUrl =
      `https://nid.naver.com/oauth2.0/authorize?` +
      `response_type=code` +
      `&client_id=${process.env.REACT_APP_NAVER_CLIENT_ID}` +
      `&redirect_uri=${encodeURIComponent(
        process.env.REACT_APP_NAVER_REDIRECT_URI
      )}` +
      `&state=${state}` +
      `&auth_type=reauthenticate` +
      `&prompt=consent` +
      `&service_provider=NAVER` +
      `&access_type=offline` +
      `&include_granted_scopes=true`;

    localStorage.setItem("naverState", state);
    window.location.href = naverAuthUrl;
  };

  const handleNaverCallback = async (code, state) => {
    const savedState = localStorage.getItem("naverState");
    if (state !== savedState) {
      console.error("State mismatch");
      return;
    }

    setLoading(true);
    try {
      const backendResponse = await fetch(
        `http://localhost:8000/auth/naver/callback/?code=${code}&state=${state}`,
        {
          method: "GET",
          headers: {
            "Authorization": `Bearer ${code.access_token}`,
            "Content-Type": "application/json",
          },
          credentials: "include",
        }
      );

      if (!backendResponse.ok) {
        const errorData = await backendResponse.json();
        throw new Error(errorData.error || "네이버 로그인 실패");
      }

      const data = await backendResponse.json();
      localStorage.setItem("accessToken", data.access_token);
      localStorage.setItem("user", JSON.stringify(data.user));

      dispatch(
        loginSuccess({
          user: data.user,
          token: data.access_token,
        })
      );

      setIsLoginModalOpen(false);
      window.history.replaceState({}, document.title, "/");
      navigate("/");
    } catch (error) {
      console.error("네이버 로그인 에러:", error);
      dispatch(loginFailure(error.message));
    } finally {
      setLoading(false);
      localStorage.removeItem("naverState");
    }
  };

  /* ===================== 구글 로그인 ===================== */
  const googleLogin = useGoogleLogin({
    onSuccess: async (codeResponse) => {
      try {
        const backendResponse = await fetch(
          "http://localhost:8000/api/auth/google/callback/",
          {
            method: "GET",
            headers: {
              "Authorization": `Bearer ${codeResponse.access_token}`,
              "Content-Type": "application/json",
            },
            credentials: "include",
          }
        );

        const data = await backendResponse.json();
        localStorage.setItem("accessToken", data.access_token);
        localStorage.setItem("user", JSON.stringify(data.user));

        dispatch(
          loginSuccess({
            user: data.user,
            token: data.access_token,
          })
        );

        setIsLoginModalOpen(false);
        window.history.replaceState({}, document.title, "/");
        navigate("/");
      } catch (error) {
        console.error("구글 로그인 에러:", error);
        dispatch(loginFailure(error.message));
      }
    },
  });

  /* ===================== 카카오 로그인 ===================== */
  const handleKakaoLogin = async () => {
    const kakaoAuthUrl =
      `https://kauth.kakao.com/oauth/authorize?` +
      `response_type=code` +
      `&client_id=${process.env.REACT_APP_KAKAO_CLIENT_ID}` +
      `&redirect_uri=${process.env.REACT_APP_KAKAO_REDIRECT_URI}` +
      `&scope=profile_nickname,account_email` +
      `&prompt=login`;
    window.location.href = kakaoAuthUrl;
  };

  const handleKakaoCallback = async () => {
    if (processedRef.current) return;
    setLoading(true);

    const code = new URLSearchParams(location.search).get("code");
    console.log("Received Kakao auth code:", code);
    if (!code) {
      setLoading(false);
      return;
    }

    processedRef.current = true;
    try {
      const response = await fetch(
        `http://localhost:8000/api/auth/kakao/callback/?code=${code}`
      );
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "카카오 로그인 실패");
      }

      const data = await response.json();
      localStorage.setItem("accessToken", data.access_token);
      localStorage.setItem("user", JSON.stringify(data.user));

      dispatch(
        loginSuccess({
          user: data.user,
          token: data.access_token,
        })
      );
      setIsLoginModalOpen(false);
      window.history.replaceState({}, document.title, "/");
      navigate("/");
    } catch (error) {
      console.error("카카오 로그인 에러:", error);
      dispatch(loginFailure(error.message));
      navigate("/");
    } finally {
      setLoading(false);
    }
  };

  // URL에 code가 있으면 OAuth 콜백 처리
  useEffect(() => {
    const queryParams = new URLSearchParams(location.search);
    const code = queryParams.get("code");
    const state = queryParams.get("state");

    if (code) {
      if (state) {
        // 네이버 로그인 콜백
        handleNaverCallback(code, state);
      } else {
        // 카카오 로그인 콜백
        handleKakaoCallback();
      }
      window.history.replaceState({}, document.title, "/");
    }
  }, [location.search]);

  // 로그인 상태에 따라 모달 닫기 및 페이지 이동
  useEffect(() => {
    if (user) {
      setIsLoginModalOpen(false);
      navigate("/");
    }
  }, [user, navigate]);

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-100">
        <div className="bg-white p-8 rounded-lg shadow-md">
          <div className="flex flex-col items-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-gray-900"></div>
            <h2 className="mt-4 text-xl font-semibold text-gray-700">
              로그인 처리중...
            </h2>
          </div>
        </div>
      </div>
    );
  }

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
            <button
              className="w-full bg-gray-300 p-2 rounded mb-2"
              onClick={googleLogin}
            >
              Google로 로그인
            </button>
            <button
              className="w-full bg-gray-300 p-2 rounded mb-2"
              onClick={handleKakaoLogin}
            >
              Kakao로 로그인
            </button>
            <button
              className="w-full bg-gray-300 p-2 rounded mb-4"
              onClick={handleNaverLogin}
            >
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
              onClick={() => alert("로그인 시도")}
            >
              로그인
            </button>
            <div className="text-xs text-gray-600 mt-2">
              아직 계정이 없나요?{" "}
              <span
                className="text-blue-500 cursor-pointer"
                onClick={() => {
                  setIsLoginModalOpen(false);
                  setIsSignupModalOpen(true);
                }}
              >
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
                if (
                  window.confirm("Google 계정으로 회원가입을 진행하시겠습니까?")
                ) {
                  googleLogin();
                }
              }}
            >
              Google로 회원가입
            </button>
            <button
              className="w-full bg-gray-300 p-2 rounded mb-2"
              onClick={() => {
                if (
                  window.confirm("Kakao 계정으로 회원가입을 진행하시겠습니까?")
                ) {
                  handleKakaoLogin();
                }
              }}
            >
              Kakao로 회원가입
            </button>
            <button
              className="w-full bg-gray-300 p-2 rounded mb-2"
              onClick={() => {
                if (
                  window.confirm("Naver 계정으로 회원가입을 진행하시겠습니까?")
                ) {
                  handleNaverLogin();
                }
              }}
            >
              Naver로 회원가입
            </button>
            <div className="text-xs text-gray-600 mt-2">
              계정이 없으신가요?{" "}
              <span
                className="text-blue-500 cursor-pointer"
                onClick={() => {
                  setIsSignupModalOpen(false);
                  setIsLoginModalOpen(true);
                }}
              >
                회원가입
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Loginbar;
