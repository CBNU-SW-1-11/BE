
// import React, { useState, useEffect, useRef } from "react";
// import { X } from "lucide-react";
// import { useDispatch, useSelector } from "react-redux";
// import { loginSuccess, loginFailure } from "../store/authSlice";
// import { useGoogleLogin } from "@react-oauth/google";
// import { useNavigate, useLocation } from "react-router-dom";

// const Loginbar = ({ isOpen, onClose }) => {
//   const [isLoginModalOpen, setIsLoginModalOpen] = useState(false);
//   const [isSignupModalOpen, setIsSignupModalOpen] = useState(false);
//   const [email, setEmail] = useState("");
//   const [password, setPassword] = useState("");
//   // 단일 로딩 상태
//   const [loading, setLoading] = useState(false);
//   const navigate = useNavigate();
//   const dispatch = useDispatch();
//   const location = useLocation();
//   const { user } = useSelector((state) => state.auth);
//   const processedRef = useRef(false);

//   /* ===================== 네이버 로그인 ===================== */
//   const handleNaverLogin = async () => {
//     localStorage.removeItem("naverState");
//     localStorage.removeItem("naverAccessToken");

//     const state = Math.random().toString(36).substr(2, 11);
//     const naverAuthUrl =
//       `https://nid.naver.com/oauth2.0/authorize?` +
//       `response_type=code` +
//       `&client_id=${process.env.REACT_APP_NAVER_CLIENT_ID}` +
//       `&redirect_uri=${encodeURIComponent(
//         process.env.REACT_APP_NAVER_REDIRECT_URI
//       )}` +
//       `&state=${state}` +
//       `&auth_type=reauthenticate` +
//       `&prompt=consent` +
//       `&service_provider=NAVER` +
//       `&access_type=offline` +
//       `&include_granted_scopes=true`;

//     localStorage.setItem("naverState", state);
//     window.location.href = naverAuthUrl;
//   };

//   const handleNaverCallback = async (code, state) => {
//     const savedState = localStorage.getItem("naverState");
//     if (state !== savedState) {
//       console.error("State mismatch");
//       return;
//     }

//     setLoading(true);
//     try {
//       const backendResponse = await fetch(
//         `http://localhost:8000/auth/naver/callback/?code=${code}&state=${state}`,
//         {
//           method: "GET",
//           headers: {
//             "Authorization": `Bearer ${code.access_token}`,
//             "Content-Type": "application/json",
//           },
//           credentials: "include",
//         }
//       );

//       if (!backendResponse.ok) {
//         const errorData = await backendResponse.json();
//         throw new Error(errorData.error || "네이버 로그인 실패");
//       }

//       const data = await backendResponse.json();
//       localStorage.setItem("accessToken", data.access_token);
//       localStorage.setItem("user", JSON.stringify(data.user));

//       dispatch(
//         loginSuccess({
//           user: data.user,
//           token: data.access_token,
//         })
//       );

//       setIsLoginModalOpen(false);
//       window.history.replaceState({}, document.title, "/");
//       navigate("/");
//     } catch (error) {
//       console.error("네이버 로그인 에러:", error);
//       dispatch(loginFailure(error.message));
//     } finally {
//       setLoading(false);
//       localStorage.removeItem("naverState");
//     }
//   };

//   /* ===================== 구글 로그인 ===================== */
//   const googleLogin = useGoogleLogin({
//     onSuccess: async (codeResponse) => {
//       try {
//         const backendResponse = await fetch(
//           "http://localhost:8000/api/auth/google/callback/",
//           {
//             method: "GET",
//             headers: {
//               "Authorization": `Bearer ${codeResponse.access_token}`,
//               "Content-Type": "application/json",
//             },
//             credentials: "include",
//           }
//         );

//         const data = await backendResponse.json();
//         localStorage.setItem("accessToken", data.access_token);
//         localStorage.setItem("user", JSON.stringify(data.user));

//         dispatch(
//           loginSuccess({
//             user: data.user,
//             token: data.access_token,
//           })
//         );

//         setIsLoginModalOpen(false);
//         window.history.replaceState({}, document.title, "/");
//         navigate("/");
//       } catch (error) {
//         console.error("구글 로그인 에러:", error);
//         dispatch(loginFailure(error.message));
//       }
//     },
//   });

//   /* ===================== 카카오 로그인 ===================== */
//   const handleKakaoLogin = async () => {
//     const kakaoAuthUrl =
//       `https://kauth.kakao.com/oauth/authorize?` +
//       `response_type=code` +
//       `&client_id=${process.env.REACT_APP_KAKAO_CLIENT_ID}` +
//       `&redirect_uri=${process.env.REACT_APP_KAKAO_REDIRECT_URI}` +
//       `&scope=profile_nickname,account_email` +
//       `&prompt=login`;
//     window.location.href = kakaoAuthUrl;
//   };

//   const handleKakaoCallback = async () => {
//     if (processedRef.current) return;
//     setLoading(true);

//     const code = new URLSearchParams(location.search).get("code");
//     console.log("Received Kakao auth code:", code);
//     if (!code) {
//       setLoading(false);
//       return;
//     }

//     processedRef.current = true;
//     try {
//       const response = await fetch(
//         `http://localhost:8000/api/auth/kakao/callback/?code=${code}`
//       );
//       if (!response.ok) {
//         const errorData = await response.json();
//         throw new Error(errorData.error || "카카오 로그인 실패");
//       }

//       const data = await response.json();
//       localStorage.setItem("accessToken", data.access_token);
//       localStorage.setItem("user", JSON.stringify(data.user));

//       dispatch(
//         loginSuccess({
//           user: data.user,
//           token: data.access_token,
//         })
//       );
//       setIsLoginModalOpen(false);
//       window.history.replaceState({}, document.title, "/");
//       navigate("/");
//     } catch (error) {
//       console.error("카카오 로그인 에러:", error);
//       dispatch(loginFailure(error.message));
//       navigate("/");
//     } finally {
//       setLoading(false);
//     }
//   };

//   // URL에 code가 있으면 OAuth 콜백 처리
//   useEffect(() => {
//     const queryParams = new URLSearchParams(location.search);
//     const code = queryParams.get("code");
//     const state = queryParams.get("state");

//     if (code) {
//       if (state) {
//         // 네이버 로그인 콜백
//         handleNaverCallback(code, state);
//       } else {
//         // 카카오 로그인 콜백
//         handleKakaoCallback();
//       }
//       window.history.replaceState({}, document.title, "/");
//     }
//   }, [location.search]);

//   // 로그인 상태에 따라 모달 닫기 및 페이지 이동
//   useEffect(() => {
//     if (user) {
//       setIsLoginModalOpen(false);
//       navigate("/");
//     }
//   }, [user, navigate]);

//   if (loading) {
//     return (
//       <div className="min-h-screen flex items-center justify-center bg-gray-100">
//         <div className="bg-white p-8 rounded-lg shadow-md">
//           <div className="flex flex-col items-center">
//             <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-gray-900"></div>
//             <h2 className="mt-4 text-xl font-semibold text-gray-700">
//               로그인 처리중...
//             </h2>
//           </div>
//         </div>
//       </div>
//     );
//   }

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
//               <div className="w-full space-y-4">
//                 <div className="bg-green-100 border border-green-400 text-green-700 p-4 rounded">
//                   <p>환영합니다, {user.nickname || user.username}님!</p>
//                   <p>이메일: {user.email}</p>
//                 </div>
//                 <button
//                   onClick={onClose}
//                   className="w-full bg-indigo-600 text-white p-2 rounded hover:bg-indigo-700"
//                 >
//                   닫기
//                 </button>
//               </div>
//             ) : (
//               <>
//                 <button
//                   className="w-full bg-gray-300 p-2 rounded mb-2"
//                   onClick={() => setIsLoginModalOpen(true)}
//                 >
//                   로그인
//                 </button>
//                 <button
//                   className="w-full bg-gray-300 p-2 rounded mb-2"
//                   onClick={() => setIsSignupModalOpen(true)}
//                 >
//                   회원가입
//                 </button>
//               </>
//             )}
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
//             <button
//               className="w-full bg-gray-300 p-2 rounded mb-2"
//               onClick={googleLogin}
//             >
//               Google로 로그인
//             </button>
//             <button
//               className="w-full bg-gray-300 p-2 rounded mb-2"
//               onClick={handleKakaoLogin}
//             >
//               Kakao로 로그인
//             </button>
//             <button
//               className="w-full bg-gray-300 p-2 rounded mb-4"
//               onClick={handleNaverLogin}
//             >
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
//               onClick={() => alert("로그인 시도")}
//             >
//               로그인
//             </button>
//             <div className="text-xs text-gray-600 mt-2">
//               아직 계정이 없나요?{" "}
//               <span
//                 className="text-blue-500 cursor-pointer"
//                 onClick={() => {
//                   setIsLoginModalOpen(false);
//                   setIsSignupModalOpen(true);
//                 }}
//               >
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
//                 if (
//                   window.confirm("Google 계정으로 회원가입을 진행하시겠습니까?")
//                 ) {
//                   googleLogin();
//                 }
//               }}
//             >
//               Google로 회원가입
//             </button>
//             <button
//               className="w-full bg-gray-300 p-2 rounded mb-2"
//               onClick={() => {
//                 if (
//                   window.confirm("Kakao 계정으로 회원가입을 진행하시겠습니까?")
//                 ) {
//                   handleKakaoLogin();
//                 }
//               }}
//             >
//               Kakao로 회원가입
//             </button>
//             <button
//               className="w-full bg-gray-300 p-2 rounded mb-2"
//               onClick={() => {
//                 if (
//                   window.confirm("Naver 계정으로 회원가입을 진행하시겠습니까?")
//                 ) {
//                   handleNaverLogin();
//                 }
//               }}
//             >
//               Naver로 회원가입
//             </button>
//             <div className="text-xs text-gray-600 mt-2">
//               계정이 없으신가요?{" "}
//               <span
//                 className="text-blue-500 cursor-pointer"
//                 onClick={() => {
//                   setIsSignupModalOpen(false);
//                   setIsLoginModalOpen(true);
//                 }}
//               >
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

  // 🔧 토큰 저장 함수 - 모든 가능한 키로 저장
  const saveAuthToken = (token, userData) => {
    // 일정 관리 시스템에서 찾는 모든 키에 저장
    localStorage.setItem('access_token', token);
    localStorage.setItem('token', token);
    localStorage.setItem('authToken', token);
    localStorage.setItem('accessToken', token);
    localStorage.setItem('user', JSON.stringify(userData));
    
    console.log('토큰 저장 완료:', {
      access_token: token.substring(0, 10) + '...',
      user: userData.username || userData.email
    });
  };

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
      
      // 🔧 통일된 토큰 저장
      saveAuthToken(data.access_token, data.user);

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
      setLoading(true);
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
        
        // 🔧 통일된 토큰 저장
        saveAuthToken(data.access_token, data.user);

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
      } finally {
        setLoading(false);
      }
    },
    onError: () => {
      console.error("구글 로그인 실패");
      setLoading(false);
    }
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
      
      // 🔧 통일된 토큰 저장
      saveAuthToken(data.access_token, data.user);

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

  // 🔧 컴포넌트 마운트 시 토큰 확인
  useEffect(() => {
    const checkTokens = () => {
      const tokens = {
        access_token: localStorage.getItem('access_token'),
        token: localStorage.getItem('token'),
        authToken: localStorage.getItem('authToken'),
        accessToken: localStorage.getItem('accessToken')
      };
      
      console.log('현재 저장된 토큰들:', tokens);
      
      // 하나라도 토큰이 있으면 모든 키에 동일하게 저장
      const availableToken = tokens.access_token || tokens.token || tokens.authToken || tokens.accessToken;
      if (availableToken && user) {
        saveAuthToken(availableToken, user);
      }
    };
    
    checkTokens();
  }, [user]);

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
                {/* 🔧 토큰 상태 표시 */}
                <div className="bg-blue-100 border border-blue-400 text-blue-700 p-2 rounded text-xs">
                  <p>토큰 상태: {localStorage.getItem('access_token') ? '✅ 정상' : '❌ 없음'}</p>
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
              disabled={loading}
            >
              {loading ? "처리중..." : "Google로 로그인"}
            </button>
            <button
              className="w-full bg-gray-300 p-2 rounded mb-2"
              onClick={handleKakaoLogin}
              disabled={loading}
            >
              {loading ? "처리중..." : "Kakao로 로그인"}
            </button>
            <button
              className="w-full bg-gray-300 p-2 rounded mb-4"
              onClick={handleNaverLogin}
              disabled={loading}
            >
              {loading ? "처리중..." : "Naver로 로그인"}
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