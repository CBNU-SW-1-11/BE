
// import React, { useEffect } from "react";
// import { Provider, useDispatch } from "react-redux";
// import { GoogleOAuthProvider } from "@react-oauth/google";
// import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
// import { store } from "./store";
// import MainPage from "./pages/MainPage";
// import KakaoCallback from "./components/KakaoCallback";
// import NaverCallback from "./components/NaverCallback";
// import OCRToolPage from "./pages/OCRToolPage"; // 새로 추가된 OCR 도구 페이지
// import { ChatProvider } from "./context/ChatContext";
// import axios from "axios";
// import { loginSuccess } from "./store/authSlice";
// import ScheduleManagement from "./services/scheduleService";
// axios.defaults.baseURL = "http://localhost:8000";

// // App 초기화 시 localStorage에 저장된 인증 정보를 Redux에 복원
// function AuthInitializer({ children }) {
//   const dispatch = useDispatch();
//   useEffect(() => {
//     const token = localStorage.getItem("accessToken");
//     const user = localStorage.getItem("user");
//     if (token && user) {
//       dispatch(loginSuccess({ token, user: JSON.parse(user) }));
//     }
//   }, [dispatch]);
//   return children;
// }

// function App() {
//   return (
//     <Provider store={store}>
//       <GoogleOAuthProvider clientId={process.env.REACT_APP_GOOGLE_CLIENT_ID}>
//         <ChatProvider>
//           <Router>
//             <AuthInitializer>
//               <Routes>
//                 <Route path="/" element={<MainPage />} />
//                 <Route path="/ocr-tool" element={<OCRToolPage />} />
//                 <Route path="/auth/kakao/callback" element={<KakaoCallback />} />
//                 <Route path="/auth/naver/callback" element={<NaverCallback />} />
//                 <Route path="/schedule-management" element={<ScheduleManagement />} />
//               </Routes>
//             </AuthInitializer>
//           </Router>
//         </ChatProvider>
//       </GoogleOAuthProvider>
//     </Provider>
//   );
// }

// export default App;
import { safeJsonParse } from './utils/safeJson';

import React, { useEffect } from "react";
import { Provider, useDispatch } from "react-redux";
import { GoogleOAuthProvider } from "@react-oauth/google";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import { store } from "./store";
import MainPage from "./pages/MainPage";
import KakaoCallback from "./components/KakaoCallback";
import NaverCallback from "./components/NaverCallback";
import OCRToolPage from "./pages/OCRToolPage";
import { ChatProvider } from "./context/ChatContext";
import axios from "axios";
import { loginSuccess } from "./store/authSlice";
import ScheduleManagement from "./services/scheduleService";

// 새로 추가: 비디오 관련 페이지들
import VideoAnalysisPage from "./pages/VideoAnalysisPage"; // 비디오 분석 메인 페이지
import IntegratedChatPage from "./pages/IntegratedChatPage"; // 통합 채팅 페이지
import VideoUploadPage from "./pages/VideoUploadPage"; // 비디오 업로드 페이지
import VideoDetailPage from "./pages/VideoDetailPage";
axios.defaults.baseURL = "http://localhost:8000";

// App 초기화 시 localStorage에 저장된 인증 정보를 Redux에 복원
function AuthInitializer({ children }) {
  const dispatch = useDispatch();
  useEffect(() => {
    const token = localStorage.getItem("accessToken");
    const user = localStorage.getItem("user");
    if (token && user) {
      dispatch(loginSuccess({ token, user: safeJsonParse(user, {}) }));
    }
  }, [dispatch]);
  return children;
}

function App() {
  return (
    <Provider store={store}>
      <GoogleOAuthProvider clientId={process.env.REACT_APP_GOOGLE_CLIENT_ID}>
        <ChatProvider>
          <Router>
            <AuthInitializer>
              <Routes>
                {/* 기존 라우트들 */}
                <Route path="/" element={<MainPage />} />
                <Route path="/ocr-tool" element={<OCRToolPage />} />
                <Route path="/auth/kakao/callback" element={<KakaoCallback />} />
                <Route path="/auth/naver/callback" element={<NaverCallback />} />
                <Route path="/schedule-management" element={<ScheduleManagement />} />
                
                {/* 새로 추가: 비디오 관련 라우트들 */}
                <Route path="/video-analysis" element={<VideoAnalysisPage />} />
                <Route path="/video-upload" element={<VideoUploadPage />} />
                <Route path="/integrated-chat" element={<IntegratedChatPage />} />
                
                {/* 선택사항: 비디오 상세 페이지 */}
                <Route path="/video/:videoId" element={<VideoDetailPage />} />
              </Routes>
            </AuthInitializer>
          </Router>
        </ChatProvider>
      </GoogleOAuthProvider>
    </Provider>
  );
}

export default App;