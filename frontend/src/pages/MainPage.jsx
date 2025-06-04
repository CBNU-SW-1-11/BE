// // import React, { useState } from "react";
// // import Sidebar from "../components/Sidebar";
// // import ChatBox from "../components/ChatBox";
// // import Settingbar from "../components/Settingbar"; // ✅ default import 가능
// // import Loginbar from "../components/Loginbar";
// // import { Menu, Search, Clock, Settings, UserCircle } from "lucide-react";

// // const MainPage = () => {
  
// //   const [isSidebarVisible, setIsSidebarVisible] = useState(false);
// //   const [isSettingVisible, setIsSettingVisible] = useState(false);
// //   const [isLoginVisible, setIsLoginVisible] = useState(false);

// //   // 설정창과 로그인창이 동시에 열리지 않도록 조정
// //   const toggleSetting = () => {
// //     setIsSettingVisible(!isSettingVisible);
// //     setIsLoginVisible(false); // 설정창을 열 때 로그인창은 닫기
// //   };

// //   const toggleLogin = () => {
// //     setIsLoginVisible(!isLoginVisible);
// //     setIsSettingVisible(false); // 로그인창을 열 때 설정창은 닫기
// //   };

  

// //   return (
// //     <div className="flex flex-col h-screen bg-white relative">
// //       {/* 상단 네비게이션 바 */}
// //       <nav className="bg-white border-b px-6 py-3 flex items-center justify-between">
// //         <div className="flex items-center space-x-4">
// //           <Menu
// //             className="w-6 h-6 text-gray-600 cursor-pointer"
// //             onClick={() => setIsSidebarVisible(!isSidebarVisible)}
// //           />
// //           <h1 className="text-xl font-semibold">AI Chatbot</h1>
// //         </div>
// //         <div className="flex items-center space-x-4">
// //           {/*<Search className="w-5 h-5 text-gray-600" />
// //           <Clock className="w-5 h-5 text-gray-600" />*/}
// //           <UserCircle
// //             className="w-5 h-5 text-gray-600 cursor-pointer"
// //             onClick={toggleLogin}
// //           />
// //           <Settings
// //             className="w-5 h-5 text-gray-600 cursor-pointer"
// //             onClick={toggleSetting}
// //           />
// //         </div>
// //       </nav>

// //       {/* 메인 컨텐츠 영역 */}
// //       <div className="flex flex-1 min-h-0 overflow-hidden"> 
// //         {isSidebarVisible && <Sidebar />}
// //         <ChatBox />
// //       </div>

// //       {/* 로그인 모달 */}
// //       {isLoginVisible && <Loginbar onClose={() => setIsLoginVisible(false)} />}

// //       {/* 설정 모달 */}
// //       <Settingbar isOpen={isSettingVisible} onClose={() => setIsSettingVisible(false)} />
// //     </div>
// //   );
// // };

// // export default MainPage;
// // React와 필요한 훅 import
// import React, { useState, useEffect } from 'react';
// import { useSelector } from 'react-redux';

// // 필요한 컴포넌트와 아이콘 import
// import Sidebar from '../components/Sidebar';
// import ChatBox from '../components/ChatBox';
// import Loginbar from '../components/Loginbar';
// import Settingbar from '../components/Settingbar';
// import { Menu, Search, Clock, Settings, UserCircle } from 'lucide-react';

// // MainPage 컴포넌트 정의
// const MainPage = () => {
//   // useState를 사용하여 상태 관리
//   const [isSidebarVisible, setIsSidebarVisible] = useState(false);
//   const [isSettingVisible, setIsSettingVisible] = useState(false);
//   const [isLoginVisible, setIsLoginVisible] = useState(false);

//   const user = useSelector(state => state.auth.user); // Redux store에서 유저 상태 접근

//   const toggleSetting = () => {
//     setIsSettingVisible(!isSettingVisible);
//     setIsLoginVisible(false);
//   };

//   const toggleLogin = () => {
//     setIsLoginVisible(!isLoginVisible);
//     setIsSettingVisible(false);
//   };

//   return (
//     <div className="flex flex-col h-screen bg-white relative">
//       <nav className="bg-white border-b px-6 py-3 flex items-center justify-between">
//         <div className="flex items-center space-x-4">
//           <Menu
//             className="w-6 h-6 text-gray-600 cursor-pointer"
//             onClick={() => setIsSidebarVisible(!isSidebarVisible)}
//           />
//           <h1 className="text-xl font-semibold">AI Chatbot</h1>
//         </div>
//         <div className="flex items-center space-x-4">
//           {user ? (
//             <div className="flex items-center space-x-2">
//               <span className="text-gray-600 cursor-pointer">{user.nickname || user.username}</span>
//               <Settings
//                 className="w-5 h-5 text-gray-600 cursor-pointer"
//                 onClick={toggleSetting}
//               />
//             </div>
//           ) : (
//             <>
//               <UserCircle
//                 className="w-5 h-5 text-gray-600 cursor-pointer"
//                 onClick={toggleLogin}
//               />
//               <Settings
//                 className="w-5 h-5 text-gray-600 cursor-pointer"
//                 onClick={toggleSetting}
//               />
//             </>
//           )}
//         </div>
//       </nav>
//       <div className="flex flex-1 min-h-0 overflow-hidden">
//         {isSidebarVisible && <Sidebar />}
//         <ChatBox />
//       </div>
//       {isLoginVisible && <Loginbar onClose={() => setIsLoginVisible(false)} />}
//       <Settingbar isOpen={isSettingVisible} onClose={() => setIsSettingVisible(false)} />
//     </div>
//   );
// };

// // export default MainPage;
// // src/pages/MainPage.js
// import React, { useState } from 'react';
// import { useSelector, useDispatch } from 'react-redux';
// import { logout } from '../store/authSlice';
// import { useNavigate } from 'react-router-dom';
// // 필요한 컴포넌트와 아이콘 import
// import Sidebar from '../components/Sidebar';
// import ChatBox from '../components/ChatBox';
// import Loginbar from '../components/Loginbar';
// import Settingbar from '../components/Settingbar';
// import { Menu, Settings, UserCircle } from 'lucide-react';

// const MainPage = () => {
//   const [isSidebarVisible, setIsSidebarVisible] = useState(false);
//   const [isSettingVisible, setIsSettingVisible] = useState(false);
//   const [isLoginVisible, setIsLoginVisible] = useState(false);

//   const user = useSelector((state) => state.auth.user);
//   const dispatch = useDispatch();
//   const navigate = useNavigate();

//   const toggleSetting = () => {
//     setIsSettingVisible(!isSettingVisible);
//     setIsLoginVisible(false);
//   };

//   const toggleLogin = () => {
//     setIsLoginVisible(!isLoginVisible);
//     setIsSettingVisible(false);
//   };

//   const handleLogout = () => {
//     // 로컬 스토리지에서 토큰과 회원 정보 제거
//     localStorage.removeItem("accessToken");
//     localStorage.removeItem("user");
//     // Redux 스토어 초기화
//     dispatch(logout());
//     navigate("/");
//   };

//   return (
//     <div className="flex flex-col h-screen bg-white relative">
//       <nav className="bg-white border-b px-6 py-3 flex items-center justify-between">
//         <div className="flex items-center space-x-4">
//           <Menu
//             className="w-6 h-6 text-gray-600 cursor-pointer"
//             onClick={() => setIsSidebarVisible(!isSidebarVisible)}
//           />
//           <h1 className="text-xl font-semibold">AI Chatbot</h1>
//         </div>
//         <div className="flex items-center space-x-4">
//           {user ? (
//             <div className="flex items-center space-x-2">
//               <span className="text-gray-600 cursor-pointer">
//                 {user.nickname || user.username}
//               </span>
//               <button
//                 onClick={handleLogout}
//                 className="text-sm text-red-500 cursor-pointer"
//               >
//                 로그아웃
//               </button>
//               <Settings
//                 className="w-5 h-5 text-gray-600 cursor-pointer"
//                 onClick={toggleSetting}
//               />
//             </div>
//           ) : (
//             <>
//               <UserCircle
//                 className="w-5 h-5 text-gray-600 cursor-pointer"
//                 onClick={toggleLogin}
//               />
//               <Settings
//                 className="w-5 h-5 text-gray-600 cursor-pointer"
//                 onClick={toggleSetting}
//               />
//             </>
//           )}
//         </div>
//       </nav>
//       <div className="flex flex-1 min-h-0 overflow-hidden">
//         {isSidebarVisible && <Sidebar />}
//         <ChatBox />
//       </div>
//       {isLoginVisible && <Loginbar onClose={() => setIsLoginVisible(false)} />}
//       <Settingbar isOpen={isSettingVisible} onClose={() => setIsSettingVisible(false)} />
//     </div>
//   );
// };

// export default MainPage;
// src/pages/MainPage.js
import React, { useState } from "react";
import { useSelector, useDispatch } from "react-redux";
import Loginbar from "../components/Loginbar";
import Settingbar from "../components/Settingbar";
import Sidebar from "../components/Sidebar";
import ChatBox from "../components/ChatBox";
import { Menu, Settings, UserCircle } from "lucide-react";
import { logout } from "../store/authSlice";
import { useNavigate } from "react-router-dom";
import ModelSelectionModal from "../components/ModelSelectionModal";
import {  CirclePlus } from "lucide-react";
import { useChat } from "../context/ChatContext";

const MainPage = () => {
  const [isSidebarVisible, setIsSidebarVisible] = useState(false);
  const [isSettingVisible, setIsSettingVisible] = useState(false);
  const [isLoginVisible, setIsLoginVisible] = useState(false);

  const user = useSelector((state) => state.auth.user);
  const dispatch = useDispatch();
  const navigate = useNavigate();

  const toggleSetting = () => {
    setIsSettingVisible(!isSettingVisible);
    setIsLoginVisible(false);
  };

  const toggleLogin = () => {
    setIsLoginVisible(!isLoginVisible);
    setIsSettingVisible(false);
  };

  const handleLogout = () => {
    localStorage.removeItem("accessToken");
    localStorage.removeItem("user");
    dispatch(logout());
    navigate("/");
  };
  const [isModelModalOpen, setIsModelModalOpen] = useState(false);
  const { selectedModels, setSelectedModels } = useChat();

  return (
    <div className="flex flex-col h-screen bg-white relative">
      <nav className="bg-white border-b px-6 py-3 flex items-center justify-between">
        <div className="flex items-center space-x-4">
        <Menu className="w-6 h-6 text-gray-600 cursor-pointer" onClick={() => setIsSidebarVisible(!isSidebarVisible)} />
          <h1 className="text-xl font-semibold">AI Chatbot</h1>
        </div>
        <div className="flex items-center space-x-4">
          {user ? (
            <div className="flex items-center space-x-2">
              <span className="text-gray-600 cursor-pointer">{user.nickname || user.username}</span>
              <button onClick={handleLogout} className="text-sm text-gray-600 cursor-pointer">로그아웃</button>
              <CirclePlus className="w-6 h-6 text-gray-600 cursor-pointer" onClick={() => setIsModelModalOpen(true)} title="AI 모델 선택" />

              <Settings className="w-5 h-5 text-gray-600 cursor-pointer" onClick={toggleSetting} />

            </div>
          ) : (
            <>
          <CirclePlus className="w-6 h-6 text-gray-600 cursor-pointer" onClick={() => setIsModelModalOpen(true)} title="AI 모델 선택" />
          <UserCircle className="w-6 h-6 text-gray-600 cursor-pointer" onClick={() => setIsLoginVisible(!isLoginVisible)} />
          <Settings className="w-6 h-6 text-gray-600 cursor-pointer" onClick={() => setIsSettingVisible(!isSettingVisible)} />
            </>
          )}
        </div>
      </nav>
      <div className="flex flex-1 min-h-0 overflow-hidden">
        {isSidebarVisible && <Sidebar />}
        <ChatBox selectedModels={selectedModels} />
      </div>
      <ModelSelectionModal 
        isOpen={isModelModalOpen} 
        onClose={() => setIsModelModalOpen(false)}
        selectedModels={selectedModels}
        onModelSelect={setSelectedModels}
      />

      {isLoginVisible && <Loginbar onClose={() => setIsLoginVisible(false)} />}
      {isLoginVisible && <Loginbar onClose={() => setIsLoginVisible(false)} />}
      <Settingbar isOpen={isSettingVisible} onClose={() => setIsSettingVisible(false)} />
    </div>
  );
};

export default MainPage;
