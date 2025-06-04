
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
