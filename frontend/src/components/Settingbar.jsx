import React, { useState, useEffect } from "react";
import { X } from "lucide-react";
import { useChat } from '../context/ChatContext';
import { useSelector, useDispatch } from 'react-redux';
import { updateUserSettings } from '../store/authSlice';

const languages = [
  "Afrikaans", "Bahasa Indonesia", "Bahasa Melayu", "Català", "Čeština", "Dansk", "Deutsch", 
  "Eesti", "English (United Kingdom)", "English (United States)", "Español (España)", "Español (Latinoamérica)", 
  "Euskara", "Filipino", "Français (Canada)", "Français (France)", "Galego", "Hrvatski", "IsiZulu", "Íslenska", 
  "Italiano", "Kiswahili", "Latviešu", "Lietuvių", "Magyar", "Nederlands", "Norsk", "Polski", 
  "Português (Brasil)", "Português (Portugal)", "Română", "Slovenčina", "Slovenščina", "Suomi", "Svenska", 
  "Tiếng Việt", "Türkçe", "Ελληνικά", "Български", "Русский", "Српски", "Українська", "Հայերեն", "עברית", 
  "اردو", "العربية", "فارسی", "मराठी", "हिन्दी", "বাংলা", "ગુજરાતી", "தமிழ்", "తెలుగు", "ಕನ್ನಡ", "മലയാളം", 
  "ไทย", "한국어", "中文 (简体)", "中文 (繁體)", "日本語"
];

const Settingbar = ({ isOpen, onClose }) => {
  const [isLanguageSelectionOpen, setIsLanguageSelectionOpen] = useState(false);
  const [selectedLanguage, setSelectedLanguage] = useState(null);
  const [showConfirmButton, setShowConfirmButton] = useState(false);
  const [showLoginAlert, setShowLoginAlert] = useState(false);

  const { 
    isSelectionModalOpen, 
    setIsSelectionModalOpen,
    selectedModel,
    setSelectedModel,
    isLoggedIn
  } = useChat();

  const dispatch = useDispatch();
  const { user } = useSelector((state) => state.auth);
  
  useEffect(() => {
    if (user?.settings) {
      setSelectedLanguage(user.settings.language || null);
      if (setSelectedModel) {
        setSelectedModel(user.settings.model || 'default');
      }
    }
  }, [user, setSelectedModel]);

  const handleSettingClick = () => {
    if (!isLoggedIn) {
      setShowLoginAlert(true);
      setTimeout(() => setShowLoginAlert(false), 3000);
      return;
    }
    // (열기만) — 실제 열기는 버튼 onClick에서 분기 처리
  };

  const saveSettings = async (settings) => {
    if (!isLoggedIn) {
      setShowLoginAlert(true);
      setTimeout(() => setShowLoginAlert(false), 3000);
      return;
    }

    try {
      const token = localStorage.getItem("accessToken");
      if (!token) {
        throw new Error("인증 토큰이 없습니다.");
      }

      const settingsData = {
        language: settings?.language ?? user?.settings?.language ?? 'en',
        // 선택적으로 모델도 저장
        model: settings?.preferredModel ?? user?.settings?.model,
      };

      const response = await fetch("http://localhost:8000/api/user/settings/", {
        method: "PUT",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Token ${token}`,
        },
        body: JSON.stringify(settingsData),
        credentials: 'include',
      });

      if (!response.ok) {
        let errorMsg = '설정 저장에 실패했습니다.';
        try {
          const errorData = await response.json();
          errorMsg = errorData.message || errorData.error || errorMsg;
        } catch (_) {}
        throw new Error(errorMsg);
      }

      const data = await response.json();
      dispatch(updateUserSettings(data.settings));
      return data;
    } catch (error) {
      console.error('Error saving settings:', error);
      throw error;
    }
  };

  const handleConfirm = async () => {
    if (!isLoggedIn) {
      setShowLoginAlert(true);
      setTimeout(() => setShowLoginAlert(false), 3000);
      return;
    }

    if (!selectedLanguage) {
      alert('언어를 선택해주세요.');
      return;
    }

    try {
      await saveSettings({ language: selectedLanguage });
      setIsLanguageSelectionOpen(false);
      onClose();
    } catch (error) {
      alert(error.message || '설정 저장에 실패했습니다. 다시 시도해주세요.');
    }
  };

  const handleModelSelection = async (model) => {
    if (!isLoggedIn) {
      setShowLoginAlert(true);
      setTimeout(() => setShowLoginAlert(false), 3000);
      return;
    }

    try {
      await saveSettings({
        language: selectedLanguage,
        preferredModel: model
      });
      setSelectedModel(model);
      setIsSelectionModalOpen(false);
    } catch (error) {
      alert(error.message || '모델 설정 저장에 실패했습니다. 다시 시도해주세요.');
    }
  };

  return (
    <>
      {/* 메인 설정 모달 — 1번 디자인 */}
      {isOpen && !isLanguageSelectionOpen && !isSelectionModalOpen && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-96 shadow-lg relative">
            <X className="absolute top-3 right-3 w-6 h-6 cursor-pointer text-gray-500 hover:text-gray-700" onClick={onClose} />
            <h3 className="text-xl font-bold mb-2 text-left" style={{ color: '#2d3e2c' }}>설정</h3>
            <p className="text-sm text-gray-600 mb-4 text-left">개인화된 AI 경험을 위해 설정을 변경하세요.</p>
            <hr className="w-full border-gray-300 mb-4" />
            <div className="space-y-4 w-full">
              <button
                className="w-full p-4 border border-gray-200 rounded-lg transition-colors font-bold"
                style={{ color: '#2d3e2c', backgroundColor: 'white' }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.backgroundColor = 'rgba(139, 168, 138, 0.05)';
                  e.currentTarget.style.borderColor = 'rgba(139, 168, 138, 0.4)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.backgroundColor = 'white';
                  e.currentTarget.style.borderColor = '#d1d5db';
                }}
                onClick={() => !isLoggedIn ? handleSettingClick() : setIsLanguageSelectionOpen(true)}
              >
                언어 선택
              </button>

              <button
                className="w-full p-4 border border-gray-200 rounded-lg transition-colors font-bold"
                style={{ color: '#2d3e2c', backgroundColor: 'white' }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.backgroundColor = 'rgba(139, 168, 138, 0.05)';
                  e.currentTarget.style.borderColor = 'rgba(139, 168, 138, 0.4)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.backgroundColor = 'white';
                  e.currentTarget.style.borderColor = '#d1d5db';
                }}
                onClick={() => !isLoggedIn ? handleSettingClick() : setIsSelectionModalOpen(true)}
              >
                최적화 모델 선택
              </button>
            </div>
          </div>
        </div>
      )}

      {/* 로그인 안내 토스트 */}
      {showLoginAlert && (
        <div className="fixed top-4 right-4 bg-blue-500 text-white px-6 py-3 rounded-lg shadow-lg z-50 animate-fade-in-out">
          로그인 후 설정이 가능합니다.
        </div>
      )}

      {/* 언어 선택 모달 — 1번 디자인 */}
      {isLoggedIn && isLanguageSelectionOpen && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div 
            className="bg-white rounded-lg w-full max-w-md max-h-[60vh] flex flex-col relative"
          >
            <div className="p-6">
              <X className="absolute top-3 right-3 w-6 h-6 cursor-pointer text-gray-500 hover:text-gray-700" onClick={() => setIsLanguageSelectionOpen(false)} />
              <h3 className="text-xl font-bold mb-2 text-left" style={{ color: '#2d3e2c' }}>언어 선택</h3>
              <p className="text-sm text-gray-600 mb-0.1 text-left">AI에게 응답받을 언어를 선택하세요.</p>
            </div>
            
            <div 
              className="flex-1 overflow-y-auto px-6 border-t"
              onScroll={(e) => setShowConfirmButton(
                e.currentTarget.scrollTop + e.currentTarget.clientHeight >= e.currentTarget.scrollHeight
              )}
            >
              <div className="grid grid-cols-2 gap-2 py-4">
                {languages.map((lang) => (
                  <button
                    key={lang}
                    onClick={() => setSelectedLanguage(lang)}
                    className="p-2 border border-gray-200 rounded-lg transition-colors"
                    style={selectedLanguage === lang ? { 
                      borderColor: 'rgba(139, 168, 138, 0.4)', 
                      backgroundColor: 'rgba(139, 168, 138, 0.05)', 
                      color: '#2d3e2c' 
                    } : { backgroundColor: 'white', color: '#2d3e2c' }}
                    onMouseEnter={(e) => {
                      if (selectedLanguage !== lang) {
                        e.currentTarget.style.backgroundColor = 'rgba(139, 168, 138, 0.05)';
                        e.currentTarget.style.borderColor = 'rgba(139, 168, 138, 0.4)';
                      }
                    }}
                    onMouseLeave={(e) => {
                      if (selectedLanguage !== lang) {
                        e.currentTarget.style.backgroundColor = 'white';
                        e.currentTarget.style.borderColor = '#d1d5db';
                      }
                    }}
                  >
                    {lang}
                  </button>
                ))}
              </div>
            </div>

            <div className="p-6 border-t">
              <button 
                className={`w-full px-6 py-3 rounded-lg transition-colors ${
                  selectedLanguage 
                    ? "text-white" 
                    : "bg-gray-300 text-gray-500 cursor-not-allowed"
                }`}
                style={selectedLanguage ? { backgroundColor: '#8ba88a' } : {}}
                onMouseEnter={(e) => selectedLanguage && (e.currentTarget.style.backgroundColor = '#5d7c5b')}
                onMouseLeave={(e) => selectedLanguage && (e.currentTarget.style.backgroundColor = '#8ba88a')}
                onClick={handleConfirm}
                disabled={!selectedLanguage}
              >
                확인
              </button>
            </div>
          </div>
        </div>
      )}

      {/* AI 선택 모달 — 1번 디자인 + 기능 유지 */}
      {isLoggedIn && isSelectionModalOpen && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-full max-w-2xl shadow-lg relative pb-20">
            <X className="absolute top-3 right-3 w-6 h-6 cursor-pointer text-gray-500 hover:text-gray-700" onClick={() => setIsSelectionModalOpen(false)} />
            <h3 className="text-xl font-bold mb-2 text-left" style={{ color: '#2d3e2c' }}>최적화 모델 선택</h3>
            <p className="text-sm text-gray-600 mb-4 text-left">최적의 응답을 생성할 AI 모델을 선택하세요.</p>
            <hr className="w-full border-gray-300 mb-4" />

            <div className="grid grid-cols-3 gap-4 mb-6">
              {["GPT-3.5", "Claude", "Mixtral"].map((model) => (
                <button
                  key={model}
                  onClick={() => {
                    setSelectedModel(model);
                    // 기존 기능 유지: 클릭 즉시 저장 시도
                    handleModelSelection(model);
                  }}
                  className="p-6 border border-gray-200 rounded-lg transition-colors"
                  style={selectedModel === model ? { 
                    borderColor: 'rgba(139, 168, 138, 0.4)', 
                    backgroundColor: 'rgba(139, 168, 138, 0.05)' 
                  } : { backgroundColor: 'white' }}
                  onMouseEnter={(e) => {
                    if (selectedModel !== model) {
                      e.currentTarget.style.backgroundColor = 'rgba(139, 168, 138, 0.05)';
                      e.currentTarget.style.borderColor = 'rgba(139, 168, 138, 0.4)';
                    }
                  }}
                  onMouseLeave={(e) => {
                    if (selectedModel !== model) {
                      e.currentTarget.style.backgroundColor = 'transparent';
                      e.currentTarget.style.borderColor = '#d1d5db';
                    }
                  }}
                >
                  <h3 className="font-bold text-lg mb-2" style={{ color: '#2d3e2c' }}>{model}</h3>
                  <p className="text-sm text-gray-600 mb-2">
                    {model === "GPT-3.5" ? "OpenAI의 GPT-3.5 모델" : 
                     model === "Claude" ? "Anthropic의 Claude 모델" : 
                     "Mixtral-8x7B 모델"}
                  </p>
                  <ul className="text-xs text-gray-500 list-disc pl-4">
                    {model === "GPT-3.5" && (<>
                      <li>빠른 응답 속도</li>
                      <li>일관된 답변 품질</li>
                      <li>다양한 주제 처리</li>
                    </>)}
                    {model === "Claude" && (<>
                      <li>높은 분석 능력</li>
                      <li>정확한 정보 제공</li>
                      <li>상세한 설명 제공</li>
                    </>)}
                    {model === "Mixtral" && (<>
                      <li>균형잡힌 성능</li>
                      <li>다국어 지원</li>
                      <li>코드 분석 특화</li>
                    </>)}
                  </ul>
                </button>
              ))}
            </div>

            {/* 선택 후 확정 버튼도 허용(네트워크 지연/실패 대비) */}
            <button
              onClick={() => selectedModel && handleModelSelection(selectedModel)}
              className={`absolute bottom-6 right-6 px-6 py-3 rounded-lg transition-colors ${
                selectedModel 
                  ? "text-white" 
                  : "bg-gray-300 text-gray-500 cursor-not-allowed"
              }`}
              style={selectedModel ? { backgroundColor: '#8ba88a' } : {}}
              onMouseEnter={(e) => selectedModel && (e.currentTarget.style.backgroundColor = '#5d7c5b')}
              onMouseLeave={(e) => selectedModel && (e.currentTarget.style.backgroundColor = '#8ba88a')}
              disabled={!selectedModel}
            >
              확인
            </button>
          </div>
        </div>
      )}
    </>
  );
};

export default Settingbar;
