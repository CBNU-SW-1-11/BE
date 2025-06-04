// // import React, { useState } from "react";
// // import { X } from "lucide-react";
// // import { useChat } from '../context/ChatContext'; // ChatContext import 추가

// // const Settingbar = ({ isOpen, onClose }) => {
// //   const [isLanguageSelectionOpen, setIsLanguageSelectionOpen] = useState(false);
// //   const [selectedLanguage, setSelectedLanguage] = useState(null);
// //   const [showConfirmButton, setShowConfirmButton] = useState(false);
// //   const { 
// //     isSelectionModalOpen, 
// //     setIsSelectionModalOpen 
// //   } = useChat(); // ChatContext의 모달 상태 사용
  
// //   const languages = [
// //     "Afrikaans", "Bahasa Indonesia", "Bahasa Melayu", "Català", "Čeština", "Dansk", "Deutsch", 
// //     "Eesti", "English (United Kingdom)", "English (United States)", "Español (España)", "Español (Latinoamérica)", 
// //     "Euskara", "Filipino", "Français (Canada)", "Français (France)", "Galego", "Hrvatski", "IsiZulu", "Íslenska", 
// //     "Italiano", "Kiswahili", "Latviešu", "Lietuvių", "Magyar", "Nederlands", "Norsk", "Polski", 
// //     "Português (Brasil)", "Português (Portugal)", "Română", "Slovenčina", "Slovenščina", "Suomi", "Svenska", 
// //     "Tiếng Việt", "Türkçe", "Ελληνικά", "Български", "Русский", "Српски", "Українська", "Հայերեն", "עברית", 
// //     "اردو", "العربية", "فارسی", "मराठी", "हिन्दी", "বাংলা", "ગુજરાતી", "தமிழ்", "తెలుగు", "ಕನ್ನಡ", "മലയാളം", 
// //     "ไทย", "한국어", "中文 (简体)", "中文 (繁體)", "日本語"
// //   ];

// //   const handleConfirm = () => {
// //     setIsLanguageSelectionOpen(false);
// //     setSelectedLanguage(null);
// //     onClose();
// //   };

// //   return (
// //     <>
// //       {/* 메인 설정 모달 */}
// //       {isOpen && !isLanguageSelectionOpen && !isSelectionModalOpen && (
// //         <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
// //           <div className="bg-white rounded-lg p-6 w-96 shadow-lg relative flex flex-col items-center">
// //             <X className="absolute top-3 right-3 w-6 h-6 cursor-pointer" onClick={onClose} />
// //             <h2 className="text-xl font-bold mb-4">설정</h2>
// //             <div className="space-y-4 w-full">
// //               <button
// //                 className="w-full p-4 border rounded-lg hover:bg-blue-50 transition-colors font-bold"
// //                 onClick={() => setIsLanguageSelectionOpen(true)}
// //               >
// //                 언어 선택
// //               </button>
// //               <button
// //                 className="w-full p-4 border rounded-lg hover:bg-blue-50 transition-colors font-bold"
// //                 onClick={() => setIsSelectionModalOpen(true)} // ChatContext의 모달 열기 함수 사용
// //               >
// //                 최적화 모델 선택
// //               </button>
// //             </div>
// //           </div>
// //         </div>
// //       )}

// //       {/* 언어 선택 모달 */}
// //       {isLanguageSelectionOpen && (
// //         <div 
// //           className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
// //         >
// //           <div 
// //             className="bg-white rounded-lg p-6 w-full max-w-md shadow-lg relative h-96 overflow-y-auto flex flex-col" 
// //             onScroll={(e) => setShowConfirmButton(
// //               e.target.scrollTop + e.target.clientHeight >= e.target.scrollHeight
// //             )}
// //           >
// //             <h2 className="text-xl font-bold mb-4">언어 선택</h2>
// //             <div className="grid grid-cols-2 gap-2 mb-6">
// //               {languages.map((lang) => (
// //                 <button
// //                   key={lang}
// //                   onClick={() => setSelectedLanguage(lang)}
// //                   className={`p-2 border rounded-lg transition-colors ${
// //                     selectedLanguage === lang ? "bg-blue-300" : "hover:bg-blue-50"
// //                   }`}
// //                 >
// //                   {lang}
// //                 </button>
// //               ))}
// //             </div>
// //             <button 
// //               className={`px-6 py-3 rounded-lg transition-colors self-end mt-auto shadow-md ${
// //                 selectedLanguage
// //                   ? "bg-blue-500 text-white hover:bg-blue-600"
// //                   : "bg-gray-300 text-gray-500 cursor-not-allowed"
// //               }`} 
// //               onClick={handleConfirm}
// //               disabled={!selectedLanguage}
// //             >
// //               확인
// //             </button>
// //           </div>
// //         </div>
// //       )}
// //     </>
// //   );
// // };

// // export default Settingbar;
// import React, { useState, useEffect } from "react";
// import { X } from "lucide-react";
// import { useChat } from '../context/ChatContext';
// import { useSelector, useDispatch } from 'react-redux';
// import { updateUserSettings } from '../store/authSlice';
//   const languages = [
//     "Afrikaans", "Bahasa Indonesia", "Bahasa Melayu", "Català", "Čeština", "Dansk", "Deutsch", 
//     "Eesti", "English (United Kingdom)", "English (United States)", "Español (España)", "Español (Latinoamérica)", 
//     "Euskara", "Filipino", "Français (Canada)", "Français (France)", "Galego", "Hrvatski", "IsiZulu", "Íslenska", 
//     "Italiano", "Kiswahili", "Latviešu", "Lietuvių", "Magyar", "Nederlands", "Norsk", "Polski", 
//     "Português (Brasil)", "Português (Portugal)", "Română", "Slovenčina", "Slovenščina", "Suomi", "Svenska", 
//     "Tiếng Việt", "Türkçe", "Ελληνικά", "Български", "Русский", "Српски", "Українська", "Հայերեն", "עברית", 
//     "اردو", "العربية", "فارسی", "मराठी", "हिन्दी", "বাংলা", "ગુજરાતી", "தமிழ்", "తెలుగు", "ಕನ್ನಡ", "മലയാളം", 
//     "ไทย", "한국어", "中文 (简体)", "中文 (繁體)", "日本語"
//   ];
// const Settingbar = ({ isOpen, onClose }) => {
//   const [isLanguageSelectionOpen, setIsLanguageSelectionOpen] = useState(false);
//   const [selectedLanguage, setSelectedLanguage] = useState(null);
//   const [showConfirmButton, setShowConfirmButton] = useState(false);
//   const { 
//     isSelectionModalOpen, 
//     setIsSelectionModalOpen,
//     selectedModel,
//     setSelectedModel 
//   } = useChat();

//   const dispatch = useDispatch();
//   const { user } = useSelector((state) => state.auth);
  
//   // 초기 설정 로드
//   useEffect(() => {
//     if (user?.settings) {
//       setSelectedLanguage(user.settings.language || null);
//       if (setSelectedModel) {
//         setSelectedModel(user.settings.preferredModel || 'default');
//       }
//     }
//   }, [user, setSelectedModel]);

//   // const saveSettings = async (settings) => {
//   //   try {
//   //     const token = localStorage.getItem("accessToken");
//   //     if (!token) {
//   //       throw new Error("인증 토큰이 없습니다.");
//   //     }
      
//   //     // 현재 유저의 기존 설정값 가져오기
//   //     const currentSettings = user?.settings || {};
      
//   //     // settings 객체 생성 (기존 설정 유지하면서 새로운 설정으로 업데이트)
//   //     const settingsData = {
//   //       language: settings?.language || currentSettings.language || selectedLanguage,
//   //       preferredModel: settings?.preferredModel || currentSettings.preferredModel || selectedModel || 'default'
//   //     };

//   //     // 필수 필드 검증
//   //     if (!settingsData.language) {
//   //       throw new Error("언어 설정이 필요합니다.");
//   //     }

//   //     console.log('Sending settings:', settingsData);
//   //     console.log('Using token:', token);

//   //     const response = await fetch("http://localhost:8000/api/user/settings/", {
//   //       method: "PUT",
//   //       headers: {
//   //         "Content-Type": "application/json",
//   //         "Authorization": `Token ${token}`,
//   //       },
//   //       body: JSON.stringify(settingsData),
//   //     });

//   //     const data = await response.json();
      
//   //     if (!response.ok) {
//   //       throw new Error(data.message || data.error || '설정 저장에 실패했습니다.');
//   //     }

//   //     console.log('Settings saved successfully:', data);
      
//   //     // Redux store 업데이트
//   //     dispatch(updateUserSettings(settingsData));

//   //     return data;
//   //   } catch (error) {
//   //     console.error('Error saving settings:', error);
//   //     throw error;
//   //   }
//   // };
//   const saveSettings = async (settings) => {
//     try {
//         const token = localStorage.getItem("accessToken");
//         if (!token) {
//             throw new Error("인증 토큰이 없습니다.");
//         }

//         const settingsData = {
//             language: settings?.language || user?.settings?.language || 'en',
//         };

//         console.log('Sending settings:', settingsData);

//         const response = await fetch("http://localhost:8000/api/user/settings/", {
//             method: "PUT",
//             headers: {
//                 "Content-Type": "application/json",
//                 "Authorization": `Token ${token}`,  // 토큰 형식 확인
//             },
//             body: JSON.stringify(settingsData),
//             credentials: 'include',  // 쿠키 포함
//         });

//         if (!response.ok) {
//             const errorData = await response.json();
//             throw new Error(errorData.message || errorData.error || '설정 저장에 실패했습니다.');
//         }

//         const data = await response.json();
//         console.log('Settings saved successfully:', data);
        
//         // Redux store 업데이트
//         dispatch(updateUserSettings(data.settings));

//         return data;
//     } catch (error) {
//         console.error('Error saving settings:', error);
//         throw error;
//     }
// };
//   const handleConfirm = async () => {
//     if (!selectedLanguage) {
//       alert('언어를 선택해주세요.');
//       return;
//     }

//     try {
//       await saveSettings({
//         language: selectedLanguage,
//       });
//       setIsLanguageSelectionOpen(false);
//       onClose();
//     } catch (error) {
//       alert(error.message || '설정 저장에 실패했습니다. 다시 시도해주세요.');
//     }
//   };

//   const handleModelSelection = async (model) => {
//     try {
//       await saveSettings({
//         language: selectedLanguage,
//         preferredModel: model
//       });
//       setSelectedModel(model);
//       setIsSelectionModalOpen(false);
//     } catch (error) {
//       alert(error.message || '모델 설정 저장에 실패했습니다. 다시 시도해주세요.');
//     }
//   };

//   return (
//     <>
//       {/* 메인 설정 모달 */}
//       {isOpen && !isLanguageSelectionOpen && !isSelectionModalOpen && (
//         <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
//           <div className="bg-white rounded-lg p-6 w-96 shadow-lg relative flex flex-col items-center">
//             <X className="absolute top-3 right-3 w-6 h-6 cursor-pointer" onClick={onClose} />
//             <h2 className="text-xl font-bold mb-4">설정</h2>
//             <div className="space-y-4 w-full">
//               <div className="w-full">
//                 <button
//                   className="w-full p-4 border rounded-lg hover:bg-blue-50 transition-colors font-bold"
//                   onClick={() => setIsLanguageSelectionOpen(true)}
//                 >
//                   언어 선택
//                 </button>
//                 <p className="text-sm text-gray-500 mt-1 ml-2">
//                   현재 언어: {user?.settings?.language || '설정되지 않음'}
//                 </p>
//               </div>
//               <div className="w-full">
//                 <button
//                   className="w-full p-4 border rounded-lg hover:bg-blue-50 transition-colors font-bold"
//                   onClick={() => setIsSelectionModalOpen(true)}
//                 >
//                   최적화 모델 선택
//                 </button>
//                 <p className="text-sm text-gray-500 mt-1 ml-2">
//                   현재 모델: {user?.settings?.preferredModel || '설정되지 않음'}
//                 </p>
//               </div>
//             </div>
//           </div>
//         </div>
//       )}

//       {/* 언어 선택 모달 */}
//       {isLanguageSelectionOpen && (
//         <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
//           <div 
//             className="bg-white rounded-lg p-6 w-full max-w-md shadow-lg relative h-96 overflow-y-auto flex flex-col"
//             onScroll={(e) => setShowConfirmButton(
//               e.target.scrollTop + e.target.clientHeight >= e.target.scrollHeight
//             )}
//           >
//             <h2 className="text-xl font-bold mb-4">언어 선택</h2>
//             <div className="grid grid-cols-2 gap-2 mb-6">
//               {languages.map((lang) => (
//                 <button
//                   key={lang}
//                   onClick={() => setSelectedLanguage(lang)}
//                   className={`p-2 border rounded-lg transition-colors ${
//                     selectedLanguage === lang ? "bg-blue-300" : "hover:bg-blue-50"
//                   }`}
//                 >
//                   {lang}
//                 </button>
//               ))}
//             </div>
//             <button 
//               className={`px-6 py-3 rounded-lg transition-colors self-end mt-auto shadow-md ${
//                 selectedLanguage
//                   ? "bg-blue-500 text-white hover:bg-blue-600"
//                   : "bg-gray-300 text-gray-500 cursor-not-allowed"
//               }`} 
//               onClick={handleConfirm}
//               disabled={!selectedLanguage}
//             >
//               확인
//             </button>
//           </div>
//         </div>
//       )}
//     </>
//   );
// };

// export default Settingbar;

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
      setTimeout(() => setShowLoginAlert(false), 3000); // 3초 후 알림 닫기
      return;
    }
    // 기존 기능 실행...
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
        language: settings?.language || user?.settings?.language || 'en',
      };
           console.log('Sending settings:', settingsData);

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
        const errorData = await response.json();
        throw new Error(errorData.message || errorData.error || '설정 저장에 실패했습니다.');
      }

      const data = await response.json();
console.log('Settings saved successfully:', data);

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
      await saveSettings({
        language: selectedLanguage,
      });
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
      {/* 메인 설정 모달 */}
      {isOpen && !isLanguageSelectionOpen && !isSelectionModalOpen && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-96 shadow-lg relative flex flex-col items-center">
            <X className="absolute top-3 right-3 w-6 h-6 cursor-pointer" onClick={onClose} />
            <h2 className="text-xl font-bold mb-4">설정</h2>
            <div className="space-y-4 w-full">
              <div className="w-full">
                <button
                  className="w-full p-4 border rounded-lg hover:bg-blue-50 transition-colors font-bold"
                  onClick={() => !isLoggedIn ? handleSettingClick() : setIsLanguageSelectionOpen(true)}
                >
                  언어 선택
                </button>
       
              </div>
              <div className="w-full">
                <button
                  className="w-full p-4 border rounded-lg hover:bg-blue-50 transition-colors font-bold"
                  onClick={() => !isLoggedIn ? handleSettingClick() : setIsSelectionModalOpen(true)}
                >
                  최적화 모델 선택
                </button>
          
              </div>
            </div>
          </div>
        </div>
      )}

      {/* 로그인 알림 메시지 */}
      {showLoginAlert && (
        <div className="fixed top-4 right-4 bg-blue-500 text-white px-6 py-3 rounded-lg shadow-lg z-50 animate-fade-in-out">
          로그인 후 설정이 가능합니다.
        </div>
      )}

      {/* 언어 선택 모달 - 로그인한 경우에만 표시 */}
      {isLoggedIn && isLanguageSelectionOpen && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div 
            className="bg-white rounded-lg p-6 w-full max-w-md shadow-lg relative h-96 overflow-y-auto flex flex-col"
            onScroll={(e) => setShowConfirmButton(
              e.target.scrollTop + e.target.clientHeight >= e.target.scrollHeight
            )}
          >
            <h2 className="text-xl font-bold mb-4">언어 선택</h2>
            <div className="grid grid-cols-2 gap-2 mb-6">
              {languages.map((lang) => (
                <button
                  key={lang}
                  onClick={() => setSelectedLanguage(lang)}
                  className={`p-2 border rounded-lg transition-colors ${
                    selectedLanguage === lang ? "bg-blue-300" : "hover:bg-blue-50"
                  }`}
                >
                  {lang}
                </button>
              ))}
            </div>
            <button 
              className={`px-6 py-3 rounded-lg transition-colors self-end mt-auto shadow-md ${
                selectedLanguage
                  ? "bg-blue-500 text-white hover:bg-blue-600"
                  : "bg-gray-300 text-gray-500 cursor-not-allowed"
              }`} 
              onClick={handleConfirm}
              disabled={!selectedLanguage}
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