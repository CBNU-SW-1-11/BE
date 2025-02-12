

// // Chat.js
// import React, { createContext, useState, useContext, useRef, useEffect } from 'react';

// // Context 생성
// const ChatContext = createContext(null);

// // 봇 선택 모달 컴포넌트
// const SelectBotModal = ({ isOpen, onClose, onSelectBot }) => {
//   if (!isOpen) return null;

//   return (
//     <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
//       <div className="bg-white rounded-lg p-6 w-full max-w-2xl">
//         <h2 className="text-xl font-bold mb-4">분석을 수행할 AI 선택</h2>
//         <div className="grid grid-cols-3 gap-4 mb-6">
//           <button
//             onClick={() => onSelectBot('gpt')}
//             className="p-6 border rounded-lg hover:bg-blue-50 transition-colors"
//           >
//             <h3 className="font-bold text-lg mb-2">GPT-3.5</h3>
//             <p className="text-sm text-gray-600 mb-2">OpenAI의 GPT-3.5 모델</p>
//             <ul className="text-xs text-gray-500 list-disc pl-4">
//               <li>빠른 응답 속도</li>
//               <li>일관된 답변 품질</li>
//               <li>다양한 주제 처리</li>
//             </ul>
//           </button>
//           <button
//             onClick={() => onSelectBot('claude')}
//             className="p-6 border rounded-lg hover:bg-blue-50 transition-colors"
//           >
//             <h3 className="font-bold text-lg mb-2">Claude</h3>
//             <p className="text-sm text-gray-600 mb-2">Anthropic의 Claude 모델</p>
//             <ul className="text-xs text-gray-500 list-disc pl-4">
//               <li>높은 분석 능력</li>
//               <li>정확한 정보 제공</li>
//               <li>상세한 설명 제공</li>
//             </ul>
//           </button>
//           <button
//             onClick={() => onSelectBot('mixtral')}
//             className="p-6 border rounded-lg hover:bg-blue-50 transition-colors"
//           >
//             <h3 className="font-bold text-lg mb-2">Mixtral</h3>
//             <p className="text-sm text-gray-600 mb-2">Mixtral-8x7B 모델</p>
//             <ul className="text-xs text-gray-500 list-disc pl-4">
//               <li>균형잡힌 성능</li>
//               <li>다국어 지원</li>
//               <li>코드 분석 특화</li>
//             </ul>
//           </button>
//         </div>
//         <div className="text-sm text-gray-600 mb-4">
//           선택한 AI가 다른 AI들의 응답을 분석하여 최적의 답변을 제공합니다.
//         </div>
//         <div className="flex justify-end">
//           <button
//             onClick={onClose}
//             className="px-4 py-2 text-gray-600 hover:text-gray-800"
//           >
//             취소
//           </button>
//         </div>
//       </div>
//     </div>
//   );
// };

// // ChatProvider 컴포넌트
// export const ChatProvider = ({ children }) => {
//   const [messages, setMessages] = useState({
//     gpt: [],
//     claude: [],
//     mixtral: [],
//   });
//   const [selectedBot, setSelectedBot] = useState(null);
//   const [isSelectionModalOpen, setIsSelectionModalOpen] = useState(true);
//   const [isLoading, setIsLoading] = useState(false);
//   const [analysisResults, setAnalysisResults] = useState({});
//   const messagesEndRef = useRef(null);

//   useEffect(() => {
//     messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
//   }, [messages]);

//   const handleBotSelection = (botName) => {
//     setSelectedBot(botName);
//     setIsSelectionModalOpen(false);
//   };

//   const addMessage = (botName, text, isUser) => {
//     setMessages((prev) => ({
//       ...prev,
//       [botName]: [...(prev[botName] || []), { text, isUser }],
//     }));
//   };

//   const sendMessage = async (userMessage) => {
//     if (!userMessage.trim() || !selectedBot) return;

//     setIsLoading(true);
//     try {
//       const response = await fetch(`http://localhost:8000/chat/${selectedBot}/`, {
//         method: "POST",
//         headers: {
//           "Content-Type": "application/json",
//         },
//         body: JSON.stringify({
//           message: userMessage,
//           compare: true
//         }),
//       });

//       if (!response.ok) {
//         throw new Error(`HTTP error! status: ${response.status}`);
//       }

//       const data = await response.json();
      
//       // 사용자 메시지 추가
//       Object.keys(messages).forEach((botName) => {
//         addMessage(botName, userMessage, true);
//       });

//       // 각 AI의 응답 추가
//       Object.entries(data.responses).forEach(([botName, response]) => {
//         addMessage(botName, response, false);
//       });

//       // 분석 결과 저장
//       setAnalysisResults((prev) => ({
//         ...prev,
//         [userMessage]: {
//           bestResponse: data.best_response,
//           analysis: data.analysis,
//           analyzer: data.analyzer // 분석한 AI 정보 저장
//         }
//       }));

//     } catch (error) {
//       console.error("Error:", error);
//       Object.keys(messages).forEach((botName) => {
//         addMessage(botName, `오류가 발생했습니다: ${error.message}`, false);
//       });
//     }
//     setIsLoading(false);
//   };

//   return (
//     <ChatContext.Provider
//       value={{
//         messages,
//         sendMessage,
//         isLoading,
//         messagesEndRef,
//         selectedBot,
//         analysisResults,
//         isSelectionModalOpen,
//         setIsSelectionModalOpen,
//         handleBotSelection
//       }}
//     >
//       {children}
//       <SelectBotModal
//         isOpen={isSelectionModalOpen}
//         onClose={() => selectedBot ? setIsSelectionModalOpen(false) : null}
//         onSelectBot={handleBotSelection}
//       />
//     </ChatContext.Provider>
//   );
// };

// // Custom Hook
// export const useChat = () => {
//   const context = useContext(ChatContext);
//   if (!context) {
//     throw new Error('useChat must be used within a ChatProvider');
//   }
//   return context;
// };

// // ChatInterface 컴포넌트
// // AnalysisModal 컴포넌트
// const AnalysisModal = ({ isOpen, onClose, result }) => {
//   if (!isOpen || !result) return null;

//   return (
//     <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
//       <div className="bg-white rounded-lg p-6 w-full max-w-4xl max-h-[90vh] overflow-y-auto">
//         <div className="flex justify-between items-center mb-4">
//           <h2 className="text-xl font-bold">분석 결과</h2>
//           <button
//             onClick={onClose}
//             className="text-gray-500 hover:text-gray-700"
//           >
//             <span className="text-2xl">×</span>
//           </button>
//         </div>

//         <div className="space-y-6">
//           {/* 최적의 답변 */}
//           <div>
//             <h3 className="font-medium text-lg mb-2">최적의 답변:</h3>
//             <div className="p-4 bg-blue-50 rounded-lg">
//               {result.bestResponse}
//             </div>
//           </div>

//           {/* AI별 분석 */}
//           <div>
//             <h3 className="font-medium text-lg mb-2">AI 별 분석:</h3>
//             <div className="grid grid-cols-3 gap-4">
//               {Object.entries(result.analysis).map(([aiName, aiAnalysis]) => (
//                 <div key={aiName} className="p-4 bg-gray-50 rounded-lg">
//                   <h4 className="font-medium mb-2">{aiName.toUpperCase()}</h4>
//                   <div className="space-y-2">
//                     <div className="text-green-600">
//                       <span className="font-medium">장점:</span> {aiAnalysis.장점}
//                     </div>
//                     <div className="text-red-600">
//                       <span className="font-medium">단점:</span> {aiAnalysis.단점}
//                     </div>
//                   </div>
//                 </div>
//               ))}
//             </div>
//           </div>

//           {/* 분석 근거 */}
//           <div>
//             <h3 className="font-medium text-lg mb-2">분석 근거:</h3>
//             <div className="p-4 bg-gray-50 rounded-lg">
//               {result.reasoning}
//             </div>
//           </div>
//         </div>
//       </div>
//     </div>
//   );
// };

// // ChatInterface 컴포넌트 수정된 부분
// export const ChatInterface = () => {
//   const {
//     messages,
//     sendMessage,
//     isLoading,
//     messagesEndRef,
//     selectedBot,
//     setSelectedBot,
//     analysisResults
//   } = useChat();
//   const [userInput, setUserInput] = useState("");
//   const [isAnalysisModalOpen, setIsAnalysisModalOpen] = useState(false);
//   const [selectedAnalysis, setSelectedAnalysis] = useState(null);

//   const handleSubmit = (e) => {
//     e.preventDefault();
//     if (userInput.trim()) {
//       sendMessage(userInput);
//       setUserInput("");
//     }
//   };

//   const handleShowAnalysis = (result) => {
//     setSelectedAnalysis(result);
//     setIsAnalysisModalOpen(true);
//   };

//   return (
//     <div className="flex flex-col h-screen max-w-7xl mx-auto p-4">
//       {/* AI 선택 영역 */}
//       <div className="mb-6 bg-white p-4 rounded-lg shadow-sm">
//         <div className="flex items-center gap-4">
//           <div>
//             <label htmlFor="bot-select" className="block text-sm font-medium text-gray-700 mb-1">
//               응답 분석 AI 선택
//             </label>
//             <select
//               id="bot-select"
//               value={selectedBot}
//               onChange={(e) => setSelectedBot(e.target.value)}
//               className="p-2 border rounded shadow-sm"
//             >
//               <option value="gpt">GPT-3.5</option>
//               <option value="claude">Claude</option>
//               <option value="mixtral">Mixtral</option>
//             </select>
//           </div>
//           <div className="text-sm text-gray-600">
//             선택한 AI가 다른 AI들의 응답을 분석하여 최적의 답변을 제공합니다.
//           </div>
//         </div>
//       </div>

//       {/* 채팅 영역 */}
//       <div className="grid grid-cols-3 gap-4 flex-1 overflow-y-auto">
//         {Object.entries(messages).map(([botName, botMessages]) => (
//           <div key={botName} className="border rounded-lg p-4 overflow-y-auto">
//             <h2 className="text-xl font-bold mb-4 sticky top-0 bg-white py-2">
//               {botName.toUpperCase()}
//             </h2>
//             <div className="space-y-4">
//               {botMessages.map((msg, idx) => (
//                 <div key={idx}>
//                   <div
//                     className={`p-3 rounded-lg ${
//                       msg.isUser 
//                         ? "bg-blue-100 ml-auto" 
//                         : "bg-gray-100"
//                     }`}
//                   >
//                     <p>{msg.text}</p>
//                   </div>
//                   {!msg.isUser && analysisResults[botMessages[idx-1]?.text] && (
//                     <button
//                       onClick={() => handleShowAnalysis(analysisResults[botMessages[idx-1].text])}
//                       className="mt-2 text-sm text-blue-600 hover:text-blue-800 flex items-center gap-1"
//                     >
//                       <span>📊</span> 최적의 답변 보기
//                     </button>
//                   )}
//                 </div>
//               ))}
//             </div>
//           </div>
//         ))}
//       </div>

//       {/* 입력 폼 */}
//       <form onSubmit={handleSubmit} className="mt-4">
//         <div className="flex gap-2">
//           <input
//             type="text"
//             value={userInput}
//             onChange={(e) => setUserInput(e.target.value)}
//             className="flex-1 p-2 border rounded shadow-sm"
//             placeholder="메시지를 입력하세요..."
//             disabled={isLoading}
//           />
//           <button
//             type="submit"
//             className="px-4 py-2 bg-blue-500 text-white rounded shadow hover:bg-blue-600 transition-colors disabled:bg-blue-300"
//             disabled={isLoading}
//           >
//             {isLoading ? "처리 중..." : "전송"}
//           </button>
//         </div>
//       </form>

//       <div ref={messagesEndRef} />

//       {/* 분석 결과 모달 */}
//       <AnalysisModal
//         isOpen={isAnalysisModalOpen}
//         onClose={() => setIsAnalysisModalOpen(false)}
//         result={selectedAnalysis}
//       />
//     </div>
//   );
// };// ChatInterface 컴포넌트 수정

// Chat.js
import React, { createContext, useState, useContext, useRef, useEffect } from 'react';

// Context 생성
const ChatContext = createContext(null);

// 봇 선택 모달 컴포넌트
const SelectBotModal = ({ isOpen, onClose, onSelectBot }) => {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg p-6 w-full max-w-2xl">
        <h2 className="text-xl font-bold mb-4">분석을 수행할 AI 선택</h2>
        <div className="grid grid-cols-3 gap-4 mb-6">
          <button
            onClick={() => onSelectBot('gpt')}
            className="p-6 border rounded-lg hover:bg-blue-50 transition-colors"
          >
            <h3 className="font-bold text-lg mb-2">GPT-3.5</h3>
            <p className="text-sm text-gray-600 mb-2">OpenAI의 GPT-3.5 모델</p>
            <ul className="text-xs text-gray-500 list-disc pl-4">
              <li>빠른 응답 속도</li>
              <li>일관된 답변 품질</li>
              <li>다양한 주제 처리</li>
            </ul>
          </button>
          <button
            onClick={() => onSelectBot('claude')}
            className="p-6 border rounded-lg hover:bg-blue-50 transition-colors"
          >
            <h3 className="font-bold text-lg mb-2">Claude</h3>
            <p className="text-sm text-gray-600 mb-2">Anthropic의 Claude 모델</p>
            <ul className="text-xs text-gray-500 list-disc pl-4">
              <li>높은 분석 능력</li>
              <li>정확한 정보 제공</li>
              <li>상세한 설명 제공</li>
            </ul>
          </button>
          <button
            onClick={() => onSelectBot('mixtral')}
            className="p-6 border rounded-lg hover:bg-blue-50 transition-colors"
          >
            <h3 className="font-bold text-lg mb-2">Mixtral</h3>
            <p className="text-sm text-gray-600 mb-2">Mixtral-8x7B 모델</p>
            <ul className="text-xs text-gray-500 list-disc pl-4">
              <li>균형잡힌 성능</li>
              <li>다국어 지원</li>
              <li>코드 분석 특화</li>
            </ul>
          </button>
        </div>
        <div className="text-sm text-gray-600 mb-4">
          선택한 AI가 다른 AI들의 응답을 분석하여 최적의 답변을 제공합니다.
        </div>
        <div className="flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 text-gray-600 hover:text-gray-800"
          >
            취소
          </button>
        </div>
      </div>
    </div>
  );
};

// ChatProvider 컴포넌트
export const ChatProvider = ({ children }) => {
  const [messages, setMessages] = useState({
    gpt: [],
    claude: [],
    mixtral: [],
  });
  const [selectedBot, setSelectedBot] = useState(null);
  const [isSelectionModalOpen, setIsSelectionModalOpen] = useState(true);
  const [isLoading, setIsLoading] = useState(false);
  const [analysisResults, setAnalysisResults] = useState({});
  const messagesEndRef = useRef(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleBotSelection = (botName) => {
    setSelectedBot(botName);
    setIsSelectionModalOpen(false);
  };

  const addMessage = (botName, text, isUser) => {
    setMessages((prev) => ({
      ...prev,
      [botName]: [...(prev[botName] || []), { text, isUser }],
    }));
  };

// ChatContext.js 의 sendMessage 함수 내부
const sendMessage = async (userMessage) => {
  if (!userMessage.trim() || !selectedBot) return;

  setIsLoading(true);
  try {
    const response = await fetch(`http://localhost:8000/chat/${selectedBot}/`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        message: userMessage,
        compare: true
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    console.log('API Response:', data); // 데이터 구조 확인을 위한 로깅
    
    // 사용자 메시지 추가
    Object.keys(messages).forEach((botName) => {
      addMessage(botName, userMessage, true);
    });

    // 각 AI의 응답 추가
    Object.entries(data.responses).forEach(([botName, response]) => {
      addMessage(botName, response, false);
    });

    // 분석 결과 저장
    setAnalysisResults((prev) => ({
      ...prev,
      [userMessage]: {
        botName: data.bot_name,
        bestResponse: data.best_response,
        analysis: data.analysis, // 직접 analysis 객체 사용
        reasoning: data.reasoning
      }
    }));

  } catch (error) {
    console.error("Error:", error);
    Object.keys(messages).forEach((botName) => {
      addMessage(botName, `오류가 발생했습니다: ${error.message}`, false);
    });
  }
  setIsLoading(false);
};

  return (
    <ChatContext.Provider
      value={{
        messages,
        sendMessage,
        isLoading,
        messagesEndRef,
        selectedBot,
        analysisResults,
        isSelectionModalOpen,
        setIsSelectionModalOpen,
        handleBotSelection
      }}
    >
      {children}
      <SelectBotModal
        isOpen={isSelectionModalOpen}
        onClose={() => selectedBot ? setIsSelectionModalOpen(false) : null}
        onSelectBot={handleBotSelection}
      />
    </ChatContext.Provider>
  );
};

// Custom Hook
export const useChat = () => {
  const context = useContext(ChatContext);
  if (!context) {
    throw new Error('useChat must be used within a ChatProvider');
  }
  return context;
};

// ChatInterface 컴포넌트
export const ChatInterface = () => {
  const {
    messages,
    sendMessage,
    isLoading,
    messagesEndRef,
    selectedBot,
    setSelectedBot,
    analysisResults
  } = useChat();
  const [userInput, setUserInput] = useState("");
  const [isAnalysisModalOpen, setIsAnalysisModalOpen] = useState(false);
  const [currentAnalysisResult, setCurrentAnalysisResult] = useState(null);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (userInput.trim()) {
      sendMessage(userInput);
      setUserInput("");
    }
  };

  const handleShowAnalysis = (messageIndex, botMessages) => {
    const userMessage = botMessages[messageIndex-1]?.text;
    if (userMessage && analysisResults[userMessage]) {
      setCurrentAnalysisResult(analysisResults[userMessage]);
      setIsAnalysisModalOpen(true);
    }
  };

  return (
    <div className="flex flex-col h-screen max-w-7xl mx-auto p-4">
      {/* AI 선택 영역 */}
      <div className="mb-6 bg-white p-4 rounded-lg shadow-sm">
        <div className="flex items-center gap-4">
          <div>
            <label htmlFor="bot-select" className="block text-sm font-medium text-gray-700 mb-1">
              응답 분석 AI 선택
            </label>
            <select
              id="bot-select"
              value={selectedBot}
              onChange={(e) => setSelectedBot(e.target.value)}
              className="p-2 border rounded shadow-sm"
            >
              <option value="gpt">GPT-3.5</option>
              <option value="claude">Claude</option>
              <option value="mixtral">Mixtral</option>
            </select>
          </div>
          <div className="text-sm text-gray-600">
            선택한 AI가 다른 AI들의 응답을 분석하여 최적의 답변을 제공합니다.
          </div>
        </div>
      </div>

      {/* 채팅 영역 */}
      <div className="grid grid-cols-3 gap-4 flex-1 overflow-y-auto">
        {Object.entries(messages).map(([botName, botMessages]) => (
          <div key={botName} className="border rounded-lg p-4 overflow-y-auto">
            <h2 className="text-xl font-bold mb-4 sticky top-0 bg-white py-2">
              {botName.toUpperCase()}
            </h2>
            <div className="space-y-4">
              {botMessages.map((msg, idx) => (
                <div key={idx} className="mb-4">
                  <div
                    className={`p-3 rounded-lg ${
                      msg.isUser 
                        ? "bg-blue-100 ml-auto" 
                        : "bg-gray-100"
                    }`}
                  >
                    <p>{msg.text}</p>
                  </div>
                  {!msg.isUser && (
                    <button
                      onClick={() => handleShowAnalysis(idx, botMessages)}
                      className="mt-2 px-3 py-1 text-sm flex items-center gap-1 border rounded-md transition-colors"
                      disabled={!botMessages[idx-1]}
                      style={{
                        backgroundColor: botMessages[idx-1] && analysisResults[botMessages[idx-1].text] ? "#EBF5FF" : "#F3F4F6",
                        borderColor: botMessages[idx-1] && analysisResults[botMessages[idx-1].text] ? "#93C5FD" : "#E5E7EB",
                        color: botMessages[idx-1] ? (analysisResults[botMessages[idx-1].text] ? "#2563EB" : "#6B7280") : "#9CA3AF"
                      }}
                    >
                      <span>{botMessages[idx-1] ? (analysisResults[botMessages[idx-1].text] ? "📊" : "⏳") : "❌"}</span>
                      {botMessages[idx-1] ? (
                        analysisResults[botMessages[idx-1].text] 
                          ? "분석 결과 보기" 
                          : "분석 중..."
                      ) : "분석 불가"}
                    </button>
                  )}
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* 입력 폼 */}
      <form onSubmit={handleSubmit} className="mt-4">
        <div className="flex gap-2">
          <input
            type="text"
            value={userInput}
            onChange={(e) => setUserInput(e.target.value)}
            className="flex-1 p-2 border rounded shadow-sm"
            placeholder="메시지를 입력하세요..."
            disabled={isLoading}
          />
          <button
            type="submit"
            className="px-4 py-2 bg-blue-500 text-white rounded shadow hover:bg-blue-600 transition-colors disabled:bg-blue-300"
            disabled={isLoading}
          >
            {isLoading ? "처리 중..." : "전송"}
          </button>
        </div>
      </form>

      <div ref={messagesEndRef} />

  
      {/* 분석 결과 모달
      <AnalysisModal
        isOpen={isAnalysisModalOpen}
        onClose={() => setIsAnalysisModalOpen(false)}
        result={currentAnalysisResult}
      /> */}
    </div>
  );
};

