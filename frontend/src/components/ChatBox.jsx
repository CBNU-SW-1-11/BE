// import React, { useState, useEffect, useRef } from "react";
// import { Send } from "lucide-react";
// import { useChat } from "../context/ChatContext"; 

// const ChatBox = () => {
//   const { messages, sendMessage, isLoading } = useChat(); 
//   const [inputMessage, setInputMessage] = useState("");
//   const messagesEndRefs = {
//     gpt: useRef(null),
//     claude: useRef(null),
//     mixtral: useRef(null)
//   };

//   // 메시지 전송
//   const handleSendMessage = (e) => {
//     e.preventDefault();
//     sendMessage(inputMessage);
//     setInputMessage("");
//   };

//   // 새 메시지가 추가될 때마다 스크롤을 맨 아래로 이동
//   useEffect(() => {
//     Object.values(messagesEndRefs).forEach(ref => {
//       ref.current?.scrollIntoView({ behavior: "smooth" });
//     });
//   }, [messages]);

//   return (
//     <div className="flex flex-col h-screen w-full bg-white">
//       {/* AI 이름 박스 - 상단 고정 */}
//       <div className="grid grid-cols-3 border-b bg-white sticky top-0 z-20">
//         {["gpt", "claude", "mixtral"].map((botName) => (
//           <div key={botName} className="p-4 text-xl font-semibold text-center border-r last:border-r-0">
//             {botName.toUpperCase()}
//           </div>
//         ))}
//       </div>

//       {/* 채팅 메시지 영역을 AI 이름 박스 아래에 위치하도록 조정 */}
//       <div className="flex-1 grid grid-cols-3 overflow-y-auto p-6 pt-12 space-y-6"> {/* 패딩 추가하여 시작 지점 조정 */}
//         {["gpt", "claude", "mixtral"].map((botName) => (
//           <div key={botName} className="flex flex-col border-r last:border-r-0 h-full overflow-hidden"> {/* GPT 칸 높이 문제 해결 */}
//             <div className="flex-1 overflow-y-auto p-4">
//               {messages[botName].map((message, index) => (
//                 <div key={index} className={`flex ${message.isUser ? "justify-end" : "justify-start"}`}>
//                   <div
//                     className={`max-w-2xl p-4 rounded-2xl ${
//                       message.isUser ? "bg-purple-600 text-white" : "bg-gray-100 text-gray-800"
//                     }`}
//                   >
//                     {message.text}
//                   </div>
//                 </div>
//               ))}
//               {isLoading && (
//                 <div className="flex justify-start">
//                   <div className="bg-gray-100 text-gray-800 p-4 rounded-2xl">입력 중...</div>
//                 </div>
//               )}
//               <div ref={messagesEndRefs[botName]} />
//             </div>
//           </div>
//         ))}
//       </div>

//       {/* 입력창을 화면 하단에 고정 */}
//       <div className="border-t p-4 w-full flex-shrink-0 bg-white sticky bottom-0 z-20">
//         <form onSubmit={handleSendMessage} className="max-w-4xl mx-auto flex items-center bg-white border rounded-xl p-2">
//           <input
//             type="text"
//             value={inputMessage}
//             onChange={(e) => setInputMessage(e.target.value)}
//             placeholder="메시지를 입력하세요..."
//             className="flex-1 px-3 py-2 focus:outline-none"
//           />
//           <button type="submit" disabled={isLoading} className="p-2 rounded-lg hover:bg-gray-100">
//             <Send className="w-5 h-5 text-gray-600" />
//           </button>
//         </form>
//       </div>
//     </div>
//   );
// };

// export default ChatBox;

// import React, { useState, useEffect, useRef } from "react";
// import { Send } from "lucide-react";
// import { useChat } from "../context/ChatContext";

// const ChatBox = () => {
//   const { messages, sendMessage, isLoading, analysisResults } = useChat();
//   const [inputMessage, setInputMessage] = useState("");
//   const messagesEndRefs = {
//     gpt: useRef(null),
//     claude: useRef(null),
//     mixtral: useRef(null),
//     optimal: useRef(null)
//   };

//   // Handle message sending
//   const handleSendMessage = (e) => {
//     e.preventDefault();
//     if (inputMessage.trim()) {
//       sendMessage(inputMessage);
//       setInputMessage("");
//     }
//   };

//   useEffect(() => {
//     Object.values(messagesEndRefs).forEach(ref => {
//       ref.current?.scrollIntoView({ behavior: "smooth" });
//     });
//   }, [messages, analysisResults]);

//   // Get optimal response for a given user message
//   const getOptimalResponse = (userMessage) => {
//     return analysisResults[userMessage]?.bestResponse || "";
//   };

//   return (
//     <div className="flex flex-col h-screen w-full bg-white">
//     {/* AI 이름 박스 - 상단 고정 */}
//     <div className="grid grid-cols-4 border-b bg-white sticky top-0 z-20">
//       {["gpt", "claude", "mixtral", "optimal"].map((botName) => (
//         <div key={botName} className="p-4 text-xl font-semibold text-center border-r last:border-r-0">
//           {botName === "optimal" ? "최적 답변" : botName.toUpperCase()}
//         </div>
//       ))}
//     </div>
//       {/* Chat messages area */}
//       <div className="grid grid-cols-4 gap-4 flex-1 overflow-hidden">
//         {["gpt", "claude", "mixtral"].map((botName) => (
//           <div key={botName} className="flex flex-col overflow-y-auto bg-gray-50 rounded-lg p-4">
//             <div className="flex-1">
//               {messages[botName].map((message, index) => (
//                 <div
//                   key={index}
//                   className={`mb-4 ${
//                     message.isUser ? "flex justify-end" : "flex justify-start"
//                   }`}
//                 >
//                   <div
//                     className={`max-w-[85%] p-3 rounded-lg ${
//                       message.isUser
//                         ? "bg-blue-100 text-blue-900"
//                         : "bg-white text-gray-900"
//                     }`}
//                   >
//                     {message.text}
//                   </div>
//                 </div>
//               ))}
//               {isLoading && (
//                 <div className="flex justify-start mb-4">
//                   <div className="bg-white p-3 rounded-lg">
//                     입력 중...
//                   </div>
//                 </div>
//               )}
//               <div ref={messagesEndRefs[botName]} />
//             </div>
//           </div>
//         ))}

//                   {/* Optimal answer column with full analysis */}
//         <div className="flex flex-col overflow-y-auto bg-gray-50 rounded-lg p-4">
//           <div className="flex-1">
//             {messages.gpt.map((message, index) => {
//               if (!message.isUser) {
//                 const userMessage = messages.gpt[index - 1]?.text;
//                 const analysis = analysisResults[userMessage];
                
//                 return analysis ? (
//                   <div key={index} className="mb-6 space-y-4">
//                     {/* Analysis Complete Header */}
//                     <div className="flex justify-between items-center text-blue-600 font-semibold">
//                       <span>✅ 분석 완료</span>
//                       <span>🤖 분석 수행 AI: {analysis.botName || "N/A"}</span>
//                     </div>
                  
//                     {/* Separator */}
//                     <div className="border-t border-gray-200 my-2"></div>
                    
//                     {/* Best Response */}
//                     <div>
//                       <div className="font-semibold text-lg mb-2">✨ 최적의 답변:</div>
//                       <div className="bg-green-50 p-3 rounded-lg border border-green-200">
//                         {analysis.bestResponse}
//                       </div>
//                     </div>
//                     {/* AI Analysis */}
// <div>
//   <div className="font-semibold text-lg mb-2">📌 각 AI 분석:</div>
//   <div className="space-y-3">
//     {Object.entries(analysis.analysis || {}).map(([ai, aiAnalysis]) => (
//       <div key={ai} className="bg-white p-3 rounded-lg">
//         <div className="font-medium mb-1">{ai.toUpperCase()}:</div>
//         <div className="text-green-600">장점: {aiAnalysis?.장점 || '정보 없음'}</div>
//         <div className="text-red-600">단점: {aiAnalysis?.단점 || '정보 없음'}</div>
//       </div>
//     ))}
//   </div>

//   {/* Analysis Reasoning */}
//   <div className="mt-4">
//     <div className="font-semibold text-lg mb-2">💡 분석 근거:</div>
//     <div className="bg-white p-3 rounded-lg">
//       {analysis.reasoning || '분석 근거 없음'}
//     </div>
//   </div>

//   {/* Separator */}
//   <div className="mt-4 border-b border-gray-300" />
// </div>
                    
                  
                      
//                       {/* Separator */}
//                       <div className="mt-4 border-b border-gray-300" />
//                     </div>
                    
                
//                 ) : null;
//               }
//               return null;
//             })}
//             <div ref={messagesEndRefs.optimal} />
//           </div>
//         </div>
//       </div>

//       {/* Input area */}
//       <form onSubmit={handleSendMessage} className="mt-4">
//         <div className="flex items-center gap-2 bg-white rounded-lg shadow-sm p-2">
//           <input
//             type="text"
//             value={inputMessage}
//             onChange={(e) => setInputMessage(e.target.value)}
//             placeholder="메시지를 입력하세요..."
//             className="flex-1 px-3 py-2 focus:outline-none"
//             disabled={isLoading}
//           />
//           <button
//             type="submit"
//             className="p-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 transition-colors disabled:bg-blue-300"
//             disabled={isLoading}
//           >
//             <Send className="w-5 h-5" />
//           </button>
//         </div>
//       </form>
//     </div>
//   );
// };

// export default ChatBox;

import React, { useState, useEffect, useRef } from "react";
import { Send } from "lucide-react";
import { useChat } from "../context/ChatContext";

const ChatBox = () => {
  const { messages, sendMessage, isLoading, analysisResults } = useChat();
  const [inputMessage, setInputMessage] = useState("");
  const messagesEndRefs = {
    gpt: useRef(null),
    claude: useRef(null),
    mixtral: useRef(null),
    optimal: useRef(null)
  };

  // 메시지 전송
  const handleSendMessage = (e) => {
    e.preventDefault();
    if (inputMessage.trim()) {
      sendMessage(inputMessage);
      setInputMessage("");
    }
  };

  // 새 메시지가 추가될 때마다 스크롤을 맨 아래로 이동
  useEffect(() => {
    Object.values(messagesEndRefs).forEach(ref => {
      ref.current?.scrollIntoView({ behavior: "smooth" });
    });
  }, [messages, analysisResults]);

  return (
    <div className="flex flex-col h-screen w-full bg-white">
      {/* AI 이름 박스 - 상단 고정 */}
      <div className="grid grid-cols-4 border-b bg-white sticky top-0 z-20">
        {["gpt", "claude", "mixtral", "optimal"].map((botName) => (
          <div key={botName} className="p-4 text-xl font-semibold text-center border-r last:border-r-0">
            {botName === "optimal" ? "최적 답변" : botName.toUpperCase()}
          </div>
        ))}
      </div>

      {/* 채팅 메시지 영역 */}
      <div className="flex-1 grid grid-cols-4 overflow-y-auto p-6 pt-12">
        {/* GPT, Claude, Mixtral 칼럼 */}
        {["gpt", "claude", "mixtral"].map((botName) => (
          <div key={botName} className="flex flex-col border-r last:border-r-0 h-full overflow-hidden">
            <div className="flex-1 overflow-y-auto p-4">
              {messages[botName].map((message, index) => (
                <div key={index} className={`flex ${message.isUser ? "justify-end" : "justify-start"}`}>
                  <div
                    className={`max-w-2xl p-4 rounded-2xl ${
                      message.isUser ? "bg-purple-600 text-white" : "bg-gray-100 text-gray-800"
                    }`}
                  >
                    {message.text}
                  </div>
                </div>
              ))}
              {isLoading && (
                <div className="flex justify-start">
                  <div className="bg-gray-100 text-gray-800 p-4 rounded-2xl">입력 중...</div>
                </div>
              )}
              <div ref={messagesEndRefs[botName]} />
            </div>
          </div>
        ))}

        {/* 최적 답변 칼럼 */}
        <div className="flex flex-col border-r last:border-r-0 h-full overflow-hidden">
          <div className="flex-1 overflow-y-auto p-4">
            {messages.gpt.map((message, index) => {
              if (!message.isUser) {
                const userMessage = messages.gpt[index - 1]?.text;
                const analysis = analysisResults[userMessage];
                
                return analysis ? (
                  <div key={index} className="mb-6">
                    <div className="flex justify-between items-center text-sm text-blue-600 mb-2">
                      <span>✅ 분석 완료</span>
                      <span>🤖 {analysis.botName || "N/A"}</span>
                    </div>
                    
                    <div className="bg-gray-100 p-4 rounded-2xl space-y-4">
                      <div>
                        <div className="font-semibold mb-2">✨ 최적의 답변:</div>
                        <div className="bg-white p-3 rounded-xl">
                          {analysis.bestResponse}
                        </div>
                      </div>
                      
                      <div>
                        <div className="font-semibold mb-2">📌 각 AI 분석:</div>
                        {Object.entries(analysis.analysis || {}).map(([ai, aiAnalysis]) => (
                          <div key={ai} className="bg-white p-3 rounded-xl mb-2">
                            <div className="font-medium">{ai.toUpperCase()}:</div>
                            <div className="text-green-600">장점: {aiAnalysis?.장점 || '정보 없음'}</div>
                            <div className="text-red-600">단점: {aiAnalysis?.단점 || '정보 없음'}</div>
                          </div>
                        ))}
                      </div>
                      
                      <div>
                        <div className="font-semibold mb-2">💡 분석 근거:</div>
                        <div className="bg-white p-3 rounded-xl">
                          {analysis.reasoning || '분석 근거 없음'}
                        </div>
                      </div>
                    </div>
                  </div>
                ) : null;
              }
              return null;
            })}
            <div ref={messagesEndRefs.optimal} />
          </div>
        </div>
      </div>

      {/* 입력창 */}
      <div className="border-t p-4 w-full flex-shrink-0 bg-white sticky bottom-0 z-20">
        <form onSubmit={handleSendMessage} className="max-w-4xl mx-auto flex items-center bg-white border rounded-xl p-2">
          <input
            type="text"
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            placeholder="메시지를 입력하세요..."
            className="flex-1 px-3 py-2 focus:outline-none"
            disabled={isLoading}
          />
          <button 
            type="submit" 
            disabled={isLoading} 
            className="p-2 rounded-lg hover:bg-gray-100"
          >
            <Send className="w-5 h-5 text-gray-600" />
          </button>
        </form>
      </div>
    </div>
  );
};

export default ChatBox;