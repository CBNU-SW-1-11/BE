

import React, { useState, useEffect, useRef } from "react";
import { useChat } from "../context/ChatContext";
import ModelSelectionModal from "./ModelSelectionModal";
import SimilarityDetailModal from "./SimilarityDetailModal";
import { Send, Settings, Layers, AlertTriangle, Globe, BookOpen, Camera } from "lucide-react";

const ChatBox = () => {
  const { 
    messages, 
    sendMessage, 
    isLoading, 
    selectedModels, 
    setSelectedModels, 
    analysisResults,
    similarityResults,
    isProcessingImage, // 추가
    imageAnalysisResults, // 추가
    processImageUpload // 추가
  } = useChat();
  const [showImageUpload, setShowImageUpload] = useState(false);
  const [inputMessage, setInputMessage] = useState("");
  const [selectedImage, setSelectedImage] = useState(null);
  const [previewURL, setPreviewURL] = useState("");
  const fileInputRef = useRef(null);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isSimilarityModalOpen, setIsSimilarityModalOpen] = useState(false);
  const [currentSimilarityData, setCurrentSimilarityData] = useState(null);
  const [messageStates, setMessageStates] = useState({});
  const [processedAnalysis, setProcessedAnalysis] = useState({});
  const [similarityGroups, setSimilarityGroups] = useState({});
  const [similarityLoadingStates, setSimilarityLoadingStates] = useState({});
  // Reference to track processed analysis results
  const processedKeys = useRef(new Set());
  // const getMessageId = (message, index) => {
  //       return message.requestId || `${message.text}-${index}`;
  //     };
  const getMessageId = (message, index) => {
    return message.requestId;    // 절대 fallback 으로 text 를 쓰지 않습니다.
  };
  
  // References for message containers of each model
  const messagesEndRefs = useRef({});

  // Update refs when selectedModels changes
  useEffect(() => {
    const allModels = [...selectedModels, "optimal"];
    allModels.forEach((modelId) => {
      if (!messagesEndRefs.current[modelId]) {
        messagesEndRefs.current[modelId] = React.createRef();
      }
    });
  }, [selectedModels]);

// 백엔드와 동일한 형식의 requestId를 생성 (예: "타임스탬프.랜덤6자리")
const generateRequestId = () => {
  const timestamp = Date.now(); // 밀리초 단위 타임스탬프
  const randomPart = Math.floor(Math.random() * 1000000)
    .toString()
    .padStart(6, "0");
  return `req-${timestamp}-${randomPart}`;
};



 useEffect(() => {
     if (!similarityResults) return;
  
     const updatedGroups = {};
     Object.values(similarityResults).forEach(data => {
       const key = data.requestId; // full requestId 만으로 매핑
       updatedGroups[key] = {
         ...data,
         messageId: key,
         similarityMatrix: convertSimilarityMatrix(data.similarityMatrix)
       };
     });
  
     setSimilarityGroups(updatedGroups);
   }, [similarityResults]);

  useEffect(() => {
    const userMessages = messages.optimal 
      ? messages.optimal.filter(msg => msg.isUser)
      : [];
      
    if (userMessages.length > 0) {
      const updatedStates = { ...messageStates };
      let hasUpdates = false;
      
      userMessages.forEach((message, index) => {
        // Message ID: use requestId if available, otherwise use text+index combination
        // const messageId = message.requestId || generateRequestId();
        const messageId = getMessageId(message, index);

        

        if (updatedStates[messageId] === undefined) {
          updatedStates[messageId] = null; // null indicates 'analyzing' state
          hasUpdates = true;
        }
      });
      
      if (hasUpdates) {
        setMessageStates(updatedStates);
      }
    }
  }, [messages.optimal]);
  
  // Update analysis results using message IDs from optimal answers
  useEffect(() => {
    if (Object.keys(analysisResults).length > 0) {
      const newResults = {};
      let hasNewResults = false;
      
      Object.entries(analysisResults).forEach(([key, value]) => {
        if (!processedKeys.current.has(key)) {
          newResults[key] = value;
          processedKeys.current.add(key);
          hasNewResults = true;
        }
      });
      
      if (hasNewResults) {
        const latestKey = Object.keys(newResults)[Object.keys(newResults).length - 1];
        const latestResult = newResults[latestKey];
        
        const userMessages = messages.optimal 
          ? messages.optimal.filter(msg => msg.isUser)
          : [];
          
        if (userMessages.length > 0) {
          const matchingMsgIndex = userMessages.findIndex(
            (msg, idx) => (msg.requestId && msg.requestId === latestKey) || `${msg.text}-${idx}` === latestKey
          );
          
          let messageId;
          if (matchingMsgIndex !== -1) {
            messageId = userMessages[matchingMsgIndex].requestId || `${userMessages[matchingMsgIndex].text}-${matchingMsgIndex}`;
          } else {
            const latestIndex = userMessages.length - 1;
            messageId = userMessages[latestIndex].requestId || `${userMessages[latestIndex].text}-${latestIndex}`;
          }
          
          setMessageStates(prev => ({
            ...prev,
            [messageId]: latestResult
          }));
          
          setProcessedAnalysis(prev => ({
            ...prev,
            [messageId]: latestResult
          }));
        }
      }
    }
  }, [analysisResults, messages.optimal]);


  const handleSimilarityClick = (messageId) => {
        console.log("유사도 분석 데이터 조회 시도:", messageId);
    console.log("사용 가능한 유사도 그룹:", Object.keys(similarityGroups));
    
    // similarityGroups 에 full requestId key 로 매핑된 데이터 꺼냄
    const data = similarityGroups[messageId];
  
    if (data) {
      setCurrentSimilarityData(data);
    } else {
      setCurrentSimilarityData({
        messageId,
        noDataAvailable: true,
        debugInfo: {
          availableMessageIds: Object.keys(similarityGroups),
          currentMessageId: messageId,
          timestamp: new Date().toISOString(),
        },
      });
    }
  
    setIsSimilarityModalOpen(true);
  };

  const convertSimilarityMatrix = (matrix) => {
    if (!matrix) return null;
    
    const result = {};
    Object.entries(matrix).forEach(([model1, similarities]) => {
      result[model1] = {};
      Object.entries(similarities).forEach(([model2, score]) => {
        // 문자열로 된 값을 숫자로 변환 (필요한 경우)
        result[model1][model2] = typeof score === 'string' ? parseFloat(score) : score;
      });
    });
    
    return result;
  };
  // Send message handler
  const handleSendMessage = (e) => {
    e.preventDefault();
    // if (inputMessage.trim()) {
    //   // 메시지 전송 전에 한 번 requestId를 생성
    //   const requestId = generateRequestId();
    //   // sendMessage 함수에 requestId를 함께 전달 (백엔드도 이를 받아 동일한 ID로 분석 결과에 포함시킴)
    //   sendMessage(inputMessage, requestId);
    //   setInputMessage("");

    // }
    const requestId = generateRequestId();

    if (selectedImage) {
      // 이미지 전송
      processImageUpload(selectedImage, requestId)
      // 리셋
      setSelectedImage(null);
      setPreviewURL("");
    } else if (inputMessage.trim()) {
      // 텍스트 전송
      sendMessage(inputMessage, requestId);
      setInputMessage("");
  }

  };
  

  // Scroll to bottom when new messages arrive
  useEffect(() => {
    const allModels = [...selectedModels, "optimal"];
    allModels.forEach((modelId) => {
      messagesEndRefs.current[modelId]?.current?.scrollIntoView({ behavior: "smooth" });
    });
  }, [messages, selectedModels]);

  // Determine model color based on similarity group
  const getModelColor = (modelId, messageId) => {
    if (!similarityGroups[messageId]) return "bg-gray-100";
    
    // 데이터 구조 적응형 처리
    const groups = similarityGroups[messageId].similarGroups;
    const mainGroup = similarityGroups[messageId].mainGroup;
    const outliers = similarityGroups[messageId].outliers;
    
    // similarGroups가 있는 경우
    if (groups && groups.length > 0) {
      const mainGroupData = groups[0] || [];
      const secondGroupData = groups[1] || [];
      
      if (mainGroupData.includes(modelId)) {
        return "bg-blue-50 border-l-4 border-blue-500";
      } else if (secondGroupData && secondGroupData.includes(modelId)) {
        return "bg-green-50 border-l-4 border-green-500";
      } else {
        return "bg-yellow-50 border-l-4 border-yellow-500";
      }
    }
    // mainGroup와 outliers가 있는 경우
    else if (mainGroup) {
      if (Array.isArray(mainGroup) && mainGroup.includes(modelId)) {
        return "bg-blue-50 border-l-4 border-blue-500";
      } else if (outliers && Array.isArray(outliers) && outliers.includes(modelId)) {
        return "bg-yellow-50 border-l-4 border-yellow-500";
      } else {
        return "bg-gray-100";
      }
    }
    
    return "bg-gray-100";
  };
  // Model status badge component
  const ModelStatusBadge = ({ modelId, messageId }) => {
    if (!similarityGroups[messageId]) return null;
    
    // 데이터 구조 적응형 처리
    const groups = similarityGroups[messageId].similarGroups;
    const mainGroup = similarityGroups[messageId].mainGroup;
    const outliers = similarityGroups[messageId].outliers;
    const semanticTags = similarityGroups[messageId].semanticTags || {};
    const responseFeatures = similarityGroups[messageId].responseFeatures || {};
    
    // 다국어 지원 관련 태그 표시 (추가된 기능)
    if (responseFeatures[modelId] && responseFeatures[modelId].detectedLang) {
      const detectedLang = responseFeatures[modelId].detectedLang;
      return (
        <>
          <span className="inline-flex items-center px-2 py-1 text-xs rounded-full bg-indigo-100 text-indigo-800 mr-1">
            <Globe size={12} className="mr-1" /> {detectedLang}
          </span>
          {semanticTags[modelId] && (
            <span className="inline-flex items-center px-2 py-1 text-xs rounded-full bg-purple-100 text-purple-800">
              <BookOpen size={12} className="mr-1" /> {semanticTags[modelId]}
            </span>
          )}
          {renderGroupBadge()}
        </>
      );
    }
    
    if (semanticTags[modelId]) {
      return (
        <span className="inline-flex items-center px-2 py-1 text-xs rounded-full bg-purple-100 text-purple-800">
          <BookOpen size={12} className="mr-1" /> {semanticTags[modelId]}
        </span>
      );
    }
    
    return renderGroupBadge();
    
    // 내부 함수: 그룹 배지 렌더링
    function renderGroupBadge() {
      // similarGroups가 있는 경우
      if (groups && groups.length > 0) {
        const mainGroupData = groups[0] || [];
        const secondGroupData = groups[1] || [];
        
        if (mainGroupData.includes(modelId)) {
          return (
            <span className="inline-flex items-center px-2 py-1 text-xs rounded-full bg-blue-100 text-blue-800">
              <Layers size={12} className="mr-1" /> 주요 그룹
            </span>
          );
        } else if (secondGroupData && secondGroupData.includes(modelId)) {
          return (
            <span className="inline-flex items-center px-2 py-1 text-xs rounded-full bg-green-100 text-green-800">
              <Layers size={12} className="mr-1" /> 부 그룹
            </span>
          );
        } else {
          return (
            <span className="inline-flex items-center px-2 py-1 text-xs rounded-full bg-yellow-100 text-yellow-800">
              <AlertTriangle size={12} className="mr-1" /> 이상치
            </span>
          );
        }
      }
      // mainGroup와 outliers가 있는 경우
      else if (mainGroup) {
        if (Array.isArray(mainGroup) && mainGroup.includes(modelId)) {
          return (
            <span className="inline-flex items-center px-2 py-1 text-xs rounded-full bg-blue-100 text-blue-800">
              <Layers size={12} className="mr-1" /> 주요 그룹
            </span>
          );
        } else if (outliers && Array.isArray(outliers) && outliers.includes(modelId)) {
          return (
            <span className="inline-flex items-center px-2 py-1 text-xs rounded-full bg-yellow-100 text-yellow-800">
              <AlertTriangle size={12} className="mr-1" /> 이상치
            </span>
          );
        }
      }
      
      return null;
    }
  };

  // Message feature badge component
  const MessageFeatureBadge = ({ modelId, messageId }) => {
    if (!similarityGroups[messageId] || !similarityGroups[messageId].responseFeatures) {
      return null;
    }
    
    const features = similarityGroups[messageId].responseFeatures[modelId];
    if (!features) return null;
    
    return (
      <div className="flex flex-wrap gap-1 mt-1">
        {/* 다국어 감지 결과 표시 (추가) */}
        {features.detectedLang && features.detectedLang !== "unknown" && (
          <span className="inline-flex items-center px-2 py-0.5 text-xs rounded-full bg-indigo-100 text-indigo-800">
            <Globe size={10} className="mr-1" /> {features.detectedLang}
          </span>
        )}
        
        {/* 언어 비율 정보가 있고 2개 이상의 언어가 감지된 경우 (추가) */}
        {features.langRatios && Object.entries(features.langRatios).filter(([lang, ratio]) => 
          ratio > 0.1 && lang !== features.detectedLang
        ).length > 0 && (
          <span className="inline-flex items-center px-2 py-0.5 text-xs rounded-full bg-blue-100 text-blue-800">
            혼합 언어
          </span>
        )}
        
        {/* 기존 코드 표시 */}
        {(features.hasCode === true || features.hasCode === "True" || features.codeBlockCount > 0) && (
          <span className="inline-flex items-center px-2 py-0.5 text-xs rounded-full bg-purple-100 text-purple-800">
            코드 포함
          </span>
        )}
        {(features.listItemCount > 0) && (
          <span className="inline-flex items-center px-2 py-0.5 text-xs rounded-full bg-gray-100 text-gray-800">
            목록 {features.listItemCount}개
          </span>
        )}
        {(features.linkCount > 0) && (
          <span className="inline-flex items-center px-2 py-0.5 text-xs rounded-full bg-indigo-100 text-indigo-800">
            링크 {features.linkCount}개
          </span>
        )}
        {features.length && (
          <span className="inline-flex items-center px-2 py-0.5 text-xs rounded-full bg-green-100 text-green-800">
            길이 {features.length}자
          </span>
        )}
        {/* 어휘 다양성 정보 표시 (추가) */}
        {features.vocabularyDiversity && (
          <span className="inline-flex items-center px-2 py-0.5 text-xs rounded-full bg-yellow-100 text-yellow-800">
            어휘 {(features.vocabularyDiversity * 100).toFixed(0)}%
          </span>
        )}
      </div>
    );
  };

  return (
    <div className="flex flex-col h-screen w-full bg-white">
      {/* AI model headers - fixed at top */}
      <div className="grid border-b bg-white sticky top-0 z-20" 
           style={{ gridTemplateColumns: `repeat(${selectedModels.length + 1}, minmax(0, 1fr))` }}>
        {selectedModels.map((modelId) => (
          <div key={modelId} className="p-4 text-xl font-semibold text-center border-r">
            {modelId.toUpperCase()}
          </div>
        ))}
        <div className="p-4 text-xl font-semibold text-center">
          최적 답변
        </div>
      </div>

      {/* Model selection button and similarity legend */}
      <div className="flex-shrink-0 px-4 py-2 border-b bg-white flex justify-between items-center">
        <div className="flex items-center gap-2">
          <button 
            onClick={() => setIsModalOpen(true)} 
            className="flex items-center gap-2 py-2 px-4 text-sm bg-purple-100 text-purple-700 rounded-lg hover:bg-purple-200 transition-colors"
          >
            <Settings size={16} />
            AI 모델 선택 ({selectedModels.length})
          </button>
          
          {/* 다국어 유사도 분석 배지 */}
          <div className="flex items-center py-2 px-4 text-sm bg-indigo-100 text-indigo-700 rounded-lg">
            <Globe size={16} className="mr-2" />
            다국어 유사도 분석 활성화
          </div>
        </div>
        
        <div className="flex items-center gap-3 text-sm">
          <div className="flex items-center">
            <div className="w-3 h-3 bg-blue-500 rounded-full mr-1"></div>
            <span>주요 그룹</span>
          </div>
          <div className="flex items-center">
            <div className="w-3 h-3 bg-green-500 rounded-full mr-1"></div>
            <span>부 그룹</span>
          </div>
          <div className="flex items-center">
            <div className="w-3 h-3 bg-yellow-500 rounded-full mr-1"></div>
            <span>이상치</span>
          </div>
        </div>
      </div>

      {/* Chat message area */}
      <div className="flex-1 grid overflow-y-auto pt-4" 
           style={{ gridTemplateColumns: `repeat(${selectedModels.length + 1}, minmax(0, 1fr))` }}>
        {/* Selected models' message areas */}
        {selectedModels.map((modelId) => (
          <div key={modelId} className="flex flex-col border-r h-full overflow-hidden">
            <div className="flex-1 overflow-y-auto p-4">
              {messages[modelId]?.map((message, index) => {
                // Create message ID (same as optimal)
                const messageId = message.requestId || generateRequestId();

                // const messageId = message.requestId || `${message.text}-${index}`;
                return (
                  <div key={index} className={`flex ${message.isUser ? "justify-end" : "justify-start"} mb-4`}>
                    <div
                      className={`max-w-[85%] p-4 rounded-2xl ${
                        message.isUser 
                          ? "bg-purple-600 text-white" 
                          : `${getModelColor(modelId, messageId)} text-gray-800`
                      }`}
                    >
                      {message.text}
                      {/* Show feature badges only for AI responses */}
                      {!message.isUser && (
                        <div className="mt-2 flex flex-wrap gap-1">
                          <ModelStatusBadge 
                            modelId={modelId} 
                            messageId={messageId}
                          />
                          <MessageFeatureBadge 
                            modelId={modelId} 
                            messageId={messageId}
                          />
                        </div>
                      )}
                    </div>
                  </div>
                );
              })}
              {isLoading && (
                <div className="flex justify-start">
                  <div className="bg-gray-100 text-gray-800 p-4 rounded-2xl">분석중...</div>
                </div>
              )}
              <div ref={messagesEndRefs.current[modelId]} />
            </div>
          </div>
        ))}

        {/* Optimal answer column */}
        <div className="flex flex-col h-full overflow-hidden">
          <div className="flex-1 overflow-y-auto p-4">
            {(() => {
              const userMessages = messages.optimal 
                ? messages.optimal.filter(msg => msg.isUser)
                : [];
              
              if (userMessages.length === 0) {
                return (
                  <div className="flex items-center justify-center h-full text-gray-500">
                    채팅을 입력하면 여기에 최적의 답변이 표시됩니다.
                  </div>
                );
              }
              
              // Render each user message
              return userMessages.map((message, index) => {
                // Create message ID
                // const messageId = message.requestId || `${message.text}-${index}`;
            // const messageId = message.requestId || `${message.text}-${index}-${Date.now()}`;
            const messageId = getMessageId(message, index);
                
                // Get analysis result for this message
                const analysisResult = messageStates[messageId];
                
                // Get similarity analysis result
                const similarityResult = similarityGroups[messageId];
                
                // User message element
                const userMessageElement = (
                  <div className="mb-2">
                    <div className="flex justify-end">
                      <div className="bg-purple-600 text-white p-3 rounded-2xl max-w-[85%]">
                        {message.text}
                      </div>
                    </div>
                  </div>
                );
                
                const similarityButtonElement = (
                  <div className="flex justify-end mb-2">
                    <button
                      onClick={() => handleSimilarityClick(messageId)}
                      className="
                        flex items-center
                        px-3 py-1.5 text-sm
                        bg-indigo-100 text-indigo-700
                        rounded-lg hover:bg-indigo-200
                        transition-colors
                      "
                    >
                      <Globe size={14} className="mr-2" />
                      다국어 유사도 분석 결과 보기
                    </button>
                  </div>
                );
                if (analysisResult) {
                  return (
                    <div key={`msg-${index}`} className="mb-6">
                      {userMessageElement}
                      {similarityButtonElement}
                      <div className="mt-4">
                        <div className="flex justify-between items-center text-sm text-blue-600 mb-2">
                          <span>✅ 분석 완료</span>
                          <span>🤖 {analysisResult.botName || analysisResult.preferredModel || "AI"}</span>
                        </div>
                        {/* 다국어 유사도 분석 기능 표시 배너 */}
                        <div className="bg-indigo-50 p-3 rounded-xl mb-3 flex items-center">
                          <Globe className="text-indigo-600 mr-2 flex-shrink-0" size={18} />
                          <div>
                            <p className="text-sm font-medium text-indigo-700">다국어 유사도 분석 활성화</p>
                            <p className="text-xs text-indigo-600">paraphrase-multilingual-MiniLM-L12-v2 모델 사용 중</p>
                          </div>
                        </div>
                        <div className="bg-gray-100 p-4 rounded-2xl space-y-4">
                          <div>
                            <div className="font-semibold mb-2">✨ 최적의 답변:</div>
                            <div className="bg-white p-3 rounded-xl">
                              {analysisResult.bestResponse || analysisResult.best_response}
                            </div>
                          </div>
                          <div>
                            <div className="font-semibold mb-2">📌 각 AI 분석:</div>
                            {Object.entries(analysisResult.analysis || {}).map(([ai, aiAnalysis]) => (
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
                              {analysisResult.reasoning || '분석 근거 없음'}
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  );
                } else {
                  return (
                    <div key={`msg-${index}`} className="mb-6">
                      {userMessageElement}
                      {similarityButtonElement}
                      <div className="mt-4">
                        <div className="flex justify-between items-center text-sm text-yellow-600 mb-2">
                          <span>⏳ 분석 중...</span>
                        </div>
                        <div className="bg-gray-100 p-4 rounded-2xl space-y-4">
                          <div className="flex items-center justify-center p-4">
                            <div className="animate-pulse flex space-x-4">
                              <div className="h-3 w-3 bg-gray-300 rounded-full"></div>
                              <div className="h-3 w-3 bg-gray-300 rounded-full"></div>
                              <div className="h-3 w-3 bg-gray-300 rounded-full"></div>
                            </div>
                          </div>
                          <div className="text-sm text-gray-500 text-center">
                            AI 답변 분석 중입니다. 잠시만 기다려주세요.
                          </div>
                        </div>
                      </div>
                    </div>
                  );
                }
              });
            })()}
            <div ref={messagesEndRefs.current.optimal} />
          </div>
        </div>
      </div>

      {/* Input area */}
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
            {/* 이미지 업로드 버튼 */}
         <button
           type="button"
           onClick={() => fileInputRef.current.click()}
           className="p-2 rounded-lg hover:bg-gray-100 mr-2"
           disabled={isLoading}
         >
           <Camera size={20} />
         </button>
         {/* 실제 파일 인풋 (숨김) */}
         <input
           type="file"
           accept="image/*"
           ref={fileInputRef}
           className="hidden"
           onChange={(e) => {
             const file = e.target.files?.[0];
             if (!file) return;
             setSelectedImage(file);
             setPreviewURL(URL.createObjectURL(file));
           }}
         />
         {/* 미리보기 */}
         {previewURL && (
           <div className="mr-2">
             <img src={previewURL} alt="preview" className="w-16 h-16 object-cover rounded-lg" />
           </div>
         )}

          <button 
            type="submit" 
            disabled={isLoading} 
            className="p-2 rounded-lg hover:bg-gray-100"
          >
            <Send className="w-5 h-5 text-gray-600" />
          </button>
        </form>
      </div>

      {/* Model selection modal */}
      <ModelSelectionModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        selectedModels={selectedModels}
        onModelSelect={setSelectedModels}
      />
      
      {/* 유사도 분석 상세 모달 */}
      <SimilarityDetailModal
        isOpen={isSimilarityModalOpen}
        onClose={() => setIsSimilarityModalOpen(false)}
        similarityData={currentSimilarityData}
      />
    </div>
  );
};

export default ChatBox;
