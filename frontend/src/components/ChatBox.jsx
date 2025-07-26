// paste.txt 수정된 내용 (주요 변경사항만)
import React, { useState, useEffect, useRef } from "react";
import { useChat } from "../context/ChatContext";
import ModelSelectionModal from "./ModelSelectionModal";
import SimilarityDetailModal from "./SimilarityDetailModal";
import { Send, Settings, Layers, AlertTriangle, Globe, BookOpen, Camera, Workflow, Zap } from "lucide-react";

const ChatBox = () => {
  const { 
    messages, 
    sendMessage, 
    isLoading, 
    selectedModels, 
    setSelectedModels, 
    analysisResults,
    similarityResults,
    isProcessingImage,
    imageAnalysisResults,
    processImageUpload
  } = useChat();
  
  // 새로 추가된 상태
  const [useWorkflow, setUseWorkflow] = useState(true); // LangGraph 워크플로우 사용 여부
  const [workflowStatus, setWorkflowStatus] = useState('ready'); // ready, running, completed, error
  
  // 기존 상태들...
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
  const processedKeys = useRef(new Set());
  
  const getMessageId = (message, index) => {
    return message.requestId;
  };
  
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

  // 기존 useEffect들... (변경 없음)
  useEffect(() => {
    const allModels = [...selectedModels, "optimal"];
    allModels.forEach((modelId) => {
      if (!messagesEndRefs.current[modelId]) {
        messagesEndRefs.current[modelId] = React.createRef();
      }
    });
  }, [selectedModels]);
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
  // 
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
  const generateRequestId = () => {
    const timestamp = Date.now();
    const randomPart = Math.floor(Math.random() * 1000000)
      .toString()
      .padStart(6, "0");
    return `req-${timestamp}-${randomPart}`;
  };

  // 워크플로우 상태에 따른 메시지 전송 처리 수정
  const handleSendMessage = (e) => {
    e.preventDefault();
    const requestId = generateRequestId();

    if (selectedImage) {
      processImageUpload(selectedImage, requestId);
      setSelectedImage(null);
      setPreviewURL("");
    } else if (inputMessage.trim()) {
      // 워크플로우 사용 여부를 sendMessage에 전달
      sendMessage(inputMessage, requestId, { useWorkflow });
      setInputMessage("");
      
      // 워크플로우 상태 업데이트
      if (useWorkflow) {
        setWorkflowStatus('running');
      }
    }
  };

  // 워크플로우 상태 배지 컴포넌트
  const WorkflowStatusBadge = () => {
    const statusConfig = {
      ready: { color: 'bg-green-100 text-green-700', icon: Workflow, text: '워크플로우 준비됨' },
      running: { color: 'bg-blue-100 text-blue-700', icon: Zap, text: '워크플로우 실행 중' },
      completed: { color: 'bg-purple-100 text-purple-700', icon: Workflow, text: '워크플로우 완료' },
      error: { color: 'bg-red-100 text-red-700', icon: AlertTriangle, text: '워크플로우 오류' }
    };
    
    const config = statusConfig[workflowStatus];
    const Icon = config.icon;
    
    return (
      <div className={`flex items-center py-2 px-4 text-sm rounded-lg ${config.color}`}>
        <Icon size={16} className="mr-2" />
        {config.text}
      </div>
    );
  };

  // 기존 컴포넌트들... (ModelStatusBadge, MessageFeatureBadge 등 변경 없음)
  const ModelStatusBadge = ({ modelId, messageId }) => {
    // 기존 코드와 동일...
    if (!similarityGroups[messageId]) return null;
    
    const groups = similarityGroups[messageId].similarGroups;
    const mainGroup = similarityGroups[messageId].mainGroup;
    const outliers = similarityGroups[messageId].outliers;
    const semanticTags = similarityGroups[messageId].semanticTags || {};
    const responseFeatures = similarityGroups[messageId].responseFeatures || {};
    
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
    
    function renderGroupBadge() {
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
      } else if (mainGroup) {
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

  // 기존 MessageFeatureBadge 함수도 동일...
  const MessageFeatureBadge = ({ modelId, messageId }) => {
    if (!similarityGroups[messageId] || !similarityGroups[messageId].responseFeatures) {
      return null;
    }
    
    const features = similarityGroups[messageId].responseFeatures[modelId];
    if (!features) return null;
    
    return (
      <div className="flex flex-wrap gap-1 mt-1">
        {features.detectedLang && features.detectedLang !== "unknown" && (
          <span className="inline-flex items-center px-2 py-0.5 text-xs rounded-full bg-indigo-100 text-indigo-800">
            <Globe size={10} className="mr-1" /> {features.detectedLang}
          </span>
        )}
        
        {features.langRatios && Object.entries(features.langRatios).filter(([lang, ratio]) => 
          ratio > 0.1 && lang !== features.detectedLang
        ).length > 0 && (
          <span className="inline-flex items-center px-2 py-0.5 text-xs rounded-full bg-blue-100 text-blue-800">
            혼합 언어
          </span>
        )}
        
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

      {/* 수정된 컨트롤 영역 - 워크플로우 토글 추가 */}
      <div className="flex-shrink-0 px-4 py-2 border-b bg-white">
        <div className="flex justify-between items-center mb-2">
          <div className="flex items-center gap-2">
            <button 
              onClick={() => setIsModalOpen(true)} 
              className="flex items-center gap-2 py-2 px-4 text-sm bg-purple-100 text-purple-700 rounded-lg hover:bg-purple-200 transition-colors"
            >
              <Settings size={16} />
              AI 모델 선택 ({selectedModels.length})
            </button>
            
            {/* LangGraph 워크플로우 토글 */}
            <div className="flex items-center gap-2">
              <label className="flex items-center cursor-pointer">
                <input
                  type="checkbox"
                  checked={useWorkflow}
                  onChange={(e) => setUseWorkflow(e.target.checked)}
                  className="sr-only"
                />
                <div className={`relative w-11 h-6 rounded-full transition-colors ${
                  useWorkflow ? 'bg-blue-600' : 'bg-gray-300'
                }`}>
                  <div className={`absolute top-0.5 left-0.5 bg-white w-5 h-5 rounded-full transition-transform ${
                    useWorkflow ? 'translate-x-5' : 'translate-x-0'
                  }`}></div>
                </div>
                <span className="ml-2 text-sm font-medium text-gray-700">
                  LangGraph 워크플로우
                </span>
              </label>
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
        
        {/* 상태 표시 영역 */}
        <div className="flex items-center gap-3">
          {useWorkflow && <WorkflowStatusBadge />}
          
          <div className="flex items-center py-2 px-4 text-sm bg-indigo-100 text-indigo-700 rounded-lg">
            <Globe size={16} className="mr-2" />
            다국어 유사도 분석 활성화
          </div>
          
          {useWorkflow && (
            <div className="flex items-center py-2 px-4 text-sm bg-green-100 text-green-700 rounded-lg">
              <Workflow size={16} className="mr-2" />
              LangChain + LangGraph 사용 중
            </div>
          )}
        </div>
      </div>

      {/* 기존 채팅 메시지 영역 - 변경 없음 */}
      <div className="flex-1 grid overflow-y-auto pt-4" 
           style={{ gridTemplateColumns: `repeat(${selectedModels.length + 1}, minmax(0, 1fr))` }}>
        {/* 기존 메시지 렌더링 로직과 동일... */}
        {selectedModels.map((modelId) => (
          <div key={modelId} className="flex flex-col border-r h-full overflow-hidden">
            <div className="flex-1 overflow-y-auto p-4">
              {messages[modelId]?.map((message, index) => {
                const messageId = message.requestId || generateRequestId();
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
                  <div className="bg-gray-100 text-gray-800 p-4 rounded-2xl">
                    {useWorkflow ? '워크플로우 실행 중...' : '분석중...'}
                  </div>
                </div>
              )}
              <div ref={messagesEndRefs.current[modelId]} />
            </div>
          </div>
        ))}

        {/* 최적 답변 컬럼도 기존과 동일하지만 워크플로우 사용 표시 추가 */}
        <div className="flex flex-col h-full overflow-hidden">
          <div className="flex-1 overflow-y-auto p-4">
            {(() => {
              const userMessages = messages.optimal 
                ? messages.optimal.filter(msg => msg.isUser)
                : [];
              
              if (userMessages.length === 0) {
                return (
                  <div className="flex items-center justify-center h-full text-gray-500">
                    <div className="text-center">
                      <div className="mb-4">
                        {useWorkflow ? (
                          <Workflow className="mx-auto text-blue-500" size={48} />
                        ) : (
                          <Zap className="mx-auto text-purple-500" size={48} />
                        )}
                      </div>
                      <p>
                        {useWorkflow 
                          ? 'LangGraph 워크플로우로 최적의 답변을 생성합니다'
                          : '기존 방식으로 최적의 답변을 생성합니다'
                        }
                      </p>
                    </div>
                  </div>
                );
              }
              
              return userMessages.map((message, index) => {
                const messageId = getMessageId(message, index);
                const analysisResult = messageStates[messageId];
                const similarityResult = similarityGroups[messageId];
                
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
                      className="flex items-center px-3 py-1.5 text-sm bg-indigo-100 text-indigo-700 rounded-lg hover:bg-indigo-200 transition-colors"
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
                          <div className="flex items-center gap-2">
                            <span>🤖 {analysisResult.botName || analysisResult.preferredModel || "AI"}</span>
                            {analysisResult.workflowUsed && (
                              <span className="px-2 py-1 text-xs bg-green-100 text-green-700 rounded-full">
                                워크플로우
                              </span>
                            )}
                          </div>
                        </div>
                        
                        {/* 다국어 유사도 분석 기능 표시 배너 */}
                        <div className="bg-indigo-50 p-3 rounded-xl mb-3 flex items-center">
                          <Globe className="text-indigo-600 mr-2 flex-shrink-0" size={18} />
                          <div className="flex-1">
                            <p className="text-sm font-medium text-indigo-700">다국어 유사도 분석 활성화</p>
                            <p className="text-xs text-indigo-600">
                              {useWorkflow 
                                ? 'LangGraph 워크플로우 + paraphrase-multilingual-MiniLM-L12-v2 사용 중'
                                : 'paraphrase-multilingual-MiniLM-L12-v2 모델 사용 중'
                              }
                            </p>
                          </div>
                          {useWorkflow && (
                            <Workflow className="text-green-600 ml-2" size={18} />
                          )}
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
                          
                          {/* 워크플로우 에러 표시 */}
                          {analysisResult.errors && analysisResult.errors.length > 0 && (
                            <div>
                              <div className="font-semibold mb-2 text-orange-600">⚠️ 워크플로우 경고:</div>
                              <div className="bg-orange-50 p-3 rounded-xl">
                                {analysisResult.errors.map((error, idx) => (
                                  <div key={idx} className="text-sm text-orange-700">{error}</div>
                                ))}
                              </div>
                            </div>
                          )}
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
                          <span>⏳ {useWorkflow ? '워크플로우 실행 중...' : '분석 중...'}</span>
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
                            {useWorkflow 
                              ? 'LangGraph 워크플로우로 AI 답변을 분석하고 있습니다...'
                              : 'AI 답변 분석 중입니다. 잠시만 기다려주세요.'
                            }
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

      {/* 기존 입력 영역 - 변경 없음 */}
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
            type="button"
            onClick={() => fileInputRef.current.click()}
            className="p-2 rounded-lg hover:bg-gray-100 mr-2"
            disabled={isLoading}
          >
            <Camera size={20} />
          </button>
          
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

      {/* 기존 모달들 */}
      <ModelSelectionModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        selectedModels={selectedModels}
        onModelSelect={setSelectedModels}
      />
      
      <SimilarityDetailModal
        isOpen={isSimilarityModalOpen}
        onClose={() => setIsSimilarityModalOpen(false)}
        similarityData={currentSimilarityData}
      />
    </div>
  );
};

export default ChatBox;