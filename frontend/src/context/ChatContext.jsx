
// export default ChatInterface;
import React, { createContext, useState, useContext, useRef, useEffect } from 'react';
import { useSelector } from 'react-redux';
import { clusterResponses, extractResponseFeatures } from "../utils/similarityAnalysis";

// Context 생성
const ChatContext = createContext(null);

// 봇 선택 모달 컴포넌트
const SelectBotModal = ({ isOpen, onClose, onSelectBot, currentModel }) => {
  const [selectedAI, setSelectedAI] = useState(null);

  useEffect(() => {
    // 현재 모델에 따라 초기 선택 설정
    const reverseMapping = {
      'gpt': 'GPT-3.5',
      'claude': 'Claude', 
      'mixtral': 'Mixtral'
    };
    setSelectedAI(reverseMapping[currentModel]);
  }, [currentModel]);

  const handleConfirm = () => {
    if (selectedAI) {
      const botMapping = {
        'GPT-3.5': 'gpt',
        'Claude': 'claude',
        'Mixtral': 'mixtral'
      };
      onSelectBot(botMapping[selectedAI]);
      onClose();
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg p-6 w-full max-w-2xl shadow-lg relative pb-20">
        <h2 className="text-xl font-bold mb-4">최적화 모델 선택</h2>
        <div className="grid grid-cols-3 gap-4 mb-6">
          {["GPT-3.5", "Claude", "Mixtral"].map((model) => (
            <button
              key={model}
              onClick={() => setSelectedAI(model)}
              className={`p-6 border rounded-lg transition-colors ${
                selectedAI === model ? "bg-blue-300" : "hover:bg-blue-50"
              }`}
            >
              <h3 className="font-bold text-lg mb-2">{model}</h3>
              <p className="text-sm text-gray-600 mb-2">
                {model === "GPT-3.5" 
                  ? "OpenAI의 GPT-3.5 모델" 
                  : model === "Claude" 
                  ? "Anthropic의 Claude 모델" 
                  : "Mixtral-8x7B 모델"}
              </p>
              <ul className="text-xs text-gray-500 list-disc pl-4">
                {model === "GPT-3.5" && (
                  <>
                    <li>빠른 응답 속도</li>
                    <li>일관된 답변 품질</li>
                    <li>다양한 주제 처리</li>
                  </>
                )}
                {model === "Claude" && (
                  <>
                    <li>높은 분석 능력</li>
                    <li>정확한 정보 제공</li>
                    <li>상세한 설명 제공</li>
                  </>
                )}
                {model === "Mixtral" && (
                  <>
                    <li>균형잡힌 성능</li>
                    <li>다국어 지원</li>
                    <li>코드 분석 특화</li>
                  </>
                )}
              </ul>
            </button>
          ))}
        </div>
        <button
          onClick={handleConfirm}
          className={`absolute bottom-6 right-6 px-6 py-3 rounded-lg transition-colors ${
            selectedAI
              ? "bg-blue-500 text-white hover:bg-blue-600"
              : "bg-gray-300 text-gray-500 cursor-not-allowed"
          }`}
          disabled={!selectedAI}
        >
          확인
        </button>
      </div>
    </div>
  );
};

export const ChatProvider = ({ children }) => {
  const [messages, setMessages] = useState({
    gpt: [],
    claude: [],
    mixtral: [],
    gemini: [],
    llama: [],
    palm: [],
    allama: [],
    deepseek: [],
    bloom: [],
    labs: [],
    optimal: []
  });
  
  const [selectedModels, setSelectedModels] = useState(['gpt', 'claude', 'mixtral']);
  const [isLoading, setIsLoading] = useState(false);
  const [analysisResults, setAnalysisResults] = useState({});
  
  // 유사도 분석 결과를 저장할 새로운 상태 추가
  const [similarityResults, setSimilarityResults] = useState({});
  
  // 모델 선택 모달용 상태
  const [selectedBot, setSelectedBot] = useState('gpt'); // 기본값: 'gpt'
  const [isSelectionModalOpen, setIsSelectionModalOpen] = useState(false);
  const messagesEndRef = useRef(null);
  
  // 응답 중인 모델 추적
  const [respondingBots, setRespondingBots] = useState({
    gpt: false,
    claude: false,
    mixtral: false,
    gemini: false,
    llama: false,
    palm: false,
    allama: false,
    deepseek: false,
    bloom: false,
    labs: false
  });
  const [isProcessingImage, setIsProcessingImage] = useState(false);
  const [imageAnalysisResults, setImageAnalysisResults] = useState({});
  // Redux에서 user 정보 가져오기
  const { user } = useSelector((state) => state.auth || { user: null });

  // 디버깅을 위한 스트림 데이터 로깅
  useEffect(() => {
    const originalFetch = window.fetch;
    window.fetch = function(...args) {
      const result = originalFetch.apply(this, args);
      if (args[0] && typeof args[0] === 'string' && args[0].includes('/chat/')) {
        // 채팅 요청인 경우 응답 스트림 모니터링
        result.then(response => {
          const reader = response.clone().body.getReader();
          const decoder = new TextDecoder();
          
          async function processStream() {
            while (true) {
              const { done, value } = await reader.read();
              if (done) break;
              
              const chunk = decoder.decode(value, { stream: true });
              console.log('Raw stream chunk:', chunk);
            }
          }
          
          processStream().catch(console.error);
        }).catch(console.error);
      }
      return result;
    };
  }, []);

  // 로그인한 사용자의 선호 모델 로드 (로그인하지 않은 경우 기본값 사용)
  useEffect(() => {
    if (user?.settings?.preferredModels) {
      try {
        const preferredModels = JSON.parse(user.settings.preferredModels);
        if (Array.isArray(preferredModels) && preferredModels.length > 0) {
          setSelectedModels(preferredModels);
        }
      } catch (error) {
        console.error("Error parsing preferred models:", error);
      }
    }
    
    // 기존 preferredModel 설정 호환성 유지
    if (user?.settings?.preferredModel) {
      const modelMapping = {
        'gpt': 'gpt',
        'claude': 'claude',
        'mixtral': 'mixtral',
        'default': 'gpt'
      };
      setSelectedBot(modelMapping[user.settings.preferredModel] || 'gpt');
    }
  }, [user]);

  const addMessage = (botName, text, isUser, requestId = null) => {
    setMessages((prev) => ({
      ...prev,
      [botName]: [...(prev[botName] || []), { text, isUser, requestId, timestamp: Date.now() }],
    }));
  };

  // 디버깅용 로깅
  useEffect(() => {
    console.log('Current similarity results:', similarityResults);
  }, [similarityResults]);

  // 봇 선택 핸들러
  const handleBotSelection = async (botName) => {
    console.log('===== 모델 변경 =====');
    console.log('이전 모델:', selectedBot);
    console.log('새로 선택된 모델:', botName);
    console.log('===================');
    
    setSelectedBot(botName);
    setIsSelectionModalOpen(false);

    // 선택한 모델을 사용자 설정에 저장 (로그인한 경우에만 저장 시도)
    try {
      const token = localStorage.getItem("accessToken");
      if (!token) {
        console.log("No token found, skipping settings save");
        return;
      }

      const settingsData = {
        preferredModel: botName,
        language: (user && user.settings && user.settings.language) || 'ko'
      };

      const response = await fetch("http://localhost:8000/api/user/settings/", {
        method: "PUT",
        headers: {
          "Content-Type": "application/json",
          "Authorization": `Token ${token}`,
        },
        body: JSON.stringify(settingsData),
      });

      if (!response.ok) {
        throw new Error("모델 설정 저장에 실패했습니다.");
      }
    } catch (error) {
      console.error("Error saving model preference:", error);
    }
  };

// 프론트엔드에서 유사도 분석 수행
useEffect(() => {
  // 모든 모델에 응답이 도착했는지 확인
  const allModelsHaveResponses = () => {
    if (!selectedModels.length) return false;

    // 사용자 메시지들만 필터링
    const userMessages = messages.optimal?.filter(msg => msg.isUser) || [];

    userMessages.forEach(userMsg => {
      // 각 모델이 이 메시지에 답했는지 확인
      const allHaveResponses = selectedModels.every(modelId => {
        return (messages[modelId] || []).some(
          msg => !msg.isUser && msg.requestId === userMsg.requestId
        );
      });

      // 아직 저장된 적 없는 메시지의 경우에만 분석 실행
      if (allHaveResponses && !similarityResults[userMsg.requestId]) {
        // 모델별 응답 수집
        const responses = {};
        selectedModels.forEach(modelId => {
          const aiMsg = (messages[modelId] || []).find(
            msg => !msg.isUser && msg.requestId === userMsg.requestId
          );
          if (aiMsg) responses[modelId] = aiMsg.text;
        });

        // 2개 이상 응답이 있으면 클러스터링
        if (Object.keys(responses).length >= 2) {
          const result = clusterResponses(responses, 0.01);

          // 응답 특성 추출
          const responseFeatures = {};
          Object.entries(responses).forEach(([modelId, text]) => {
               const features = extractResponseFeatures(text);
               features.length = text.length;           // ← 여기서 글자 수 할당
               responseFeatures[modelId] = features;
            });

          // 최종 유사도 분석 결과 구성
          const analysisResult = {
            ...result,
            responseFeatures,
            timestamp: Date.now(),
            requestId: userMsg.requestId,
            userMessage: userMsg.text
          };

          // requestId 를 key 로 해서 저장
          setSimilarityResults(prev => ({
            ...prev,
            [userMsg.requestId]: analysisResult
          }));
        }
      }
    });
  };

  allModelsHaveResponses();
}, [messages, selectedModels]);

  const sendMessage = async (userMessage) => {
    if (!userMessage.trim() || selectedModels.length === 0) return;

    setIsLoading(true);
    
    // 각 요청마다 고유 ID 생성
    const requestId = `req-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    
    // 응답 중인 상태로 설정
    const respondingModels = {};
    selectedModels.forEach(model => {
      respondingModels[model] = true;
    });
    setRespondingBots({...respondingBots, ...respondingModels});
    
    try {
      const token = localStorage.getItem("accessToken");
      const headers = {
        "Content-Type": "application/json"
      };

      // 토큰이 있는 경우에만 Authorization 헤더 추가
      if (token) {
        headers["Authorization"] = `Token ${token}`;
      }
      
      // 선택된 모델들에게만 사용자 메시지 추가 (requestId 포함)
      selectedModels.forEach((modelId) => {
        addMessage(modelId, userMessage, true, requestId);
      });
      
      // optimal 칼럼에도 사용자 메시지 추가
      addMessage('optimal', userMessage, true, requestId);

      // 분석을 위한 모델 선택 (선호 모델 또는 첫 번째 선택된 모델)
      const analyzerModel = selectedModels.includes(selectedBot) 
        ? selectedBot 
        : selectedModels[0] || 'gpt';
      
      console.log('===== 분석 모델 =====');
      console.log('선택된 모델들:', selectedModels);
      console.log('분석 수행 모델:', analyzerModel);
      console.log('요청 ID:', requestId);
      console.log('====================');
      
      // 응답 스트림 가져오기
      const response = await fetch(`http://localhost:8000/chat/${analyzerModel}/`, {
        method: "POST",
        headers,
        body: JSON.stringify({
          message: userMessage,
          compare: true, // 비교 기능 활성화
          selectedModels: selectedModels, // 선택된 모델 전달
          language: (user && user.settings && user.settings.language) || 'ko', // 기본값: 'ko'
          requestId: requestId // 요청 ID 전달
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      // 스트림 리더 생성
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      
      // 임시 분석 결과 저장
      let currentAnalysis = null;

      // 스트림 처리
      while (true) {
        const { done, value } = await reader.read();
        
        if (done) {
          console.log('Stream complete');
          break;
        }
        
        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split('\n').filter(line => line.trim() !== '');
        
        for (const line of lines) {
          try {
            const data = JSON.parse(line);
            console.log('Received data:', data);
            
            if (data.type === 'bot_response' && selectedModels.includes(data.botId)) {
              // 각 모델의 응답을 받는 즉시 표시
              addMessage(data.botId, data.response, false, data.requestId || requestId);
              
              // 모델 응답 완료 상태 업데이트
              setRespondingBots(prev => ({
                ...prev,
                [data.botId]: false
              }));
            } 
            else if (data.type === 'bot_error' && selectedModels.includes(data.botId)) {
              // 에러 메시지 추가
              addMessage(data.botId, `오류가 발생했습니다: ${data.error}`, false, data.requestId || requestId);
              
              // 오류 발생한 모델 응답 완료 처리
              setRespondingBots(prev => ({
                ...prev,
                [data.botId]: false
              }));
            }
            // 유사도 분석 결과 처리 추가
            else if (data.type === 'similarity_analysis') {
              console.log('Received similarity analysis:', data.result);
            
              const similarityData = {
                requestId,
                timestamp: data.timestamp,
                userMessage: data.userMessage || userMessage,
                ...data.result
              };
              
              // similarityResults 상태 업데이트
              setSimilarityResults(prev => ({
                        ...prev,
                         [requestId]: similarityData
                       }));
                      }
            else if (data.type === 'analysis') {
              // 분석 결과 저장
              currentAnalysis = {
                botName: data.preferredModel,
                bestResponse: data.best_response,
                analysis: data.analysis,
                reasoning: data.reasoning,
                timestamp: Date.now(),
                requestId: data.requestId || requestId,
                userMessage: data.userMessage || userMessage
              };
              
              // 중요: 매번 새 분석 결과 생성을 위해, 타임스탬프를 키에 포함
              // 이렇게 하면 동일한 질문에 대해서도 매번 다른 키로 저장됨
              const analysisKey = `${userMessage}_${Date.now()}`;
              
              // 전체 분석 결과 업데이트
              setAnalysisResults(prev => {
                // 이전 결과는 그대로 유지하고 새 결과만 추가
                return {
                  ...prev,
                  [analysisKey]: currentAnalysis
                };
              });
            }
          } catch (e) {
            console.error('Error parsing stream chunk:', e, line);
          }
        }
      }
    } catch (error) {
      console.error("Message sending error:", error);
      // 에러 메시지를 선택된 모든 봇에 표시
      selectedModels.forEach((modelId) => {
        addMessage(modelId, `오류가 발생했습니다: ${error.message}`, false, requestId);
      });
      
      // 모든 선택된 모델 응답 완료 상태로 설정
      const completedModels = {};
      selectedModels.forEach(model => {
        completedModels[model] = false;
      });
      setRespondingBots({...respondingBots, ...completedModels});
    }
    
    setIsLoading(false);
  };
  const processImageUpload = async (imageFile, analysisMode) => {
    if (!imageFile || selectedModels.length === 0) return;
    
    setIsLoading(true);
    setIsProcessingImage(true);
    
    // 각 요청마다 고유 ID 생성
    const requestId = `img-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
    
    // 이미지 분석 모드에 따른 프롬프트 생성
    let promptText = '';
    switch (analysisMode) {
      case 'describe':
        promptText = '이 이미지를 자세히 설명해주세요. 이미지에서 보이는 모든 중요한 요소를 포함하세요.';
        break;
      case 'ocr':
        promptText = '이 이미지에서 모든 텍스트를 추출하고 원본 레이아웃을 최대한 유지하여 표시해주세요.';
        break;
      case 'objects':
        promptText = '이 이미지에서 인식 가능한 모든 객체를 나열하고 각각에 대해 간략히 설명해주세요.';
        break;
      default:
        promptText = '이 이미지를 분석해주세요.';
    }
    
    // 이미지 파일 처리를 위한 FormData 생성
    const formData = new FormData();
    formData.append('image', imageFile);
    formData.append('prompt', promptText);
    formData.append('analysisMode', analysisMode);
    formData.append('requestId', requestId);
    formData.append('selectedModels', JSON.stringify(selectedModels));
    formData.append('language', (user && user.settings && user.settings.language) || 'ko');
    
    // 각 모델과 optimal 컬럼에 사용자 메시지 추가
    const userImageMessage = `[이미지 업로드됨] - ${analysisMode === 'describe' ? '이미지 설명' : analysisMode === 'ocr' ? '텍스트 추출' : '객체 인식'} 요청`;
    
    selectedModels.forEach((modelId) => {
      addMessage(modelId, userImageMessage, true, requestId);
    });
    
    // optimal 칼럼에도 사용자 메시지 추가
    addMessage('optimal', userImageMessage, true, requestId);
    
    try {
      const token = localStorage.getItem("accessToken");
      const headers = {};
      
      // 토큰이 있는 경우에만 Authorization 헤더 추가
      if (token) {
        headers["Authorization"] = `Token ${token}`;
      }
      
      // 응답 중인 상태로 설정
      const respondingModels = {};
      selectedModels.forEach(model => {
        respondingModels[model] = true;
      });
      setRespondingBots({...respondingBots, ...respondingModels});
      
      // 이미지 분석 API 엔드포인트로 요청 
      // 참고: 백엔드에 '/analyze-image/' 엔드포인트를 구현해야 함
      const response = await fetch(`http://localhost:8000/analyze-image/`, {
        method: "POST",
        headers, // Content-Type은 FormData 사용 시 자동 설정됨
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      // 스트림 응답 처리 (기존 sendMessage 함수와 유사)
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      
      while (true) {
        const { done, value } = await reader.read();
        
        if (done) {
          console.log('Image analysis stream complete');
          break;
        }
        
        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split('\n').filter(line => line.trim() !== '');
        
        for (const line of lines) {
          try {
            const data = JSON.parse(line);
            console.log('Received image analysis data:', data);
            
            if (data.type === 'bot_response' && selectedModels.includes(data.botId)) {
              // 각 모델의 응답을 받는 즉시 표시
              addMessage(data.botId, data.response, false, data.requestId || requestId);
              
              // 모델 응답 완료 상태 업데이트
              setRespondingBots(prev => ({
                ...prev,
                [data.botId]: false
              }));
            } 
            else if (data.type === 'bot_error' && selectedModels.includes(data.botId)) {
              // 에러 메시지 추가
              addMessage(data.botId, `오류가 발생했습니다: ${data.error}`, false, data.requestId || requestId);
              
              // 오류 발생한 모델 응답 완료 처리
              setRespondingBots(prev => ({
                ...prev,
                [data.botId]: false
              }));
            }
            else if (data.type === 'similarity_analysis') {
              // 유사도 분석 결과 저장
              setSimilarityResults(prev => ({
                ...prev,
                [requestId]: {
                  requestId,
                  timestamp: data.timestamp,
                  userMessage: data.userMessage || userImageMessage,
                  ...data.result
                }
              }));
            }
            else if (data.type === 'analysis') {
              // 분석 결과 저장
              const imageAnalysis = {
                botName: data.preferredModel,
                bestResponse: data.best_response,
                analysis: data.analysis,
                reasoning: data.reasoning,
                timestamp: Date.now(),
                requestId: data.requestId || requestId,
                userMessage: data.userMessage || userImageMessage,
                imageAnalysisMode: analysisMode
              };
              
              // 이미지 분석 결과 업데이트
              setImageAnalysisResults(prev => ({
                ...prev,
                [requestId]: imageAnalysis
              }));
              
              // 일반 분석 결과도 업데이트
              setAnalysisResults(prev => ({
                ...prev,
                [requestId]: imageAnalysis
              }));
            }
          } catch (e) {
            console.error('Error parsing image analysis chunk:', e, line);
          }
        }
      }
    } catch (error) {
      console.error("Image analysis error:", error);
      // 에러 메시지를 선택된 모든 봇에 표시
      selectedModels.forEach((modelId) => {
        addMessage(modelId, `이미지 분석 오류: ${error.message}`, false, requestId);
      });
      
      // 모든 선택된 모델 응답 완료 상태로 설정
      const completedModels = {};
      selectedModels.forEach(model => {
        completedModels[model] = false;
      });
      setRespondingBots({...respondingBots, ...completedModels});
    } finally {
      setIsLoading(false);
      setIsProcessingImage(false);
    }
  };
  
  // 모델 선택 상태 변경 시 사용자 설정 저장
  useEffect(() => {
    const saveUserPreferences = async () => {
      try {
        const token = localStorage.getItem("accessToken");
        if (!token) {
          console.log("No token found, skipping settings save");
          return;
        }

        const settingsData = {
          preferredModels: JSON.stringify(selectedModels),
          language: (user && user.settings && user.settings.language) || 'ko'
        };

        const response = await fetch("http://localhost:8000/api/user/settings/", {
          method: "PUT",
          headers: {
            "Content-Type": "application/json",
            "Authorization": `Token ${token}`,
          },
          body: JSON.stringify(settingsData),
        });

        if (!response.ok) {
          throw new Error("모델 설정 저장에 실패했습니다.");
        }
      } catch (error) {
        console.error("Error saving model preferences:", error);
      }
    };

    // 사용자가 로그인한 경우에만 설정 저장
    if (user) {
      saveUserPreferences();
    }
  }, [selectedModels, user]);

  return (
    <ChatContext.Provider
      value={{
        messages,
        sendMessage,
        isLoading,
        analysisResults,
        similarityResults, // 추가된 유사도 분석 결과
        selectedModels,
        setSelectedModels,
        respondingBots,
        isLoggedIn: !!user,
        messagesEndRef,
        // 기존 모델 선택 모달 관련 함수와 상태 추가
        selectedBot,
        setSelectedBot,
        isSelectionModalOpen,
        setIsSelectionModalOpen,
        handleBotSelection,
        processImageUpload,
       isProcessingImage,
     imageAnalysisResults,
        
      }}
    >
      {children}
      
      {/* 로그인 여부와 상관없이 모델 선택 모달 표시 */}
      <SelectBotModal
        isOpen={isSelectionModalOpen}
        onClose={() => setIsSelectionModalOpen(false)}
        onSelectBot={handleBotSelection}
        currentModel={selectedBot}
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