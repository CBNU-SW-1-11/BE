import React, { useState, useRef, useEffect } from 'react';
import { videoAnalysisService } from '../../services/videoAnalysisService';

const ChatInterface = ({ selectedVideo, onFrameReceived }) => {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  const quickActions = [
    { text: '비디오 요약해줘', icon: '📝' },
    { text: '하이라이트 찾아줘', icon: '⭐' },
    { text: '사람 찾아줘', icon: '👤' },
    { text: '빨간색 객체 찾아줘', icon: '🔴' },
    { text: '자동차 찾아줘', icon: '🚗' },
    { text: '비디오 정보 알려줘', icon: 'ℹ️' }
  ];

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    if (selectedVideo) {
      setMessages([
        {
          type: 'system',
          content: `🎬 "${selectedVideo.filename}" 비디오가 선택되었습니다. 무엇을 도와드릴까요?`,
          timestamp: new Date()
        }
      ]);
    }
  }, [selectedVideo]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const sendMessage = async (message = inputValue) => {
    if (!message.trim() || isLoading) return;

    const userMessage = {
      type: 'user',
      content: message,
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInputValue('');
    setIsLoading(true);

    try {
      const response = await videoAnalysisService.sendChatMessage(message, selectedVideo.id);
      
      const botMessage = {
        type: 'bot',
        content: response.response,
        timestamp: new Date(),
        frameImage: response.frame_image,
        frameNumber: response.frame_number,
        frameTimestamp: response.timestamp,
        responseType: response.type,
        searchResults: response.search_results,
        enhancedInfo: response.enhanced_info
      };

      setMessages(prev => [...prev, botMessage]);

      // 프레임 정보가 있으면 부모 컴포넌트에 전달
      if (response.frame_image && onFrameReceived) {
        onFrameReceived({
          imageUrl: response.frame_image,
          frameNumber: response.frame_number,
          timestamp: response.timestamp,
          responseType: response.type,
          enhancedInfo: response.enhanced_info
        });
      }

    } catch (error) {
      const errorMessage = {
        type: 'error',
        content: `죄송합니다. 오류가 발생했습니다: ${error.message}`,
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const formatTimestamp = (timestamp) => {
    return timestamp.toLocaleTimeString('ko-KR', { 
      hour: '2-digit', 
      minute: '2-digit' 
    });
  };

  const renderMessage = (message, index) => {
    const isUser = message.type === 'user';
    const isSystem = message.type === 'system';
    const isError = message.type === 'error';

    return (
      <div key={index} className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`}>
        <div className={`max-w-xs lg:max-w-md px-4 py-2 rounded-lg ${
          isUser 
            ? 'bg-blue-600 text-white' 
            : isError
            ? 'bg-red-100 text-red-800 border border-red-200'
            : isSystem
            ? 'bg-gray-100 text-gray-700'
            : 'bg-white border border-gray-200 text-gray-900'
        }`}>
          {!isUser && !isSystem && (
            <div className="flex items-center mb-1">
              <span className="text-lg mr-1">🤖</span>
              <span className="text-xs text-gray-500">AI Assistant</span>
            </div>
          )}
          
          <div className="whitespace-pre-wrap">{message.content}</div>
          
          {/* 프레임 정보 표시 */}
          {message.frameNumber && (
            <div className="mt-2 text-xs opacity-75">
              📍 프레임 {message.frameNumber} ({message.frameTimestamp?.toFixed(1)}초)
            </div>
          )}
          
          {/* 검색 결과 요약 */}
          {message.searchResults && message.searchResults.length > 0 && (
            <div className="mt-2 text-xs opacity-75">
              🔍 {message.searchResults.length}개 결과 발견
            </div>
          )}
          
          <div className={`text-xs mt-1 ${isUser ? 'text-blue-200' : 'text-gray-500'}`}>
            {formatTimestamp(message.timestamp)}
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="bg-white rounded-lg shadow flex flex-col h-96">
      {/* Header */}
      <div className="px-4 py-3 border-b border-gray-200 bg-gray-50 rounded-t-lg">
        <h3 className="font-medium text-gray-900">💬 AI 비디오 분석 채팅</h3>
        <p className="text-sm text-gray-600">
          {selectedVideo.filename} • {selectedVideo.enhanced_analysis ? 'Enhanced' : 'Basic'} 분석
        </p>
      </div>

      {/* Quick Actions */}
      <div className="px-4 py-2 border-b border-gray-100">
        <div className="flex flex-wrap gap-2">
          {quickActions.map((action, index) => (
            <button
              key={index}
              onClick={() => sendMessage(action.text)}
              disabled={isLoading}
              className="inline-flex items-center px-2 py-1 text-xs bg-gray-100 text-gray-700 rounded hover:bg-gray-200 disabled:opacity-50 transition-colors"
            >
              <span className="mr-1">{action.icon}</span>
              {action.text}
            </button>
          ))}
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-2">
        {messages.map((message, index) => renderMessage(message, index))}
        
        {isLoading && (
          <div className="flex justify-start mb-4">
            <div className="bg-gray-100 px-4 py-2 rounded-lg">
              <div className="flex items-center space-x-2">
                <div className="animate-spin h-4 w-4 border-2 border-blue-600 border-t-transparent rounded-full"></div>
                <span className="text-gray-600">AI가 분석 중입니다...</span>
              </div>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="px-4 py-3 border-t border-gray-200">
        <div className="flex space-x-2">
          <input
            ref={inputRef}
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="비디오에 대해 질문하세요... (예: '빨간 자동차 찾아줘')"
            disabled={isLoading}
            className="flex-1 px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-100"
          />
          <button
            onClick={() => sendMessage()}
            disabled={!inputValue.trim() || isLoading}
            className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:bg-gray-400 transition-colors"
          >
            {isLoading ? '⏳' : '전송'}
          </button>
        </div>
        
        <div className="mt-2 text-xs text-gray-500">
          💡 <strong>사용 예시:</strong> "사람 찾아줘", "하이라이트 보여줘", "비디오 요약해줘"
        </div>
      </div>
    </div>
  );
};

export default ChatInterface;