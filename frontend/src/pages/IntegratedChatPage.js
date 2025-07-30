import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { videoAnalysisService } from '../services/videoAnalysisService';
import FrameViewer from '../components/FrameViewer'; // 새로 추가된 컴포넌트
import './IntegratedChatPage.css';

const IntegratedChatPage = () => {
  const navigate = useNavigate();
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [videos, setVideos] = useState([]);
  const [selectedVideo, setSelectedVideo] = useState(null);
  const [searchResults, setSearchResults] = useState([]);
  const [error, setError] = useState(null);
  const messagesEndRef = useRef(null);
  
  // 고급 분석 관련 상태
  const [analysisInsights, setAnalysisInsights] = useState({});
  const [advancedSearchMode, setAdvancedSearchMode] = useState(false);
  const [searchFilters, setSearchFilters] = useState({
    includeClipAnalysis: true,
    includeOcrText: true,
    includeVqaResults: false,
    includeSceneGraph: false,
    confidenceThreshold: 0.7
  });
  const [chatMode, setChatMode] = useState('general');
  
  // ✅ 새로 추가: 프레임 뷰어 상태
  const [selectedFrame, setSelectedFrame] = useState(null);
  const [frameViewerOpen, setFrameViewerOpen] = useState(false);

  useEffect(() => {
    loadVideos();
    addWelcomeMessage();
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const loadVideos = async () => {
    try {
      const response = await videoAnalysisService.getVideoList();
      const analyzedVideos = response.videos.filter(v => v.is_analyzed);
      setVideos(analyzedVideos);
      
      if (analyzedVideos.length > 0 && !selectedVideo) {
        const advancedVideo = analyzedVideos.find(v => v.advanced_features_used && 
          Object.values(v.advanced_features_used).some(Boolean)) || analyzedVideos[0];
        setSelectedVideo(advancedVideo);
        
        if (advancedVideo.advanced_features_used) {
          loadVideoInsights(advancedVideo.id);
        }
      }
    } catch (error) {
      console.error('비디오 목록 로드 실패:', error);
      setError('비디오 목록을 불러올 수 없습니다.');
    }
  };

  const loadVideoInsights = async (videoId) => {
    try {
      const insights = await videoAnalysisService.sendVideoChatMessage(
        '이 비디오의 고급 분석 결과와 주요 특징들을 요약해주세요. 사용된 AI 기능들과 발견된 주요 내용을 포함해서 설명해주세요.',
        videoId
      );
      
      setAnalysisInsights(prev => ({
        ...prev,
        [videoId]: insights.response || insights
      }));
    } catch (error) {
      console.error('비디오 인사이트 로드 실패:', error);
    }
  };

  const addWelcomeMessage = () => {
    const welcomeMessage = {
      id: Date.now(),
      type: 'bot',
      content: `🧠 고급 AI 비디오 분석 채팅에 오신 것을 환영합니다!

저는 Scene Graph, VQA, OCR, CLIP 등 최신 AI 기술을 활용하여 비디오를 분석하고, 여러분의 질문에 답변해드립니다.

✨ **새로운 고급 기능들:**
🖼️ **CLIP 씬 분석** - 이미지의 의미적 컨텍스트 이해
📝 **OCR 텍스트 추출** - 비디오 내 텍스트 인식
❓ **VQA 질문답변** - 이미지에 대한 지능적 질문과 답변
🕸️ **Scene Graph** - 객체간 관계 및 상호작용 분석
📦 **바운딩 박스 표시** - 감지된 객체를 시각적으로 강조 표시

**사용법:**
• "사람이 있는 장면 찾아줘" - 바운딩 박스와 함께 결과 표시
• "차가 있는 부분 보여줘" - 차량 감지 결과를 시각적으로 표시
• "텍스트가 있는 장면은?" - OCR 결과와 함께 표시
• "가장 중요한 장면들을 보여줘" - 하이라이트 장면 분석

어떤 도움이 필요하신가요?`,
      timestamp: new Date().toLocaleTimeString()
    };
    
    setMessages([welcomeMessage]);
  };

  const handleVideoSelect = (video) => {
    setSelectedVideo(video);
    setSearchResults([]);
    
    const videoChangeMessage = {
      id: Date.now(),
      type: 'system',
      content: `📹 "${video.original_name}" 비디오로 변경되었습니다.${video.advanced_features_used ? 
        '\n🚀 이 비디오는 고급 분석이 완료되어 더 정확한 결과를 제공할 수 있습니다.' : ''}`,
      timestamp: new Date().toLocaleTimeString()
    };
    
    setMessages(prev => [...prev, videoChangeMessage]);
    
    if (video.advanced_features_used && !analysisInsights[video.id]) {
      loadVideoInsights(video.id);
    }
  };

  const sendMessage = async () => {
    if (!inputMessage.trim() || !selectedVideo) return;

    const userMessage = {
      id: Date.now(),
      type: 'user',
      content: inputMessage,
      timestamp: new Date().toLocaleTimeString()
    };

    setMessages(prev => [...prev, userMessage]);
    setIsLoading(true);
    setError(null);

    try {
      let response;
      
      if (advancedSearchMode && isSearchQuery(inputMessage)) {
        response = await videoAnalysisService.searchVideoAdvanced(
          selectedVideo.id, 
          inputMessage, 
          searchFilters
        );
        
        if (response.search_results) {
          setSearchResults(response.search_results);
        }
      } else {
        response = await videoAnalysisService.sendVideoChatMessage(inputMessage, selectedVideo.id);
        
        if (response.search_results) {
          setSearchResults(response.search_results);
        }
      }

      const botMessage = {
        id: Date.now() + 1,
        type: 'bot',
        content: response.response || response.insights || '응답을 받지 못했습니다.',
        timestamp: new Date().toLocaleTimeString(),
        searchResults: response.search_results,
        analysisType: response.search_type || 'general',
        fallbackUsed: response.fallback_used,
        hasBboxAnnotations: response.has_bbox_annotations || false // ✅ 바운딩 박스 정보 추가
      };

      setMessages(prev => [...prev, botMessage]);
      
      if (response.search_results && response.search_results.length > 0) {
        setChatMode('search');
      } else if (response.insights) {
        setChatMode('insights');
      }

    } catch (error) {
      console.error('메시지 전송 실패:', error);
      
      const errorMessage = {
        id: Date.now() + 1,
        type: 'error',
        content: `오류가 발생했습니다: ${error.message}`,
        timestamp: new Date().toLocaleTimeString()
      };
      
      setMessages(prev => [...prev, errorMessage]);
      setError(error.message);
    } finally {
      setIsLoading(false);
      setInputMessage('');
    }
  };

  const isSearchQuery = (message) => {
    const searchKeywords = ['찾아', '검색', '어디', 'find', 'search', 'where', '보여줘', '있는'];
    return searchKeywords.some(keyword => message.toLowerCase().includes(keyword));
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  // ✅ 새로 추가: 프레임 보기 함수 (바운딩 박스 포함)
  const viewFrameWithBbox = async (result) => {
    try {
      console.log('🖼️ 바운딩 박스와 함께 프레임 보기:', result);
      
      // 프레임 이미지 URL 생성
      const frameUrl = videoAnalysisService.getFrameImageUrl(selectedVideo.id, result.frame_id);
      
      // 바운딩 박스 어노테이션 준비
      const bboxAnnotations = result.bbox_annotations || [];
      
      // 고급 프레임 데이터 로드 시도
      let frameData = null;
      try {
        const response = await fetch(`/api/frame/${selectedVideo.id}/${result.frame_id}/enhanced/`);
        if (response.ok) {
          frameData = await response.json();
        }
      } catch (error) {
        console.warn('고급 프레임 데이터 로드 실패, 기본 데이터 사용:', error);
        // 기본 데이터 사용
        frameData = {
          caption: result.caption,
          timestamp: result.timestamp,
          detected_objects: result.detected_objects || [],
          advanced_analysis: {
            clip_analysis: result.clip_analysis,
            ocr_text: result.ocr_text,
            vqa_results: result.vqa_results
          }
        };
      }
      
      // 프레임 뷰어 상태 설정
      setSelectedFrame({
        frameId: result.frame_id,
        frameUrl: frameUrl,
        frameData: frameData,
        bboxAnnotations: bboxAnnotations,
        timestamp: result.timestamp
      });
      
      setFrameViewerOpen(true);
      
      console.log('✅ 프레임 뷰어 열기 완료:', {
        frameId: result.frame_id,
        bboxCount: bboxAnnotations.length,
        hasAdvancedData: !!frameData?.advanced_analysis
      });
      
    } catch (error) {
      console.error('❌ 프레임 보기 실패:', error);
      alert('프레임을 불러오는 중 오류가 발생했습니다.');
    }
  };

  // ✅ 프레임 뷰어 닫기
  const closeFrameViewer = () => {
    setFrameViewerOpen(false);
    setSelectedFrame(null);
  };

  const renderAdvancedFeatures = (video) => {
    if (!video.advanced_features_used) return null;

    const features = video.advanced_features_used;
    const featureIcons = {
      clip: '🖼️',
      ocr: '📝',
      vqa: '❓',
      scene_graph: '🕸️'
    };

    return (
      <div className="video-advanced-features">
        <span className="features-label">고급 기능:</span>
        {Object.entries(features).map(([feature, enabled]) => {
          if (!enabled) return null;
          return (
            <span key={feature} className="feature-badge enabled">
              {featureIcons[feature]} {feature.toUpperCase()}
            </span>
          );
        })}
      </div>
    );
  };

  const renderSearchFilters = () => {
    if (!advancedSearchMode) return null;

    return (
      <div className="search-filters">
        <h4>🔧 고급 검색 옵션</h4>
        <div className="filter-options">
          <label className="filter-option">
            <input
              type="checkbox"
              checked={searchFilters.includeClipAnalysis}
              onChange={(e) => setSearchFilters(prev => ({
                ...prev,
                includeClipAnalysis: e.target.checked
              }))}
            />
            <span>🖼️ CLIP 씬 분석 포함</span>
          </label>
          
          <label className="filter-option">
            <input
              type="checkbox"
              checked={searchFilters.includeOcrText}
              onChange={(e) => setSearchFilters(prev => ({
                ...prev,
                includeOcrText: e.target.checked
              }))}
            />
            <span>📝 OCR 텍스트 포함</span>
          </label>
          
          <label className="filter-option">
            <input
              type="checkbox"
              checked={searchFilters.includeVqaResults}
              onChange={(e) => setSearchFilters(prev => ({
                ...prev,
                includeVqaResults: e.target.checked
              }))}
            />
            <span>❓ VQA 결과 포함</span>
          </label>
          
          <label className="filter-option">
            <input
              type="checkbox"
              checked={searchFilters.includeSceneGraph}
              onChange={(e) => setSearchFilters(prev => ({
                ...prev,
                includeSceneGraph: e.target.checked
              }))}
            />
            <span>🕸️ Scene Graph 포함</span>
          </label>
        </div>
        
        <div className="confidence-slider">
          <label>신뢰도 임계값: {searchFilters.confidenceThreshold}</label>
          <input
            type="range"
            min="0.1"
            max="1.0"
            step="0.1"
            value={searchFilters.confidenceThreshold}
            onChange={(e) => setSearchFilters(prev => ({
              ...prev,
              confidenceThreshold: parseFloat(e.target.value)
            }))}
          />
        </div>
      </div>
    );
  };

  // ✅ 수정된 검색 결과 렌더링 (바운딩 박스 정보 포함)
  const renderSearchResults = () => {
    if (!searchResults || searchResults.length === 0) return null;

    return (
      <div className="search-results-container">
        <h4>🔍 검색 결과 ({searchResults.length}개)</h4>
        {searchResults.some(r => r.bbox_annotations?.length > 0) && (
          <div className="bbox-info-banner">
            📦 바운딩 박스 표시 가능한 결과가 있습니다. 프레임을 클릭하여 상세히 확인하세요.
          </div>
        )}
        <div className="search-results-grid">
          {searchResults.slice(0, 6).map((result, index) => (
            <div key={index} className={`search-result-card ${result.bbox_annotations?.length > 0 ? 'has-bbox' : ''}`}>
              <div className="result-header">
                <span className="frame-number">프레임 #{result.frame_id}</span>
                <span className="result-score">점수: {result.match_score?.toFixed(2) || 'N/A'}</span>
                {result.bbox_annotations?.length > 0 && (
                  <span className="bbox-indicator">📦 {result.bbox_annotations.length}개 객체</span>
                )}
              </div>
              
              <div className="result-info">
                <div className="timestamp">
                  ⏱️ {Math.floor(result.timestamp / 60)}:{String(Math.floor(result.timestamp % 60)).padStart(2, '0')}
                </div>
                
                {result.detected_objects && result.detected_objects.length > 0 && (
                  <div className="detected-objects">
                    <strong>객체:</strong> {result.detected_objects.slice(0, 3).join(', ')}
                  </div>
                )}
                
                {result.caption && (
                  <div className="result-caption">
                    <strong>설명:</strong> {result.caption.substring(0, 100)}...
                  </div>
                )}
                
                {/* 바운딩 박스 객체 미리보기 */}
                {result.bbox_annotations?.length > 0 && (
                  <div className="bbox-preview">
                    <strong>감지된 객체:</strong>
                    <div className="bbox-objects">
                      {result.bbox_annotations.map((annotation, idx) => (
                        <span key={idx} className="bbox-object-tag">
                          {annotation.match} ({(annotation.confidence * 100).toFixed(0)}%)
                        </span>
                      ))}
                    </div>
                  </div>
                )}
                
                {/* 고급 분석 결과 표시 */}
                {result.clip_analysis && (
                  <div className="advanced-result">
                    <span className="advanced-tag">🖼️ CLIP</span>
                    <span>{result.clip_analysis.scene_type || 'Scene analyzed'}</span>
                  </div>
                )}
                
                {result.ocr_text && result.ocr_text.full_text && (
                  <div className="advanced-result">
                    <span className="advanced-tag">📝 OCR</span>
                    <span>{result.ocr_text.full_text.substring(0, 50)}...</span>
                  </div>
                )}
                
                {result.vqa_insights && (
                  <div className="advanced-result">
                    <span className="advanced-tag">❓ VQA</span>
                    <span>질문답변 분석 완료</span>
                  </div>
                )}
                
                {result.match_reasons && result.match_reasons.length > 0 && (
                  <div className="match-reasons">
                    <strong>매칭 이유:</strong> {result.match_reasons.join(', ')}
                  </div>
                )}
              </div>
              
              {/* ✅ 수정된 버튼 - 바운딩 박스 표시 기능 */}
              <button 
                onClick={() => viewFrameWithBbox(result)}
                className={`view-frame-button ${result.bbox_annotations?.length > 0 ? 'with-bbox' : ''}`}
                title={result.bbox_annotations?.length > 0 ? 
                  `바운딩 박스와 함께 프레임 보기 (${result.bbox_annotations.length}개 객체)` : 
                  '프레임 보기'}
              >
                {result.bbox_annotations?.length > 0 ? '📦 바운딩 박스와 함께 보기' : '🖼️ 프레임 보기'}
              </button>
            </div>
          ))}
        </div>
      </div>
    );
  };

  const renderMessage = (message) => {
    return (
      <div key={message.id} className={`message ${message.type}`}>
        <div className="message-content">
          <div className="message-text">
            {message.content}
            {message.fallbackUsed && (
              <div className="fallback-notice">
                ⚠️ 고급 분석 기능이 일시적으로 사용 불가하여 기본 분석 결과를 사용했습니다.
              </div>
            )}
            {message.hasBboxAnnotations && (
              <div className="bbox-notice">
                📦 바운딩 박스 표시 가능한 결과가 포함되어 있습니다.
              </div>
            )}
          </div>
          <div className="message-timestamp">{message.timestamp}</div>
          {message.analysisType && (
            <div className="analysis-type-badge">
              {message.analysisType === 'enhanced_search' ? '🔍 고급 검색' : 
               message.analysisType === 'analysis_insights' ? '💡 분석 인사이트' : '💬 일반 채팅'}
            </div>
          )}
        </div>
      </div>
    );
  };

  const renderQuickActions = () => {
    if (!selectedVideo) return null;

    const quickActions = [
      {
        text: "사람이 있는 장면 찾아줘",
        icon: "👥",
        description: "사람 감지 + 바운딩 박스 표시"
      },
      {
        text: "차가 있는 부분 보여줘",
        icon: "🚗",
        description: "차량 감지 + 바운딩 박스 표시"
      },
      {
        text: "텍스트가 있는 장면은?",
        icon: "📝",
        description: "OCR 텍스트 추출 결과"
      },
      {
        text: "주요 객체들을 알려줘",
        icon: "🎯",
        description: "감지된 주요 객체 분석"
      },
      {
        text: "가장 중요한 장면은?",
        icon: "⭐",
        description: "하이라이트 장면 분석"
      },
      {
        text: "동물이 있는 장면 찾아줘",
        icon: "🐕",
        description: "동물 감지 + 바운딩 박스"
      }
    ];

    return (
      <div className="quick-actions">
        <h4>⚡ 빠른 질문</h4>
        <div className="quick-actions-grid">
          {quickActions.map((action, index) => (
            <button
              key={index}
              className="quick-action-button"
              onClick={() => setInputMessage(action.text)}
              title={action.description}
            >
              <span className="action-icon">{action.icon}</span>
              <span className="action-text">{action.text}</span>
            </button>
          ))}
        </div>
      </div>
    );
  };

  const renderVideoInsights = () => {
    if (!selectedVideo || !analysisInsights[selectedVideo.id]) return null;

    return (
      <div className="video-insights-panel">
        <h4>💡 AI 분석 인사이트</h4>
        <div className="insights-content">
          <div className="insight-text">
            {analysisInsights[selectedVideo.id]}
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="integrated-chat-page">
      <div className="chat-container">
        {/* 사이드바 */}
        <div className="chat-sidebar">
          <div className="sidebar-header">
            <h3>🧠 고급 AI 채팅</h3>
            <div className="chat-mode-selector">
              <button 
                className={`mode-button ${!advancedSearchMode ? 'active' : ''}`}
                onClick={() => setAdvancedSearchMode(false)}
              >
                💬 일반 채팅
              </button>
              <button 
                className={`mode-button ${advancedSearchMode ? 'active' : ''}`}
                onClick={() => setAdvancedSearchMode(true)}
              >
                🔍 고급 검색
              </button>
            </div>
          </div>

          {/* 비디오 선택 */}
          <div className="video-selection">
            <h4>📹 분석된 비디오</h4>
            {videos.length === 0 ? (
              <div className="no-videos">
                <p>분석된 비디오가 없습니다.</p>
                <button onClick={() => navigate('/video-upload')}>
                  비디오 업로드하기
                </button>
              </div>
            ) : (
              <div className="video-list">
                {videos.map(video => (
                  <div 
                    key={video.id}
                    className={`video-item ${selectedVideo?.id === video.id ? 'selected' : ''} ${video.advanced_features_used ? 'advanced' : ''}`}
                    onClick={() => handleVideoSelect(video)}
                  >
                    <div className="video-info">
                      <div className="video-name">{video.original_name}</div>
                      <div className="video-stats">
                        <span>🎯 {video.unique_objects || 0}개 객체</span>
                        {video.analysis_type && (
                          <span className="analysis-type">
                            {video.analysis_type === 'comprehensive' ? '🧠' : 
                             video.analysis_type === 'enhanced' ? '⚡' : '🔍'} 
                            {video.analysis_type}
                          </span>
                        )}
                      </div>
                      {renderAdvancedFeatures(video)}
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* 고급 검색 필터 */}
          {renderSearchFilters()}

          {/* 빠른 액션 */}
          {renderQuickActions()}

          {/* 비디오 인사이트 */}
          {renderVideoInsights()}
        </div>

        {/* 메인 채팅 영역 */}
        <div className="chat-main">
          {/* 채팅 헤더 */}
          <div className="chat-header">
            <div className="current-video-info">
              {selectedVideo ? (
                <>
                  <span className="video-name">📹 {selectedVideo.original_name}</span>
                  {selectedVideo.advanced_features_used && (
                    <span className="advanced-badge">🚀 고급 분석 완료</span>
                  )}
                </>
              ) : (
                <span>비디오를 선택해주세요</span>
              )}
            </div>
            <div className="chat-actions">
              <button onClick={() => setMessages([])}>
                🗑️ 채팅 지우기
              </button>
              <button onClick={() => navigate('/video-analysis')}>
                📊 분석 현황
              </button>
            </div>
          </div>

          {/* 오류 표시 */}
          {error && (
            <div className="error-banner">
              <span>⚠️ {error}</span>
              <button onClick={() => setError(null)}>✕</button>
            </div>
          )}

          {/* 메시지 영역 */}
          <div className="messages-container">
            {messages.map(renderMessage)}
            {isLoading && (
              <div className="message bot loading">
                <div className="loading-indicator">
                  <div className="loading-spinner"></div>
                  <span>AI가 분석 중입니다...</span>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* 검색 결과 */}
          {renderSearchResults()}

          {/* 입력 영역 */}
          <div className="input-container">
            <div className="input-wrapper">
              <textarea
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder={
                  !selectedVideo ? "먼저 비디오를 선택해주세요..." :
                  advancedSearchMode ? "고급 검색어를 입력하세요 (예: 사람이 있는 장면 찾아줘)" :
                  "질문을 입력하세요 (예: 사람이 있는 장면 찾아줘)"
                }
                disabled={!selectedVideo}
                rows={2}
              />
              <button 
                onClick={sendMessage}
                disabled={!inputMessage.trim() || !selectedVideo || isLoading}
                className="send-button"
              >
                {isLoading ? '🔄' : '🚀'}
              </button>
            </div>
            
            <div className="input-hints">
              {advancedSearchMode ? (
                <span>💡 고급 검색 모드: CLIP, OCR, VQA 결과를 활용한 정밀 검색 + 바운딩 박스 표시</span>
              ) : (
                <span>💡 "사람이 있는 장면 찾아줘"처럼 입력하면 바운딩 박스와 함께 결과를 보여드립니다</span>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* ✅ 프레임 뷰어 모달 (바운딩 박스 포함) */}
      {frameViewerOpen && selectedFrame && (
        <FrameViewer
          frameUrl={selectedFrame.frameUrl}
          frameData={selectedFrame.frameData}
          bboxAnnotations={selectedFrame.bboxAnnotations}
          frameId={selectedFrame.frameId}
          timestamp={selectedFrame.timestamp}
          onClose={closeFrameViewer}
        />
      )}
    </div>
  );
};

export default IntegratedChatPage;