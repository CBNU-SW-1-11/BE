// EnhancedIntegratedChatPage.jsx - 기존 IntegratedChatPage 확장

import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { videoAnalysisService, SearchTypes, SearchFilters } from '../services/videoAnalysisService';
import FrameViewer from '../components/FrameViewer';
import StatisticsChart from '../components/StatisticsChart'; // 새로운 컴포넌트
import TrackingTimeline from '../components/TrackingTimeline'; // 새로운 컴포넌트
import SearchTypeIndicator from '../components/SearchTypeIndicator'; // 새로운 컴포넌트
import './EnhancedIntegratedChatPage.css';

const EnhancedIntegratedChatPage = () => {
  const navigate = useNavigate();
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [videos, setVideos] = useState([]);
  const [selectedVideo, setSelectedVideo] = useState(null);
  const [error, setError] = useState(null);
  const messagesEndRef = useRef(null);
  
  // 고급 검색 관련 상태
  const [searchMode, setSearchMode] = useState('intelligent'); // 'intelligent', 'cross_video', 'in_video', 'temporal'
  const [searchResults, setSearchResults] = useState([]);
  const [systemCapabilities, setSystemCapabilities] = useState({});
  const [searchHistory, setSearchHistory] = useState([]);
  const [popularQueries, setPopularQueries] = useState([]);
  
  // 검색 결과 전용 상태
  const [crossVideoResults, setCrossVideoResults] = useState([]);
  const [trackingResults, setTrackingResults] = useState([]);
  const [statisticsResults, setStatisticsResults] = useState(null);
  
  // UI 상태
  const [selectedFrame, setSelectedFrame] = useState(null);
  const [frameViewerOpen, setFrameViewerOpen] = useState(false);
  const [activeResultTab, setActiveResultTab] = useState('all'); // 'all', 'cross_video', 'tracking', 'statistics'
  const [searchFilters, setSearchFilters] = useState(SearchFilters.DEFAULT);

  useEffect(() => {
    loadInitialData();
    addWelcomeMessage();
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const loadInitialData = async () => {
    try {
      // 기본 데이터 로드
      await Promise.all([
        loadVideos(),
        loadSystemCapabilities(),
        loadSearchHistory(),
        loadPopularQueries()
      ]);
    } catch (error) {
      console.error('초기 데이터 로드 실패:', error);
      setError('시스템 초기화 중 오류가 발생했습니다.');
    }
  };

  const loadVideos = async () => {
    try {
      const response = await videoAnalysisService.getVideoList();
      const analyzedVideos = response.videos.filter(v => v.is_analyzed);
      setVideos(analyzedVideos);
      
      if (analyzedVideos.length > 0 && !selectedVideo) {
        const bestVideo = analyzedVideos.find(v => 
          v.advanced_features_used && Object.values(v.advanced_features_used).some(Boolean)
        ) || analyzedVideos[0];
        setSelectedVideo(bestVideo);
      }
    } catch (error) {
      console.error('비디오 목록 로드 실패:', error);
      setError('비디오 목록을 불러올 수 없습니다.');
    }
  };

  const loadSystemCapabilities = async () => {
    try {
      const capabilities = await videoAnalysisService.getSystemCapabilities();
      setSystemCapabilities(capabilities);
    } catch (error) {
      console.error('시스템 능력 조회 실패:', error);
    }
  };

  const loadSearchHistory = async () => {
    try {
      const history = await videoAnalysisService.getSearchHistory(10);
      setSearchHistory(history.queries || []);
    } catch (error) {
      console.error('검색 히스토리 로드 실패:', error);
    }
  };

  const loadPopularQueries = async () => {
    try {
      const popular = await videoAnalysisService.getPopularQueries();
      setPopularQueries(popular.queries || []);
    } catch (error) {
      console.error('인기 검색어 로드 실패:', error);
    }
  };

  const addWelcomeMessage = () => {
    const welcomeMessage = {
      id: Date.now(),
      type: 'bot',
      content: `🧠 **고급 AI 비디오 분석 시스템에 오신 것을 환영합니다!**

저는 다음과 같은 **3가지 핵심 기능**으로 여러분의 비디오 분석을 도와드립니다:

## 🎯 **주요 기능들**

### 1️⃣ **영상 간 검색** 🔍
- *"비가오는 밤에 촬영된 영상을 찾아줘"*
- *"사람이 많이 나오는 낮 영상 보여줘"*
- *"실외에서 촬영된 영상들 찾아줘"*

### 2️⃣ **영상 내 추적** 🎯
- *"주황색 상의를 입은 남성이 지나간 장면을 추적해줘"*
- *"빨간 옷을 입은 사람 찾아줘"*
- *"자전거 탄 사람이 나타나는 시점 알려줘"*

### 3️⃣ **시간별 통계 분석** 📊
- *"3:00~5:00 분 사이에 지나간 사람들의 성비 분포는?"*
- *"처음 2분 동안 나타난 객체들 분석해줘"*
- *"마지막 1분간 활동량은 어떻게 돼?"*

## ⚡ **새로운 기능들**
- **바운딩 박스 시각화**: 감지된 객체를 정확히 표시
- **실시간 추적**: 같은 사람/객체의 움직임 추적  
- **통계 차트**: 성별, 색상, 시간별 분포 시각화
- **지능형 검색**: 자동으로 최적의 검색 방법 선택

어떤 분석이 필요하신가요? 자연스럽게 말씀해주시면 됩니다! 🚀`,
      timestamp: new Date().toLocaleTimeString(),
      searchType: 'welcome'
    };
    
    setMessages([welcomeMessage]);
  };

  const handleVideoSelect = (video) => {
    setSelectedVideo(video);
    clearResults();
    
    const videoChangeMessage = {
      id: Date.now(),
      type: 'system',
      content: `📹 "${video.original_name}" 비디오로 변경되었습니다.${
        video.advanced_features_used ? 
        '\n🚀 고급 분석 완료 - 모든 기능을 사용할 수 있습니다.' : 
        '\n⚡ 기본 분석 완료 - 일부 기능이 제한될 수 있습니다.'
      }`,
      timestamp: new Date().toLocaleTimeString()
    };
    
    setMessages(prev => [...prev, videoChangeMessage]);
  };

  const sendMessage = async () => {
    if (!inputMessage.trim()) return;

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
      const detectedSearchType = detectSearchType(inputMessage);
      
      console.log('🔍 감지된 검색 타입:', detectedSearchType);

      // 검색 타입에 따른 처리
      switch (detectedSearchType) {
        case SearchTypes.CROSS_VIDEO:
          response = await handleCrossVideoSearch(inputMessage);
          break;
          
        case SearchTypes.IN_VIDEO_TRACKING:
          if (!selectedVideo) {
            throw new Error('영상 내 추적을 위해서는 비디오를 먼저 선택해주세요.');
          }
          response = await handleInVideoTracking(inputMessage);
          break;
          
        case SearchTypes.TEMPORAL_ANALYSIS:
          if (!selectedVideo) {
            throw new Error('시간별 분석을 위해서는 비디오를 먼저 선택해주세요.');
          }
          response = await handleTemporalAnalysis(inputMessage);
          break;
          
        default:
          // 지능형 검색 또는 기본 채팅
          if (searchMode === 'intelligent') {
            response = await videoAnalysisService.intelligentSearch(
              inputMessage, 
              selectedVideo?.id,
              { maxResults: 20 }
            );
          } else {
            response = await videoAnalysisService.sendVideoChatMessage(
              inputMessage, 
              selectedVideo?.id
            );
          }
      }

      // 응답 메시지 생성
      const botMessage = createBotMessage(response, detectedSearchType);
      setMessages(prev => [...prev, botMessage]);
      
      // 검색 결과 상태 업데이트
      updateSearchResults(response, detectedSearchType);

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

  const detectSearchType = (message) => {
    const lowerMessage = message.toLowerCase();
    
    // 영상 간 검색 키워드
    const crossVideoKeywords = [
      '비가오는', '밤에 촬영', '영상을 찾', '영상 찾', '비디오 찾',
      '실외', '실내', '날씨', '시간대', '낮에', '저녁에',
      '사람이 많이', '차가 많이', '동물이 있는'
    ];
    
    // 영상 내 추적 키워드
    const trackingKeywords = [
      '추적해줘', '찾아줘', '따라가', '지나간', '상의를 입은',
      '하의를 입은', '색깔 옷', '남성이', '여성이', '사람이 나타',
      '객체를 추', '이동하는', '움직이는'
    ];
    
    // 시간별 분석 키워드
    const temporalKeywords = [
      '성비', '분포는', '시간 사이', '분 사이', '~', '-',
      '처음', '마지막', '통계', '분석해줘', '어떻게 돼',
      '얼마나', '몇 명', '몇 개'
    ];
    
    // 우선순위에 따른 검색 타입 결정
    if (temporalKeywords.some(keyword => lowerMessage.includes(keyword))) {
      // 시간 패턴이 있는지 확인
      const timePattern = /\d+:\d+.*\d+:\d+|\d+분.*\d+분|처음.*분|마지막.*분/;
      if (timePattern.test(lowerMessage)) {
        return SearchTypes.TEMPORAL_ANALYSIS;
      }
    }
    
    if (trackingKeywords.some(keyword => lowerMessage.includes(keyword))) {
      return SearchTypes.IN_VIDEO_TRACKING;
    }
    
    if (crossVideoKeywords.some(keyword => lowerMessage.includes(keyword))) {
      return SearchTypes.CROSS_VIDEO;
    }
    
    return SearchTypes.INTELLIGENT;
  };

  const handleCrossVideoSearch = async (query) => {
    console.log('🔍 영상 간 검색 실행:', query);
    
    const result = await videoAnalysisService.searchAcrossVideos(query, {
      matchThreshold: 0.3,
      maxResults: 20,
      includeMetadata: true
    });
    
    setCrossVideoResults(result.results || []);
    setActiveResultTab('cross_video');
    
    return {
      searchType: SearchTypes.CROSS_VIDEO,
      content: result.aiInsights || `${result.results?.length || 0}개의 매칭 영상을 찾았습니다.`,
      results: result.results,
      totalSearched: result.totalVideosSearched,
      searchDuration: result.searchDuration
    };
  };

  const handleInVideoTracking = async (query) => {
    console.log('🎯 영상 내 추적 실행:', query);
    
    const result = await videoAnalysisService.trackPersonInVideo(
      selectedVideo.id, 
      query,
      { confidenceThreshold: 0.6, maxTracks: 10 }
    );
    
    setTrackingResults(result.tracks || []);
    setActiveResultTab('tracking');
    
    return {
      searchType: SearchTypes.IN_VIDEO_TRACKING,
      content: result.aiInterpretation || `${result.tracks?.length || 0}개의 추적 결과를 찾았습니다.`,
      tracks: result.tracks,
      summary: result.summary,
      trackingConditions: result.trackingConditions
    };
  };

  const handleTemporalAnalysis = async (query) => {
    console.log('📊 시간별 분석 실행:', query);
    
    const result = await videoAnalysisService.analyzeTemporalStatistics(
      selectedVideo.id,
      query,
      { analysisGranularity: 'medium', includeDetailedStats: true }
    );
    
    setStatisticsResults(result);
    setActiveResultTab('statistics');
    
    return {
      searchType: SearchTypes.TEMPORAL_ANALYSIS,
      content: result.aiInterpretation || '시간별 통계 분석이 완료되었습니다.',
      statistics: result.statistics,
      chartData: result.chartData,
      timeRange: result.timeRange,
      totalPersons: result.totalPersons
    };
  };

  const createBotMessage = (response, searchType) => {
    return {
      id: Date.now() + 1,
      type: 'bot',
      content: response.content || response.response || '분석이 완료되었습니다.',
      timestamp: new Date().toLocaleTimeString(),
      searchType: searchType,
      searchResults: response.results || response.tracks || response.statistics,
      analysisType: response.searchType || searchType,
      metadata: {
        totalResults: response.results?.length || response.tracks?.length || 0,
        searchDuration: response.searchDuration || 0,
        cached: response.cached || false
      }
    };
  };

  const updateSearchResults = (response, searchType) => {
    switch (searchType) {
      case SearchTypes.CROSS_VIDEO:
        setCrossVideoResults(response.results || []);
        break;
      case SearchTypes.IN_VIDEO_TRACKING:
        setTrackingResults(response.tracks || []);
        break;
      case SearchTypes.TEMPORAL_ANALYSIS:
        setStatisticsResults(response);
        break;
    }
  };

  const clearResults = () => {
    setSearchResults([]);
    setCrossVideoResults([]);
    setTrackingResults([]);
    setStatisticsResults(null);
    setActiveResultTab('all');
  };

  const viewFrameWithBbox = async (result, videoId = null) => {
    try {
      const targetVideoId = videoId || selectedVideo?.id;
      if (!targetVideoId) return;

      console.log('🖼️ 프레임 보기:', { result, targetVideoId });
      
      const frameUrl = videoAnalysisService.getFrameImageUrl(
        targetVideoId, 
        result.frame_id || result.frameId,
        { withBbox: true }
      );
      
      const frameData = {
        frameId: result.frame_id || result.frameId,
        timestamp: result.timestamp,
        caption: result.caption || result.enhanced_caption || '',
        detected_objects: result.detected_objects || [],
        bbox_annotations: result.bbox_annotations || [],
        advanced_analysis: {
          clip_analysis: result.clip_analysis || {},
          ocr_text: result.ocr_text || {},
          vqa_results: result.vqa_results || {}
        }
      };
      
      setSelectedFrame({
        frameId: result.frame_id || result.frameId,
        frameUrl: frameUrl,
        frameData: frameData,
        bboxAnnotations: result.bbox_annotations || [],
        timestamp: result.timestamp,
        videoId: targetVideoId
      });
      
      setFrameViewerOpen(true);
      
    } catch (error) {
      console.error('❌ 프레임 보기 실패:', error);
      alert('프레임을 불러오는 중 오류가 발생했습니다.');
    }
  };

  const closeFrameViewer = () => {
    setFrameViewerOpen(false);
    setSelectedFrame(null);
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  // 렌더링 함수들
  const renderMessage = (message) => {
    return (
      <div key={message.id} className={`message ${message.type}`}>
        <div className="message-content">
          {message.searchType && (
            <SearchTypeIndicator type={message.searchType} />
          )}
          <div className="message-text">
            {message.content}
          </div>
          <div className="message-timestamp">{message.timestamp}</div>
          {message.metadata && (
            <div className="message-metadata">
              {message.metadata.totalResults > 0 && (
                <span className="result-count">
                  📊 {message.metadata.totalResults}개 결과
                </span>
              )}
              {message.metadata.searchDuration > 0 && (
                <span className="duration">
                  ⏱️ {message.metadata.searchDuration}ms
                </span>
              )}
              {message.metadata.cached && (
                <span className="cached">💾 캐시</span>
              )}
            </div>
          )}
        </div>
      </div>
    );
  };

  const renderSearchModeSelector = () => (
    <div className="search-mode-selector">
      <h4>🔍 검색 모드</h4>
      <div className="mode-buttons">
        {[
          { id: 'intelligent', label: '🧠 지능형', desc: '자동 판단' },
          { id: 'cross_video', label: '🎬 영상간', desc: '영상 찾기' },
          { id: 'in_video', label: '🎯 추적', desc: '객체 추적' },
          { id: 'temporal', label: '📊 통계', desc: '시간 분석' }
        ].map(mode => (
          <button
            key={mode.id}
            className={`mode-button ${searchMode === mode.id ? 'active' : ''}`}
            onClick={() => setSearchMode(mode.id)}
            title={mode.desc}
          >
            {mode.label}
          </button>
        ))}
      </div>
    </div>
  );

  const renderQuickActions = () => (
    <div className="quick-actions">
      <h4>⚡ 빠른 질문</h4>
      <div className="action-categories">
        <div className="action-category">
          <h5>🎬 영상 검색</h5>
          {[
            "비가오는 밤에 촬영된 영상을 찾아줘",
            "사람이 많이 나오는 낮 영상 보여줘",
            "실외에서 촬영된 영상들을 찾아줘"
          ].map((action, idx) => (
            <button
              key={idx}
              className="quick-action-button small"
              onClick={() => setInputMessage(action)}
            >
              {action}
            </button>
          ))}
        </div>
        
        <div className="action-category">
          <h5>🎯 객체 추적</h5>
          {[
            "주황색 상의를 입은 남성이 지나간 장면을 추적해줘",
            "빨간 옷을 입은 사람 찾아줘",
            "자전거 탄 사람이 나타나는 시점 알려줘"
          ].map((action, idx) => (
            <button
              key={idx}
              className="quick-action-button small"
              onClick={() => setInputMessage(action)}
            >
              {action}
            </button>
          ))}
        </div>
        
        <div className="action-category">
          <h5>📊 시간 분석</h5>
          {[
            "3:00~5:00 분 사이에 지나간 사람들의 성비 분포는?",
            "처음 2분 동안 나타난 객체들 분석해줘",
            "마지막 1분간 활동량은 어떻게 돼?"
          ].map((action, idx) => (
            <button
              key={idx}
              className="quick-action-button small"
              onClick={() => setInputMessage(action)}
            >
              {action}
            </button>
          ))}
        </div>
      </div>
    </div>
  );

  const renderResultTabs = () => {
    const tabs = [
      { id: 'all', label: '전체', count: searchResults.length },
      { id: 'cross_video', label: '영상검색', count: crossVideoResults.length },
      { id: 'tracking', label: '추적결과', count: trackingResults.length },
      { id: 'statistics', label: '통계분석', count: statisticsResults ? 1 : 0 }
    ];

    return (
      <div className="result-tabs">
        {tabs.map(tab => (
          <button
            key={tab.id}
            className={`result-tab ${activeResultTab === tab.id ? 'active' : ''}`}
            onClick={() => setActiveResultTab(tab.id)}
            disabled={tab.count === 0}
          >
            {tab.label} {tab.count > 0 && `(${tab.count})`}
          </button>
        ))}
      </div>
    );
  };

  const renderCrossVideoResults = () => {
    if (crossVideoResults.length === 0) return null;

    return (
      <div className="cross-video-results">
        <h4>🎬 영상 검색 결과 ({crossVideoResults.length}개)</h4>
        <div className="video-results-grid">
          {crossVideoResults.map((video, index) => (
            <div key={index} className="video-result-card">
              <div className="video-info">
                <h5>{video.video_name}</h5>
                <div className="match-info">
                  <span className="match-score">
                    매칭도: {(video.match_score * 100).toFixed(1)}%
                  </span>
                  <div className="match-reasons">
                    {video.match_reasons?.map((reason, idx) => (
                      <span key={idx} className="reason-tag">{reason}</span>
                    ))}
                  </div>
                </div>
              </div>
              <div className="video-actions">
                <button 
                  onClick={() => {
                    // 해당 비디오로 전환
                    const targetVideo = videos.find(v => v.id === video.video_id);
                    if (targetVideo) {
                      handleVideoSelect(targetVideo);
                    }
                  }}
                  className="select-video-button"
                >
                  📹 이 영상 선택
                </button>
                {video.relevant_frames?.length > 0 && (
                  <button
                    onClick={() => viewFrameWithBbox(
                      video.relevant_frames[0], 
                      video.video_id
                    )}
                    className="view-frame-button"
                  >
                    🖼️ 대표 프레임 보기
                  </button>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  };

  const renderTrackingResults = () => {
    if (trackingResults.length === 0) return null;

    return (
      <div className="tracking-results">
        <h4>🎯 추적 결과 ({trackingResults.length}개)</h4>
        <div className="tracking-grid">
          {trackingResults.map((track, index) => (
            <div key={index} className="tracking-result-card">
              <div className="track-header">
                <h5>추적 #{track.track_id}</h5>
                <span className="match-score">
                  매칭도: {(track.avg_match_score * 100).toFixed(1)}%
                </span>
              </div>
              
              <div className="track-info">
                <div className="time-info">
                  ⏱️ {Math.floor(track.first_appearance / 60)}:{String(Math.floor(track.first_appearance % 60)).padStart(2, '0')} 
                  ~ {Math.floor(track.last_appearance / 60)}:{String(Math.floor(track.last_appearance % 60)).padStart(2, '0')}
                </div>
                <div className="attributes">
                  {track.attributes.gender !== 'unknown' && (
                    <span className="attr-tag">
                      👤 {track.attributes.gender}
                    </span>
                  )}
                  {track.attributes.upper_body_color !== 'unknown' && (
                    <span className="attr-tag">
                      👕 {track.attributes.upper_body_color}
                    </span>
                  )}
                </div>
                <div className="match-reasons">
                  {track.match_reasons?.map((reason, idx) => (
                    <span key={idx} className="reason-tag">{reason}</span>
                  ))}
                </div>
              </div>

              {/* 타임라인 */}
              <TrackingTimeline 
                track={track}
                videoDuration={selectedVideo?.duration || 0}
                onFrameClick={(frame) => viewFrameWithBbox(frame)}
              />

              {/* 키 프레임들 */}
              <div className="key-frames">
                <h6>주요 프레임들</h6>
                <div className="frames-grid">
                  {track.key_frames?.slice(0, 5).map((frame, frameIdx) => (
                    <div key={frameIdx} className="key-frame">
                      <div className="frame-info">
                        <span>#{frame.frame_id}</span>
                        <span>{Math.floor(frame.timestamp / 60)}:{String(Math.floor(frame.timestamp % 60)).padStart(2, '0')}</span>
                      </div>
                      <button
                        onClick={() => viewFrameWithBbox(frame)}
                        className="view-frame-button small"
                      >
                        📦 바운딩박스로 보기
                      </button>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    );
  };

  const renderStatisticsResults = () => {
    if (!statisticsResults) return null;

    return (
      <div className="statistics-results">
        <h4>📊 시간별 통계 분석</h4>
        
        <div className="stats-summary">
          <div className="summary-cards">
            <div className="summary-card">
              <h5>분석 기간</h5>
              <p>{statisticsResults.timeRange}</p>
            </div>
            <div className="summary-card">
              <h5>총 감지 인원</h5>
              <p>{statisticsResults.totalPersons}명</p>
            </div>
            <div className="summary-card">
              <h5>분석 품질</h5>
              <p>{statisticsResults.cached ? '캐시됨' : '실시간'}</p>
            </div>
          </div>
        </div>

        {/* 차트들 */}
        {statisticsResults.chartData && (
          <div className="charts-container">
            {statisticsResults.chartData.genderDistribution && (
              <StatisticsChart
                type="pie"
                title="성별 분포"
                data={statisticsResults.chartData.genderDistribution}
              />
            )}
            
            {statisticsResults.chartData.timeDensity && (
              <StatisticsChart
                type="line"
                title="시간별 활동 밀도"
                data={statisticsResults.chartData.timeDensity}
              />
            )}
            
            {statisticsResults.chartData.colorDistribution && (
              <StatisticsChart
                type="bar"
                title="의상 색상 분포"
                data={statisticsResults.chartData.colorDistribution}
              />
            )}
          </div>
        )}
      </div>
    );
  };

  const renderActiveResults = () => {
    switch (activeResultTab) {
      case 'cross_video':
        return renderCrossVideoResults();
      case 'tracking':
        return renderTrackingResults();
      case 'statistics':
        return renderStatisticsResults();
      default:
        return (
          <div className="all-results">
            {renderCrossVideoResults()}
            {renderTrackingResults()}
            {renderStatisticsResults()}
          </div>
        );
    }
  };

  const renderVideoSelector = () => (
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
                  <span>📊 {video.analysis_type || 'basic'}</span>
                </div>
                {video.advanced_features_used && (
                  <div className="advanced-features">
                    {Object.entries(video.advanced_features_used).map(([feature, enabled]) => 
                      enabled && (
                        <span key={feature} className="feature-badge">
                          {feature.toUpperCase()}
                        </span>
                      )
                    )}
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );

  return (
    <div className="enhanced-integrated-chat-page">
      <div className="chat-container">
        {/* 사이드바 */}
        <div className="chat-sidebar">
          <div className="sidebar-header">
            <h3>🧠 고급 AI 분석</h3>
            <div className="system-status">
              {systemCapabilities.system_status?.analyzer_available ? 
                <span className="status-good">🟢 시스템 정상</span> :
                <span className="status-error">🔴 시스템 오류</span>
              }
            </div>
          </div>

          {renderSearchModeSelector()}
          {renderVideoSelector()}
          {renderQuickActions()}

          {/* 검색 히스토리 */}
          {searchHistory.length > 0 && (
            <div className="search-history">
              <h4>📋 최근 검색</h4>
              <div className="history-list">
                {searchHistory.slice(0, 5).map((query, idx) => (
                  <button
                    key={idx}
                    className="history-item"
                    onClick={() => setInputMessage(query.query_text)}
                  >
                    {query.query_text.substring(0, 30)}...
                  </button>
                ))}
              </div>
            </div>
          )}
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
                    <span className="advanced-badge">🚀 고급 분석</span>
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
              <button onClick={clearResults}>
                🔄 결과 지우기
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

          {/* 검색 결과 영역 */}
          {(crossVideoResults.length > 0 || trackingResults.length > 0 || statisticsResults) && (
            <div className="results-container">
              {renderResultTabs()}
              <div className="results-content">
                {renderActiveResults()}
              </div>
            </div>
          )}

          {/* 입력 영역 */}
          <div className="input-container">
            <div className="input-wrapper">
              <textarea
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder={
                  !selectedVideo ? "먼저 비디오를 선택해주세요..." :
                  searchMode === 'cross_video' ? "영상 검색: '비가오는 밤에 촬영된 영상을 찾아줘'" :
                  searchMode === 'in_video' ? "객체 추적: '주황색 상의를 입은 남성이 지나간 장면을 추적해줘'" :
                  searchMode === 'temporal' ? "시간 분석: '3:00~5:00 분 사이에 지나간 사람들의 성비 분포는?'" :
                  "자연스럽게 질문해주세요 (예: 사람이 있는 장면 찾아줘)"
                }
                disabled={!selectedVideo && searchMode !== 'cross_video'}
                rows={2}
              />
              <button 
                onClick={sendMessage}
                disabled={!inputMessage.trim() || (!selectedVideo && searchMode !== 'cross_video') || isLoading}
                className="send-button"
              >
                {isLoading ? '🔄' : '🚀'}
              </button>
            </div>
            
            <div className="input-hints">
              <span>
                💡 {searchMode === 'intelligent' ? '지능형 모드: AI가 자동으로 최적의 분석 방법을 선택합니다' :
                     searchMode === 'cross_video' ? '영상 검색 모드: 조건에 맞는 영상들을 찾습니다' :
                     searchMode === 'in_video' ? '추적 모드: 영상 내에서 특정 객체를 추적합니다' :
                     '통계 모드: 시간별 통계 분석을 수행합니다'}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* 프레임 뷰어 모달 */}
      {frameViewerOpen && selectedFrame && (
        <FrameViewer
          frameUrl={selectedFrame.frameUrl}
          frameData={selectedFrame.frameData}
          bboxAnnotations={selectedFrame.bboxAnnotations}
          frameId={selectedFrame.frameId}
          timestamp={selectedFrame.timestamp}
          videoId={selectedFrame.videoId}
          onClose={closeFrameViewer}
        />
      )}
    </div>
  );
};

export default EnhancedIntegratedChatPage;

// EnhancedIntegratedChatPage.css (주요 스타일만)
const enhancedChatPageCSS = `
.enhanced-integrated-chat-page {
  display: flex;
  height: 100vh;
  background: #f8fafc;
  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
}

.chat-container {
  display: flex;
  width: 100%;
  max-width: 1400px;
  margin: 0 auto;
  box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
  border-radius: 12px;
  overflow: hidden;
}

.chat-sidebar {
  width: 320px;
  background: white;
  border-right: 1px solid #e5e7eb;
  display: flex;
  flex-direction: column;
  overflow-y: auto;
}

.sidebar-header {
  padding: 16px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
}

.system-status {
  margin-top: 8px;
  font-size: 12px;
}

.status-good {
  color: #10b981;
}

.status-error {
  color: #ef4444;
}

.search-mode-selector {
  padding: 16px;
  border-bottom: 1px solid #f3f4f6;
}

.mode-buttons {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 8px;
  margin-top: 8px;
}

.mode-button {
  padding: 8px;
  border: 2px solid #e5e7eb;
  border-radius: 6px;
  background: white;
  font-size: 12px;
  cursor: pointer;
  transition: all 0.2s ease;
  text-align: center;
}

.mode-button:hover {
  border-color: #d1d5db;
  background: #f9fafb;
}

.mode-button.active {
  border-color: #4f46e5;
  background: #eef2ff;
  color: #4f46e5;
  font-weight: 600;
}

.quick-actions {
  padding: 16px;
  border-bottom: 1px solid #f3f4f6;
}

.action-categories {
  margin-top: 12px;
}

.action-category {
  margin-bottom: 16px;
}

.action-category h5 {
  margin: 0 0 8px 0;
  font-size: 13px;
  color: #374151;
  font-weight: 600;
}

.quick-action-button {
  display: block;
  width: 100%;
  padding: 8px;
  margin-bottom: 4px;
  border: 1px solid #e5e7eb;
  border-radius: 4px;
  background: white;
  text-align: left;
  font-size: 11px;
  cursor: pointer;
  transition: all 0.2s ease;
  line-height: 1.3;
}

.quick-action-button:hover {
  background: #f3f4f6;
  border-color: #d1d5db;
}

.quick-action-button.small {
  padding: 6px;
  font-size: 10px;
}

.chat-main {
  flex: 1;
  display: flex;
  flex-direction: column;
  background: white;
}

.chat-header {
  padding: 16px;
  background: #f9fafb;
  border-bottom: 1px solid #e5e7eb;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.current-video-info {
  display: flex;
  align-items: center;
  gap: 8px;
}

.video-name {
  font-weight: 600;
  color: #374151;
}

.advanced-badge {
  background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
  color: white;
  padding: 2px 8px;
  border-radius: 12px;
  font-size: 12px;
  font-weight: 500;
}

.chat-actions {
  display: flex;
  gap: 8px;
}

.chat-actions button {
  padding: 6px 12px;
  border: 1px solid #d1d5db;
  border-radius: 6px;
  background: white;
  font-size: 12px;
  cursor: pointer;
  transition: all 0.2s ease;
}

.chat-actions button:hover {
  background: #f3f4f6;
}

.messages-container {
  flex: 1;
  overflow-y: auto;
  padding: 16px;
  max-height: 400px;
}

.message {
  margin-bottom: 16px;
  display: flex;
  max-width: 85%;
}

.message.user {
  margin-left: auto;
}

.message.user .message-content {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  margin-left: auto;
}

.message.bot .message-content {
  background: #f3f4f6;
  color: #374151;
}

.message.system .message-content {
  background: #fef3c7;
  color: #92400e;
  border-left: 3px solid #f59e0b;
}

.message.error .message-content {
  background: #fef2f2;
  color: #dc2626;
  border-left: 3px solid #ef4444;
}

.message-content {
  padding: 12px 16px;
  border-radius: 12px;
  max-width: 100%;
  line-height: 1.5;
}

.message-text {
  white-space: pre-wrap;
  word-wrap: break-word;
}

.message-timestamp {
  font-size: 10px;
  opacity: 0.7;
  margin-top: 4px;
}

.message-metadata {
  display: flex;
  gap: 8px;
  margin-top: 6px;
  font-size: 10px;
}

.result-count, .duration, .cached {
  background: rgba(255, 255, 255, 0.2);
  padding: 2px 6px;
  border-radius: 10px;
}

.results-container {
  background: #f8fafc;
  border-top: 1px solid #e5e7eb;
  max-height: 50vh;
  overflow-y: auto;
}

.result-tabs {
  display: flex;
  background: white;
  border-bottom: 1px solid #e5e7eb;
  padding: 0 16px;
}

.result-tab {
  padding: 12px 16px;
  border: none;
  background: none;
  cursor: pointer;
  font-size: 13px;
  color: #6b7280;
  border-bottom: 2px solid transparent;
  transition: all 0.2s ease;
}

.result-tab:hover {
  color: #374151;
}

.result-tab.active {
  color: #4f46e5;
  border-bottom-color: #4f46e5;
  font-weight: 600;
}

.result-tab:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.results-content {
  padding: 16px;
}

.input-container {
  padding: 16px;
  background: white;
  border-top: 1px solid #e5e7eb;
}

.input-wrapper {
  display: flex;
  gap: 8px;
  align-items: flex-end;
}

.input-wrapper textarea {
  flex: 1;
  padding: 12px;
  border: 2px solid #e5e7eb;
  border-radius: 8px;
  font-size: 14px;
  resize: vertical;
  min-height: 44px;
  max-height: 120px;
  font-family: inherit;
  transition: border-color 0.2s ease;
}

.input-wrapper textarea:focus {
  outline: none;
  border-color: #4f46e5;
}

.send-button {
  padding: 12px 16px;
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;
  border-radius: 8px;
  font-size: 16px;
  cursor: pointer;
  transition: all 0.2s ease;
  min-width: 60px;
}

.send-button:hover:not(:disabled) {
  transform: translateY(-1px);
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
}

.send-button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.input-hints {
  margin-top: 8px;
  font-size: 12px;
  color: #6b7280;
  text-align: center;
}

.loading-indicator {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 16px;
}

.loading-spinner {
  width: 16px;
  height: 16px;
  border: 2px solid #e5e7eb;
  border-top-color: #4f46e5;
  border-radius: 50%;
  animation: spin 1s linear infinite;
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}

.error-banner {
  background: #fef2f2;
  color: #dc2626;
  padding: 12px 16px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  border-bottom: 1px solid #fecaca;
}

.error-banner button {
  background: none;
  border: none;
  color: #dc2626;
  cursor: pointer;
  font-size: 18px;
  padding: 0;
  width: 24px;
  height: 24px;
  display: flex;
  align-items: center;
  justify-content: center;
}

/* 반응형 디자인 */
@media (max-width: 768px) {
  .chat-container {
    flex-direction: column;
    height: 100vh;
  }
  
  .chat-sidebar {
    width: 100%;
    height: 200px;
    border-right: none;
    border-bottom: 1px solid #e5e7eb;
  }
  
  .mode-buttons {
    grid-template-columns: 1fr 1fr 1fr 1fr;
  }
  
  .mode-button {
    font-size: 10px;
    padding: 6px;
  }
  
  .messages-container {
    max-height: 300px;
  }
  
  .results-container {
    max-height: 40vh;
  }
}
`;
