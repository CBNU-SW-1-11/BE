// pages/IntegratedChatPage.js - 중복 제거된 버전
import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { videoAnalysisService } from '../services/videoAnalysisService';

const IntegratedChatPage = () => {
  const navigate = useNavigate();
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [loading, setLoading] = useState(false);
  const [currentVideo, setCurrentVideo] = useState(null);
  const [availableVideos, setAvailableVideos] = useState([]);
  const chatContainerRef = useRef(null);

  useEffect(() => {
    loadAvailableVideos();
  }, []);

  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [messages]);

  const loadAvailableVideos = async () => {
    try {
      const data = await videoAnalysisService.getVideoList();
      const analyzedVideos = data.videos?.filter(v => v.is_analyzed) || [];
      setAvailableVideos(analyzedVideos);
      
      if (analyzedVideos.length > 0 && !currentVideo) {
        setCurrentVideo(analyzedVideos[0]);
      }
    } catch (error) {
      console.error('비디오 목록 로드 실패:', error);
    }
  };

  // 시간 범위 파싱 함수 개선
  const parseTimeRange = (message) => {
    // "3:00~5:00", "3:00-5:00", "3분~5분" 등의 패턴 감지
    const timePatterns = [
      /(\d+):(\d+)\s*[-~]\s*(\d+):(\d+)/,  // 3:00-5:00 형태
      /(\d+)분\s*[-~]\s*(\d+)분/,          // 3분-5분 형태
      /(\d+)\s*[-~]\s*(\d+)분/,            // 3-5분 형태
    ];

    for (const pattern of timePatterns) {
      const match = message.match(pattern);
      if (match) {
        if (pattern.source.includes(':')) {
          return {
            start: `${match[1]}:${match[2]}`,
            end: `${match[3]}:${match[4]}`
          };
        } else {
          return {
            start: `${match[1]}:00`,
            end: `${match[2]}:00`
          };
        }
      }
    }
    return null;
  };

  // 검색 타입 감지 개선
  const detectSearchIntent = (message) => {
    const messageLower = message.toLowerCase();
    
    // 시간대별 분석 키워드
    const timeAnalysisKeywords = ['성비', '분포', '통계', '비율', '몇명', '얼마나'];
    // 객체 추적 키워드  
    const trackingKeywords = ['추적', '지나간', '상의', '모자', '색깔', '옷'];
    
    const hasTimeRange = parseTimeRange(message) !== null;
    const hasTimeAnalysis = timeAnalysisKeywords.some(keyword => messageLower.includes(keyword));
    const hasTracking = trackingKeywords.some(keyword => messageLower.includes(keyword));

    if (hasTimeRange && hasTimeAnalysis) {
      return 'time-analysis';
    } else if (hasTracking || messageLower.includes('남성') || messageLower.includes('여성')) {
      return 'object-tracking';
    } else {
      return 'general-search';
    }
  };

  // 검색 응답 포맷팅 함수
  const formatSearchResponse = (query, searchResults) => {
    if (!searchResults || searchResults.length === 0) {
      return `'${query}' 검색 결과를 찾을 수 없습니다.`;
    }

    let response_text = `'${query}' 검색 결과 ${searchResults.length}개를 찾았습니다.\n\n`;
    
    searchResults.slice(0, 5).forEach((result, index) => {
      const timeStr = videoAnalysisService.timeUtils.secondsToTimeString(result.timestamp || 0);
      response_text += `${index + 1}. 프레임 #${result.frame_id} (${timeStr})\n`;
      
      if (result.caption) {
        response_text += `   ${result.caption.substring(0, 100)}...\n`;
      }
      
      response_text += '\n';
    });
    
    if (searchResults.length > 5) {
      response_text += `... 외 ${searchResults.length - 5}개 프레임 더\n\n`;
    }
    
    response_text += '🖼️ 아래에서 실제 프레임 이미지를 확인하세요!';
    
    return response_text;
  };

  // 메시지 전송 함수 (단일 정의)
  const handleSendMessage = async () => {
    if (!inputMessage.trim() || loading) return;
    if (!currentVideo) {
      alert('분석된 비디오를 먼저 선택해주세요.');
      return;
    }

    const userMessage = inputMessage.trim();
    setInputMessage('');
    setLoading(true);

    // 사용자 메시지 추가
    const newUserMessage = {
      id: Date.now(),
      type: 'user',
      content: userMessage,
      timestamp: new Date()
    };
    setMessages(prev => [...prev, newUserMessage]);

    try {
      console.log('🔍 메시지 분석 시작:', userMessage);
      
      // 검색 의도 감지
      const searchIntent = detectSearchIntent(userMessage);
      console.log('📋 감지된 검색 의도:', searchIntent);
      
      let response;
      let searchResults = null;
      let searchType = 'general';

      if (searchIntent === 'time-analysis') {
        // 시간대별 분석
        const timeRange = parseTimeRange(userMessage);
        console.log('⏰ 파싱된 시간 범위:', timeRange);
        
        if (timeRange) {
          try {
            response = await videoAnalysisService.analyzeTimeBasedData(
              currentVideo.id,
              timeRange,
              userMessage
            );
            searchType = 'time-analysis';
            searchResults = response.result;
            
            // 결과 포맷팅
            if (response.result && response.result.total_persons !== undefined) {
              const result = response.result;
              response.formatted_response = 
                `📊 ${timeRange.start}~${timeRange.end} 시간대 분석 결과:\n\n` +
                `👥 총 인원: ${result.total_persons}명\n` +
                `👨 남성: ${result.male_count}명 (${result.gender_ratio?.male || 0}%)\n` +
                `👩 여성: ${result.female_count}명 (${result.gender_ratio?.female || 0}%)\n\n`;
              
              if (result.clothing_colors && Object.keys(result.clothing_colors).length > 0) {
                response.formatted_response += `👕 주요 의상 색상:\n`;
                Object.entries(result.clothing_colors).slice(0, 3).forEach(([color, count]) => {
                  response.formatted_response += `   • ${color}: ${count}명\n`;
                });
              }
              
              if (result.peak_times && result.peak_times.length > 0) {
                response.formatted_response += `\n⏰ 활동 피크 시간: ${result.peak_times.join(', ')}`;
              }
            }
          } catch (error) {
            console.error('❌ 시간대별 분석 실패:', error);
            response = { error: error.message };
          }
        } else {
          response = { error: '시간 범위를 명확히 지정해주세요 (예: 3:00~5:00)' };
        }
        
      } else if (searchIntent === 'object-tracking') {
        // 객체 추적
        const timeRange = parseTimeRange(userMessage);
        console.log('🎯 객체 추적 시작, 시간 범위:', timeRange);
        
        try {
          response = await videoAnalysisService.trackObjectInVideo(
            currentVideo.id,
            userMessage,
            timeRange || {}
          );
          searchType = 'object-tracking';
          searchResults = response.tracking_results;
          
          // 결과 포맷팅
          if (response.tracking_results && response.tracking_results.length > 0) {
            const results = response.tracking_results;
            response.formatted_response = 
              `🎯 "${response.tracking_target}" 추적 결과:\n\n` +
              `📍 총 ${results.length}개 장면에서 발견\n\n`;
            
            results.slice(0, 5).forEach((result, index) => {
              const timeStr = videoAnalysisService.timeUtils.secondsToTimeString(result.timestamp);
              response.formatted_response += 
                `${index + 1}. ${timeStr} - ${result.description} (신뢰도: ${(result.confidence * 100).toFixed(1)}%)\n`;
            });
            
            if (results.length > 5) {
              response.formatted_response += `\n... 외 ${results.length - 5}개 장면 더`;
            }
          } else {
            response.formatted_response = `🔍 "${response.tracking_target}"에 해당하는 객체를 찾을 수 없습니다.`;
          }
        } catch (error) {
          console.error('❌ 객체 추적 실패:', error);
          response = { error: error.message };
        }
        
      } else {
        // 일반 채팅 또는 프레임 검색
        try {
          response = await videoAnalysisService.sendVideoChatMessage(userMessage, currentVideo.id);
          
          // 검색 결과가 있으면 프레임 검색으로 처리
          if (response.search_results && Array.isArray(response.search_results)) {
            searchType = 'frame-search';
            searchResults = response.search_results;
            
            // 검색 응답 포맷팅
            response.formatted_response = formatSearchResponse(userMessage, searchResults);
          }
        } catch (error) {
          console.error('❌ 일반 채팅 실패:', error);
          response = { error: error.message };
        }
      }

      // 응답 메시지 생성
      const botMessage = {
        id: Date.now() + 1,
        type: 'bot',
        content: response.formatted_response || response.response || response.error || '응답을 생성할 수 없습니다.',
        timestamp: new Date(),
        searchType: searchType,
        searchResults: searchResults,
        originalResponse: response
      };

      setMessages(prev => [...prev, botMessage]);
      
      console.log('✅ 메시지 처리 완료:', {
        searchType,
        hasResults: !!searchResults,
        responseLength: botMessage.content.length
      });

    } catch (error) {
      console.error('❌ 메시지 전송 실패:', error);
      
      const errorMessage = {
        id: Date.now() + 1,
        type: 'bot',
        content: `오류가 발생했습니다: ${error.message}`,
        timestamp: new Date(),
        isError: true
      };
      
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    // 시간 범위 파싱 함수 개선
    const parseTimeRange = (message) => {
      // "3:00~5:00", "3:00-5:00", "3분~5분" 등의 패턴 감지
      const timePatterns = [
        /(\d+):(\d+)\s*[-~]\s*(\d+):(\d+)/,  // 3:00-5:00 형태
        /(\d+)분\s*[-~]\s*(\d+)분/,          // 3분-5분 형태
        /(\d+)\s*[-~]\s*(\d+)분/,            // 3-5분 형태
      ];

      for (const pattern of timePatterns) {
        const match = message.match(pattern);
        if (match) {
          if (pattern.source.includes(':')) {
            return {
              start: `${match[1]}:${match[2]}`,
              end: `${match[3]}:${match[4]}`
            };
          } else {
            return {
              start: `${match[1]}:00`,
              end: `${match[2]}:00`
            };
          }
        }
      }
      return null;
    };

    // 검색 타입 감지 개선
    const detectSearchIntent = (message) => {
      const messageLower = message.toLowerCase();
      
      // 시간대별 분석 키워드
      const timeAnalysisKeywords = ['성비', '분포', '통계', '비율', '몇명', '얼마나'];
      // 객체 추적 키워드  
      const trackingKeywords = ['추적', '지나간', '상의', '모자', '색깔', '옷'];
      
      const hasTimeRange = parseTimeRange(message) !== null;
      const hasTimeAnalysis = timeAnalysisKeywords.some(keyword => messageLower.includes(keyword));
      const hasTracking = trackingKeywords.some(keyword => messageLower.includes(keyword));

      if (hasTimeRange && hasTimeAnalysis) {
        return 'time-analysis';
      } else if (hasTracking || messageLower.includes('남성') || messageLower.includes('여성')) {
        return 'object-tracking';
      } else {
        return 'general-search';
      }
    };

    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

  const formatTimestamp = (timestamp) => {
    return timestamp.toLocaleDateString('ko-KR', { 
      hour: '2-digit', 
      minute: '2-digit' 
    });
  };

  // 검색 결과 렌더링 (이미지 포함)
  const renderSearchResults = (message) => {
    if (!message.searchResults) return null;

    if (message.searchType === 'time-analysis') {
      const result = message.searchResults;
      return (
        <div style={{ 
          marginTop: '10px', 
          padding: '10px', 
          backgroundColor: '#f0f8ff', 
          borderRadius: '5px',
          fontSize: '14px'
        }}>
          <strong>📊 상세 분석 데이터:</strong>
          <div style={{ marginTop: '5px' }}>
            {result.analysis_period && <div>📅 분석 기간: {result.analysis_period}</div>}
            {result.movement_patterns && <div>🔄 이동 패턴: {result.movement_patterns}</div>}
          </div>
        </div>
      );
    }

    if (message.searchType === 'object-tracking' && Array.isArray(message.searchResults)) {
      return (
        <div style={{ 
          marginTop: '10px', 
          padding: '10px', 
          backgroundColor: '#f0fff0', 
          borderRadius: '5px',
          fontSize: '14px'
        }}>
          <strong>🎯 추적된 위치들:</strong>
          {message.searchResults.slice(0, 3).map((result, index) => (
            <div key={index} style={{ marginTop: '5px' }}>
              {videoAnalysisService.timeUtils.secondsToTimeString(result.timestamp)} 
              {result.match_reasons && result.match_reasons.length > 0 && 
                ` - ${result.match_reasons.join(', ')}`
              }
            </div>
          ))}
        </div>
      );
    }

    // 프레임 검색 결과 - 이미지 포함
    if (message.searchType === 'frame-search' && Array.isArray(message.searchResults)) {
      return (
        <div style={{ 
          marginTop: '15px', 
          padding: '10px', 
          backgroundColor: '#f8f9fa', 
          borderRadius: '8px'
        }}>
          <strong>🖼️ 검색된 프레임들:</strong>
          <div style={{ 
            display: 'grid', 
            gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', 
            gap: '10px', 
            marginTop: '10px' 
          }}>
            {message.searchResults.slice(0, 6).map((result, index) => (
              <div key={index} style={{ 
                border: '1px solid #ddd', 
                borderRadius: '8px', 
                overflow: 'hidden',
                backgroundColor: 'white'
              }}>
                <img 
                  src={videoAnalysisService.getFrameImageUrl(currentVideo?.id, result.frame_id)}
                  alt={`프레임 ${result.frame_id}`}
                  style={{ 
                    width: '100%', 
                    height: '120px', 
                    objectFit: 'cover',
                    cursor: 'pointer'
                  }}
                  onClick={() => {
                    // 클릭시 큰 이미지로 보기
                    const newWindow = window.open('', '_blank');
                    newWindow.document.write(`
                      <html>
                        <head><title>프레임 ${result.frame_id}</title></head>
                        <body style="margin:0; display:flex; justify-content:center; align-items:center; min-height:100vh; background:#000;">
                          <img src="${videoAnalysisService.getFrameImageUrl(currentVideo?.id, result.frame_id)}" 
                               style="max-width:90%; max-height:90%; object-fit:contain;" />
                        </body>
                      </html>
                    `);
                  }}
                />
                <div style={{ padding: '8px', fontSize: '12px' }}>
                  <div><strong>프레임 #{result.frame_id}</strong></div>
                  <div style={{ color: '#666' }}>
                    {videoAnalysisService.timeUtils.secondsToTimeString(result.timestamp)}
                  </div>
                  <div style={{ color: '#666', marginTop: '4px' }}>
                    신뢰도: {(result.match_score * 100).toFixed(1)}%
                  </div>
                  {result.matches && result.matches.length > 0 && (
                    <div style={{ color: '#007bff', fontSize: '11px', marginTop: '2px' }}>
                      {result.matches[0].match} 감지
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
          {message.searchResults.length > 6 && (
            <div style={{ textAlign: 'center', marginTop: '10px', fontSize: '12px', color: '#666' }}>
              ... 외 {message.searchResults.length - 6}개 프레임 더
            </div>
          )}
        </div>
      );
    }

    return null;
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', fontFamily: 'Arial, sans-serif' }}>
      {/* 헤더 */}
      <div style={{ 
        padding: '20px', 
        backgroundColor: '#4a90e2', 
        color: 'white',
        boxShadow: '0 2px 4px rgba(0,0,0,0.1)'
      }}>
        <h1 style={{ margin: 0, fontSize: '24px' }}>🤖 AI 비디오 분석 채팅</h1>
        
        {/* 비디오 선택 */}
        <div style={{ marginTop: '10px' }}>
          <label style={{ fontSize: '14px', marginRight: '10px' }}>현재 비디오:</label>
          <select 
            value={currentVideo?.id || ''} 
            onChange={(e) => {
              const video = availableVideos.find(v => v.id === parseInt(e.target.value));
              setCurrentVideo(video);
            }}
            style={{ 
              padding: '5px 10px', 
              borderRadius: '4px', 
              border: 'none',
              fontSize: '14px'
            }}
          >
            {availableVideos.map(video => (
              <option key={video.id} value={video.id}>
                {video.original_name}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* 채팅 영역 */}
      <div 
        ref={chatContainerRef}
        style={{ 
          flex: 1, 
          overflowY: 'auto', 
          padding: '20px',
          backgroundColor: '#f8f9fa'
        }}
      >
        {messages.length === 0 && (
          <div style={{ 
            textAlign: 'center',
            color: '#666',
            marginTop: '50px'
          }}>
            <h3>💬 AI와 대화를 시작해보세요!</h3>
            <p>예시 질문:</p>
            <div style={{ textAlign: 'left', maxWidth: '400px', margin: '0 auto' }}>
              <p>• "이 영상에서 주황색 상의를 입은 남성이 지나간 장면을 추적해줘"</p>
              <p>• "3:00~5:00 사이에 지나간 사람들의 성비 분포는?"</p>
              <p>• "사람이 나오는 장면 찾아"</p>
            </div>
          </div>
        )}

        {messages.map((message) => (
          <div 
            key={message.id} 
            style={{ 
              marginBottom: '15px',
              display: 'flex',
              justifyContent: message.type === 'user' ? 'flex-end' : 'flex-start'
            }}
          >
            <div style={{ 
              maxWidth: '70%',
              padding: '12px 16px',
              borderRadius: '18px',
              backgroundColor: message.type === 'user' ? '#4a90e2' : '#ffffff',
              color: message.type === 'user' ? 'white' : '#333',
              boxShadow: '0 1px 2px rgba(0,0,0,0.1)',
              whiteSpace: 'pre-line'
            }}>
              <div>{message.content}</div>
              
              {/* 검색 결과 표시 */}
              {renderSearchResults(message)}
              
              <div style={{ 
                fontSize: '12px', 
                opacity: 0.7, 
                marginTop: '5px',
                textAlign: 'right'
              }}>
                {formatTimestamp(message.timestamp)}
                {message.searchType && (
                  <span style={{ marginLeft: '10px' }}>
                    {message.searchType === 'time-analysis' ? '⏰' : 
                     message.searchType === 'object-tracking' ? '🎯' : 
                     message.searchType === 'frame-search' ? '🖼️' : '💬'}
                  </span>
                )}
              </div>
            </div>
          </div>
        ))}

        {loading && (
          <div style={{ 
            display: 'flex', 
            justifyContent: 'flex-start',
            marginBottom: '15px'
          }}>
            <div style={{ 
              padding: '12px 16px',
              borderRadius: '18px',
              backgroundColor: '#ffffff',
              boxShadow: '0 1px 2px rgba(0,0,0,0.1)'
            }}>
              <span>🤖 분석 중...</span>
            </div>
          </div>
        )}
      </div>

      {/* 입력 영역 */}
      <div style={{ 
        padding: '20px',
        backgroundColor: 'white',
        borderTop: '1px solid #e0e0e0',
        display: 'flex',
        gap: '10px'
      }}>
        <input
          type="text"
          value={inputMessage}
          onChange={(e) => setInputMessage(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="메시지를 입력하세요..."
          disabled={loading || !currentVideo}
          style={{ 
            flex: 1,
            padding: '12px 16px',
            borderRadius: '25px',
            border: '1px solid #ddd',
            fontSize: '16px',
            outline: 'none'
          }}
        />
        <button
          onClick={handleSendMessage}
          disabled={loading || !inputMessage.trim() || !currentVideo}
          style={{ 
            padding: '12px 24px',
            borderRadius: '25px',
            border: 'none',
            backgroundColor: loading ? '#ccc' : '#4a90e2',
            color: 'white',
            fontSize: '16px',
            cursor: loading ? 'not-allowed' : 'pointer'
          }}
        >
          {loading ? '⏳' : '전송'}
        </button>
      </div>

      {/* 하단 네비게이션 */}
      <div style={{ 
        padding: '10px 20px',
        backgroundColor: '#f8f9fa',
        borderTop: '1px solid #e0e0e0',
        display: 'flex',
        gap: '10px',
        justifyContent: 'center'
      }}>
        <button onClick={() => navigate('/')} style={{ padding: '8px 16px', borderRadius: '4px', border: '1px solid #ddd', backgroundColor: 'white' }}>
          🏠 홈
        </button>
        <button onClick={() => navigate('/video-analysis')} style={{ padding: '8px 16px', borderRadius: '4px', border: '1px solid #ddd', backgroundColor: 'white' }}>
          📊 분석 현황
        </button>
        <button onClick={() => navigate('/video-upload')} style={{ padding: '8px 16px', borderRadius: '4px', border: '1px solid #ddd', backgroundColor: 'white' }}>
          📁 업로드
        </button>
      </div>
    </div>
  );
};

export default IntegratedChatPage;