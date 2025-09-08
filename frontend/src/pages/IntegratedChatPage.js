
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
  const [expandedResults, setExpandedResults] = useState(new Set());
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
    
    const timeAnalysisKeywords = ['성비', '분포', '통계', '비율', '몇명', '얼마나'];
    const trackingKeywords = ['추적', '지나간', '상의', '모자', '색깔', '옷', '찾아줘', '찾아'];
    
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

  // 추적 결과를 표시 가능한 아이템으로 변환
  const convertTrackingResultsToItems = (trackingResults, videoId) => {
    if (!trackingResults || !Array.isArray(trackingResults)) {
      return [];
    }

    return trackingResults.slice(0, 12).map((result, index) => ({
      time: videoAnalysisService.timeUtils.secondsToTimeString(result.timestamp),
      seconds: result.timestamp,
      frame_id: result.frame_id,
      desc: result.description || `객체 감지`,
      score: result.confidence || 0.5,
      reasons: result.match_reasons || [],
      thumbUrl: videoAnalysisService.getFrameImageUrl(videoId, result.frame_id),
      thumbBBoxUrl: videoAnalysisService.getFrameImageUrl(videoId, result.frame_id, true),
      clipUrl: videoAnalysisService.getClipUrl(videoId, result.timestamp, 4)
    }));
  };

  // 메시지 전송 함수
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
      let displayItems = [];
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
          
          // 추적 결과를 표시 가능한 아이템으로 변환
          if (response.tracking_results && Array.isArray(response.tracking_results)) {
            displayItems = convertTrackingResultsToItems(response.tracking_results, currentVideo.id);
            
            const resultCount = response.tracking_results.length;
            response.formatted_response = 
              `🎯 "${response.tracking_target || userMessage}" 추적 결과:\n\n` +
              `📍 총 ${resultCount}개 장면에서 발견\n\n`;
            
            response.tracking_results.slice(0, 5).forEach((result, index) => {
              const timeStr = videoAnalysisService.timeUtils.secondsToTimeString(result.timestamp);
              const confidenceStr = (result.confidence * 100).toFixed(1);
              response.formatted_response += 
                `${index + 1}. ${timeStr} - ${result.description} (신뢰도: ${confidenceStr}%)\n`;
            });
            
            if (resultCount > 5) {
              response.formatted_response += `\n... 외 ${resultCount - 5}개 장면 더`;
            }

            if (displayItems.length > 0) {
              response.formatted_response += `\n\n📸 아래에서 실제 프레임 이미지를 확인하세요!`;
            }
          } else {
            response.formatted_response = `🔍 "${response.tracking_target || userMessage}"에 해당하는 객체를 찾을 수 없습니다.`;
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
            displayItems = response.search_results.map(result => ({
              time: videoAnalysisService.timeUtils.secondsToTimeString(result.timestamp || 0),
              seconds: result.timestamp || 0,
              frame_id: result.frame_id,
              desc: result.caption || result.description || '프레임',
              score: result.match_score || result.score || 0.5,
              thumbUrl: videoAnalysisService.getFrameImageUrl(currentVideo.id, result.frame_id),
              thumbBBoxUrl: videoAnalysisService.getFrameImageUrl(currentVideo.id, result.frame_id, true),
              clipUrl: videoAnalysisService.getClipUrl(currentVideo.id, result.timestamp || 0, 4)
            }));
            
            // 검색 응답 포맷팅
            response.formatted_response = formatSearchResponse(userMessage, response.search_results);
          }
          
          // 검색 결과가 없지만 객체 검색 의도가 있는 경우, 객체 추적도 시도해보기
          else if (userMessage.includes('찾아') && (!response.search_results || response.search_results.length === 0)) {
            console.log('🔄 일반 검색에서 결과가 없어서 객체 추적도 시도해봅니다...');
            try {
              const trackingResponse = await videoAnalysisService.trackObjectInVideo(
                currentVideo.id,
                userMessage,
                {}
              );
              
              if (trackingResponse.tracking_results && trackingResponse.tracking_results.length > 0) {
                searchType = 'object-tracking';
                displayItems = convertTrackingResultsToItems(trackingResponse.tracking_results, currentVideo.id);
                response.formatted_response = 
                  `🎯 "${trackingResponse.tracking_target || userMessage}" 객체 추적 결과:\n\n` +
                  `📍 총 ${trackingResponse.tracking_results.length}개 장면에서 발견\n\n`;
                
                trackingResponse.tracking_results.slice(0, 5).forEach((result, index) => {
                  const timeStr = videoAnalysisService.timeUtils.secondsToTimeString(result.timestamp);
                  const confidenceStr = (result.confidence * 100).toFixed(1);
                  response.formatted_response += 
                    `${index + 1}. ${timeStr} - ${result.description} (신뢰도: ${confidenceStr}%)\n`;
                });
                
                response.formatted_response += `\n📸 아래에서 실제 프레임 이미지를 확인하세요!`;
              }
            } catch (trackingError) {
              console.log('🔄 객체 추적도 실패:', trackingError.message);
            }
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
        items: displayItems, // 표시할 아이템들
        originalResponse: response
      };

      setMessages(prev => [...prev, botMessage]);
      
      console.log('✅ 메시지 처리 완료:', {
        searchType,
        hasItems: displayItems.length > 0,
        itemCount: displayItems.length,
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

  // 검색 결과 렌더링 (가로 슬라이드 형식)
  const renderSearchResults = (message) => {
    if (!message.items || message.items.length === 0) return null;

    if (message.searchType === 'time-analysis') {
      const result = message.originalResponse?.result;
      if (result) {
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
      return null;
    }

    // 가로 슬라이드 형식의 검색 결과
    return (
      <div style={{ 
        marginTop: '15px', 
        padding: '10px', 
        backgroundColor: '#f8f9fa', 
        borderRadius: '8px'
      }}>
        <div style={{ 
          display: 'flex', 
          justifyContent: 'space-between', 
          alignItems: 'center',
          marginBottom: '10px'
        }}>
          <strong>
            {message.searchType === 'object-tracking' ? '🎯 추적된 장면들' : '🖼️ 검색된 프레임들'}
          </strong>
          <span style={{ fontSize: '12px', color: '#666' }}>
            총 {message.items.length}개 (신뢰도 순)
          </span>
        </div>
        
        {/* 가로 스크롤 컨테이너 */}
        <div style={{ 
          display: 'flex',
          overflowX: 'auto',
          gap: '12px',
          paddingBottom: '8px',
          scrollbarWidth: 'thin',
          scrollbarColor: '#888 #f1f1f1'
        }}>
          {message.items.map((item, index) => (
            <div 
              key={index} 
              style={{ 
                minWidth: '200px',
                width: '200px',
                border: '1px solid #ddd', 
                borderRadius: '8px', 
                overflow: 'hidden',
                backgroundColor: 'white',
                boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
                cursor: 'pointer',
                transition: 'transform 0.2s ease, box-shadow 0.2s ease'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.transform = 'translateY(-2px)';
                e.currentTarget.style.boxShadow = '0 4px 8px rgba(0,0,0,0.15)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.transform = 'translateY(0)';
                e.currentTarget.style.boxShadow = '0 2px 4px rgba(0,0,0,0.1)';
              }}
            >
              {/* 신뢰도 배지 */}
              <div style={{
                position: 'relative'
              }}>
                <div style={{
                  position: 'absolute',
                  top: '5px',
                  right: '5px',
                  backgroundColor: item.score >= 0.8 ? '#28a745' : item.score >= 0.6 ? '#ffc107' : '#dc3545',
                  color: 'white',
                  padding: '2px 6px',
                  borderRadius: '12px',
                  fontSize: '10px',
                  fontWeight: 'bold',
                  zIndex: 1
                }}>
                  {Math.round(item.score * 100)}%
                </div>
                
                {item.thumbUrl && (
                  <img 
                    src={item.thumbUrl}
                    alt={`프레임 ${item.frame_id}`}
                    style={{ 
                      width: '100%', 
                      height: '120px', 
                      objectFit: 'cover'
                    }}
                    onClick={() => {
                      const newWindow = window.open('', '_blank');
                      newWindow.document.write(`
                        <html>
                          <head><title>프레임 ${item.frame_id} - ${item.time}</title></head>
                          <body style="margin:0; display:flex; flex-direction:column; justify-content:center; align-items:center; min-height:100vh; background:#000; color:white; font-family:Arial;">
                            <img src="${item.thumbUrl}" style="max-width:90%; max-height:80%; object-fit:contain; margin-bottom:20px;" />
                            <div style="max-width:80%; text-align:center; padding:20px;">
                              <h3>${item.time} 시점 (신뢰도: ${Math.round(item.score * 100)}%)</h3>
                              <p>${item.desc}</p>
                              ${item.reasons && item.reasons.length > 0 ? `<small>매칭 이유: ${item.reasons.join(', ')}</small>` : ''}
                            </div>
                          </body>
                        </html>
                      `);
                    }}
                  />
                )}
              </div>
              
              <div style={{ padding: '8px', fontSize: '12px' }}>
                <div style={{ 
                  fontWeight: 'bold', 
                  display: 'flex', 
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  marginBottom: '4px'
                }}>
                  <span>#{item.frame_id}</span>
                  <span style={{ fontSize: '11px', color: '#666' }}>{item.time}</span>
                </div>
                
                <div style={{ 
                  color: '#333', 
                  lineHeight: '1.3',
                  fontSize: '11px',
                  marginBottom: '6px',
                  display: '-webkit-box',
                  WebkitLineClamp: 2,
                  WebkitBoxOrient: 'vertical',
                  overflow: 'hidden'
                }}>
                  {item.desc}
                </div>
                
                {item.reasons && item.reasons.length > 0 && (
                  <div style={{ 
                    color: '#007bff', 
                    fontSize: '10px', 
                    marginBottom: '6px',
                    backgroundColor: '#e7f3ff',
                    padding: '2px 4px',
                    borderRadius: '3px'
                  }}>
                    {item.reasons[0]}
                  </div>
                )}
                
                <div style={{ 
                  display: 'flex', 
                  gap: '6px',
                  justifyContent: 'space-between'
                }}>
                  {item.clipUrl && (
                    <a 
                      href={item.clipUrl} 
                      target="_blank" 
                      rel="noreferrer"
                      style={{ 
                        fontSize: '10px', 
                        color: '#007bff', 
                        textDecoration: 'none',
                        padding: '2px 6px',
                        backgroundColor: '#e7f3ff',
                        borderRadius: '3px',
                        border: '1px solid #007bff'
                      }}
                      onMouseEnter={(e) => {
                        e.target.style.backgroundColor = '#007bff';
                        e.target.style.color = 'white';
                      }}
                      onMouseLeave={(e) => {
                        e.target.style.backgroundColor = '#e7f3ff';
                        e.target.style.color = '#007bff';
                      }}
                    >
                      📹 클립
                    </a>
                  )}
                  {item.thumbBBoxUrl && (
                    <a 
                      href={item.thumbBBoxUrl} 
                      target="_blank" 
                      rel="noreferrer"
                      style={{ 
                        fontSize: '10px', 
                        color: '#28a745', 
                        textDecoration: 'none',
                        padding: '2px 6px',
                        backgroundColor: '#e8f5e8',
                        borderRadius: '3px',
                        border: '1px solid #28a745'
                      }}
                      onMouseEnter={(e) => {
                        e.target.style.backgroundColor = '#28a745';
                        e.target.style.color = 'white';
                      }}
                      onMouseLeave={(e) => {
                        e.target.style.backgroundColor = '#e8f5e8';
                        e.target.style.color = '#28a745';
                      }}
                    >
                      🔍 박스
                    </a>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
        
        {/* 스크롤 힌트 */}
        {message.items.length > 3 && (
          <div style={{ 
            textAlign: 'center', 
            fontSize: '11px', 
            color: '#999', 
            marginTop: '8px',
            fontStyle: 'italic'
          }}>
            ← 좌우로 스크롤하여 더 많은 결과를 확인하세요 →
          </div>
        )}
        
        {/* 상위 결과 요약 */}
        <div style={{ 
          marginTop: '10px',
          padding: '8px',
          backgroundColor: 'white',
          borderRadius: '5px',
          fontSize: '12px'
        }}>
          <strong>💡 검색 요약:</strong>
          <div style={{ marginTop: '4px', color: '#666' }}>
            {message.items.length > 0 && (
              <>
                가장 높은 신뢰도: {Math.round(message.items[0].score * 100)}% 
                {message.items.length > 1 && (
                  <> • 평균 신뢰도: {Math.round(message.items.reduce((sum, item) => sum + item.score, 0) / message.items.length * 100)}%</>
                )}
              </>
            )}
          </div>
        </div>
      </div>
    );
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
              <p>• "초록 옷 입은 사람 찾아줘"</p>
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
                     message.searchType === 'frame-search' ? '🖼️' : 
                     message.searchType === 'gender-analysis' ? '👥' : '💬'}
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