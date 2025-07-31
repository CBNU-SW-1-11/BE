import React, { useState, useEffect } from 'react';

// 고급 비디오 검색 및 분석 컴포넌트
const AdvancedVideoSearch = () => {
  const [searchMode, setSearchMode] = useState('cross-video'); // cross-video, intra-video, time-analysis
  const [query, setQuery] = useState('');
  const [videos, setVideos] = useState([]);
  const [selectedVideo, setSelectedVideo] = useState(null);
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [timeRange, setTimeRange] = useState({ start: '', end: '' });
  const [trackingTarget, setTrackingTarget] = useState('');

  // 샘플 비디오 데이터
  useEffect(() => {
    setVideos([
      { id: 1, name: '도심 CCTV 영상_01.mp4', weather: 'rainy', time: 'night', analyzed: true },
      { id: 2, name: '거리 보행자_02.mp4', weather: 'sunny', time: 'day', analyzed: true },
      { id: 3, name: '교통상황_03.mp4', weather: 'cloudy', time: 'evening', analyzed: true }
    ]);
  }, []);

  // 영상 간 검색 (크로스 비디오)
  const handleCrossVideoSearch = async () => {
    setLoading(true);
    try {
      // 샘플 결과 (실제로는 API 호출)
      const mockResults = [
        {
          video_id: 1,
          video_name: '도심 CCTV 영상_01.mp4',
          match_reason: '비가오는 밤 조건에 매칭',
          confidence: 0.85,
          weather: 'rainy',
          time: 'night',
          thumbnail: '/api/frame/1/100/'
        }
      ];
      setResults(mockResults);
    } catch (error) {
      console.error('크로스 비디오 검색 실패:', error);
    }
    setLoading(false);
  };

  // 영상 내 검색 (객체 추적)
  const handleIntraVideoSearch = async () => {
    setLoading(true);
    try {
      // 샘플 결과
      const mockResults = [
        {
          frame_id: 150,
          timestamp: 12.5,
          confidence: 0.78,
          bbox: [0.3, 0.2, 0.6, 0.8],
          description: '주황색 상의를 입은 남성 감지',
          tracking_id: 'person_001'
        },
        {
          frame_id: 180,
          timestamp: 15.0,
          confidence: 0.82,
          bbox: [0.4, 0.25, 0.65, 0.85],
          description: '동일 인물 계속 추적됨',
          tracking_id: 'person_001'
        }
      ];
      setResults(mockResults);
    } catch (error) {
      console.error('영상 내 검색 실패:', error);
    }
    setLoading(false);
  };

  // 시간대별 분석
  const handleTimeAnalysis = async () => {
    setLoading(true);
    try {
      // 샘플 분석 결과
      const mockAnalysis = {
        total_persons: 24,
        male_count: 13,
        female_count: 11,
        gender_ratio: { male: 54.2, female: 45.8 },
        age_distribution: { young: 8, adult: 12, elderly: 4 },
        clothing_colors: { 
          'black': 8, 'white': 6, 'blue': 4, 'red': 3, 'orange': 2, 'other': 1 
        },
        peak_times: ['3:15-3:30', '4:20-4:35'],
        movement_patterns: 'left_to_right_dominant'
      };
      setResults([mockAnalysis]);
    } catch (error) {
      console.error('시간대별 분석 실패:', error);
    }
    setLoading(false);
  };

  // 검색 실행
  const executeSearch = () => {
    if (searchMode === 'cross-video') {
      handleCrossVideoSearch();
    } else if (searchMode === 'intra-video') {
      handleIntraVideoSearch();
    } else if (searchMode === 'time-analysis') {
      handleTimeAnalysis();
    }
  };

  // 크로스 비디오 검색 UI
  const renderCrossVideoSearch = () => (
    <div style={{ border: '1px solid #ddd', padding: '15px', margin: '10px 0', borderRadius: '5px' }}>
      <h3 style={{ margin: '0 0 15px 0', fontSize: '16px' }}>🔍 영상 간 검색</h3>
      <input
        type="text"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="예: 비가오는 밤에 촬영된 영상"
        style={{ 
          width: '70%', 
          padding: '8px', 
          border: '1px solid #ccc', 
          borderRadius: '3px',
          marginRight: '10px' 
        }}
      />
      <button 
        onClick={executeSearch}
        disabled={!query.trim() || loading}
        style={{ 
          padding: '8px 15px', 
          backgroundColor: loading ? '#ccc' : '#007bff', 
          color: 'white', 
          border: 'none', 
          borderRadius: '3px',
          cursor: loading ? 'not-allowed' : 'pointer'
        }}
      >
        {loading ? '검색중...' : '검색'}
      </button>
      
      <div style={{ marginTop: '10px', fontSize: '12px', color: '#666' }}>
        💡 예시: "비가오는 밤 영상", "낮에 촬영된 도로 영상", "실내 영상"
      </div>
    </div>
  );

  // 영상 내 검색 UI
  const renderIntraVideoSearch = () => (
    <div style={{ border: '1px solid #ddd', padding: '15px', margin: '10px 0', borderRadius: '5px' }}>
      <h3 style={{ margin: '0 0 15px 0', fontSize: '16px' }}>🎯 영상 내 객체 추적</h3>
      
      <div style={{ marginBottom: '10px' }}>
        <select 
          value={selectedVideo?.id || ''}
          onChange={(e) => setSelectedVideo(videos.find(v => v.id === parseInt(e.target.value)))}
          style={{ padding: '8px', border: '1px solid #ccc', borderRadius: '3px', marginRight: '10px' }}
        >
          <option value="">영상 선택</option>
          {videos.map(video => (
            <option key={video.id} value={video.id}>{video.name}</option>
          ))}
        </select>
      </div>

      <input
        type="text"
        value={trackingTarget}
        onChange={(e) => setTrackingTarget(e.target.value)}
        placeholder="예: 주황색 상의를 입은 남성"
        style={{ 
          width: '70%', 
          padding: '8px', 
          border: '1px solid #ccc', 
          borderRadius: '3px',
          marginRight: '10px' 
        }}
      />
      <button 
        onClick={executeSearch}
        disabled={!selectedVideo || !trackingTarget.trim() || loading}
        style={{ 
          padding: '8px 15px', 
          backgroundColor: loading || !selectedVideo ? '#ccc' : '#28a745', 
          color: 'white', 
          border: 'none', 
          borderRadius: '3px',
          cursor: loading || !selectedVideo ? 'not-allowed' : 'pointer'
        }}
      >
        {loading ? '추적중...' : '추적'}
      </button>

      <div style={{ marginTop: '10px', fontSize: '12px', color: '#666' }}>
        💡 예시: "빨간 모자 여성", "검은색 차량", "강아지를 산책시키는 사람"
      </div>
    </div>
  );

  // 시간대별 분석 UI
  const renderTimeAnalysis = () => (
    <div style={{ border: '1px solid #ddd', padding: '15px', margin: '10px 0', borderRadius: '5px' }}>
      <h3 style={{ margin: '0 0 15px 0', fontSize: '16px' }}>📊 시간대별 분석</h3>
      
      <div style={{ marginBottom: '10px' }}>
        <select 
          value={selectedVideo?.id || ''}
          onChange={(e) => setSelectedVideo(videos.find(v => v.id === parseInt(e.target.value)))}
          style={{ padding: '8px', border: '1px solid #ccc', borderRadius: '3px', marginRight: '10px' }}
        >
          <option value="">영상 선택</option>
          {videos.map(video => (
            <option key={video.id} value={video.id}>{video.name}</option>
          ))}
        </select>
      </div>

      <div style={{ marginBottom: '10px' }}>
        <input
          type="text"
          value={timeRange.start}
          onChange={(e) => setTimeRange({...timeRange, start: e.target.value})}
          placeholder="시작 시간 (예: 3:00)"
          style={{ 
            width: '120px', 
            padding: '6px', 
            border: '1px solid #ccc', 
            borderRadius: '3px',
            marginRight: '10px' 
          }}
        />
        <span style={{ margin: '0 5px' }}>~</span>
        <input
          type="text"
          value={timeRange.end}
          onChange={(e) => setTimeRange({...timeRange, end: e.target.value})}
          placeholder="종료 시간 (예: 5:00)"
          style={{ 
            width: '120px', 
            padding: '6px', 
            border: '1px solid #ccc', 
            borderRadius: '3px',
            marginRight: '10px' 
          }}
        />
      </div>

      <input
        type="text"
        value={query}
        onChange={(e) => setQuery(e.target.value)}
        placeholder="분석 내용 (예: 지나간 사람들의 성비 분포)"
        style={{ 
          width: '60%', 
          padding: '8px', 
          border: '1px solid #ccc', 
          borderRadius: '3px',
          marginRight: '10px' 
        }}
      />
      <button 
        onClick={executeSearch}
        disabled={!selectedVideo || !timeRange.start || !timeRange.end || !query.trim() || loading}
        style={{ 
          padding: '8px 15px', 
          backgroundColor: loading || !selectedVideo ? '#ccc' : '#dc3545', 
          color: 'white', 
          border: 'none', 
          borderRadius: '3px',
          cursor: loading || !selectedVideo ? 'not-allowed' : 'pointer'
        }}
      >
        {loading ? '분석중...' : '분석'}
      </button>

      <div style={{ marginTop: '10px', fontSize: '12px', color: '#666' }}>
        💡 예시: "사람들의 성비 분포", "차량 유형별 통계", "보행자 이동 패턴"
      </div>
    </div>
  );

  // 결과 렌더링
  const renderResults = () => {
    if (!results.length) return null;

    return (
      <div style={{ marginTop: '20px', border: '1px solid #ddd', padding: '15px', borderRadius: '5px' }}>
        <h3 style={{ margin: '0 0 15px 0', fontSize: '16px' }}>📋 분석 결과</h3>
        
        {searchMode === 'cross-video' && (
          <div>
            {results.map((result, idx) => (
              <div key={idx} style={{ 
                padding: '10px', 
                border: '1px solid #eee', 
                borderRadius: '3px', 
                marginBottom: '10px',
                backgroundColor: '#f9f9f9'
              }}>
                <div style={{ fontWeight: 'bold', marginBottom: '5px' }}>
                  📹 {result.video_name}
                </div>
                <div style={{ fontSize: '14px', marginBottom: '5px' }}>
                  매칭 이유: {result.match_reason}
                </div>
                <div style={{ fontSize: '12px', color: '#666' }}>
                  신뢰도: {(result.confidence * 100).toFixed(1)}% | 
                  날씨: {result.weather} | 시간: {result.time}
                </div>
              </div>
            ))}
          </div>
        )}

        {searchMode === 'intra-video' && (
          <div>
            {results.map((result, idx) => (
              <div key={idx} style={{ 
                padding: '10px', 
                border: '1px solid #eee', 
                borderRadius: '3px', 
                marginBottom: '10px',
                backgroundColor: '#f9f9f9'
              }}>
                <div style={{ fontWeight: 'bold', marginBottom: '5px' }}>
                  🎯 프레임 #{result.frame_id} ({Math.floor(result.timestamp / 60)}:{(result.timestamp % 60).toFixed(0).padStart(2, '0')})
                </div>
                <div style={{ fontSize: '14px', marginBottom: '5px' }}>
                  {result.description}
                </div>
                <div style={{ fontSize: '12px', color: '#666' }}>
                  신뢰도: {(result.confidence * 100).toFixed(1)}% | 추적 ID: {result.tracking_id}
                </div>
              </div>
            ))}
          </div>
        )}

        {searchMode === 'time-analysis' && results[0] && (
          <div style={{ backgroundColor: '#f9f9f9', padding: '15px', borderRadius: '3px' }}>
            <div style={{ marginBottom: '15px' }}>
              <h4 style={{ margin: '0 0 10px 0', fontSize: '14px' }}>👥 성비 분석</h4>
              <div style={{ fontSize: '14px' }}>
                총 {results[0].total_persons}명 (남성: {results[0].male_count}명, 여성: {results[0].female_count}명)
              </div>
              <div style={{ fontSize: '12px', color: '#666' }}>
                비율 - 남성: {results[0].gender_ratio.male}%, 여성: {results[0].gender_ratio.female}%
              </div>
            </div>

            <div style={{ marginBottom: '15px' }}>
              <h4 style={{ margin: '0 0 10px 0', fontSize: '14px' }}>👕 의상 색상 분포</h4>
              <div style={{ fontSize: '12px' }}>
                {Object.entries(results[0].clothing_colors).map(([color, count]) => (
                  <span key={color} style={{ marginRight: '15px' }}>
                    {color}: {count}명
                  </span>
                ))}
              </div>
            </div>

            <div style={{ marginBottom: '15px' }}>
              <h4 style={{ margin: '0 0 10px 0', fontSize: '14px' }}>📈 피크 시간대</h4>
              <div style={{ fontSize: '12px' }}>
                {results[0].peak_times.join(', ')}
              </div>
            </div>

            <div>
              <h4 style={{ margin: '0 0 10px 0', fontSize: '14px' }}>🚶 이동 패턴</h4>
              <div style={{ fontSize: '12px' }}>
                {results[0].movement_patterns === 'left_to_right_dominant' ? 
                  '좌→우 이동이 주된 패턴' : results[0].movement_patterns}
              </div>
            </div>
          </div>
        )}
      </div>
    );
  };

  return (
    <div style={{ maxWidth: '800px', margin: '20px auto', padding: '20px', fontFamily: 'Arial, sans-serif' }}>
      <h2 style={{ fontSize: '20px', marginBottom: '20px', textAlign: 'center' }}>
        🧠 고급 비디오 검색 & 분석
      </h2>

      {/* 모드 선택 */}
      <div style={{ marginBottom: '20px', textAlign: 'center' }}>
        <button
          onClick={() => setSearchMode('cross-video')}
          style={{
            padding: '10px 15px',
            margin: '0 5px',
            backgroundColor: searchMode === 'cross-video' ? '#007bff' : '#f8f9fa',
            color: searchMode === 'cross-video' ? 'white' : '#333',
            border: '1px solid #ccc',
            borderRadius: '3px',
            cursor: 'pointer'
          }}
        >
          영상 간 검색
        </button>
        <button
          onClick={() => setSearchMode('intra-video')}
          style={{
            padding: '10px 15px',
            margin: '0 5px',
            backgroundColor: searchMode === 'intra-video' ? '#28a745' : '#f8f9fa',
            color: searchMode === 'intra-video' ? 'white' : '#333',
            border: '1px solid #ccc',
            borderRadius: '3px',
            cursor: 'pointer'
          }}
        >
          영상 내 추적
        </button>
        <button
          onClick={() => setSearchMode('time-analysis')}
          style={{
            padding: '10px 15px',
            margin: '0 5px',
            backgroundColor: searchMode === 'time-analysis' ? '#dc3545' : '#f8f9fa',
            color: searchMode === 'time-analysis' ? 'white' : '#333',
            border: '1px solid #ccc',
            borderRadius: '3px',
            cursor: 'pointer'
          }}
        >
          시간대별 분석
        </button>
      </div>

      {/* 검색 UI */}
      {searchMode === 'cross-video' && renderCrossVideoSearch()}
      {searchMode === 'intra-video' && renderIntraVideoSearch()}
      {searchMode === 'time-analysis' && renderTimeAnalysis()}

      {/* 결과 표시 */}
      {renderResults()}

      {/* 사용법 안내 */}
      <div style={{ 
        marginTop: '30px', 
        padding: '15px', 
        backgroundColor: '#f8f9fa', 
        borderRadius: '5px',
        fontSize: '12px',
        color: '#666'
      }}>
        <h4 style={{ margin: '0 0 10px 0', fontSize: '14px' }}>💡 사용법</h4>
        <div><strong>영상 간 검색:</strong> 여러 비디오 중에서 조건에 맞는 영상 찾기</div>
        <div><strong>영상 내 추적:</strong> 특정 영상에서 객체나 사람 추적하기</div>
        <div><strong>시간대별 분석:</strong> 지정된 시간 구간의 통계 분석</div>
      </div>
    </div>
  );
};

export default AdvancedVideoSearch;