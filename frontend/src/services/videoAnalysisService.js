// services/videoAnalysisService.js - 오류 수정된 버전

// 기본 API 설정
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// 유틸리티 함수들
const timeUtils = {
  parseTimeToSeconds(timeStr) {
    if (!timeStr) return 0;
    try {
      if (timeStr.includes(':')) {
        const [minutes, seconds = 0] = timeStr.split(':').map(Number);
        return minutes * 60 + seconds;
      } else {
        return parseInt(timeStr);
      }
    } catch {
      return 0;
    }
  },

  secondsToTimeString(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  }
};

// 메인 서비스 객체
const videoAnalysisService = {
  
  // ========== 기존 기본 메서드들 ==========
  
  async getVideoList() {
    try {
      const response = await fetch(`${API_BASE_URL}/videos/`);
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      return await response.json();
    } catch (error) {
      console.error('비디오 목록 조회 실패:', error);
      throw error;
    }
  },

  async uploadVideo(videoFile, onProgress) {
    try {
      const formData = new FormData();
      formData.append('video', videoFile);

      const response = await fetch(`${API_BASE_URL}/upload_video/`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Upload failed: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('비디오 업로드 실패:', error);
      throw error;
    }
  },

  async analyzeVideoEnhanced(videoId, options = {}) {
    try {
      const response = await fetch(`${API_BASE_URL}/analyze_video_enhanced/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          video_id: videoId,
          analysisType: options.analysisType || 'enhanced',
          analysisConfig: options.analysisConfig || {},
          enhancedAnalysis: options.enhancedAnalysis !== false
        })
      });

      if (!response.ok) {
        throw new Error(`분석 시작 실패: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('고급 분석 시작 실패:', error);
      throw error;
    }
  },

  async getAnalysisStatus(videoId) {
    try {
      const response = await fetch(`${API_BASE_URL}/analysis_status/${videoId}/`);
      if (!response.ok) {
        throw new Error(`분석 상태 조회 실패: ${response.status}`);
      }
      return await response.json();
    } catch (error) {
      console.error('분석 상태 조회 실패:', error);
      throw error;
    }
  },

  async sendVideoChatMessage(message, videoId) {
    try {
      const response = await fetch(`${API_BASE_URL}/video/chat/enhanced/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: message,
          video_id: videoId
        })
      });

      if (!response.ok) {
        throw new Error(`채팅 메시지 전송 실패: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('채팅 메시지 전송 실패:', error);
      throw error;
    }
  },

  // ========== 수정된 고급 검색 메서드들 ==========

  async performAdvancedSearch(query, options = {}) {
    try {
      console.log('🔍 고급 검색 시작:', { query, options });

      // 쿼리 타입 자동 감지
      const searchType = this.detectSearchType(query, options);
      console.log('📋 감지된 검색 타입:', searchType);

      let result;

      switch (searchType) {
        case 'cross-video':
          result = await this.searchCrossVideo(query, options.filters);
          break;
        
        case 'object-tracking':
          if (!options.videoId) {
            throw new Error('객체 추적에는 비디오 ID가 필요합니다.');
          }
          result = await this.trackObjectInVideo(
            options.videoId, 
            query, 
            options.timeRange
          );
          break;
        
        case 'time-analysis':
          if (!options.videoId || !options.timeRange) {
            throw new Error('시간대별 분석에는 비디오 ID와 시간 범위가 필요합니다.');
          }
          result = await this.analyzeTimeBasedData(
            options.videoId, 
            options.timeRange, 
            query
          );
          break;
        
        default:
          // 기본 검색으로 fallback
          result = await this.searchVideoAdvanced(
            options.videoId || null, 
            query, 
            options.searchOptions || {}
          );
      }

      console.log('✅ 고급 검색 완료:', result);
      return {
        ...result,
        detected_search_type: searchType,
        original_query: query
      };

    } catch (error) {
      console.error('❌ 고급 검색 실패:', error);
      throw error;
    }
  },

  detectSearchType(query, options) {
    const queryLower = query.toLowerCase();
    
    // 시간대별 분석 키워드
    const timeAnalysisKeywords = [
      '성비', '분포', '통계', '시간대', '구간', '사이', 
      '몇명', '얼마나', '평균', '비율', '패턴'
    ];
    
    // 객체 추적 키워드
    const trackingKeywords = [
      '추적', '따라가', '이동', '경로', '지나간', 
      '상의', '모자', '색깔', '옷', '사람'
    ];
    
    // 영상 간 검색 키워드
    const crossVideoKeywords = [
      '촬영된', '영상', '비디오', '찾아', '비가', '밤', 
      '낮', '실내', '실외', '장소'
    ];
    
    // 시간 범위가 있고 분석 키워드가 있으면 시간대별 분석
    if ((options.timeRange && options.timeRange.start && options.timeRange.end) || 
        timeAnalysisKeywords.some(keyword => queryLower.includes(keyword))) {
      return 'time-analysis';
    }
    
    // 특정 비디오 ID가 있고 추적 키워드가 있으면 객체 추적
    if (options.videoId && trackingKeywords.some(keyword => queryLower.includes(keyword))) {
      return 'object-tracking';
    }
    
    // 크로스 비디오 키워드가 있으면 영상 간 검색
    if (crossVideoKeywords.some(keyword => queryLower.includes(keyword))) {
      return 'cross-video';
    }
    
    // 기본값: 특정 비디오가 선택되어 있으면 객체 추적, 아니면 크로스 비디오
    return options.videoId ? 'object-tracking' : 'cross-video';
  },

  async searchCrossVideo(query, filters = {}) {
    try {
      const response = await fetch(`${API_BASE_URL}/search/cross-video/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          query: query,
          filters: filters
        })
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `크로스 비디오 검색 실패: ${response.status}`);
      }

      const data = await response.json();
      console.log('✅ 크로스 비디오 검색 성공:', data);
      return data;

    } catch (error) {
      console.error('❌ 크로스 비디오 검색 오류:', error);
      throw error;
    }
  },

  async trackObjectInVideo(videoId, trackingTarget, timeRange = {}) {
    try {
      const response = await fetch(`${API_BASE_URL}/search/object-tracking/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          video_id: videoId,
          tracking_target: trackingTarget,
          time_range: timeRange
        })
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `객체 추적 실패: ${response.status}`);
      }

      const data = await response.json();
      console.log('✅ 객체 추적 성공:', data);
      return data;

    } catch (error) {
      console.error('❌ 객체 추적 오류:', error);
      throw error;
    }
  },

  async analyzeTimeBasedData(videoId, timeRange, analysisType) {
    try {
      const response = await fetch(`${API_BASE_URL}/analysis/time-based/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          video_id: videoId,
          time_range: timeRange,
          analysis_type: analysisType
        })
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `시간대별 분석 실패: ${response.status}`);
      }

      const data = await response.json();
      console.log('✅ 시간대별 분석 성공:', data);
      return data;

    } catch (error) {
      console.error('❌ 시간대별 분석 오류:', error);
      throw error;
    }
  },

  async searchVideoAdvanced(videoId, query, searchOptions = {}) {
    try {
      const response = await fetch(`${API_BASE_URL}/videos/search/advanced/`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          video_id: videoId,
          query: query,
          search_options: searchOptions
        })
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `고급 검색 실패: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      console.error('고급 검색 실패:', error);
      throw error;
    }
  },

  // ========== 삭제 관련 메서드들 ==========

  async deleteVideo(videoId) {
    try {
      const response = await fetch(`${API_BASE_URL}/videos/${videoId}/delete/`, {
        method: 'DELETE',
        headers: {
          'Content-Type': 'application/json'
        }
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.error || `삭제 실패: ${response.status}`);
      }

      const result = await response.json();
      console.log('✅ 비디오 삭제 성공:', result);
      return result;

    } catch (error) {
      console.error('❌ 비디오 삭제 실패:', error);
      throw error;
    }
  },

  async checkVideoExists(videoId) {
    try {
      const response = await fetch(`${API_BASE_URL}/videos/${videoId}/exists/`);
      return await response.json();
    } catch (error) {
      console.error('비디오 존재 확인 실패:', error);
      return { exists: false };
    }
  },

  // ========== 유틸리티 메서드들 ==========

  canDeleteVideo(video) {
    const canDelete = video.analysis_status !== 'processing';
    const reason = canDelete ? null : '분석 중인 비디오는 삭제할 수 없습니다.';
    return { canDelete, reason };
  },

  getFrameImageUrl(videoId, frameNumber, withBbox = false) {
    const baseUrl = `${API_BASE_URL}/frame/${videoId}/${frameNumber}/`;
    return withBbox ? `${baseUrl}bbox/` : baseUrl;
  },

  async getScenes(videoId) {
    try {
      const response = await fetch(`${API_BASE_URL}/scenes/${videoId}/enhanced/`);
      if (!response.ok) {
        throw new Error(`씬 정보 조회 실패: ${response.status}`);
      }
      return await response.json();
    } catch (error) {
      console.error('씬 정보 조회 실패:', error);
      throw error;
    }
  },

  timeUtils: timeUtils
};

// 삭제 관련 유틸리티
const deleteUtils = {
  getSelectableVideos(videos) {
    return videos.filter(video => video.analysis_status !== 'processing');
  },

  getNonSelectableVideos(videos) {
    return videos.filter(video => video.analysis_status === 'processing');
  },

  formatFileSize(bytes) {
    if (!bytes) return '0 B';
    
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    
    return Math.round(bytes / Math.pow(1024, i) * 100) / 100 + ' ' + sizes[i];
  }
};

// 진행률 추적기
const advancedProgressMonitor = {
  async monitorAdvancedProgress(videoId, callback) {
    const pollInterval = 2000; // 2초마다 체크
    
    const checkProgress = async () => {
      try {
        const status = await videoAnalysisService.getAnalysisStatus(videoId);
        
        if (callback) {
          callback({
            progress: status.progress || 0,
            step: status.currentStep || '분석 중',
            currentFeature: status.currentFeature || '',
            completedFeatures: status.completedFeatures || [],
            totalFeatures: status.totalFeatures || 4,
            processedFrames: status.processedFrames || 0,
            totalFrames: status.totalFrames || 0
          });
        }
        
        if (status.status === 'processing' && status.progress < 100) {
          setTimeout(checkProgress, pollInterval);
        }
        
      } catch (error) {
        console.error('진행률 모니터링 오류:', error);
        if (callback) {
          callback({ error: error.message });
        }
      }
    };
    
    checkProgress();
  }
};

export { videoAnalysisService, deleteUtils, advancedProgressMonitor };