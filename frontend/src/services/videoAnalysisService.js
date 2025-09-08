// services/videoAnalysisService.js — 통합 & 개선 버전
// - 백엔드 라우트와 일치
// - 고급 검색/추적/분석 포함
// - 하위호환 util 함수 유지 (getFrameImageUrl / getClipUrl / sendVideoChatMessage)

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// -------------------------
// 시간 유틸
// -------------------------
const timeUtils = {
  parseTimeToSeconds(timeStr) {
    if (!timeStr) return 0;
    try {
      if (timeStr.includes(':')) {
        const [m, s = 0] = timeStr.split(':').map(Number);
        return m * 60 + s;
      }
      return parseInt(timeStr, 10);
    } catch {
      return 0;
    }
  },
  secondsToTimeString(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  },
};

// -------------------------
// 내부 공통 fetch 래퍼
// -------------------------
async function jsonFetch(url, options = {}) {
  const res = await fetch(url, options);
  if (!res.ok) {
    let payload = {};
    try {
      payload = await res.json();
    } catch (_) {}
    const msg = payload.error || payload.message || `HTTP ${res.status}`;
    throw new Error(msg);
  }
  return res.json();
}

// -------------------------
// 하위호환 유틸 (백엔드에 맞게 경로 고정)
// -------------------------
function getFrameImageUrl(videoId, frameNumber, withBbox = false) {
  const base = `${API_BASE_URL}/api/frame/${encodeURIComponent(videoId)}/${encodeURIComponent(frameNumber)}/`;
  return withBbox ? `${base}bbox/` : base;
}

function getClipUrl(videoId, seconds, duration = 4) {
  return `${API_BASE_URL}/api/clip/${encodeURIComponent(videoId)}/${encodeURIComponent(seconds)}/?duration=${duration}`;
}

async function sendVideoChatMessage(message, videoId) {
  return jsonFetch(`${API_BASE_URL}/api/video_chat/`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, video_id: videoId }),
  });
}

// -------------------------
// 메인 서비스
// -------------------------
const videoAnalysisService = {
  // ▶ 비디오 목록
  async getVideoList() {
    try {
      return await jsonFetch(`${API_BASE_URL}/videos/`);
    } catch (err) {
      console.error('비디오 목록 조회 실패:', err);
      throw err;
    }
  },

  // ▶ 업로드
  async uploadVideo(file) {
    try {
      const fd = new FormData();
      fd.append('video', file);
      const res = await fetch(`${API_BASE_URL}/upload_video/`, { method: 'POST', body: fd });
      if (!res.ok) throw new Error(`Upload failed: ${res.status}`);
      return res.json();
    } catch (err) {
      console.error('비디오 업로드 실패:', err);
      throw err;
    }
  },

  // ▶ 고급 분석 시작
  async analyzeVideoEnhanced(videoId, options = {}) {
    try {
      return await jsonFetch(
        `${API_BASE_URL}/api/analyze/enhanced/${encodeURIComponent(videoId)}/`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            analysisType: options.analysisType || 'enhanced',
            analysisConfig: options.analysisConfig || {},
            enhancedAnalysis: options.enhancedAnalysis !== false,
          }),
        }
      );
    } catch (err) {
      console.error('고급 분석 시작 실패:', err);
      throw err;
    }
  },

  // ▶ 분석 상태
  async getAnalysisStatus(videoId) {
    try {
      return await jsonFetch(
        `${API_BASE_URL}/api/analyze/status/${encodeURIComponent(videoId)}/`
      );
    } catch (err) {
      console.error('분석 상태 조회 실패:', err);
      throw err;
    }
  },

  // ▶ 자연어 채팅 (자연스러운 Q&A + 썸네일/클립 동반)
  async chat(message, videoId) {
    return sendVideoChatMessage(message, videoId);
  },

  // ▶ 하위호환을 위해 sendVideoChatMessage도 메서드로 추가
  async sendVideoChatMessage(message, videoId) {
    return sendVideoChatMessage(message, videoId);
  },
  

  // ============================
  // 고급 검색/추적/분석
  // ============================

  // 고급 검색 라우팅
  async performAdvancedSearch(query, options = {}) {
    try {
      const searchType = this.detectSearchType(query, options);
      let result;

      switch (searchType) {
        case 'cross-video':
          result = await this.searchCrossVideo(query, options.filters);
          break;

        case 'object-tracking':
          if (!options.videoId) throw new Error('객체 추적에는 비디오 ID가 필요합니다.');
          result = await this.trackObjectInVideo(options.videoId, query, options.timeRange);
          break;

        case 'time-analysis':
          if (!options.videoId || !options.timeRange)
            throw new Error('시간대별 분석에는 비디오 ID와 시간 범위가 필요합니다.');
          result = await this.analyzeTimeBasedData(options.videoId, options.timeRange, query);
          break;

        default:
          result = await this.searchVideoAdvanced(
            options.videoId || null,
            query,
            options.searchOptions || {}
          );
      }

      return { ...result, detected_search_type: searchType, original_query: query };
    } catch (err) {
      console.error('❌ 고급 검색 실패:', err);
      throw err;
    }
  },

  // 검색 타입 감지
  detectSearchType(query, options) {
    const q = (query || '').toLowerCase();

    const timeAnalysisKeywords = ['성비', '분포', '통계', '시간대', '구간', '사이', '몇명', '얼마나', '평균', '비율', '패턴'];
    const trackingKeywords = ['추적', '따라가', '이동', '경로', '지나간', '상의', '모자', '색깔', '옷', '사람'];
    const crossVideoKeywords = ['촬영된', '영상', '비디오', '찾아', '비가', '밤', '낮', '실내', '실외', '장소'];

    if ((options.timeRange && options.timeRange.start && options.timeRange.end) ||
        timeAnalysisKeywords.some(k => q.includes(k))) {
      return 'time-analysis';
    }
    if (options.videoId && trackingKeywords.some(k => q.includes(k))) {
      return 'object-tracking';
    }
    if (crossVideoKeywords.some(k => q.includes(k))) {
      return 'cross-video';
    }
    return options.videoId ? 'object-tracking' : 'cross-video';
  },

  // ▶ 영상 간 검색
  async searchCrossVideo(query, filters = {}) {
    try {
      return await jsonFetch(`${API_BASE_URL}/search/cross-video/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, filters }),
      });
    } catch (err) {
      console.error('크로스 비디오 검색 실패:', err);
      throw err;
    }
  },

  // ▶ 객체 추적
  async trackObjectInVideo(videoId, trackingTarget, timeRange = {}) {
    try {
      return await jsonFetch(`${API_BASE_URL}/search/object-tracking/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          video_id: videoId,
          tracking_target: trackingTarget,
          time_range: timeRange || {},
        }),
      });
    } catch (err) {
      console.error('객체 추적 오류:', err);
      throw err;
    }
  },

  // ▶ 시간대별 분석 (예: 성비)
  async analyzeTimeBasedData(videoId, timeRange, analysisType) {
    try {
      return await jsonFetch(`${API_BASE_URL}/analysis/time-based/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ video_id: videoId, time_range: timeRange, analysis_type: analysisType }),
      });
    } catch (err) {
      console.error('시간대별 분석 오류:', err);
      throw err;
    }
  },

  // ▶ 고급 검색 (단일/선택 영상 내)
  async searchVideoAdvanced(videoId, query, searchOptions = {}) {
    try {
      return await jsonFetch(`${API_BASE_URL}/videos/search/advanced/`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ video_id: videoId, query, search_options: searchOptions }),
      });
    } catch (err) {
      console.error('고급 검색 실패:', err);
      throw err;
    }
  },

  // ============================
  // 부가 기능
  // ============================

  // ▶ 씬 정보 (향상판)
  async getScenes(videoId) {
    try {
      return await jsonFetch(`${API_BASE_URL}/scenes/${encodeURIComponent(videoId)}/enhanced/`);
    } catch (err) {
      console.error('씬 정보 조회 실패:', err);
      throw err;
    }
  },

  // ▶ 존재 확인
  async checkVideoExists(videoId) {
    try {
      return await jsonFetch(`${API_BASE_URL}/videos/${encodeURIComponent(videoId)}/exists/`);
    } catch (err) {
      console.error('비디오 존재 확인 실패:', err);
      return { exists: false };
    }
  },

  // ▶ 삭제 가능 여부
  canDeleteVideo(video) {
    const canDelete = video.analysis_status !== 'processing';
    return { canDelete, reason: canDelete ? null : '분석 중인 비디오는 삭제할 수 없습니다.' };
  },

  // ▶ 프레임/클립 URL (컴포넌트에서 직접 쓰기 좋게 노출)
  getFrameImageUrl,
  getClipUrl,

  // ▶ 시간 유틸 노출
  timeUtils,
};

// -------------------------
// 삭제 관련 유틸
// -------------------------
const deleteUtils = {
  getSelectableVideos(videos) {
    return (videos || []).filter(v => v.analysis_status !== 'processing');
  },
  getNonSelectableVideos(videos) {
    return (videos || []).filter(v => v.analysis_status === 'processing');
  },
  formatFileSize(bytes) {
    if (!bytes) return '0 B';
    const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return Math.round((bytes / Math.pow(1024, i)) * 100) / 100 + ' ' + sizes[i];
  },
};

// -------------------------
// 진행률 모니터
// -------------------------
const advancedProgressMonitor = {
  async monitorAdvancedProgress(videoId, callback) {
    const pollInterval = 2000;
    const tick = async () => {
      try {
        const status = await videoAnalysisService.getAnalysisStatus(videoId);
        callback?.({
          progress: status.progress ?? 0,
          step: status.currentStep || '분석 중',
          currentFeature: status.currentFeature || '',
          completedFeatures: status.completedFeatures || [],
          totalFeatures: status.totalFeatures || 4,
          processedFrames: status.processedFrames || 0,
          totalFrames: status.totalFrames || 0,
        });
        if (status.status === 'processing' && (status.progress ?? 0) < 100) {
          setTimeout(tick, pollInterval);
        }
      } catch (err) {
        console.error('진행률 모니터링 오류:', err);
        callback?.({ error: err.message });
      }
    };
    tick();
  },
};

// -------------------------
// exports
// -------------------------
export {
  API_BASE_URL,
  getFrameImageUrl,    // 하위호환
  getClipUrl,          // 하위호환
  sendVideoChatMessage, // 하위호환(= videoAnalysisService.chat)
  videoAnalysisService,
  deleteUtils,
  advancedProgressMonitor,
};