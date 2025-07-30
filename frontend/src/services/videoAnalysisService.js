// videoAnalysisService.js - 완전한 버전 (모든 필수 함수 포함)

import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

// Create axios instance with default config
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  timeout: 120000,  // 고급 분석을 위해 타임아웃 증가
  headers: {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
  },
});

// Request interceptor with better debugging
apiClient.interceptors.request.use(
  (config) => {
    console.log(`🔍 API 요청: ${config.method?.toUpperCase()} ${config.url}`);
    console.log('📋 요청 헤더:', config.headers);
    console.log('📦 요청 데이터:', config.data);
    
    return config;
  },
  (error) => {
    console.error('❌ 요청 인터셉터 오류:', error);
    return Promise.reject(error);
  }
);

// Response interceptor with detailed error handling
apiClient.interceptors.response.use(
  (response) => {
    console.log(`✅ API 응답: ${response.config.method?.toUpperCase()} ${response.config.url} - ${response.status}`);
    console.log('📊 응답 데이터:', response.data);
    return response;
  },
  (error) => {
    console.error('❌ API 오류 상세:', {
      url: error.config?.url,
      method: error.config?.method,
      status: error.response?.status,
      statusText: error.response?.statusText,
      responseData: error.response?.data,
      requestHeaders: error.config?.headers,
      responseHeaders: error.response?.headers,
      message: error.message
    });
    
    return Promise.reject(error);
  }
);

export const videoAnalysisService = {
  // ========== 연결 및 기본 기능 ==========
  
  // 연결 테스트 함수
  testConnection: async () => {
    const testUrls = [
      `${API_BASE_URL}/api_status/`,
      `${API_BASE_URL}/videos/`,
      `${API_BASE_URL}/analysis_capabilities/`,
      `${API_BASE_URL}/`,
    ];
    
    for (const url of testUrls) {
      try {
        console.log(`🔧 테스트 중: ${url}`);
        
        const response = await fetch(url, {
          method: 'GET',
          headers: {
            'Content-Type': 'application/json',
          },
          mode: 'cors',
          credentials: 'omit',
        });
        
        console.log(`📊 ${url} 결과: ${response.status} ${response.statusText}`);
        
        if (response.ok) {
          console.log(`✅ 성공한 URL: ${url}`);
          return true;
        }
      } catch (error) {
        console.error(`❌ ${url} 실패:`, error.message);
      }
    }
    
    return false;
  },

  // API 상태 확인
  getApiStatus: async () => {
    try {
      const response = await apiClient.get('/api_status/');
      return response.data;
    } catch (error) {
      throw new Error(`API 상태 확인 실패: ${error.response?.data?.error || error.message}`);
    }
  },

  // ========== 시스템 분석 기능 ==========

  // 시스템 분석 기능 확인
  getAnalysisCapabilities: async () => {
    try {
      console.log('🔧 시스템 분석 기능 확인 중...');
      
      const response = await apiClient.get('/analysis_capabilities/');
      
      console.log('✅ 분석 기능 확인 성공:', response.data);
      return response.data;
    } catch (error) {
      console.error('❌ 분석 기능 확인 실패:', error);
      
      // 기본값 반환
      return {
        object_detection: true,
        clip_analysis: false,
        ocr: false,
        vqa: false,
        scene_graph: false,
        enhanced_caption: true,
        analysis_types: {
          basic: { available: true },
          enhanced: { available: false },
          comprehensive: { available: false }
        }
      };
    }
  },

  // ========== 비디오 목록 및 상태 ==========

  // 비디오 목록 조회
  getVideoList: async () => {
    try {
      console.log('📡 비디오 목록 요청 시작...');
      
      // 먼저 연결 테스트
      const isConnected = await videoAnalysisService.testConnection();
      if (!isConnected) {
        throw new Error('서버에 연결할 수 없습니다. 서버가 실행 중인지 확인해주세요.');
      }
      
      const response = await apiClient.get('/videos/');
      
      console.log('✅ 비디오 목록 조회 성공:', response.data);
      
      // 분석 중인 비디오들의 진행률 정보 추가 조회
      const videos = response.data.videos || [];
      const processingVideos = videos.filter(v => v.analysis_status === 'processing');
      
      if (processingVideos.length > 0) {
        console.log(`🔄 ${processingVideos.length}개 비디오의 진행률 조회 중...`);
        const progressData = await videoAnalysisService.getMultipleAnalysisProgress(
          processingVideos.map(v => v.id)
        );
        
        // 진행률 정보를 비디오 데이터에 병합
        videos.forEach(video => {
          if (progressData[video.id]) {
            video.progress_info = progressData[video.id];
          }
        });
      }
      
      return response.data;
    } catch (error) {
      console.error('❌ 비디오 목록 조회 실패:', error);
      
      // 구체적인 에러 메시지 제공
      let errorMessage = '비디오 목록을 불러올 수 없습니다.';
      
      if (error.response?.status === 403) {
        errorMessage = '권한이 없습니다. Django 서버 설정을 확인해주세요.';
      } else if (error.response?.status === 404) {
        errorMessage = 'API 엔드포인트를 찾을 수 없습니다. URL을 확인해주세요.';
      } else if (error.response?.status === 500) {
        errorMessage = '서버 내부 오류가 발생했습니다.';
      } else if (error.code === 'ECONNREFUSED') {
        errorMessage = '서버에 연결할 수 없습니다. 서버가 실행 중인지 확인해주세요.';
      }
      
      throw new Error(`${errorMessage} (${error.response?.status || error.code})`);
    }
  },

  // ========== 분석 상태 및 진행률 ==========

  // 분석 상태 확인
  getAnalysisStatus: async (videoId) => {
    try {
      console.log(`📊 분석 상태 확인: ID=${videoId}`);
      
      const response = await apiClient.get(`/analysis_status/${videoId}/`);
      
      console.log('✅ 분석 상태 조회 성공:', response.data);
      return response.data;
    } catch (error) {
      console.error('❌ 분석 상태 조회 실패:', error);
      
      let errorMessage = '분석 상태를 확인할 수 없습니다.';
      
      if (error.response?.status === 404) {
        errorMessage = '비디오를 찾을 수 없습니다.';
      } else if (error.response?.data?.error) {
        errorMessage = error.response.data.error;
      }
      
      throw new Error(`${errorMessage} (${error.response?.status || error.code})`);
    }
  },

  // 진행률 전용 조회 함수
//   getAnalysisProgress: async (videoId) => {
//     try {
//       console.log(`🔄 분석 진행률 확인: ID=${videoId}`);
      
//       const response = await apiClient.get(`/analysis_progress/${videoId}/`);
      
//       console.log('✅ 분석 진행률 조회 성공:', response.data);
//       return response.data;
//     } catch (error) {
//       console.error('❌ 분석 진행률 조회 실패:', error);
      
//       // 진행 중인 분석이 없는 경우는 오류가 아님
//       if (error.response?.status === 404) {
//         return null;
//       }
      
//       let errorMessage = '분석 진행률을 확인할 수 없습니다.';
      
//       if (error.response?.data?.error) {
//         errorMessage = error.response.data.error;
//       }
      
//       throw new Error(`${errorMessage} (${error.response?.status || error.code})`);
//     }
//   },

  // 여러 비디오의 진행률을 한번에 조회
  getMultipleAnalysisProgress: async (videoIds) => {
    try {
      const progressPromises = videoIds.map(id => 
        videoAnalysisService.getAnalysisProgress(id).catch(() => null)
      );
      
      const progressResults = await Promise.all(progressPromises);
      
      const progressMap = {};
      videoIds.forEach((id, index) => {
        if (progressResults[index]) {
          progressMap[id] = progressResults[index];
        }
      });
      
      return progressMap;
    } catch (error) {
      console.error('❌ 다중 진행률 조회 실패:', error);
      return {};
    }
  },

  // ========== 비디오 분석 ==========

  // 고급 비디오 분석 시작
  analyzeVideoEnhanced: async (videoId, options = {}) => {
    try {
      const {
        analysisType = 'enhanced',
        analysisConfig = {},
        enhancedAnalysis = true
      } = options;
      
      console.log(`🚀 고급 비디오 분석 시작: ID=${videoId}, 타입=${analysisType}`);
      console.log('📋 분석 설정:', analysisConfig);
      
      const response = await apiClient.post('/analyze_video_enhanced/', {
        video_id: videoId,
        analysisType: analysisType,
        analysisConfig: analysisConfig,
        enhancedAnalysis: enhancedAnalysis,
      });
      
      console.log('✅ 고급 비디오 분석 시작 성공:', response.data);
      return response.data;
    } catch (error) {
      console.error('❌ 고급 비디오 분석 시작 실패:', error);
      
      let errorMessage = '고급 비디오 분석을 시작할 수 없습니다.';
      
      if (error.response?.status === 404) {
        errorMessage = '비디오를 찾을 수 없습니다.';
      } else if (error.response?.status === 400) {
        errorMessage = error.response.data.error || '잘못된 요청입니다.';
      } else if (error.response?.status === 500) {
        errorMessage = '서버 내부 오류가 발생했습니다.';
      } else if (error.response?.data?.error) {
        errorMessage = error.response.data.error;
      }
      
      throw new Error(`${errorMessage} (${error.response?.status || error.code})`);
    }
  },

  // 기존 분석 시작 (호환성 유지)
  analyzeVideo: async (videoId, enableEnhanced = true) => {
    try {
      console.log(`📹 기본 비디오 분석 시작: ID=${videoId}, Enhanced=${enableEnhanced}`);
      
      // 고급 분석이 활성화된 경우 새로운 API 사용
      if (enableEnhanced) {
        return await videoAnalysisService.analyzeVideoEnhanced(videoId, {
          analysisType: 'enhanced',
          enhancedAnalysis: true
        });
      }
      
      // 기본 분석
      const response = await apiClient.post('/analyze_video/', {
        video_id: videoId,
        enable_enhanced_analysis: enableEnhanced,
      });
      
      console.log('✅ 기본 비디오 분석 시작 성공:', response.data);
      return response.data;
    } catch (error) {
      console.error('❌ 비디오 분석 시작 실패:', error);
      throw new Error(`분석 시작 실패: ${error.response?.data?.error || error.message}`);
    }
  },

  // ========== 비디오 채팅 및 검색 ==========

// ========== 비디오 채팅 및 검색 ==========

  // 고급 비디오 채팅
  sendVideoChatMessage: async (message, videoId = null) => {
    try {
      console.log('💬 고급 비디오 채팅 요청:', message);
      
      const response = await apiClient.post('/video/chat/enhanced/', {
        message,
        video_id: videoId,
      });
      
      console.log('✅ 고급 비디오 채팅 성공:', response.data);
      return response.data;
    } catch (error) {
      console.error('❌ 고급 비디오 채팅 실패:', error);
      
      // 고급 채팅 실패시 기본 채팅으로 fallback
      try {
        const fallbackResponse = await apiClient.post('/video/chat/', {
          message,
          video_id: videoId,
        });
        
        console.log('✅ 기본 비디오 채팅 fallback 성공');
        return {
          ...fallbackResponse.data,
          fallback_used: true,
          fallback_reason: '고급 분석 기능 일시 사용 불가'
        };
      } catch (fallbackError) {
        throw new Error(`비디오 채팅 요청 실패: ${error.response?.data?.error || error.message}`);
      }
    }
  },

  // ✅ 새로 추가: 고급 비디오 검색 (바운딩 박스 정보 포함)
  searchVideoAdvanced: async (videoId, query, searchOptions = {}) => {
    try {
      console.log('🔍 고급 비디오 검색 요청:', { videoId, query, searchOptions });
      
      const response = await apiClient.post('/videos/search/advanced/', {
        video_id: videoId,
        query: query,
        search_options: searchOptions
      });
      
      console.log('✅ 고급 비디오 검색 성공:', response.data);
      return response.data;
    } catch (error) {
      console.error('❌ 고급 비디오 검색 실패:', error);
      
      // 고급 검색 실패시 기본 채팅으로 fallback
      try {
        console.log('🔄 기본 채팅으로 fallback 시도');
        const fallbackResponse = await videoAnalysisService.sendVideoChatMessage(
          `찾아줘 ${query}`,
          videoId
        );
        
        console.log('✅ 기본 채팅 fallback 성공');
        return {
          ...fallbackResponse,
          search_type: 'fallback_search',
          fallback_used: true,
          fallback_reason: '고급 검색 기능 일시 사용 불가'
        };
      } catch (fallbackError) {
        console.error('❌ Fallback도 실패:', fallbackError);
        throw new Error(`비디오 검색 실패: ${error.response?.data?.error || error.message}`);
      }
    }
  },

  // 객체별 검색
  searchByObject: async (objectType, videoId = null) => {
    try {
      console.log('🎯 객체별 검색:', { objectType, videoId });
      
      const params = new URLSearchParams();
      params.append('object', objectType);
      if (videoId) {
        params.append('video_id', videoId);
      }
      
      const response = await apiClient.get(`/search/objects/?${params.toString()}`);
      
      console.log('✅ 객체별 검색 성공:', response.data);
      return response.data;
    } catch (error) {
      console.error('❌ 객체별 검색 실패:', error);
      throw new Error(`객체별 검색 실패: ${error.response?.data?.error || error.message}`);
    }
  },

  // 텍스트 검색 (OCR 기반)
  searchByText: async (searchText, videoId = null) => {
    try {
      console.log('📝 텍스트 검색:', { searchText, videoId });
      
      const params = new URLSearchParams();
      params.append('text', searchText);
      if (videoId) {
        params.append('video_id', videoId);
      }
      
      const response = await apiClient.get(`/search/text/?${params.toString()}`);
      
      console.log('✅ 텍스트 검색 성공:', response.data);
      return response.data;
    } catch (error) {
      console.error('❌ 텍스트 검색 실패:', error);
      throw new Error(`텍스트 검색 실패: ${error.response?.data?.error || error.message}`);
    }
  },

  // 씬 타입별 검색
  searchByScene: async (sceneType, videoId = null) => {
    try {
      console.log('🎬 씬 타입별 검색:', { sceneType, videoId });
      
      const params = new URLSearchParams();
      params.append('scene', sceneType);
      if (videoId) {
        params.append('video_id', videoId);
      }
      
      const response = await apiClient.get(`/search/scenes/?${params.toString()}`);
      
      console.log('✅ 씬 타입별 검색 성공:', response.data);
      return response.data;
    } catch (error) {
      console.error('❌ 씬 타입별 검색 실패:', error);
      throw new Error(`씬 타입별 검색 실패: ${error.response?.data?.error || error.message}`);
    }
  },

  // ========== 비디오 업로드 ==========

  // 비디오 업로드
  uploadVideo: async (videoFile, onProgress) => {
    try {
      const formData = new FormData();
      formData.append('video', videoFile);

      const response = await apiClient.post('/upload_video/', formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
        onUploadProgress: (progressEvent) => {
          if (onProgress) {
            const percentCompleted = Math.round(
              (progressEvent.loaded * 100) / progressEvent.total
            );
            onProgress(percentCompleted);
          }
        },
        timeout: 600000,  // 10분으로 증가
      });
      return response.data;
    } catch (error) {
      throw new Error(`비디오 업로드 실패: ${error.response?.data?.error || error.message}`);
    }
  },

  // ========== 씬 관련 ==========

  // 씬 목록 조회
  getScenes: async (videoId) => {
    try {
      console.log(`🎬 씬 목록 조회: ID=${videoId}`);
      
      const response = await apiClient.get(`/scenes/${videoId}/`);
      
      console.log('✅ 씬 목록 조회 성공:', response.data);
      return response.data;
    } catch (error) {
      console.error('❌ 씬 목록 조회 실패:', error);
      throw new Error(`씬 목록을 가져올 수 없습니다: ${error.response?.data?.error || error.message}`);
    }
  },

  // 프레임 이미지 URL 생성
  getFrameImageUrl: (videoId, frameNumber, isAnnotated = false, targetClass = '') => {
    const baseUrl = `${API_BASE_URL}/frame/${videoId}/${frameNumber}/`;
    
    if (isAnnotated) {
      const params = new URLSearchParams();
      if (targetClass) {
        params.append('class', targetClass);
      }
      return `${baseUrl}annotated/?${params.toString()}`;
    }
    return baseUrl;
  },

  // ========== 삭제 관련 함수들 ==========

  // 개별 비디오 삭제 (여러 엔드포인트 시도)
  deleteVideo: async (videoId) => {
    try {
      console.log(`🗑️ 비디오 삭제 요청: ID=${videoId}`);
      
      // videoId 유효성 검사
      if (!videoId || videoId === 'undefined' || videoId === 'null') {
        throw new Error('유효하지 않은 비디오 ID입니다.');
      }
      
      // 먼저 비디오 존재 여부 확인
      try {
        await apiClient.get(`/videos/${videoId}/`);
      } catch (checkError) {
        if (checkError.response?.status === 404) {
          throw new Error('해당 비디오가 존재하지 않습니다.');
        }
      }
      
      // 여러 가능한 엔드포인트 시도
      const deleteEndpoints = [
        `/videos/${videoId}/delete/`,  // 현재 사용 중인 엔드포인트
        `/videos/${videoId}/`,         // RESTful 방식
        `/delete_video/${videoId}/`,   // 대안 방식
        `/api/videos/${videoId}/delete/`, // API prefix 있는 경우
      ];
      
      let lastError = null;
      
      for (const endpoint of deleteEndpoints) {
        try {
          console.log(`🔄 삭제 엔드포인트 시도: ${endpoint}`);
          const response = await apiClient.delete(endpoint);
          
          console.log('✅ 비디오 삭제 성공:', response.data);
          return response.data;
          
        } catch (error) {
          lastError = error;
          console.warn(`❌ ${endpoint} 실패:`, error.response?.status);
          
          // 404가 아닌 다른 오류면 즉시 중단
          if (error.response?.status && error.response.status !== 404) {
            throw error;
          }
        }
      }
      
      // 모든 엔드포인트 실패시
      throw lastError;
      
    } catch (error) {
      console.error('❌ 비디오 삭제 실패:', error);
      
      let errorMessage = '비디오를 삭제할 수 없습니다.';
      
      if (error.response?.status === 404) {
        errorMessage = '해당 비디오를 찾을 수 없습니다. 이미 삭제되었거나 존재하지 않습니다.';
      } else if (error.response?.status === 400) {
        errorMessage = error.response.data.error || '삭제할 수 없는 비디오입니다.';
      } else if (error.response?.status === 403) {
        errorMessage = '비디오 삭제 권한이 없습니다.';
      } else if (error.response?.status === 409) {
        errorMessage = '비디오가 현재 사용 중이어서 삭제할 수 없습니다.';
      } else if (error.response?.data?.error) {
        errorMessage = error.response.data.error;
      } else if (error.message.includes('유효하지 않은')) {
        errorMessage = error.message;
      }
      
      throw new Error(`${errorMessage} (${error.response?.status || error.code})`);
    }
  },

  // 일괄 삭제
  batchDeleteVideos: async (videoIds) => {
    try {
      console.log(`🗑️ 일괄 비디오 삭제 요청: IDs=${videoIds}`);
      
      if (!Array.isArray(videoIds) || videoIds.length === 0) {
        throw new Error('삭제할 비디오 ID 목록이 유효하지 않습니다.');
      }
      
      // 유효하지 않은 ID 필터링
      const validIds = videoIds.filter(id => id && id !== 'undefined' && id !== 'null');
      
      if (validIds.length === 0) {
        throw new Error('유효한 비디오 ID가 없습니다.');
      }
      
      console.log(`📊 유효한 ID: ${validIds.length}/${videoIds.length}`);
      
      // 여러 엔드포인트 시도
      const batchEndpoints = [
        '/videos/batch_delete/',
        '/videos/bulk_delete/',
        '/api/videos/batch_delete/',
        '/delete_videos/',
      ];
      
      let lastError = null;
      
      for (const endpoint of batchEndpoints) {
        try {
          console.log(`🔄 일괄 삭제 엔드포인트 시도: ${endpoint}`);
          
          const response = await apiClient.post(endpoint, {
            video_ids: validIds
          });
          
          console.log('✅ 일괄 비디오 삭제 성공:', response.data);
          return response.data;
          
        } catch (error) {
          lastError = error;
          console.warn(`❌ ${endpoint} 실패:`, error.response?.status);
          
          if (error.response?.status && error.response.status !== 404) {
            throw error;
          }
        }
      }
      
      // 일괄 삭제 API가 없는 경우 개별 삭제로 fallback
      console.log('🔄 일괄 삭제 API 없음, 개별 삭제로 fallback');
      return await videoAnalysisService.fallbackBatchDelete(validIds);
      
    } catch (error) {
      console.error('❌ 일괄 비디오 삭제 실패:', error);
      
      let errorMessage = '비디오들을 삭제할 수 없습니다.';
      
      if (error.response?.data?.error) {
        errorMessage = error.response.data.error;
      } else if (error.message.includes('유효하지 않은') || error.message.includes('유효한')) {
        errorMessage = error.message;
      }
      
      throw new Error(`${errorMessage} (${error.response?.status || error.code})`);
    }
  },

  // 개별 삭제로 일괄 처리 (fallback)
  fallbackBatchDelete: async (videoIds) => {
    const results = {
      total_requested: videoIds.length,
      total_deleted: 0,
      total_failed: 0,
      deleted_videos: [],
      failed_videos: [],
      errors: []
    };
    
    console.log(`🔄 ${videoIds.length}개 비디오 개별 삭제 시작`);
    
    for (const videoId of videoIds) {
      try {
        const deleteResult = await videoAnalysisService.deleteVideo(videoId);
        
        results.total_deleted++;
        results.deleted_videos.push({
          id: videoId,
          success: true,
          message: '삭제 완료'
        });
        
        console.log(`✅ 비디오 ${videoId} 삭제 성공`);
        
        // 서버 부하 방지를 위한 지연
        await new Promise(resolve => setTimeout(resolve, 200));
        
      } catch (error) {
        results.total_failed++;
        results.failed_videos.push({
          id: videoId,
          success: false,
          error: error.message
        });
        results.errors.push(`ID ${videoId}: ${error.message}`);
        
        console.error(`❌ 비디오 ${videoId} 삭제 실패:`, error.message);
      }
    }
    
    console.log(`📊 일괄 삭제 완료: 성공 ${results.total_deleted}, 실패 ${results.total_failed}`);
    return results;
  },

  // 저장 공간 정리
  cleanupStorage: async () => {
    try {
      console.log('🧹 저장 공간 정리 요청');
      
      const response = await apiClient.post('/videos/cleanup/');
      
      console.log('✅ 저장 공간 정리 성공:', response.data);
      return response.data;
    } catch (error) {
      console.error('❌ 저장 공간 정리 실패:', error);
      
      let errorMessage = '저장 공간을 정리할 수 없습니다.';
      
      if (error.response?.data?.error) {
        errorMessage = error.response.data.error;
      }
      
      throw new Error(`${errorMessage} (${error.response?.status || error.code})`);
    }
  },

  // 비디오 존재 여부 확인
  checkVideoExists: async (videoId) => {
    try {
      const response = await apiClient.get(`/videos/${videoId}/`);
      return {
        exists: true,
        video: response.data
      };
    } catch (error) {
      if (error.response?.status === 404) {
        return {
          exists: false,
          error: '비디오를 찾을 수 없습니다.'
        };
      }
      throw error;
    }
  },

  // 삭제 가능 여부 확인
  canDeleteVideo: (video) => {
    // 비디오 객체 유효성 검사
    if (!video || !video.id) {
      return {
        canDelete: false,
        reason: '유효하지 않은 비디오 정보입니다.'
      };
    }
    
    // 분석 중인 비디오는 삭제 불가
    if (video.analysis_status === 'processing') {
      return {
        canDelete: false,
        reason: '분석 중인 비디오는 삭제할 수 없습니다.'
      };
    }
    
    // 업로드 중인 비디오는 삭제 불가
    if (video.upload_status === 'uploading') {
      return {
        canDelete: false,
        reason: '업로드 중인 비디오는 삭제할 수 없습니다.'
      };
    }
    
    return {
      canDelete: true,
      reason: null
    };
  },

  // 확인 메시지 생성
  getDeleteConfirmMessage: (video) => {
    if (!video) {
      return '비디오 정보를 확인할 수 없습니다.';
    }
    
    const analysisInfo = video.is_analyzed ? 
      '\n(분석 결과와 관련 데이터도 함께 삭제됩니다)' : 
      '';
    
    const sizeInfo = video.file_size ? 
      `\n파일 크기: ${deleteUtils.formatFileSize(video.file_size)}` : 
      '';
    
    return `정말로 비디오 "${video.original_name}"을(를) 삭제하시겠습니까?${analysisInfo}${sizeInfo}`;
  },

  // 일괄 삭제 확인 메시지
  getBatchDeleteConfirmMessage: (videos) => {
    if (!Array.isArray(videos) || videos.length === 0) {
      return '삭제할 비디오가 선택되지 않았습니다.';
    }
    
    const totalCount = videos.length;
    const analyzedCount = videos.filter(v => v.is_analyzed).length;
    const processingCount = videos.filter(v => v.analysis_status === 'processing').length;
    const totalSize = videos.reduce((sum, v) => sum + (v.file_size || 0), 0);
    
    let message = `총 ${totalCount}개의 비디오를 삭제하시겠습니까?`;
    
    if (totalSize > 0) {
      message += `\n총 용량: ${deleteUtils.formatFileSize(totalSize)}`;
    }
    
    if (analyzedCount > 0) {
      message += `\n(${analyzedCount}개의 분석 결과도 함께 삭제됩니다)`;
    }
    
    if (processingCount > 0) {
      message += `\n⚠️ ${processingCount}개의 비디오는 분석 중이므로 삭제되지 않습니다.`;
    }
    
    return message;
  }
};

// ========== 유틸리티 함수들 ==========

// 삭제 유틸리티
export const deleteUtils = {
  // 삭제 가능한 비디오 필터링
  getSelectableVideos: (videos) => {
    if (!Array.isArray(videos)) return [];
    
    return videos.filter(video => {
      const { canDelete } = videoAnalysisService.canDeleteVideo(video);
      return canDelete;
    });
  },

  // 삭제 불가능한 비디오 필터링
  getNonSelectableVideos: (videos) => {
    if (!Array.isArray(videos)) return [];
    
    return videos.filter(video => {
      const { canDelete } = videoAnalysisService.canDeleteVideo(video);
      return !canDelete;
    });
  },

  // 파일 크기 포맷팅
  formatFileSize: (bytes) => {
    if (!bytes || bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  },

  // 삭제된 용량 계산
  calculateFreedSpace: (videos) => {
    if (!Array.isArray(videos)) return 0;
    
    return videos.reduce((total, video) => {
      return total + (video.file_size || 0);
    }, 0);
  },

  // 삭제 상태 추적
  createDeleteTracker: () => {
    const tracker = {
      total: 0,
      completed: 0,
      failed: 0,
      results: [],
      
      addResult: (videoId, success, message) => {
        tracker.results.push({
          videoId,
          success,
          message,
          timestamp: new Date().toISOString()
        });
        
        if (success) {
          tracker.completed++;
        } else {
          tracker.failed++;
        }
      },
      
      getProgress: () => {
        return tracker.total > 0 ? 
          Math.round((tracker.completed + tracker.failed) / tracker.total * 100) : 
          0;
      },
      
      getSummary: () => ({
        total: tracker.total,
        completed: tracker.completed,
        failed: tracker.failed,
        progress: tracker.getProgress(),
        results: tracker.results
      })
    };
    
    return tracker;
  }
};

// 고급 진행률 모니터링 헬퍼
export const advancedProgressMonitor = {
  // 고급 분석 진행률 모니터링
  startEnhancedMonitoring: (videoId, onUpdate, interval = 2000) => {
    let intervalId;
    
    const checkProgress = async () => {
      try {
        const progress = await videoAnalysisService.getAnalysisProgress(videoId);
        if (progress) {
          // 고급 정보 포함하여 업데이트
          onUpdate({
            ...progress,
            analysis_type: progress.analysisType || 'enhanced',
            current_feature: progress.currentFeature || '',
            completed_features: progress.completedFeatures || [],
            total_features: progress.totalFeatures || 4,
            feature_progress: progress.completedFeatures ? 
              (progress.completedFeatures.length / progress.totalFeatures) * 100 : 0
          });
          
          // 완료되면 모니터링 중지
          if (progress.progress >= 100) {
            clearInterval(intervalId);
            onUpdate({ 
              ...progress, 
              completed: true, 
              progress: 100,
              analysis_complete: true
            });
          }
        } else {
          // 진행 중인 분석이 없으면 모니터링 중지
          clearInterval(intervalId);
          onUpdate({ 
            completed: true, 
            progress: 100,
            no_analysis_running: true
          });
        }
      } catch (error) {
        console.error('고급 진행률 모니터링 오류:', error);
        // 오류가 발생해도 계속 모니터링 (서버가 일시적으로 응답하지 않을 수 있음)
      }
    };
    
    // 즉시 한 번 실행
    checkProgress();
    
    // 주기적으로 실행
    intervalId = setInterval(checkProgress, interval);
    
    // 모니터링 중지 함수 반환
    return () => {
      if (intervalId) {
        clearInterval(intervalId);
      }
    };
  }
};

export default videoAnalysisService;