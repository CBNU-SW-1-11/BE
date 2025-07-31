// pages/VideoAnalysisPage.js - 삭제 기능 추가 버전
import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { videoAnalysisService, deleteUtils, advancedProgressMonitor } from '../services/videoAnalysisService';
import './VideoAnalysisPage.css';

const VideoAnalysisPage = () => {
  const [deleteTracker, setDeleteTracker] = useState(null);
  const [showDeleteProgress, setShowDeleteProgress] = useState(false);
  const navigate = useNavigate();
  const [videos, setVideos] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [refreshing, setRefreshing] = useState(false);
  const [selectedVideo, setSelectedVideo] = useState(null);
  const [videoDetails, setVideoDetails] = useState(null);
  const [analysisProgress, setAnalysisProgress] = useState({});
  const [systemCapabilities, setSystemCapabilities] = useState({});
  
  // 삭제 관련 상태 추가
  const [selectedVideos, setSelectedVideos] = useState([]);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [deleteTarget, setDeleteTarget] = useState(null); // 'single' | 'batch'
  const [videoToDelete, setVideoToDelete] = useState(null);
  const [deleting, setDeleting] = useState(false);
  const [showBatchActions, setShowBatchActions] = useState(false);
  const [cleanupResults, setCleanupResults] = useState(null);
  const [showCleanupDialog, setShowCleanupDialog] = useState(false);
  
  // 기존 상태들
  const [analysisInsights, setAnalysisInsights] = useState({});
  const [showAdvancedDetails, setShowAdvancedDetails] = useState({});

  useEffect(() => {
    loadVideos();
    loadSystemCapabilities();
    
    const hasProcessing = videos.some(v => v.analysis_status === 'processing');
    const interval = setInterval(loadVideos, hasProcessing ? 2000 : 5000);
    return () => clearInterval(interval);
  }, [videos.length]);

  const loadSystemCapabilities = async () => {
    try {
      const capabilities = await videoAnalysisService.getAnalysisCapabilities();
      setSystemCapabilities(capabilities);
    } catch (error) {
      console.error('시스템 기능 확인 실패:', error);
    }
  };
 const loadVideos = async () => {
  try {
    console.log('📡 비디오 목록 로딩 시작...');
    setError(null);
    
    const data = await videoAnalysisService.getVideoList();
    console.log('📊 서버에서 받은 비디오 목록:', data.videos?.length, '개');
    console.log('📋 비디오 ID 목록:', data.videos?.map(v => `${v.id}:${v.original_name}`) || []);
    
    setVideos(data.videos || []);
    
      console.log(`📋 비디오 목록 업데이트: ${data.videos?.length || 0}개`);
      
      const processingVideos = data.videos?.filter(v => v.analysis_status === 'processing') || [];
      await updateAnalysisProgress(processingVideos);
      
      if (data.analysis_capabilities) {
        setSystemCapabilities(data.analysis_capabilities);
      }
      
    } catch (error) {
      console.error('❌ 비디오 목록 조회 실패:', error);
      setError(error.message);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };
  
  const updateAnalysisProgress = async (processingVideos) => {
    const progressData = {};
    
    for (const video of processingVideos) {
      try {
        const status = await videoAnalysisService.getAnalysisStatus(video.id);
        progressData[video.id] = {
          progress: status.progress || 0,
          currentStep: status.currentStep || '분석 준비중',
          estimatedTime: status.estimatedTime || null,
          processedFrames: status.processedFrames || 0,
          totalFrames: status.totalFrames || 0,
          startTime: status.startTime || null,
          analysisType: status.analysisType || 'enhanced',
          currentFeature: status.currentFeature || '',
          completedFeatures: status.completedFeatures || [],
          totalFeatures: status.totalFeatures || 4
        };
      } catch (error) {
        console.warn(`비디오 ${video.id} 진행 상황 조회 실패:`, error);
      }
    }
    
    setAnalysisProgress(progressData);
  };

  // ========== 삭제 관련 함수들 ==========

  const handleSelectVideo = (videoId, checked) => {
    setSelectedVideos(prev => {
      if (checked) {
        return [...prev, videoId];
      } else {
        return prev.filter(id => id !== videoId);
      }
    });
  };

  const handleSelectAll = (checked) => {
    if (checked) {
      const selectableVideos = deleteUtils.getSelectableVideos(videos);
      setSelectedVideos(selectableVideos.map(v => v.id));
    } else {
      setSelectedVideos([]);
    }
  };

const handleDeleteVideo = async (video) => {
  try {
    console.log('🔍 삭제 프로세스 디버깅 시작');
    console.log('1️⃣ 삭제 대상 비디오:', video);
    
    // 삭제 전 상태 확인
    console.log('2️⃣ 삭제 전 비디오 목록 개수:', videos.length);
    console.log('3️⃣ 현재 선택된 비디오들:', selectedVideos);
    
    // 실제 삭제 API 호출
    console.log('4️⃣ 삭제 API 호출 시작...');
    const deleteResponse = await videoAnalysisService.deleteVideo(video.id);
    console.log('5️⃣ 삭제 API 응답:', deleteResponse);
    
    // 삭제 후 즉시 확인
    console.log('6️⃣ 삭제 후 검증 시작...');
    try {
      const checkResponse = await fetch(`http://localhost:8000/videos/${video.id}/`, {
        method: 'GET',
        headers: { 'Content-Type': 'application/json' }
      });
      
      if (checkResponse.status === 404) {
        console.log('✅ 검증 성공: 비디오가 실제로 삭제됨 (404)');
      } else if (checkResponse.ok) {
        const stillExists = await checkResponse.json();
        console.error('❌ 검증 실패: 비디오가 여전히 존재함:', stillExists);
        alert('⚠️ 서버에서 실제로 삭제되지 않았습니다!\n\nDjango 서버 로그를 확인해주세요.');
        return; // 삭제되지 않았으므로 UI 업데이트 하지 않음
      }
    } catch (checkError) {
      if (checkError.message.includes('404')) {
        console.log('✅ 검증 성공: 네트워크 404 오류로 확인됨');
      } else {
        console.warn('⚠️ 검증 중 오류:', checkError);
      }
    }
    
    // UI 업데이트
    console.log('7️⃣ UI 업데이트 시작...');
    setVideos(prev => {
      const newVideos = prev.filter(v => v.id !== video.id);
      console.log('8️⃣ UI 업데이트 완료:', {
        이전: prev.length,
        이후: newVideos.length,
        삭제된비디오ID: video.id
      });
      return newVideos;
    });
    
    setSelectedVideos(prev => prev.filter(id => id !== video.id));
    
    alert(`✅ 비디오 "${video.original_name}"이(가) 성공적으로 삭제되었습니다.`);
    
    // 3초 후 목록 새로고침하여 실제 상태 확인
    console.log('9️⃣ 3초 후 목록 새로고침 예약...');
    setTimeout(async () => {
      console.log('🔄 목록 새로고침으로 최종 검증...');
      await loadVideos();
      console.log('🔍 새로고침 후 비디오 목록 개수:', videos.length);
    }, 3000);
    
  } catch (error) {
    console.error('❌ 삭제 프로세스 오류:', error);
    alert(`삭제 실패: ${error.message}`);
  }
};
const handleBatchDelete = () => {
    if (selectedVideos.length === 0) {
      alert('삭제할 비디오를 선택해주세요.');
      return;
    }
    
    // 선택된 비디오들의 유효성 검사
    const selectedVideoObjects = videos.filter(v => selectedVideos.includes(v.id));
    const validVideos = selectedVideoObjects.filter(v => v && v.id);
    
    if (validVideos.length === 0) {
      alert('유효한 비디오가 선택되지 않았습니다.');
      return;
    }
    
    if (validVideos.length !== selectedVideos.length) {
      console.warn(`선택된 ${selectedVideos.length}개 중 ${validVideos.length}개만 유효함`);
    }
    
    setDeleteTarget('batch');
    setShowDeleteConfirm(true);
  };
const confirmDelete = async () => {
    try {
      setDeleting(true);
      setShowDeleteConfirm(false);
      
      if (deleteTarget === 'single' && videoToDelete) {
        await handleSingleDelete(videoToDelete);
      } else if (deleteTarget === 'batch' && selectedVideos.length > 0) {
        await handleBatchDeleteProcess();
      }
      
    } catch (error) {
      console.error('삭제 실행 오류:', error);
      alert(`삭제 실행 중 오류가 발생했습니다: ${error.message}`);
    } finally {
      setDeleting(false);
      setVideoToDelete(null);
      setDeleteTarget(null);
    }
  };
  const handleSingleDelete = async (video) => {
    try {
      console.log(`🗑️ 단일 비디오 삭제 시작: ${video.original_name} (ID: ${video.id})`);
      
      // 1단계: 실제 서버에서 삭제 실행
      const deleteResult = await videoAnalysisService.deleteVideo(video.id);
      console.log('✅ 서버 삭제 응답:', deleteResult);
      
      // 2단계: 삭제 후 실제로 삭제되었는지 검증
      let isActuallyDeleted = false;
      let verificationAttempts = 0;
      const maxVerificationAttempts = 3;
      
      while (!isActuallyDeleted && verificationAttempts < maxVerificationAttempts) {
        verificationAttempts++;
        console.log(`🔍 삭제 검증 시도 ${verificationAttempts}/${maxVerificationAttempts}`);
        
        // 잠시 대기 후 검증 (서버 처리 시간 고려)
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        try {
          const existsCheck = await videoAnalysisService.checkVideoExists(video.id);
          
          if (!existsCheck.exists) {
            isActuallyDeleted = true;
            console.log('✅ 삭제 검증 성공: 비디오가 실제로 삭제됨');
          } else {
            console.warn(`⚠️ 삭제 검증 실패 (시도 ${verificationAttempts}): 비디오가 여전히 존재함`);
          }
        } catch (checkError) {
          if (checkError.message.includes('404') || checkError.message.includes('찾을 수 없습니다')) {
            isActuallyDeleted = true;
            console.log('✅ 삭제 검증 성공: 404 오류로 확인됨');
          } else {
            console.warn('⚠️ 삭제 검증 중 오류:', checkError.message);
          }
        }
      }
      
      if (isActuallyDeleted) {
        // 3단계: 실제 삭제가 확인된 경우에만 UI에서 제거
        setVideos(prev => {
          const newVideos = prev.filter(v => v.id !== video.id);
          console.log(`📋 UI 업데이트: ${prev.length} → ${newVideos.length}개 비디오`);
          return newVideos;
        });
        setSelectedVideos(prev => prev.filter(id => id !== video.id));
        
        alert(`비디오 "${video.original_name}"이(가) 성공적으로 삭제되었습니다.`);
        
        // 4단계: 추가 검증을 위해 목록 새로고침 (지연 후)
        setTimeout(() => {
          console.log('🔄 삭제 후 목록 검증을 위한 새로고침');
          loadVideos();
        }, 2000);
        
      } else {
        // 삭제 검증 실패
        console.error('❌ 삭제 검증 실패: 서버에서 실제로 삭제되지 않음');
        alert(`삭제에 실패했습니다. 비디오가 서버에서 실제로 삭제되지 않았습니다.\n\n가능한 원인:\n- 서버 권한 문제\n- 파일 시스템 오류\n- 데이터베이스 트랜잭션 실패\n\n서버 로그를 확인해주세요.`);
        
        // 실패한 경우 목록을 즉시 새로고침하여 정확한 상태 반영
        await loadVideos();
      }
      
    } catch (error) {
      console.error('단일 비디오 삭제 실패:', error);
      
      // 404 오류의 경우 이미 삭제된 것으로 간주
      if (error.message.includes('404') || error.message.includes('찾을 수 없습니다')) {
        console.log('🔍 404 오류 - 이미 삭제된 비디오로 간주하여 UI에서 제거');
        setVideos(prev => prev.filter(v => v.id !== video.id));
        setSelectedVideos(prev => prev.filter(id => id !== video.id));
        alert(`비디오 "${video.original_name}"은(는) 이미 삭제된 상태입니다. 목록을 업데이트합니다.`);
        
        // 목록 새로고침으로 정확한 상태 확인
        setTimeout(() => loadVideos(), 1000);
      } else {
        alert(`삭제 실패: ${error.message}`);
      }
    }
  };const handleBatchDeleteProcess = async () => {
    try {
      console.log(`🗑️ 일괄 삭제 시작: ${selectedVideos.length}개 비디오`);
      
      // 삭제 진행률 추적 설정
      const tracker = deleteUtils.createDeleteTracker();
      tracker.total = selectedVideos.length;
      setDeleteTracker(tracker);
      setShowDeleteProgress(true);
      
      let actuallyDeletedIds = [];
      let failedIds = [];
      
      // 1단계: 서버에서 일괄 삭제 시도
      try {
        const result = await videoAnalysisService.batchDeleteVideos(selectedVideos);
        console.log('📊 일괄 삭제 서버 응답:', result);
        
        // 성공한 ID들 수집
        if (result.deleted_videos) {
          actuallyDeletedIds = result.deleted_videos
            .filter(v => v.success !== false)
            .map(v => v.id || v);
        }
        
        // 실패한 ID들 수집
        if (result.failed_videos) {
          failedIds = result.failed_videos.map(v => v.id || v);
        }
        
      } catch (batchError) {
        console.warn('일괄 삭제 API 실패, 개별 삭제로 fallback:', batchError.message);
        
        // 개별 삭제로 fallback
        for (const videoId of selectedVideos) {
          try {
            await videoAnalysisService.deleteVideo(videoId);
            actuallyDeletedIds.push(videoId);
            console.log(`✅ 개별 삭제 성공: ${videoId}`);
          } catch (individualError) {
            failedIds.push(videoId);
            console.error(`❌ 개별 삭제 실패: ${videoId}`, individualError.message);
          }
          
          // 진행률 업데이트
          tracker.addResult(videoId, !failedIds.includes(videoId), '처리 완료');
          setDeleteTracker({...tracker});
          
          // 서버 부하 방지
          await new Promise(resolve => setTimeout(resolve, 300));
        }
      }
      
      // 2단계: 실제 삭제 검증
      console.log(`🔍 ${actuallyDeletedIds.length}개 비디오 삭제 검증 시작`);
      
      const verifiedDeletedIds = [];
      const notActuallyDeletedIds = [];
      
      for (const videoId of actuallyDeletedIds) {
        try {
          const existsCheck = await videoAnalysisService.checkVideoExists(videoId);
          
          if (!existsCheck.exists) {
            verifiedDeletedIds.push(videoId);
            console.log(`✅ 삭제 검증 성공: ${videoId}`);
          } else {
            notActuallyDeletedIds.push(videoId);
            console.warn(`⚠️ 삭제 검증 실패: ${videoId} - 여전히 존재함`);
          }
        } catch (checkError) {
          if (checkError.message.includes('404')) {
            verifiedDeletedIds.push(videoId);
            console.log(`✅ 삭제 검증 성공 (404): ${videoId}`);
          } else {
            console.warn(`⚠️ 삭제 검증 오류: ${videoId}`, checkError.message);
            notActuallyDeletedIds.push(videoId);
          }
        }
      }
      
      // 3단계: 검증된 삭제만 UI에 반영
      if (verifiedDeletedIds.length > 0) {
        setVideos(prev => {
          const newVideos = prev.filter(v => !verifiedDeletedIds.includes(v.id));
          console.log(`📋 UI 업데이트: ${verifiedDeletedIds.length}개 비디오 제거됨`);
          return newVideos;
        });
      }
      
      setSelectedVideos([]);
      
      // 4단계: 결과 메시지 생성
      let message = `${verifiedDeletedIds.length}개의 비디오가 실제로 삭제되었습니다.`;
      
      if (notActuallyDeletedIds.length > 0) {
        message += `\n\n⚠️ ${notActuallyDeletedIds.length}개의 비디오는 삭제 요청은 성공했지만 실제로는 삭제되지 않았습니다.`;
        message += `\n(서버 설정이나 권한을 확인해주세요)`;
      }
      
      if (failedIds.length > 0) {
        message += `\n\n❌ ${failedIds.length}개의 비디오는 삭제 요청 자체가 실패했습니다.`;
      }
      
      alert(message);
      
      // 5단계: 최종 목록 새로고침으로 정확한 상태 확인
      setTimeout(() => {
        console.log('🔄 일괄 삭제 후 최종 목록 검증');
        loadVideos();
      }, 2000);
      
      console.log('✅ 일괄 삭제 완료:', {
        total: selectedVideos.length,
        actuallyDeleted: verifiedDeletedIds.length,
        notDeleted: notActuallyDeletedIds.length,
        failed: failedIds.length
      });
      
    } catch (error) {
      console.error('일괄 삭제 실패:', error);
      alert(`일괄 삭제 실패: ${error.message}`);
      
      // 오류 발생시에도 목록 새로고침
      await loadVideos();
    } finally {
      setShowDeleteProgress(false);
      setDeleteTracker(null);
    }
  };
  const handleCleanupStorage = async () => {
    try {
      setDeleting(true);
      const result = await videoAnalysisService.cleanupStorage();
      setCleanupResults(result);
      setShowCleanupDialog(true);
      
      // 목록 새로고침
      await loadVideos();
      
    } catch (error) {
      console.error('저장 공간 정리 실패:', error);
      alert(`저장 공간 정리 실패: ${error.message}`);
    } finally {
      setDeleting(false);
    }
  };

  const getDeleteConfirmMessage = () => {
    if (deleteTarget === 'single' && videoToDelete) {
      return videoAnalysisService.getDeleteConfirmMessage(videoToDelete);
    } else if (deleteTarget === 'batch') {
      const selectedVideoObjects = videos.filter(v => selectedVideos.includes(v.id));
      return videoAnalysisService.getBatchDeleteConfirmMessage(selectedVideoObjects);
    }
    return '삭제하시겠습니까?';
  };
    const renderDeleteProgress = () => {
    if (!showDeleteProgress || !deleteTracker) return null;

    const progress = deleteTracker.getProgress();
    
    return (
      <div className="modal-overlay">
        <div className="modal-content delete-progress">
          <div className="modal-header">
            <h3>🗑️ 비디오 삭제 진행 중</h3>
          </div>
          <div className="modal-body">
            <div className="progress-info">
              <p>삭제 진행률: {deleteTracker.completed + deleteTracker.failed}/{deleteTracker.total}</p>
              <div className="progress-bar">
                <div 
                  className="progress-fill" 
                  style={{ width: `${progress}%` }}
                ></div>
              </div>
              <p className="progress-text">{progress}% 완료</p>
            </div>
            
            {deleteTracker.failed > 0 && (
              <div className="failure-info">
                <p className="error-text">⚠️ {deleteTracker.failed}개 삭제 실패</p>
              </div>
            )}
          </div>
        </div>
      </div>
    );
  };

  // 개선된 오류 처리를 위한 에러 바운더리
  const handleDeleteError = (error, context) => {
    console.error(`삭제 오류 (${context}):`, error);
    
    let userMessage = '삭제 중 오류가 발생했습니다.';
    
    if (error.message.includes('404') || error.message.includes('찾을 수 없습니다')) {
      userMessage = '비디오를 찾을 수 없습니다. 이미 삭제되었을 수 있습니다.';
    } else if (error.message.includes('403') || error.message.includes('권한')) {
      userMessage = '삭제 권한이 없습니다.';
    } else if (error.message.includes('409') || error.message.includes('사용 중')) {
      userMessage = '비디오가 현재 사용 중이어서 삭제할 수 없습니다.';
    } else if (error.message.includes('연결') || error.message.includes('네트워크')) {
      userMessage = '서버 연결에 문제가 있습니다. 네트워크 상태를 확인해주세요.';
    }
    
    return userMessage;
  };

  // ========== 기존 함수들 유지 ==========

  const handleRefresh = () => {
    setRefreshing(true);
    loadVideos();
  };

  const startAnalysis = async (videoId, analysisType = 'enhanced') => {
    try {
      await videoAnalysisService.analyzeVideoEnhanced(videoId, {
        analysisType: analysisType,
        enhancedAnalysis: analysisType !== 'basic'
      });
      alert(`${getAnalysisTypeName(analysisType)} 분석이 시작되었습니다.`);
      loadVideos();
    } catch (error) {
      alert(`분석 시작 실패: ${error.message}`);
    }
  };

  const viewVideoDetails = async (video) => {
    setSelectedVideo(video);
    try {
      const details = await videoAnalysisService.getAnalysisStatus(video.id);
      setVideoDetails(details);
      
      if (video.is_analyzed && video.advanced_features_used) {
        loadAnalysisInsights(video.id);
      }
    } catch (error) {
      console.error('비디오 상세 정보 조회 실패:', error);
    }
  };

  const loadAnalysisInsights = async (videoId) => {
    try {
      const insights = await videoAnalysisService.sendVideoChatMessage(
        '이 비디오의 분석 결과와 주요 인사이트를 요약해주세요',
        videoId
      );
      
      setAnalysisInsights(prev => ({
        ...prev,
        [videoId]: insights
      }));
    } catch (error) {
      console.error('분석 인사이트 로드 실패:', error);
    }
  };

  const toggleAdvancedDetails = (videoId) => {
    setShowAdvancedDetails(prev => ({
      ...prev,
      [videoId]: !prev[videoId]
    }));
  };

  // 유틸리티 함수들
  const getAnalysisTypeName = (type) => {
    const names = {
      basic: '기본 분석',
      enhanced: '향상된 분석',
      comprehensive: '종합 분석',
      custom: '사용자 정의'
    };
    return names[type] || '향상된 분석';
  };

  const formatDuration = (seconds) => {
    if (!seconds) return '0초';
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins}분 ${secs}초`;
  };

  const formatFileSize = (bytes) => {
    return deleteUtils.formatFileSize(bytes);
  };

  const getStatusBadge = (video) => {
    const status = video.analysis_status;
    const progress = analysisProgress[video.id];
    
    const statusMap = {
      'pending': { text: '대기중', class: 'status-pending' },
      'processing': { 
        text: progress ? `분석중 (${progress.progress}%)` : '분석중', 
        class: 'status-processing' 
      },
      'completed': { text: '완료', class: 'status-completed' },
      'failed': { text: '실패', class: 'status-failed' }
    };
    
    const statusInfo = statusMap[status] || { text: status, class: 'status-unknown' };
    return <span className={`status-badge ${statusInfo.class}`}>{statusInfo.text}</span>;
  };



  const renderBatchActions = () => {
    const selectableVideos = deleteUtils.getSelectableVideos(videos);
    const nonSelectableVideos = deleteUtils.getNonSelectableVideos(videos);
    const allSelectableSelected = selectableVideos.length > 0 && 
      selectableVideos.every(v => selectedVideos.includes(v.id));
    
    return (
      <div className="batch-actions">
        <div className="batch-controls">
          <label className="select-all-checkbox">
            <input
              type="checkbox"
              checked={allSelectableSelected}
              onChange={(e) => handleSelectAll(e.target.checked)}
              disabled={selectableVideos.length === 0}
            />
            <span>전체 선택 ({selectableVideos.length}개 선택 가능)</span>
          </label>
          
          {nonSelectableVideos.length > 0 && (
            <span className="non-selectable-info">
              {nonSelectableVideos.length}개 비디오는 선택할 수 없습니다 (분석 중)
            </span>
          )}
        </div>
        
        {selectedVideos.length > 0 && (
          <div className="batch-action-buttons">
            <span className="selected-count">
              {selectedVideos.length}개 선택됨
            </span>
            <button 
              onClick={handleBatchDelete}
              className="batch-delete-button danger"
              disabled={deleting}
            >
              🗑️ 선택한 비디오 삭제
            </button>
          </div>
        )}
      </div>
    );
  };

  const renderAdvancedProgressBar = (video) => {
    const progress = analysisProgress[video.id];
    
    if (!progress || video.analysis_status !== 'processing') {
      return null;
    }

    return (
      <div className="analysis-progress-container advanced">
        <div className="progress-header">
          <span className="progress-step">{progress.currentStep}</span>
          <span className="progress-percentage">{progress.progress}%</span>
          <span className="analysis-type-badge">{getAnalysisTypeName(progress.analysisType)}</span>
        </div>
        
        <div className="progress-bar">
          <div 
            className="progress-fill" 
            style={{ width: `${progress.progress}%` }}
          ></div>
        </div>
        
        {progress.totalFeatures > 0 && (
          <div className="feature-progress">
            <div className="feature-progress-header">
              <span>분석 기능: {progress.completedFeatures?.length || 0}/{progress.totalFeatures}</span>
              {progress.currentFeature && (
                <span className="current-feature">현재: {progress.currentFeature}</span>
              )}
            </div>
            <div className="feature-progress-bar">
              <div 
                className="feature-progress-fill"
                style={{ width: `${((progress.completedFeatures?.length || 0) / progress.totalFeatures) * 100}%` }}
              ></div>
            </div>
            <div className="completed-features">
              {progress.completedFeatures?.map((feature, idx) => (
                <span key={idx} className="completed-feature-tag">✓ {feature}</span>
              ))}
            </div>
          </div>
        )}
      </div>
    );
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
      <div className="advanced-features">
        <h5>🚀 사용된 고급 기능</h5>
        <div className="feature-tags">
          {Object.entries(features).map(([feature, enabled]) => {
            if (!enabled) return null;
            return (
              <span key={feature} className="feature-tag enabled">
                {featureIcons[feature] || '🔧'} {feature.toUpperCase()}
              </span>
            );
          })}
        </div>
        
        {video.scene_types && video.scene_types.length > 0 && (
          <div className="scene-types">
            <span>감지된 씬 타입: </span>
            {video.scene_types.slice(0, 3).map((scene, idx) => (
              <span key={idx} className="scene-type-tag">{scene}</span>
            ))}
          </div>
        )}
      </div>
    );
  };

  if (loading) {
    return (
      <div className="video-analysis-page">
        <div className="loading-container">
          <div className="loading-spinner"></div>
          <p>비디오 목록을 불러오는 중...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="video-analysis-page">
      <div className="analysis-container">
        {/* 헤더 */}
        <div className="analysis-header">
          <h1>🧠 고급 비디오 분석 현황</h1>
          <div className="header-actions">
            <button 
              onClick={handleRefresh} 
              className="refresh-button"
              disabled={refreshing}
            >
              {refreshing ? '🔄' : '🔄'} 새로고침
            </button>
            <button 
              onClick={() => navigate('/video-upload')} 
              className="upload-button"
            >
              📁 비디오 업로드
            </button>
            <button 
              onClick={() => setShowBatchActions(!showBatchActions)}
              className="batch-toggle-button"
            >
              {showBatchActions ? '📝 일반 모드' : '☑️ 선택 모드'}
            </button>
            <button 
              onClick={handleCleanupStorage}
              className="cleanup-button secondary"
              disabled={deleting}
            >
              🧹 저장공간 정리
            </button>
          </div>
        </div>

    

        {/* 오류 메시지 */}
        {error && (
          <div className="error-banner">
            <span className="error-icon">⚠️</span>
            {error}
            <button onClick={() => setError(null)} className="error-close">✕</button>
          </div>
        )}

        {/* 일괄 작업 UI */}
        {showBatchActions && renderBatchActions()}

        {/* 통계 카드 */}
        <div className="stats-cards">
          <div className="stat-card">
            <div className="stat-number">{videos.length}</div>
            <div className="stat-label">전체 비디오</div>
          </div>
          <div className="stat-card">
            <div className="stat-number">
              {videos.filter(v => v.is_analyzed).length}
            </div>
            <div className="stat-label">분석 완료</div>
          </div>
          <div className="stat-card processing">
            <div className="stat-number">
              {videos.filter(v => v.analysis_status === 'processing').length}
            </div>
            <div className="stat-label">분석 중</div>
            {videos.filter(v => v.analysis_status === 'processing').length > 0 && (
              <div className="processing-indicator">🔄</div>
            )}
          </div>
          <div className="stat-card advanced">
            <div className="stat-number">
              {videos.filter(v => v.advanced_features_used && Object.values(v.advanced_features_used).some(Boolean)).length}
            </div>
            <div className="stat-label">고급 분석</div>
          </div>
          {selectedVideos.length > 0 && (
            <div className="stat-card selected">
              <div className="stat-number">{selectedVideos.length}</div>
              <div className="stat-label">선택됨</div>
            </div>
          )}
        </div>

        {/* 비디오 목록 */}
        <div className="video-list">
          {videos.length === 0 ? (
            <div className="empty-state">
              <div className="empty-icon">📹</div>
              <h3>업로드된 비디오가 없습니다</h3>
              <p>첫 번째 비디오를 업로드하여 고급 AI 분석을 시작해보세요.</p>
              <button 
                onClick={() => navigate('/video-upload')} 
                className="primary-button"
              >
                비디오 업로드하기
              </button>
            </div>
          ) : (
            <div className="video-grid">
              {videos.map((video) => {
                const { canDelete } = videoAnalysisService.canDeleteVideo(video);
                
                return (
                  <div key={video.id} className={`video-card ${video.analysis_status === 'processing' ? 'processing' : ''} ${video.advanced_features_used ? 'advanced' : ''} ${selectedVideos.includes(video.id) ? 'selected' : ''}`}>
                    {/* 선택 체크박스 */}
                    {showBatchActions && (
                      <div className="video-select-checkbox">
                        <input
                          type="checkbox"
                          checked={selectedVideos.includes(video.id)}
                          onChange={(e) => handleSelectVideo(video.id, e.target.checked)}
                          disabled={!canDelete}
                        />
                      </div>
                    )}
                    
                    <div className="video-card-header">
                      <h4>{video.original_name}</h4>
                      <div className="status-and-type">
                        {getStatusBadge(video)}
                        {video.analysis_type && (
                          <span className="analysis-type-badge">{getAnalysisTypeName(video.analysis_type)}</span>
                        )}
                      </div>
                    </div>
                    
                    {/* 고급 진행률 표시 */}
                    {renderAdvancedProgressBar(video)}
                    
                    <div className="video-info">
                      <div className="info-row">
                        <span>길이:</span>
                        <span>{formatDuration(video.duration)}</span>
                      </div>
                      <div className="info-row">
                        <span>크기:</span>
                        <span>{formatFileSize(video.file_size)}</span>
                      </div>
                      <div className="info-row">
                        <span>업로드:</span>
                        <span>{new Date(video.uploaded_at).toLocaleDateString()}</span>
                      </div>
                      {video.unique_objects !== undefined && (
                        <div className="info-row">
                          <span>감지된 객체:</span>
                          <span>{video.unique_objects}종류</span>
                        </div>
                      )}
                    </div>

                    {/* 고급 기능 표시 */}
                    {renderAdvancedFeatures(video)}

                    {/* 고급 분석 상세 정보 */}
                    {video.is_analyzed && video.advanced_features_used && (
                      <div className="advanced-details-section">
                        <button 
                          onClick={() => toggleAdvancedDetails(video.id)}
                          className="toggle-details-button"
                        >
                          {showAdvancedDetails[video.id] ? '간단히 보기' : '상세 분석 결과 보기'} 
                          {showAdvancedDetails[video.id] ? '🔼' : '🔽'}
                        </button>
                        
                        {showAdvancedDetails[video.id] && (
                          <div className="advanced-details">
                            {video.scene_types && video.scene_types.length > 0 && (
                              <div className="detail-section">
                                <h5>🎬 씬 분석 결과</h5>
                                <div className="scene-types-detail">
                                  {video.scene_types.map((scene, idx) => (
                                    <span key={idx} className="scene-detail-tag">{scene}</span>
                                  ))}
                                </div>
                              </div>
                            )}
                            
                            {analysisInsights[video.id] && (
                              <div className="detail-section">
                                <h5>💡 AI 분석 인사이트</h5>
                                <p className="insight-text">
                                  {analysisInsights[video.id].response || '인사이트를 불러오는 중...'}
                                </p>
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    )}

                    <div className="video-actions">
                      {!video.is_analyzed && video.analysis_status !== 'processing' && (
                        <>
                          <button 
                            onClick={() => startAnalysis(video.id, 'enhanced')}
                            className="action-button primary"
                            disabled={!systemCapabilities.clip_analysis}
                          >
                            향상된 분석
                          </button>
                          <button 
                            onClick={() => startAnalysis(video.id, 'comprehensive')}
                            className="action-button advanced"
                            disabled={!systemCapabilities.vqa}
                          >
                            종합 분석
                          </button>
                        </>
                      )}
                      
                      <button 
                        onClick={() => viewVideoDetails(video)}
                        className="action-button secondary"
                      >
                        상세 보기
                      </button>
                      
                      {video.is_analyzed && (
                        <button 
                          onClick={() => navigate(`/integrated-chat`)}
                          className="action-button success"
                        >
                          AI 채팅 탐색
                        </button>
                      )}
                      
                      {/* 삭제 버튼 */}
                      {!showBatchActions && (
                        <button 
                          onClick={() => handleDeleteVideo(video)}
                          className={`action-button danger ${!canDelete ? 'disabled' : ''}`}
                          disabled={!canDelete || deleting}
                          title={!canDelete ? '분석 중인 비디오는 삭제할 수 없습니다' : '비디오 삭제'}
                        >
                          🗑️ 삭제
                        </button>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        {/* 삭제 확인 다이얼로그 */}
        {showDeleteConfirm && (
          <div className="modal-overlay">
            <div className="modal-content delete-confirm">
              <div className="modal-header">
                <h3>🗑️ 비디오 삭제 확인</h3>
              </div>
              <div className="modal-body">
                <div className="confirm-message">
                  {getDeleteConfirmMessage()}
                </div>
                <div className="warning-message">
                  ⚠️ 이 작업은 되돌릴 수 없습니다.
                </div>
              </div>
              <div className="modal-actions">
                <button 
                  onClick={() => {
                    setShowDeleteConfirm(false);
                    setVideoToDelete(null);
                    setDeleteTarget(null);
                  }}
                  className="button secondary"
                  disabled={deleting}
                >
                  취소
                </button>
                <button 
                  onClick={confirmDelete}
                  className="button danger"
                  disabled={deleting}
                >
                  {deleting ? '삭제 중...' : '삭제'}
                </button>
              </div>
            </div>
          </div>
        )}

        {/* 정리 결과 다이얼로그 */}
        {showCleanupDialog && cleanupResults && (
          <div className="modal-overlay">
            <div className="modal-content cleanup-results">
              <div className="modal-header">
                <h3>🧹 저장 공간 정리 완료</h3>
                <button 
                  onClick={() => setShowCleanupDialog(false)}
                  className="modal-close"
                >
                  ✕
                </button>
              </div>
              <div className="modal-body">
                <div className="cleanup-summary">
                  <div className="cleanup-stat">
                    <span className="stat-label">해제된 용량:</span>
                    <span className="stat-value">{formatFileSize(cleanupResults.cleanup_results.total_freed_space)}</span>
                  </div>
                  <div className="cleanup-stat">
                    <span className="stat-label">삭제된 고아 파일:</span>
                    <span className="stat-value">{cleanupResults.cleanup_results.orphaned_files.length}개</span>
                  </div>
                  <div className="cleanup-stat">
                    <span className="stat-label">삭제된 임시 파일:</span>
                    <span className="stat-value">{cleanupResults.cleanup_results.temp_files.length}개</span>
                  </div>
                  <div className="cleanup-stat">
                    <span className="stat-label">삭제된 오래된 파일:</span>
                    <span className="stat-value">{cleanupResults.cleanup_results.old_analysis_files.length}개</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* 기존 비디오 상세 정보 모달 유지 */}
        {selectedVideo && (
          <div className="modal-overlay" onClick={() => setSelectedVideo(null)}>
            <div className="modal-content enhanced" onClick={(e) => e.stopPropagation()}>
              <div className="modal-header">
                <h3>🧠 {selectedVideo.original_name}</h3>
                <button 
                  onClick={() => setSelectedVideo(null)}
                  className="modal-close"
                >
                  ✕
                </button>
              </div>
              
              <div className="modal-body">
                {videoDetails ? (
                  <div className="video-details enhanced">
                    <div className="detail-section">
                      <h4>기본 정보</h4>
                      <div className="detail-grid">
                        <div>파일명: {selectedVideo.filename}</div>
                        <div>상태: {getStatusBadge(selectedVideo)}</div>
                        <div>분석 완료: {videoDetails.is_analyzed ? '예' : '아니오'}</div>
                        {selectedVideo.analysis_type && (
                          <div>분석 타입: {getAnalysisTypeName(selectedVideo.analysis_type)}</div>
                        )}
                      </div>
                    </div>

                    {/* 고급 분석 정보 */}
                    {selectedVideo.advanced_features_used && (
                      <div className="detail-section">
                        <h4>🚀 고급 분석 기능</h4>
                        <div className="advanced-features-detail">
                          {Object.entries(selectedVideo.advanced_features_used).map(([feature, enabled]) => (
                            <div key={feature} className={`feature-status ${enabled ? 'enabled' : 'disabled'}`}>
                              <span className="feature-icon">{enabled ? '✅' : '❌'}</span>
                              <span>{feature.toUpperCase()}</span>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {/* 나머지 기존 섹션들 유지 */}
                    {videoDetails.enhanced_analysis !== undefined && (
                      <div className="detail-section">
                        <h4>분석 통계</h4>
                        <div className="detail-grid">
                          <div>분석 타입: {videoDetails.enhanced_analysis ? 'Enhanced' : 'Basic'}</div>
                          <div>성공률: {videoDetails.success_rate}%</div>
                          <div>처리 시간: {videoDetails.processing_time}초</div>
                        </div>
                      </div>
                    )}

                    {videoDetails.stats && (
                      <div className="detail-section">
                        <h4>콘텐츠 통계</h4>
                        <div className="detail-grid">
                          <div>감지된 객체: {videoDetails.stats.objects}개</div>
                          <div>씬 수: {videoDetails.stats.scenes}개</div>
                          <div>캡션: {videoDetails.stats.captions}개</div>
                        </div>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="loading-details">
                    <div className="loading-spinner small"></div>
                    <p>상세 정보를 불러오는 중...</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* 네비게이션 */}
        <div className="navigation">
          <button onClick={() => navigate('/')} className="nav-button">
            🏠 홈으로
          </button>
          <button onClick={() => navigate('/video-upload')} className="nav-button">
            📁 업로드
          </button>
          <button onClick={() => navigate('/integrated-chat')} className="nav-button">
            💬 AI 채팅
          </button>
        </div>
      </div>
    </div>
  );
};

export default VideoAnalysisPage;