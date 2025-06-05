import React, { useState, useEffect } from 'react';
import { 
  Calendar, Clock, Plus, Users, MapPin, AlertCircle, CheckCircle, X, 
  ChevronLeft, ChevronRight, Grid3X3, List, ChevronDown, ChevronUp, 
  Brain, Star, TrendingUp, Edit3, Trash2, Save
} from 'lucide-react';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
const getAuthHeaders = () => {
    // 모든 가능한 토큰 키를 확인
    const possibleTokenKeys = ['access_token', 'token', 'authToken', 'accessToken'];
    let token = null;
    
    // 순차적으로 토큰을 찾음
    for (const key of possibleTokenKeys) {
      const storedToken = localStorage.getItem(key);
      if (storedToken && storedToken.trim()) {
        token = storedToken.trim();
        console.log(`토큰 발견: ${key} = ${token.substring(0, 10)}...`);
        break;
      }
    }
    
    if (!token) {
      // 🔧 더 자세한 디버깅 정보
      console.log('저장된 모든 localStorage 키들:', Object.keys(localStorage));
      console.log('토큰 관련 값들:', {
        access_token: localStorage.getItem('access_token'),
        token: localStorage.getItem('token'),
        authToken: localStorage.getItem('authToken'),
        accessToken: localStorage.getItem('accessToken')
      });
      
      throw new Error('인증 토큰이 없습니다. 로그인이 필요합니다.');
    }
    
    return {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json',
    };
  };
  

// 날짜 파싱 함수 개선
const parseDateFromRequest = (requestText) => {
    const today = new Date();
    const tomorrow = new Date(today);
    tomorrow.setDate(today.getDate() + 1);
    const dayAfterTomorrow = new Date(today);
    dayAfterTomorrow.setDate(today.getDate() + 2);
  
    const setKoreanMidnight = (date) => {
      date.setHours(0, 0, 0, 0); // 한국 시간 기준 자정으로 설정
      return date;
    };
  
    if (requestText.includes('오늘')) {
      return setKoreanMidnight(today);
    } else if (requestText.includes('내일')) {
      return setKoreanMidnight(tomorrow);
    } else if (requestText.includes('모레') || requestText.includes('모래')) {
      return setKoreanMidnight(dayAfterTomorrow);
    } else if (requestText.includes('이번 주')) {
      const daysUntilFriday = (5 - today.getDay() + 7) % 7;
      const thisWeekFriday = new Date(today);
      thisWeekFriday.setDate(today.getDate() + (daysUntilFriday === 0 ? 7 : daysUntilFriday));
      return setKoreanMidnight(thisWeekFriday);
    } else if (requestText.includes('다음 주')) {
      const nextWeek = new Date(today);
      nextWeek.setDate(today.getDate() + 7);
      return setKoreanMidnight(nextWeek);
    }
  
    const datePatterns = [
      /(\d{1,2})월\s*(\d{1,2})일/,
      /(\d{1,2})\/(\d{1,2})/,
      /(\d{4})-(\d{1,2})-(\d{1,2})/
    ];
  
    for (const pattern of datePatterns) {
      const match = requestText.match(pattern);
      if (match) {
        if (pattern.source.includes('년')) {
          return setKoreanMidnight(new Date(parseInt(match[1]), parseInt(match[2]) - 1, parseInt(match[3])));
        } else {
          const month = parseInt(match[1]) - 1;
          const day = parseInt(match[2]);
          const year = month < today.getMonth() ? today.getFullYear() + 1 : today.getFullYear();
          return setKoreanMidnight(new Date(year, month, day));
        }
      }
    }
  
    return setKoreanMidnight(tomorrow);
  };

// 여러 일정 파싱 함수
const parseMultipleSchedules = (requestText) => {
  const separators = /[,，]|그리고|및|\s+그리고\s+|\s+및\s+|\s*,\s*|\s+와\s+|\s+과\s+/;
  const parts = requestText.split(separators);
  
  if (parts.length > 1) {
    const requests = [];
    parts.forEach(part => {
      const trimmed = part.trim();
      if (trimmed && trimmed.length > 2) {
        requests.push(trimmed);
      }
    });
    if (requests.length > 1) return requests;
  }
  
  // 숫자로 구분된 경우 처리
  const numberedPattern = /(\d+\s*[.)]?\s*([^0-9]+?)(?=\d+\s*[.)]?|$))/g;
  const matches = [...requestText.matchAll(numberedPattern)];
  
  if (matches.length > 1) {
    return matches.map(match => match[2].trim()).filter(item => item.length > 2);
  }
  
  return [requestText];
};
const scheduleAPI = {
    async fetchSchedules(params = {}) {
      try {
        const queryParams = new URLSearchParams(params).toString();
        const url = `${API_BASE_URL}/api/schedule/${queryParams ? `?${queryParams}` : ''}`;
        
        console.log('일정 조회 시도 중...');
        const headers = getAuthHeaders(); // 🔧 개선된 함수 사용
        console.log('사용할 헤더:', headers);
        
        const response = await fetch(url, {
          method: 'GET',
          headers: headers,
        });
  
        if (response.status === 401) {
          throw new Error('로그인이 필요합니다. 다시 로그인해주세요.');
        }
  
        if (!response.ok) {
          throw new Error(`서버 응답 오류: ${response.status}`);
        }
  
        const data = await response.json();
        console.log('일정 조회 성공:', data.length, '개의 일정');
        return data;
      } catch (error) {
        console.error('일정 조회 실패:', error);
        throw error;
      }
    },
  
    async createScheduleRequest(requestData) {
      try {
        const url = `${API_BASE_URL}/api/schedule/`;
        console.log('Creating schedule request to:', url);
        console.log('Request data:', requestData);
        
        const response = await fetch(url, {
          method: 'POST',
          headers: getAuthHeaders(), // 🔧 개선된 함수 사용
          body: JSON.stringify(requestData),
        });
  
        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}));
          throw new Error(errorData.error || `서버 응답 오류: ${response.status}`);
        }
  
        return await response.json();
      } catch (error) {
        console.error('AI 요청 실패:', error);
        throw error;
      }
    },
  
    async createSchedule(scheduleData) {
      try {
        const url = `${API_BASE_URL}/api/schedule/create/`;
        
        const response = await fetch(url, {
          method: 'POST',
          headers: getAuthHeaders(), // 🔧 개선된 함수 사용
          body: JSON.stringify(scheduleData),
        });
  
        if (response.status === 401) {
          throw new Error('로그인이 필요합니다. 다시 로그인해주세요.');
        }
  
        if (!response.ok) {
          throw new Error(`일정 생성 실패: ${response.status}`);
        }
  
        return await response.json();
      } catch (error) {
        console.error('일정 생성 실패:', error);
        throw error;
      }
    },
  
    async confirmSchedule(requestId, aiSuggestionData) {
        try {
          const url = `${API_BASE_URL}/api/schedule/confirm/${requestId}/`;
          
          const response = await fetch(url, {
            method: 'POST',
            headers: getAuthHeaders(),
            body: JSON.stringify({
              ai_suggestion: aiSuggestionData  // ✅ AI 제안 데이터 전송
            }),
          });
      
          if (!response.ok) {
            const errorData = await response.json().catch(() => ({}));
            throw new Error(errorData.error || `일정 확정 실패: ${response.status}`);
          }
      
          return await response.json();
        } catch (error) {
          console.error('일정 확정 실패:', error);
          throw error;
        }
      },
  
    async deleteSchedule(scheduleId) {
      try {
        const url = `${API_BASE_URL}/api/schedule/${scheduleId}/`;
        
        const response = await fetch(url, {
            method: 'DELETE',
            headers: getAuthHeaders(),
          });
          
          if (!response.ok) {
            throw new Error(`일정 삭제 실패: ${response.status}`);
          }
          
          // 204 응답이면 body가 없으므로 굳이 파싱할 필요 없음
          return { success: true };
          
        
      } catch (error) {
        console.error('일정 삭제 실패:', error);
        throw error;
      }
    },
  
    async updateSchedule(scheduleId, scheduleData) {
      try {
        const url = `${API_BASE_URL}/api/schedule/${scheduleId}/`;
        
        const response = await fetch(url, {
          method: 'PUT',
          headers: getAuthHeaders(), // 🔧 개선된 함수 사용
          body: JSON.stringify(scheduleData),
        });
  
        if (!response.ok) {
          throw new Error(`일정 수정 실패: ${response.status}`);
        }
  
        return await response.json();
      } catch (error) {
        console.error('일정 수정 실패:', error);
        throw error;
      }
    }
  };

// 수동 일정 추가 모달 컴포넌트
const ManualScheduleModal = ({ isOpen, onClose, onSave, editingSchedule = null }) => {
  const [formData, setFormData] = useState({
    title: '',
    description: '',
    start_date: '',
    start_time: '',
    end_time: '',
    location: '',
    priority: 'MEDIUM'
  });

  useEffect(() => {
    if (editingSchedule) {
      const startDate = new Date(editingSchedule.start_time);
      const endDate = new Date(editingSchedule.end_time);
      
      setFormData({
        title: editingSchedule.title,
        description: editingSchedule.description,
        start_date: startDate.toISOString().split('T')[0],
        start_time: startDate.toTimeString().slice(0, 5),
        end_time: endDate.toTimeString().slice(0, 5),
        location: editingSchedule.location || '',
        priority: editingSchedule.priority
      });
    } else {
      const today = new Date();
      setFormData({
        title: '',
        description: '',
        start_date: today.toISOString().split('T')[0],
        start_time: '09:00',
        end_time: '10:00',
        location: '',
        priority: 'MEDIUM'
      });
    }
  }, [editingSchedule, isOpen]);

  const handleSubmit = (e) => {
    e.preventDefault();
    
    const startDateTime = new Date(`${formData.start_date}T${formData.start_time}`);
    const endDateTime = new Date(`${formData.start_date}T${formData.end_time}`);
    
    const scheduleData = {
      title: formData.title,
      description: formData.description,
      start_time: startDateTime.toISOString(),
      end_time: endDateTime.toISOString(),
      location: formData.location,
      priority: formData.priority,
      attendees: '[]'
    };
    
    onSave(scheduleData);
    onClose();
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg p-6 max-w-md w-full mx-4">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-800">
            {editingSchedule ? '일정 수정' : '수동 일정 추가'}
          </h3>
          <button onClick={onClose} className="text-gray-400 hover:text-gray-600">
            <X className="h-5 w-5" />
          </button>
        </div>
        
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              일정 제목 *
            </label>
            <input
              type="text"
              required
              value={formData.title}
              onChange={(e) => setFormData({...formData, title: e.target.value})}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
              placeholder="일정 제목을 입력하세요"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              설명
            </label>
            <textarea
              value={formData.description}
              onChange={(e) => setFormData({...formData, description: e.target.value})}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
              rows="2"
              placeholder="일정 설명 (선택사항)"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              날짜 *
            </label>
            <input
              type="date"
              required
              value={formData.start_date}
              onChange={(e) => setFormData({...formData, start_date: e.target.value})}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
            />
          </div>
          
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                시작 시간 *
              </label>
              <input
                type="time"
                required
                value={formData.start_time}
                onChange={(e) => setFormData({...formData, start_time: e.target.value})}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                종료 시간 *
              </label>
              <input
                type="time"
                required
                value={formData.end_time}
                onChange={(e) => setFormData({...formData, end_time: e.target.value})}
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
              />
            </div>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              장소
            </label>
            <input
              type="text"
              value={formData.location}
              onChange={(e) => setFormData({...formData, location: e.target.value})}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
              placeholder="장소 (선택사항)"
            />
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              우선순위
            </label>
            <select
              value={formData.priority}
              onChange={(e) => setFormData({...formData, priority: e.target.value})}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500"
            >
              <option value="LOW">낮음</option>
              <option value="MEDIUM">보통</option>
              <option value="HIGH">높음</option>
              <option value="URGENT">긴급</option>
            </select>
          </div>
          
          <div className="flex gap-2 pt-4">
            <button
              type="button"
              onClick={onClose}
              className="flex-1 px-4 py-2 bg-gray-300 text-gray-700 rounded-lg hover:bg-gray-400"
            >
              취소
            </button>
            <button
              type="submit"
              className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 flex items-center justify-center"
            >
              <Save className="mr-2 h-4 w-4" />
              {editingSchedule ? '수정' : '추가'}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

// AI 개별 응답 카드 컴포넌트
const AIResponseCard = ({ suggestion, isSelected, onSelect }) => {
  const getAIIcon = (source) => {
    switch (source.toLowerCase()) {
      case 'gpt': return '🤖';
      case 'claude': return '🧠';
      case 'mixtral': return '⚡';
      default: return '🔮';
    }
  };

  const getAIColor = (source) => {
    switch (source.toLowerCase()) {
      case 'gpt': return 'border-green-200 bg-green-50';
      case 'claude': return 'border-purple-200 bg-purple-50';
      case 'mixtral': return 'border-orange-200 bg-orange-50';
      default: return 'border-gray-200 bg-gray-50';
    }
  };

  return (
    <div 
      className={`p-4 rounded-lg border-2 cursor-pointer transition-all ${
        isSelected 
          ? 'border-blue-500 bg-blue-50' 
          : getAIColor(suggestion.source)
      } hover:shadow-md`}
      onClick={onSelect}
    >
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center">
          <span className="text-2xl mr-2">{getAIIcon(suggestion.source)}</span>
          <h4 className="font-semibold text-gray-800 capitalize">
            {suggestion.source.toUpperCase()}
          </h4>
          {isSelected && <Star className="ml-2 h-4 w-4 text-blue-500 fill-current" />}
        </div>
      </div>
      
      <div className="space-y-2">
        <h5 className="font-medium text-gray-900">{suggestion.title}</h5>
        <p className="text-sm text-gray-600">{suggestion.description}</p>
        
        <div className="grid grid-cols-2 gap-2 text-xs">
          <div className="flex items-center">
            <Calendar className="h-3 w-3 mr-1 text-gray-400" />
            {suggestion.suggested_date}
          </div>
          <div className="flex items-center">
            <Clock className="h-3 w-3 mr-1 text-gray-400" />
            {suggestion.suggested_start_time} - {suggestion.suggested_end_time}
          </div>
        </div>
        
        {suggestion.location && (
          <div className="flex items-center text-xs">
            <MapPin className="h-3 w-3 mr-1 text-gray-400" />
            {suggestion.location}
          </div>
        )}
        
        <div className="mt-3 p-2 bg-white rounded text-xs">
          <strong>AI 분석:</strong> {suggestion.reasoning}
        </div>
      </div>
    </div>
  );
};

const ScheduleManagement = () => {
  const [schedules, setSchedules] = useState([]);
  const [newRequest, setNewRequest] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [aiSuggestion, setAiSuggestion] = useState(null);
  const [conflicts, setConflicts] = useState([]);
  const [showConflictModal, setShowConflictModal] = useState(false);
  const [currentDate, setCurrentDate] = useState(new Date());
  const [viewMode, setViewMode] = useState('calendar');
  const [error, setError] = useState(null);
  const [connectionStatus, setConnectionStatus] = useState('checking');
  const [showIndividualResponses, setShowIndividualResponses] = useState(false);
  const [selectedAIResponse, setSelectedAIResponse] = useState(null);
  
  // 수동 일정 관리 상태
  const [showManualModal, setShowManualModal] = useState(false);
  const [editingSchedule, setEditingSchedule] = useState(null);

  useEffect(() => {
    fetchSchedules();
  }, []);

  const fetchSchedules = async () => {
    try {
      setError(null);
      setConnectionStatus('checking');
      console.log('Fetching schedules...');
      
      const data = await scheduleAPI.fetchSchedules();
      console.log('Schedules fetched:', data);
      setSchedules(data);
      setConnectionStatus('connected');
    } catch (error) {
      console.error('일정 불러오기 실패:', error);
      setError('서버에 연결할 수 없습니다. 네트워크 연결을 확인하세요.');
      setConnectionStatus('disconnected');
      setSchedules([]); // 빈 배열로 초기화
    }
  };

  const handleRequestSchedule = async () => {
    if (!newRequest.trim()) return;
    
    setIsLoading(true);
    setError(null);
    setSelectedAIResponse(null);
    
    try {
      console.log('Creating schedule request...');
      
      // 현재 날짜와 시간을 정확하게 전송
 // ✅ 한국 시간 기준으로 now 정의
const koreaNow2 = new Date(new Date().toLocaleString('en-US', { timeZone: 'Asia/Seoul' }));

const currentDateTime = {
  current_date: koreaNow2.toLocaleDateString('ko-KR', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    timeZone: 'Asia/Seoul'
  }).replace(/\. /g, '-').replace('.', ''),  // YYYY-MM-DD 형식

  current_time: koreaNow2.toLocaleTimeString('ko-KR', {
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
    timeZone: 'Asia/Seoul'
  }),

  current_datetime: `${koreaNow2.toLocaleDateString('ko-KR', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    timeZone: 'Asia/Seoul'
  }).replace(/\. /g, '-').replace('.', '')}T${koreaNow2.toLocaleTimeString('en-GB', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    hour12: false,
    timeZone: 'Asia/Seoul'
  })}`,

  timezone: 'Asia/Seoul',
  weekday: koreaNow2.getDay()
};
// 1. 먼저 requestedDate 생성
const requestedDate = parseDateFromRequest(newRequest);

// 2. 그 다음에 requestedDateInfo 구성
const requestedDateKST = new Date(requestedDate.toLocaleString('en-US', { timeZone: 'Asia/Seoul' }));
requestedDateKST.setSeconds(0, 0);

const koreaNow = new Date(new Date().toLocaleString('en-US', { timeZone: 'Asia/Seoul' }));
koreaNow.setHours(0, 0, 0, 0);
requestedDateKST.setHours(0, 0, 0, 0);

const daysFromToday = Math.floor((requestedDateKST - koreaNow) / (1000 * 60 * 60 * 24));

const requested_date = `${requestedDateKST.toLocaleDateString('ko-KR', {
  year: 'numeric',
  month: 'long',
  day: 'numeric',
  weekday: 'long',
  timeZone: 'Asia/Seoul'
})} ${requestedDateKST.toLocaleTimeString('ko-KR', {
  hour: '2-digit',
  minute: '2-digit',
  hour12: false,
  timeZone: 'Asia/Seoul'
})}`;

const requestedDateInfo = {
  requested_date,
  requested_weekday: requestedDateKST.getDay(),
  days_from_today: daysFromToday,

};


      // 여러 일정 파싱
      const scheduleRequests = parseMultipleSchedules(newRequest);
      
      // 기존 일정 컨텍스트
      const currentScheduleContext = schedules.map(schedule => ({
        title: schedule.title,
        start_time: schedule.start_time,
        end_time: schedule.end_time,
        date: new Date(schedule.start_time).toISOString().split('T')[0],
        priority: schedule.priority
      }));
      const todayKST = new Date().toLocaleDateString('ko-KR', {
        timeZone: 'Asia/Seoul',
        year: 'numeric',
        month: '2-digit',
        day: '2-digit'
      }).replace(/\. /g, '-').replace('.', '');

      const requestData = {
        request_text: newRequest,
        existing_schedules: currentScheduleContext,
        current_datetime_info: currentDateTime,
        requested_date_info: requestedDateInfo,
        is_multiple_schedule: scheduleRequests.length > 1,
        schedule_requests: scheduleRequests
      };

      console.log('Request data with date context:', requestData);

      const data = await scheduleAPI.createScheduleRequest(requestData);
      console.log('AI suggestion received:', data);
      
      setAiSuggestion(data);
      
      // 첫 번째 AI 응답을 기본 선택으로 설정
      if (data.individual_suggestions && data.individual_suggestions.length > 0) {
        setSelectedAIResponse(0);
      }
      
      if (data.has_conflicts) {
        setConflicts(data.conflicts);
        setShowConflictModal(true);
      }
    } catch (error) {
      console.error('AI 제안 요청 실패:', error);
      setError(`AI 제안을 받는데 실패했습니다: ${error.message}`);
    } finally {
      setIsLoading(false);
    }
  };

  const confirmSchedule = async (requestId) => {
    try {
      setError(null);
      console.log('Confirming schedule with ID:', requestId);
      
      // ✅ AI 제안 데이터를 함께 전송
      const data = await scheduleAPI.confirmSchedule(requestId, aiSuggestion);
      console.log('Schedule confirmed:', data);
      
      // 서버에서 생성된 일정을 가져와서 상태 업데이트
      await fetchSchedules();
      
      if (aiSuggestion?.is_multiple_schedule) {
        alert(`${aiSuggestion.multiple_schedules.length}개의 일정이 성공적으로 생성되었습니다!`);
      } else {
        alert(data.message || '일정이 성공적으로 생성되었습니다!');
      }
      
      setAiSuggestion(null);
      setNewRequest('');
      setSelectedAIResponse(null);
    } catch (error) {
      console.error('일정 확정 실패:', error);
      setError(`일정 생성에 실패했습니다: ${error.message}`);
    }
  };

  // 수동 일정 추가
// 수동 일정 추가
const handleManualScheduleAdd = async (scheduleData) => {
    try {
      // 🔧 실제 API 호출로 변경
      const response = await scheduleAPI.createSchedule(scheduleData);
      await fetchSchedules(); // 서버에서 최신 데이터 가져오기
      alert('일정이 성공적으로 추가되었습니다!');
    } catch (error) {
      console.error('수동 일정 추가 실패:', error);
      if (error.message.includes('로그인')) {
        setError('로그인이 필요합니다. 다시 로그인해주세요.');
      } else {
        setError(`일정 추가에 실패했습니다: ${error.message}`);
      }
    }
  };

  // 일정 수정
  const handleScheduleEdit = async (scheduleId, scheduleData) => {
    try {
      await scheduleAPI.updateSchedule(scheduleId, scheduleData);
      await fetchSchedules(); // 서버에서 최신 데이터 다시 가져오기
      alert('일정이 성공적으로 수정되었습니다!');
      setEditingSchedule(null);
    } catch (error) {
      console.error('일정 수정 실패:', error);
      setError(`일정 수정에 실패했습니다: ${error.message}`);
    }
  };

  // 일정 삭제
  const handleScheduleDelete = async (scheduleId) => {
    if (!window.confirm('정말로 이 일정을 삭제하시겠습니까?')) {
      return;
    }
    
    try {
      await scheduleAPI.deleteSchedule(scheduleId);
      await fetchSchedules(); // 서버에서 최신 데이터 다시 가져오기
      alert('일정이 성공적으로 삭제되었습니다!');
    } catch (error) {
      console.error('일정 삭제 실패:', error);
      setError(`일정 삭제에 실패했습니다: ${error.message}`);
    }
  };
            

  const testConnection = async () => {
    try {
      setConnectionStatus('checking');
      await scheduleAPI.fetchSchedules();
      setConnectionStatus('connected');
      alert('서버 연결 성공!');
    } catch (error) {
      console.error('Connection test failed:', error);
      setConnectionStatus('disconnected');
      alert(`연결 테스트 실패: ${error.message}`);
    }
  };

  // 캘린더 관련 함수들
  const getDaysInMonth = (date) => {
    const year = date.getFullYear();
    const month = date.getMonth();
    const firstDay = new Date(year, month, 1);
    const lastDay = new Date(year, month + 1, 0);
    const daysInMonth = lastDay.getDate();
    const startingDayOfWeek = firstDay.getDay();

    const days = [];
    
    for (let i = 0; i < startingDayOfWeek; i++) {
      days.push(null);
    }
    
    for (let day = 1; day <= daysInMonth; day++) {
      days.push(new Date(year, month, day));
    }
    
    return days;
  };

  
  const getLocalDateYMD = (input) => {
    const d = new Date(input);
    const year = d.getFullYear();
    const month = (d.getMonth() + 1).toString().padStart(2, '0');
    const day = d.getDate().toString().padStart(2, '0');
    return `${year}-${month}-${day}`;
  };
  
  const getSchedulesForDate = (date) => {
    if (!date) return [];
  
    const dateStr = getLocalDateYMD(date);  // 클릭한 날짜
  
    return schedules.filter(schedule => {
      const scheduleDate = getLocalDateYMD(schedule.start_time); // ✅ 여기 중요
      return scheduleDate === dateStr;
    });
  };
  function getKoreanDateString(dateString) {
    const [year, month, day] = dateString.split('-').map(Number);
    const utcDate = new Date(Date.UTC(year, month - 1, day));  // UTC 자정 기준 생성
    return utcDate.toLocaleDateString('ko-KR', {
      timeZone: 'Asia/Seoul',
      year: 'numeric',
      month: 'long',
      day: 'numeric',
      weekday: 'long',
    });
  }
  
  

  const navigateMonth = (direction) => {
    setCurrentDate(prev => {
      const newDate = new Date(prev);
      newDate.setMonth(prev.getMonth() + direction);
      return newDate;
    });
  };

  const getPriorityColor = (priority) => {
    switch (priority) {
      case 'URGENT': return 'bg-red-100 text-red-800 border-red-200';
      case 'HIGH': return 'bg-orange-100 text-orange-800 border-orange-200';
      case 'MEDIUM': return 'bg-yellow-100 text-yellow-800 border-yellow-200';
      case 'LOW': return 'bg-green-100 text-green-800 border-green-200';
      default: return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  const formatTime = (timeString) => {
    return new Date(timeString).toLocaleTimeString('ko-KR', {
  timeZone: 'Asia/Seoul',
  hour: '2-digit',
  minute: '2-digit'});
  };

  

  const isToday = (date) => {
    if (!date) return false;
    const today = new Date();
    const targetDate = new Date(date);
    
    return today.getFullYear() === targetDate.getFullYear() &&
           today.getMonth() === targetDate.getMonth() &&
           today.getDate() === targetDate.getDate();
  };

  const getConnectionStatusInfo = () => {
    switch (connectionStatus) {
      case 'connected':
        return { color: 'bg-green-500', text: '서버 연결됨' };
      case 'disconnected':
        return { color: 'bg-red-500', text: '서버 연결 끊김' };
      case 'checking':
        return { color: 'bg-yellow-500', text: '연결 확인 중' };
      default:
        return { color: 'bg-gray-500', text: '상태 불명' };
    }
  };

  const weekDays = ['일', '월', '화', '수', '목', '금', '토'];
  const monthNames = [
    '1월', '2월', '3월', '4월', '5월', '6월',
    '7월', '8월', '9월', '10월', '11월', '12월'
  ];

  const statusInfo = getConnectionStatusInfo();

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-7xl mx-auto">
        {/* 연결 상태 알림 */}
        {connectionStatus === 'disconnected' && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
            <div className="flex items-center">
              <AlertCircle className="h-5 w-5 text-red-400 mr-2" />
              <p className="text-red-700">
                <strong>서버 연결 오류:</strong> 서버에 연결할 수 없습니다. 네트워크 연결을 확인하고 다시 시도해주세요.
              </p>
            </div>
          </div>
        )}

        {/* 에러 메시지 */}
        {error && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4 mb-6">
            <div className="flex items-center">
              <AlertCircle className="h-5 w-5 text-red-400 mr-2" />
              <p className="text-red-700">{error}</p>
              <button 
                onClick={() => setError(null)}
                className="ml-auto text-red-600 hover:text-red-800"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
          </div>
        )}

        {/* 헤더 */}
        <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
          <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between mb-4">
            <h1 className="text-2xl font-bold text-gray-800 mb-4 lg:mb-0 flex items-center">
              <Calendar className="mr-2" />
              AI 일정 관리 시스템
            </h1>
            
            {/* 연결 상태 및 컨트롤 */}
            <div className="flex items-center gap-4">
              <div className="flex items-center">
                <div className={`w-2 h-2 rounded-full mr-2 ${statusInfo.color}`}></div>
                <span className="text-sm text-gray-600">{statusInfo.text}</span>
              </div>
              
              <button
                onClick={testConnection}
                className="text-xs px-2 py-1 bg-gray-200 hover:bg-gray-300 rounded"
              >
                연결 테스트
              </button>
              
              {/* 수동 일정 추가 버튼 */}
              <button
                onClick={() => setShowManualModal(true)}
                className="px-3 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 flex items-center text-sm"
              >
                <Plus className="mr-1 h-4 w-4" />
                수동 추가
              </button>
              
              {/* 뷰 모드 토글 */}
              <div className="flex items-center gap-2">
                <button
                  onClick={() => setViewMode('calendar')}
                  className={`px-3 py-2 rounded-lg flex items-center ${
                    viewMode === 'calendar' 
                      ? 'bg-blue-600 text-white' 
                      : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                  }`}
                >
                  <Grid3X3 className="mr-1 h-4 w-4" />
                  캘린더
                </button>
                <button
                  onClick={() => setViewMode('list')}
                  className={`px-3 py-2 rounded-lg flex items-center ${
                    viewMode === 'list' 
                      ? 'bg-blue-600 text-white' 
                      : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
                  }`}
                >
                  <List className="mr-1 h-4 w-4" />
                  목록
                </button>
              </div>
            </div>
          </div>
          
          {/* 현재 날짜 정보 표시 */}

          <div className="bg-blue-50 rounded-lg p-3 mb-4">
            <div className="flex items-center text-sm text-blue-800">
              <Clock className="h-4 w-4 mr-2" />
              <span>현재: {new Date().toLocaleDateString('ko-KR', {
                year: 'numeric',
                month: 'long', 
                day: 'numeric',
                weekday: 'long'
              })} {new Date().toLocaleTimeString('ko-KR', {
                hour: '2-digit',
                minute: '2-digit'
              })}</span>
            </div>
          </div>
          {/* 새 일정 요청 */}
          <div className="flex gap-3">
            <input
              type="text"
              value={newRequest}
              onChange={(e) => setNewRequest(e.target.value)}
              placeholder="원하는 일정을 자연어로 입력하세요 (예: 내일 2시간 운동, 오늘 오후에 팀미팅, 12월 25일에 약속)"
              className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              onKeyPress={(e) => e.key === 'Enter' && handleRequestSchedule()}
            />
            <button
              onClick={handleRequestSchedule}
              disabled={isLoading || !newRequest.trim() || connectionStatus === 'disconnected'}
              className="px-6 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 flex items-center"
            >
              {isLoading ? (
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
              ) : (
                <Plus className="mr-2 h-4 w-4" />
              )}
              {isLoading ? 'AI 분석 중...' : 'AI 제안 받기'}
            </button>
          </div>

          {/* AI 제안 결과 */}
          {aiSuggestion && (
            <div className="bg-gradient-to-r from-blue-50 to-purple-50 border border-blue-200 rounded-lg p-6 mt-4">
              {/* 여러 일정 표시 */}
              {aiSuggestion.is_multiple_schedule && aiSuggestion.multiple_schedules && (
                <div className="mb-4">
                  <h3 className="text-lg font-semibold text-gray-800 mb-3 flex items-center">
                    <Brain className="mr-2 h-5 w-5 text-blue-600" />
                    여러 일정 생성 결과 ({aiSuggestion.multiple_schedules.length}개)
                  </h3>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                    {aiSuggestion.multiple_schedules.map((schedule, index) => (
                      <div key={index} className="bg-white rounded-lg p-4 border-2 border-green-300">
                        <div className="flex items-center mb-2">
                          <span className="bg-green-600 text-white rounded-full w-6 h-6 flex items-center justify-center text-sm font-bold mr-2">
                            {index + 1}
                          </span>
                          <h4 className="font-semibold text-gray-800">{schedule.title}</h4>
                        </div>
                        <p className="text-sm text-gray-600 mb-2">{schedule.description}</p>
                        <div className="space-y-1 text-sm">
                          <div className="flex items-center">
                            <Calendar className="h-3 w-3 mr-1 text-gray-500" />
                            {schedule.suggested_date}
                          </div>
                          <div className="flex items-center">
                            <Clock className="h-3 w-3 mr-1 text-gray-500" />
                            {schedule.suggested_start_time} - {schedule.suggested_end_time}
                          </div>
                          {schedule.location && (
                            <div className="flex items-center">
                              <MapPin className="h-3 w-3 mr-1 text-gray-500" />
                              {schedule.location}
                            </div>
                          )}
                        </div>
                        <div className="mt-2 p-2 bg-green-50 rounded text-xs">
                          <strong>AI 분석:</strong> {schedule.reasoning}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* 최적화된 제안 헤더 */}
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-xl font-semibold text-gray-800 flex items-center">
                  <Brain className="mr-2 h-6 w-6 text-blue-600" />
                  AI 최적화 결과
                </h3>
                <div className="flex items-center text-sm text-gray-600">
                  <TrendingUp className="mr-1 h-4 w-4" />
                  신뢰도: {(aiSuggestion.confidence_score * 100).toFixed(1)}%
                </div>
              </div>

              {/* 최적화된 제안 - 단일 일정인 경우만 표시 */}
              {!aiSuggestion.is_multiple_schedule && aiSuggestion.optimized_suggestion && (
                <div className="bg-white rounded-lg p-4 mb-4 border-2 border-blue-300">
                  <div className="flex items-center mb-2">
                    <Star className="h-5 w-5 text-yellow-500 fill-current mr-2" />
                    <h4 className="font-semibold text-gray-800">최종 최적화 제안</h4>
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                      <h5 className="font-medium text-gray-800">{aiSuggestion.optimized_suggestion.title}</h5>
                      <p className="text-sm text-gray-600 mt-1">{aiSuggestion.optimized_suggestion.description}</p>
                    </div>
                    <div className="space-y-2 text-sm">
                    <div className="flex items-center">
  <Calendar className="h-4 w-4 mr-2 text-gray-500" />
  {getKoreanDateString(aiSuggestion.optimized_suggestion.suggested_date)}
</div>


                      <div className="flex items-center">
                        <Clock className="h-4 w-4 mr-2 text-gray-500" />
                        {aiSuggestion.optimized_suggestion.suggested_start_time} - {aiSuggestion.optimized_suggestion.suggested_end_time}
                      </div>
                      {aiSuggestion.optimized_suggestion.location && (
                        <div className="flex items-center">
                          <MapPin className="h-4 w-4 mr-2 text-gray-500" />
                          {aiSuggestion.optimized_suggestion.location}
                        </div>
                      )}
                      <div className="flex items-center">
                        <span className={`px-2 py-1 rounded-full text-xs border ${getPriorityColor(aiSuggestion.optimized_suggestion.priority)}`}>
                          {aiSuggestion.optimized_suggestion.priority}
                        </span>
                      </div>
                    </div>
                  </div>
                  <div className="mt-3 p-3 bg-blue-50 rounded text-sm">
                    <strong>최적화 근거:</strong> {aiSuggestion.optimized_suggestion.reasoning}
                  </div>
                </div>
              )}

              {/* AI 분석 요약 */}
              {aiSuggestion.ai_analysis && (
                <div className="bg-gray-50 rounded-lg p-4 mb-4">
                  <h4 className="font-semibold text-gray-800 mb-2 flex items-center">
                    <Brain className="mr-2 h-4 w-4" />
                    종합 분석 결과
                  </h4>
                  <p className="text-sm text-gray-700 mb-2">
                    <strong>분석 요약:</strong> {aiSuggestion.ai_analysis.analysis_summary}
                  </p>
                  <p className="text-sm text-gray-700">
                    <strong>선택 근거:</strong> {aiSuggestion.ai_analysis.reasoning}
                  </p>
                  {aiSuggestion.ai_analysis.models_used && (
                    <div className="mt-2 flex flex-wrap gap-1">
                      {aiSuggestion.ai_analysis.models_used.map(model => (
                        <span key={model} className="px-2 py-1 bg-white text-xs rounded-full border">
                          {model.toUpperCase()}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {/* 개별 AI 응답 토글 */}
              {aiSuggestion.individual_suggestions && aiSuggestion.individual_suggestions.length > 0 && (
                <div className="mb-4">
                  <button
                    onClick={() => setShowIndividualResponses(!showIndividualResponses)}
                    className="flex items-center justify-between w-full p-3 bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors"
                  >
                    <span className="font-medium text-gray-800 flex items-center">
                      <Users className="mr-2 h-4 w-4" />
                      각 AI 모델의 개별 제안 ({aiSuggestion.individual_suggestions.length}개)
                    </span>
                    {showIndividualResponses ? 
                      <ChevronUp className="h-5 w-5 text-gray-600" /> : 
                      <ChevronDown className="h-5 w-5 text-gray-600" />
                    }
                  </button>

                  {/* 개별 AI 응답들 */}
                  {showIndividualResponses && (
                    <div className="mt-4 space-y-4">
                      <div className="text-sm text-gray-600 mb-3">
                        💡 각 AI 모델의 제안을 클릭하여 비교해보세요. 선택한 제안을 최종 결과에 반영할 수 있습니다.
                      </div>
                      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
                        {aiSuggestion.individual_suggestions.slice(0, 3).map((suggestion, index) => (
                          <AIResponseCard
                            key={index}
                            suggestion={suggestion}
                            isSelected={selectedAIResponse === index}
                            onSelect={() => setSelectedAIResponse(index)}
                          />
                        ))}
                      </div>
                      
                      {/* 선택된 AI 응답 적용 버튼 */}
                      {selectedAIResponse !== null && !aiSuggestion.is_multiple_schedule && (
                        <div className="flex items-center justify-center mt-4">
                          <button
                            onClick={() => {
                              const selectedSuggestion = aiSuggestion.individual_suggestions[selectedAIResponse];
                              setAiSuggestion({
                                ...aiSuggestion,
                                optimized_suggestion: selectedSuggestion
                              });
                              alert(`${selectedSuggestion.source.toUpperCase()}의 제안이 최종 결과에 반영되었습니다!`);
                            }}
                            className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 flex items-center"
                          >
                            <CheckCircle className="mr-2 h-4 w-4" />
                            선택한 AI 제안 적용
                          </button>
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}
              
              {/* 액션 버튼들 */}
              <div className="flex justify-between items-center">
                <div className="text-sm text-gray-600">
                  {aiSuggestion.analysis_summary || `${aiSuggestion.individual_suggestions?.length || 0}개 AI 모델 분석 완료`}
                </div>
                <div className="flex gap-2">
                  <button
                    onClick={() => {
                      setAiSuggestion(null);
                      setSelectedAIResponse(null);
                      setShowIndividualResponses(false);
                    }}
                    className="px-4 py-2 bg-gray-300 text-gray-700 rounded-lg hover:bg-gray-400"
                  >
                    취소
                  </button>
                  <button
                    onClick={() => confirmSchedule(aiSuggestion.request_id)}
                    className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 flex items-center"
                  >
                    <CheckCircle className="mr-2 h-4 w-4" />
                    {aiSuggestion.is_multiple_schedule ? '모든 일정 확정' : '일정 확정'}
                  </button>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* 메인 컨텐츠 */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* 캘린더 또는 일정 목록 */}
          <div className="lg:col-span-2">
            {viewMode === 'calendar' ? (
              /* 캘린더 뷰 */
              <div className="bg-white rounded-lg shadow-lg p-6">
                <div className="flex items-center justify-between mb-6">
                  <h2 className="text-xl font-bold text-gray-800">
                    {currentDate.getFullYear()}년 {monthNames[currentDate.getMonth()]}
                  </h2>
                  <div className="flex gap-2">
                    <button
                      onClick={() => navigateMonth(-1)}
                      className="p-2 hover:bg-gray-100 rounded-lg"
                    >
                      <ChevronLeft className="h-5 w-5" />
                    </button>
                    <button
                      onClick={() => setCurrentDate(new Date())}
                      className="px-3 py-2 text-sm bg-blue-100 text-blue-700 rounded-lg hover:bg-blue-200"
                    >
                      오늘
                    </button>
                    <button
                      onClick={() => navigateMonth(1)}
                      className="p-2 hover:bg-gray-100 rounded-lg"
                    >
                      <ChevronRight className="h-5 w-5" />
                    </button>
                  </div>
                </div>

                <div className="grid grid-cols-7 gap-1 mb-2">
                  {weekDays.map(day => (
                    <div key={day} className="p-2 text-center text-sm font-semibold text-gray-600">
                      {day}
                    </div>
                  ))}
                </div>

                <div className="grid grid-cols-7 gap-1">
                  {getDaysInMonth(currentDate).map((date, index) => {
                    const daySchedules = getSchedulesForDate(date);
                    const isCurrentDay = isToday(date);
                    
                    return (
                      <div
                        key={index}
                        className={`min-h-[80px] p-1 border border-gray-200 ${
                          date ? 'bg-white hover:bg-gray-50' : 'bg-gray-100'
                        } ${isCurrentDay ? 'bg-blue-100 border-blue-400 ring-2 ring-blue-300' : ''}`}
                      >
                        {date && (
                          <>
                            <div className={`text-sm mb-1 ${
                              isCurrentDay ? 'font-bold text-blue-700 bg-blue-200 rounded-full w-6 h-6 flex items-center justify-center mx-auto' : 'text-gray-700'
                            }`}>
                              {date.getDate()}
                            </div>
                            <div className="space-y-1">
                              {daySchedules.slice(0, 2).map(schedule => (
                                <div
                                  key={schedule.id}
                                  className={`text-xs p-1 rounded border truncate ${getPriorityColor(schedule.priority)}`}
                                  title={`${schedule.title} (${formatTime(schedule.start_time)} - ${formatTime(schedule.end_time)})`}
                                >
                                  {schedule.title}
                                </div>
                              ))}
                              {daySchedules.length > 2 && (
                                <div className="text-xs text-gray-500 text-center">
                                  +{daySchedules.length - 2}개 더
                                </div>
                              )}
                            </div>
                          </>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            ) : (
              /* 목록 뷰 */
              <div className="bg-white rounded-lg shadow-lg p-6">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-xl font-bold text-gray-800">일정 목록</h2>
                  <div className="text-sm text-gray-600">
                    총 {schedules.length}개 일정
                  </div>
                </div>
                <div className="space-y-3">
                  {schedules.length === 0 ? (
                    <div className="text-center py-8">
                      <Calendar className="h-12 w-12 text-gray-300 mx-auto mb-4" />
                      <p className="text-gray-500">등록된 일정이 없습니다.</p>
                      <p className="text-sm text-gray-400 mt-1">위에서 AI에게 일정을 요청하거나 수동으로 추가해보세요!</p>
                    </div>
                  ) : (
                    schedules.map((schedule) => (
                      <div key={schedule.id} className={`border rounded-lg p-4 hover:bg-gray-50 ${getPriorityColor(schedule.priority)}`}>
                        <div className="flex justify-between items-start">
                          <div className="flex-1">
                            <h3 className="font-medium text-gray-800">{schedule.title}</h3>
                            <p className="text-sm text-gray-600 mt-1">{schedule.description}</p>
                            <div className="flex items-center mt-2 text-sm text-gray-500 space-x-4">
                              <div className="flex items-center">
                                <Calendar className="h-4 w-4 mr-1" />
                                {new Date(schedule.start_time).toLocaleDateString('ko-KR', {
                                  year: 'numeric',
                                  month: 'long',
                                  day: 'numeric',
                                  weekday: 'short'
                                })}
                              </div>
                              <div className="flex items-center">
                                <Clock className="h-4 w-4 mr-1" />
                                {formatTime(schedule.start_time)} - {formatTime(schedule.end_time)}
                              </div>
                              {schedule.location && (
                                <div className="flex items-center">
                                  <MapPin className="h-4 w-4 mr-1" />
                                  {schedule.location}
                                </div>
                              )}
                            </div>
                          </div>
                          <div className="flex items-center gap-2 ml-4">
                            <span className={`px-2 py-1 rounded-full text-xs border ${getPriorityColor(schedule.priority)}`}>
                              {schedule.priority}
                            </span>
                            <button
                              onClick={() => {
                                setEditingSchedule(schedule);
                                setShowManualModal(true);
                              }}
                              className="p-1 text-blue-600 hover:bg-blue-100 rounded"
                              title="일정 수정"
                            >
                              <Edit3 className="h-4 w-4" />
                            </button>
                            <button
                              onClick={() => handleScheduleDelete(schedule.id)}
                              className="p-1 text-red-600 hover:bg-red-100 rounded"
                              title="일정 삭제"
                            >
                              <Trash2 className="h-4 w-4" />
                            </button>
                          </div>
                        </div>
                      </div>
                    ))
                  )}
                </div>
              </div>
            )}
          </div>

          {/* 사이드 패널 */}
          <div className="space-y-6">
            {/* 오늘의 일정 */}
            <div className="bg-white rounded-lg shadow-lg p-6">
              <h3 className="text-lg font-semibold text-gray-800 mb-4">오늘의 일정</h3>
              <div className="space-y-3">
                {getSchedulesForDate(new Date()).length === 0 ? (
                  <p className="text-gray-500 text-sm">오늘 등록된 일정이 없습니다.</p>
                ) : (
                  getSchedulesForDate(new Date()).map(schedule => (
                    <div key={schedule.id} className={`p-3 rounded-lg border ${getPriorityColor(schedule.priority)}`}>
                      <div className="flex justify-between items-start">
                        <div className="flex-1">
                          <h4 className="font-medium text-sm">{schedule.title}</h4>
                          <p className="text-xs text-gray-600 mt-1">
                            {formatTime(schedule.start_time)} - {formatTime(schedule.end_time)}
                          </p>
                          {schedule.location && (
                            <p className="text-xs text-gray-500 mt-1 flex items-center">
                              <MapPin className="h-3 w-3 mr-1" />
                              {schedule.location}
                            </p>
                          )}
                        </div>
                        <div className="flex gap-1 ml-2">
                          <button
                            onClick={() => {
                              setEditingSchedule(schedule);
                              setShowManualModal(true);
                            }}
                            className="p-1 text-blue-600 hover:bg-blue-100 rounded"
                          >
                            <Edit3 className="h-3 w-3" />
                          </button>
                          <button
                            onClick={() => handleScheduleDelete(schedule.id)}
                            className="p-1 text-red-600 hover:bg-red-100 rounded"
                          >
                            <Trash2 className="h-3 w-3" />
                          </button>
                        </div>
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>

            {/* 빠른 일정 추가 */}
            <div className="bg-white rounded-lg shadow-lg p-6">
              <h3 className="text-lg font-semibold text-gray-800 mb-4">빠른 제안</h3>
              <div className="space-y-2">
                <button
                  onClick={() => setNewRequest('오늘 빈 시간에 1시간 운동 일정 잡아줘')}
                  className="w-full text-left p-2 text-sm bg-gray-50 hover:bg-gray-100 rounded"
                >
                  🏃‍♂️ 오늘 운동 시간 찾기
                </button>
                <button
                  onClick={() => setNewRequest('내일 오전에 2시간 집중 작업 시간 잡아줘')}
                  className="w-full text-left p-2 text-sm bg-gray-50 hover:bg-gray-100 rounded"
                >
                  💻 내일 집중 작업 시간
                </button>
                <button
                  onClick={() => setNewRequest('이번 주 빈 시간에 친구와 커피 약속 잡아줘')}
                  className="w-full text-left p-2 text-sm bg-gray-50 hover:bg-gray-100 rounded"
                >
                  ☕ 이번 주 커피 약속
                </button>
                <button
                  onClick={() => setNewRequest('내일 2시간 운동, 1시간 팀미팅 일정 잡아줘')}
                  className="w-full text-left p-2 text-sm bg-gray-50 hover:bg-gray-100 rounded"
                >
                  🔥 여러 일정 한번에
                </button>
                <button
                  onClick={() => setNewRequest('12월 25일 오후에 크리스마스 파티 일정 잡아줘')}
                  className="w-full text-left p-2 text-sm bg-gray-50 hover:bg-gray-100 rounded"
                >
                  🎄 특정 날짜 일정
                </button>
              </div>
            </div>

            {/* 일정 통계 */}
            <div className="bg-white rounded-lg shadow-lg p-6">
              <h3 className="text-lg font-semibold text-gray-800 mb-4">일정 통계</h3>
              <div className="space-y-3">
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">이번 달 일정:</span>
                  <span className="font-medium">
                    {schedules.filter(s => {
                      const scheduleMonth = new Date(s.start_time).getMonth();
                      return scheduleMonth === currentDate.getMonth();
                    }).length}개
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">오늘 일정:</span>
                  <span className="font-medium text-blue-600">
                    {getSchedulesForDate(new Date()).length}개
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">긴급 일정:</span>
                  <span className="font-medium text-red-600">
                    {schedules.filter(s => s.priority === 'URGENT').length}개
                  </span>
                </div>
                <div className="flex justify-between items-center">
                  <span className="text-sm text-gray-600">이번 주 일정:</span>
                  <span className="font-medium text-green-600">
                    {schedules.filter(s => {
                      const scheduleDate = new Date(s.start_time);
                      const today = new Date();
                      const weekStart = new Date(today.setDate(today.getDate() - today.getDay()));
                      const weekEnd = new Date(today.setDate(today.getDate() - today.getDay() + 6));
                      return scheduleDate >= weekStart && scheduleDate <= weekEnd;
                    }).length}개
                  </span>
                </div>
              </div>
            </div>

            {/* AI 모델 상태 */}
            <div className="bg-white rounded-lg shadow-lg p-6">
              <h3 className="text-lg font-semibold text-gray-800 mb-4">AI 모델 상태</h3>
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center">
                    <span className="text-lg mr-2">🤖</span>
                    <span className="text-sm font-medium">GPT-3.5</span>
                  </div>
                  <span className={`px-2 py-1 text-xs rounded-full ${
                    connectionStatus === 'connected' 
                      ? 'bg-green-100 text-green-800' 
                      : 'bg-red-100 text-red-800'
                  }`}>
                    {connectionStatus === 'connected' ? '활성' : '비활성'}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <div className="flex items-center">
                    <span className="text-lg mr-2">🧠</span>
                    <span className="text-sm font-medium">Claude</span>
                  </div>
                  <span className={`px-2 py-1 text-xs rounded-full ${
                    connectionStatus === 'connected' 
                      ? 'bg-green-100 text-green-800' 
                      : 'bg-red-100 text-red-800'
                  }`}>
                    {connectionStatus === 'connected' ? '활성' : '비활성'}
                  </span>
                </div>
                <div className="flex items-center justify-between">
                  <div className="flex items-center">
                    <span className="text-lg mr-2">⚡</span>
                    <span className="text-sm font-medium">Llama3</span>
                  </div>
                  <span className={`px-2 py-1 text-xs rounded-full ${
                    connectionStatus === 'connected' 
                      ? 'bg-green-100 text-green-800' 
                      : 'bg-red-100 text-red-800'
                  }`}>
                    {connectionStatus === 'connected' ? '활성' : '비활성'}
                  </span>
                </div>
              </div>
              
              <div className="mt-4 p-3 bg-blue-50 rounded-lg">
                <p className="text-xs text-blue-700">
                  <strong>AI 분석 기능:</strong><br/>
                  • 정확한 날짜 파싱 및 해석<br/>
                  • 기존 일정과의 충돌 방지<br/>
                  • 여러 일정 동시 생성<br/>
                  • 개인화된 시간 최적화<br/>
                  • 실시간 날짜/시간 기반 제안
                </p>
              </div>
            </div>

            {/* 시스템 정보 */}
            <div className="bg-white rounded-lg shadow-lg p-6">
              <h3 className="text-lg font-semibold text-gray-800 mb-4">시스템 정보</h3>
              <div className="space-y-2 text-sm">
                <div className="flex justify-between">
                  <span>연결 상태:</span>
                  <span className={
                    connectionStatus === 'connected' ? 'text-green-600' : 
                    connectionStatus === 'disconnected' ? 'text-red-600' : 'text-yellow-600'
                  }>
                    {statusInfo.text}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>일정 개수:</span>
                  <span>{schedules.length}</span>
                </div>
                <div className="flex justify-between">
                  <span>AI 응답:</span>
                  <span className="text-green-600">
                    {aiSuggestion?.individual_suggestions?.length || 0}개
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>현재 날짜:</span>
                  <span className="text-blue-600">
                    {new Date().toLocaleDateString('ko-KR', { timeZone: 'Asia/Seoul' })}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span>현재 시간:</span>
                  <span className="text-blue-600">
                    {new Date().toLocaleTimeString('ko-KR', {
                      hour: '2-digit',
                      minute: '2-digit'
                    })}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* 수동 일정 추가/수정 모달 */}
      <ManualScheduleModal
        isOpen={showManualModal}
        onClose={() => {
          setShowManualModal(false);
          setEditingSchedule(null);
        }}
        onSave={editingSchedule ? 
          (scheduleData) => handleScheduleEdit(editingSchedule.id, scheduleData) : 
          handleManualScheduleAdd
        }
        editingSchedule={editingSchedule}
      />

      {/* 충돌 모달 */}
      {showConflictModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 max-w-md w-full mx-4">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-red-600 flex items-center">
                <AlertCircle className="mr-2" />
                일정 충돌 발생
              </h3>
              <button
                onClick={() => setShowConflictModal(false)}
                className="text-gray-400 hover:text-gray-600"
              >
                <X className="h-5 w-5" />
              </button>
            </div>
            
            <p className="text-gray-600 mb-4">
              제안된 일정이 기존 일정과 충돌합니다:
            </p>
            
            <div className="space-y-2 mb-4">
              {conflicts.map((conflict) => (
                <div key={conflict.id} className="bg-red-50 border border-red-200 rounded p-3">
                  <div className="font-medium text-red-800">{conflict.title}</div>
                  <div className="text-sm text-red-600">
                    {new Date(conflict.start_time).toLocaleString()}
                  </div>
                </div>
              ))}
            </div>
            
            <div className="flex gap-2">
              <button
                onClick={() => setShowConflictModal(false)}
                className="flex-1 px-4 py-2 bg-gray-300 text-gray-700 rounded-lg hover:bg-gray-400"
              >
                취소
              </button>
              <button
                onClick={() => {
                  setShowConflictModal(false);
                  // 다른 시간대로 다시 요청
                  handleRequestSchedule();
                }}
                className="flex-1 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
              >
                다른 시간 제안받기
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ScheduleManagement; 