# performance_optimization.py - 성능 최적화 및 캐싱 시스템

import os
import json
import time
import hashlib
import pickle
import threading
from datetime import datetime, timedelta
from collections import defaultdict, OrderedDict
from django.core.cache import cache
from django.conf import settings
from django.db import connection
from django.db.models import Prefetch, Q

import numpy as np
from .models import (
    Video, VideoAnalysis, Frame, VideoMetadata, 
    PersonDetection, TemporalStatistics, ObjectTracking, 
    VideoSearchIndex, SearchQuery
)


class AdvancedCacheManager:
    """고급 캐시 관리자"""
    
    def __init__(self):
        self.memory_cache = OrderedDict()
        self.cache_stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0
        }
        self.max_memory_items = 1000
        self.cache_lock = threading.RLock()
        
        # 캐시 TTL 설정 (초)
        self.ttl_config = {
            'video_metadata': 60 * 60,      # 1시간
            'search_results': 30 * 60,      # 30분
            'analysis_results': 60 * 60,    # 1시간
            'temporal_stats': 45 * 60,      # 45분
            'person_attributes': 60 * 60,   # 1시간
            'frame_data': 15 * 60,          # 15분
            'system_status': 5 * 60         # 5분
        }
        
        print("💾 고급 캐시 관리자 초기화 완료")
    
    def get(self, cache_key, cache_type='default'):
        """캐시에서 데이터 조회"""
        with self.cache_lock:
            # 메모리 캐시 확인
            if cache_key in self.memory_cache:
                item = self.memory_cache.pop(cache_key)
                
                # TTL 확인
                if time.time() < item['expires_at']:
                    # 최근 사용 항목으로 이동
                    self.memory_cache[cache_key] = item
                    self.cache_stats['hits'] += 1
                    return item['data']
                else:
                    # 만료된 항목 제거
                    self.cache_stats['evictions'] += 1
            
            # Django 캐시 확인
            django_cache_key = f"advanced_cache:{cache_type}:{cache_key}"
            cached_data = cache.get(django_cache_key)
            
            if cached_data:
                # 메모리 캐시에도 저장
                self._set_memory_cache(cache_key, cached_data, cache_type)
                self.cache_stats['hits'] += 1
                return cached_data
            
            self.cache_stats['misses'] += 1
            return None
    
    def set(self, cache_key, data, cache_type='default', ttl=None):
        """캐시에 데이터 저장"""
        if ttl is None:
            ttl = self.ttl_config.get(cache_type, 30 * 60)  # 기본 30분
        
        # Django 캐시에 저장
        django_cache_key = f"advanced_cache:{cache_type}:{cache_key}"
        cache.set(django_cache_key, data, ttl)
        
        # 메모리 캐시에도 저장
        self._set_memory_cache(cache_key, data, cache_type, ttl)
        
    def _set_memory_cache(self, cache_key, data, cache_type, ttl=None):
        """메모리 캐시에 데이터 저장"""
        with self.cache_lock:
            if ttl is None:
                ttl = self.ttl_config.get(cache_type, 30 * 60)
            
            # 캐시 크기 제한
            if len(self.memory_cache) >= self.max_memory_items:
                # 가장 오래된 항목 제거 (LRU)
                oldest_key = next(iter(self.memory_cache))
                del self.memory_cache[oldest_key]
                self.cache_stats['evictions'] += 1
            
            # 새 항목 추가
            self.memory_cache[cache_key] = {
                'data': data,
                'expires_at': time.time() + ttl,
                'cache_type': cache_type
            }
    
    def invalidate(self, pattern=None, cache_type=None):
        """캐시 무효화"""
        with self.cache_lock:
            # 메모리 캐시 정리
            if pattern:
                keys_to_remove = [k for k in self.memory_cache.keys() if pattern in k]
                for key in keys_to_remove:
                    del self.memory_cache[key]
            elif cache_type:
                keys_to_remove = [
                    k for k, v in self.memory_cache.items() 
                    if v.get('cache_type') == cache_type
                ]
                for key in keys_to_remove:
                    del self.memory_cache[key]
            else:
                self.memory_cache.clear()
        
        # Django 캐시 정리는 패턴 매칭이 어려우므로 전체 정리
        if not pattern and not cache_type:
            cache.clear()
    
    def get_stats(self):
        """캐시 통계 반환"""
        total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = (self.cache_stats['hits'] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            'memory_cache_size': len(self.memory_cache),
            'cache_stats': self.cache_stats,
            'hit_rate_percentage': round(hit_rate, 2),
            'memory_usage_mb': self._calculate_memory_usage()
        }
    
    def _calculate_memory_usage(self):
        """메모리 사용량 계산 (근사값)"""
        try:
            total_size = 0
            for item in self.memory_cache.values():
                # 간단한 크기 추정
                total_size += len(pickle.dumps(item['data']))
            return round(total_size / (1024 * 1024), 2)  # MB 단위
        except:
            return 0


class DatabaseOptimizer:
    """데이터베이스 최적화 관리자"""
    
    def __init__(self):
        self.query_cache = {}
        self.slow_query_threshold = 1.0  # 1초
        self.query_stats = defaultdict(list)
        
    def optimize_video_queries(self):
        """비디오 관련 쿼리 최적화"""
        
        # 자주 사용되는 쿼리들에 대한 최적화된 메서드들
        optimized_queries = {
            'analyzed_videos': self._get_analyzed_videos_optimized,
            'video_with_analysis': self._get_video_with_analysis_optimized,
            'person_detections_by_video': self._get_person_detections_optimized,
            'temporal_stats_by_video': self._get_temporal_stats_optimized,
            'search_index_data': self._get_search_index_optimized
        }
        
        return optimized_queries
    
    def _get_analyzed_videos_optimized(self):
        """분석된 비디오 목록 최적화 조회"""
        cache_key = "analyzed_videos_list"
        cached_result = cache.get(cache_key)
        
        if cached_result:
            return cached_result
        
        # 최적화된 쿼리: 필요한 관련 데이터를 한 번에 가져오기
        videos = Video.objects.filter(is_analyzed=True).select_related(
            'analysis', 'metadata'
        ).prefetch_related(
            Prefetch(
                'analysis',
                queryset=VideoAnalysis.objects.select_related('video')
            )
        ).order_by('-uploaded_at')
        
        # 결과를 직렬화 가능한 형태로 변환
        result = []
        for video in videos:
            video_data = {
                'id': video.id,
                'filename': video.filename,
                'original_name': video.original_name,
                'duration': video.duration,
                'is_analyzed': video.is_analyzed,
                'analysis_status': video.analysis_status,
                'uploaded_at': video.uploaded_at.isoformat(),
                'file_size': video.file_size
            }
            
            # 분석 정보 추가
            if hasattr(video, 'analysis') and video.analysis:
                analysis = video.analysis
                video_data.update({
                    'enhanced_analysis': analysis.enhanced_analysis,
                    'success_rate': analysis.success_rate,
                    'processing_time': analysis.processing_time_seconds,
                    'analysis_type': analysis.analysis_statistics.get('analysis_type', 'basic'),
                    'unique_objects': analysis.analysis_statistics.get('unique_objects', 0),
                    'advanced_features_used': {
                        'weather_analysis': analysis.analysis_statistics.get('weather_analysis', False),
                        'person_attributes': analysis.analysis_statistics.get('person_attribute_analysis', False),
                        'temporal_analysis': analysis.analysis_statistics.get('temporal_analysis', False),
                        'object_tracking': analysis.analysis_statistics.get('object_tracking', False)
                    }
                })
            
            # 메타데이터 정보 추가
            if hasattr(video, 'metadata') and video.metadata:
                metadata = video.metadata
                video_data.update({
                    'weather_detected': metadata.dominant_weather,
                    'time_period_detected': metadata.dominant_time_period,
                    'location_type': metadata.location_type
                })
            
            result.append(video_data)
        
        # 캐시에 저장 (5분)
        cache.set(cache_key, result, 300)
        return result
    
    def _get_video_with_analysis_optimized(self, video_id):
        """특정 비디오의 분석 결과 최적화 조회"""
        cache_key = f"video_analysis:{video_id}"
        cached_result = cache.get(cache_key)
        
        if cached_result:
            return cached_result
        
        try:
            # 모든 관련 데이터를 한 번의 쿼리로 가져오기
            video = Video.objects.select_related(
                'analysis', 'metadata'
            ).prefetch_related(
                'temporal_stats',
                'object_tracks',
                'search_index',
                Prefetch(
                    'frames',
                    queryset=Frame.objects.order_by('timestamp')[:50]  # 최대 50개 프레임
                ),
                Prefetch(
                    'scenes',
                    queryset=Scene.objects.order_by('scene_id')
                )
            ).get(id=video_id)
            
            # 결과 구성
            result = {
                'video': {
                    'id': video.id,
                    'original_name': video.original_name,
                    'duration': video.duration,
                    'is_analyzed': video.is_analyzed,
                    'analysis_status': video.analysis_status
                },
                'analysis': None,
                'metadata': None,
                'frames_sample': [],
                'scenes': [],
                'temporal_stats': [],
                'object_tracks': [],
                'search_index': None
            }
            
            # 분석 결과
            if hasattr(video, 'analysis') and video.analysis:
                result['analysis'] = {
                    'enhanced_analysis': video.analysis.enhanced_analysis,
                    'success_rate': video.analysis.success_rate,
                    'processing_time_seconds': video.analysis.processing_time_seconds,
                    'analysis_statistics': video.analysis.analysis_statistics,
                    'caption_statistics': video.analysis.caption_statistics
                }
            
            # 메타데이터
            if hasattr(video, 'metadata') and video.metadata:
                result['metadata'] = {
                    'dominant_weather': video.metadata.dominant_weather,
                    'weather_confidence': video.metadata.weather_confidence,
                    'dominant_time_period': video.metadata.dominant_time_period,
                    'time_confidence': video.metadata.time_confidence,
                    'location_type': video.metadata.location_type,
                    'overall_brightness': video.metadata.overall_brightness
                }
            
            # 프레임 샘플
            result['frames_sample'] = [
                {
                    'image_id': frame.image_id,
                    'timestamp': frame.timestamp,
                    'caption': frame.final_caption or frame.enhanced_caption or frame.caption,
                    'detected_objects': frame.detected_objects
                }
                for frame in video.frames.all()
            ]
            
            # 씬 정보
            result['scenes'] = [
                {
                    'scene_id': scene.scene_id,
                    'start_time': scene.start_time,
                    'end_time': scene.end_time,
                    'dominant_objects': scene.dominant_objects
                }
                for scene in video.scenes.all()
            ]
            
            # 시간별 통계
            result['temporal_stats'] = [
                {
                    'start_timestamp': stat.start_timestamp,
                    'end_timestamp': stat.end_timestamp,
                    'male_count': stat.male_count,
                    'female_count': stat.female_count,
                    'activity_density': stat.activity_density,
                    'top_colors': stat.top_colors
                }
                for stat in video.temporal_stats.all()
            ]
            
            # 객체 추적
            result['object_tracks'] = [
                {
                    'track_id': track.track_id,
                    'object_class': track.object_class,
                    'first_appearance': track.first_appearance,
                    'last_appearance': track.last_appearance,
                    'total_duration': track.total_duration,
                    'movement_distance': track.movement_distance
                }
                for track in video.object_tracks.all()
            ]
            
            # 검색 인덱스
            if hasattr(video, 'search_index') and video.search_index:
                result['search_index'] = {
                    'dominant_objects': video.search_index.dominant_objects,
                    'weather_tags': video.search_index.weather_tags,
                    'time_tags': video.search_index.time_tags,
                    'location_tags': video.search_index.location_tags,
                    'clothing_colors': video.search_index.clothing_colors
                }
            
            # 캐시에 저장 (30분)
            cache.set(cache_key, result, 1800)
            return result
            
        except Video.DoesNotExist:
            return None
    
    def _get_person_detections_optimized(self, video_id, filters=None):
        """사람 감지 데이터 최적화 조회"""
        cache_key = f"person_detections:{video_id}:{hash(str(filters))}"
        cached_result = cache.get(cache_key)
        
        if cached_result:
            return cached_result
        
        # 기본 쿼리
        queryset = PersonDetection.objects.filter(
            frame__video_id=video_id
        ).select_related('frame').order_by('frame__timestamp')
        
        # 필터 적용
        if filters:
            if 'gender' in filters:
                queryset = queryset.filter(gender_estimation=filters['gender'])
            if 'min_confidence' in filters:
                queryset = queryset.filter(confidence__gte=filters['min_confidence'])
            if 'color' in filters:
                queryset = queryset.filter(
                    Q(upper_body_color=filters['color']) | 
                    Q(lower_body_color=filters['color'])
                )
        
        # 결과 구성
        result = []
        for detection in queryset:
            result.append({
                'frame_id': detection.frame.image_id,
                'timestamp': detection.frame.timestamp,
                'person_id': detection.person_id,
                'track_id': detection.track_id,
                'bbox': [
                    detection.bbox_x1, detection.bbox_y1,
                    detection.bbox_x2, detection.bbox_y2
                ],
                'confidence': detection.confidence,
                'gender': detection.gender_estimation,
                'gender_confidence': detection.gender_confidence,
                'upper_body_color': detection.upper_body_color,
                'lower_body_color': detection.lower_body_color,
                'posture': detection.posture
            })
        
        # 캐시에 저장 (15분)
        cache.set(cache_key, result, 900)
        return result
    
    def _get_temporal_stats_optimized(self, video_id, time_range=None):
        """시간별 통계 최적화 조회"""
        cache_key = f"temporal_stats:{video_id}:{hash(str(time_range))}"
        cached_result = cache.get(cache_key)
        
        if cached_result:
            return cached_result
        
        # 기본 쿼리
        queryset = TemporalStatistics.objects.filter(
            video_id=video_id
        ).order_by('start_timestamp')
        
        # 시간 범위 필터 적용
        if time_range:
            start_time = time_range.get('start', 0)
            end_time = time_range.get('end', float('inf'))
            queryset = queryset.filter(
                start_timestamp__gte=start_time,
                end_timestamp__lte=end_time
            )
        
        # 결과 구성
        result = []
        for stat in queryset:
            result.append({
                'start_timestamp': stat.start_timestamp,
                'end_timestamp': stat.end_timestamp,
                'duration': stat.duration,
                'total_persons_detected': stat.total_persons_detected,
                'male_count': stat.male_count,
                'female_count': stat.female_count,
                'activity_density': stat.activity_density,
                'top_colors': stat.top_colors,
                'detailed_statistics': stat.detailed_statistics
            })
        
        # 캐시에 저장 (20분)
        cache.set(cache_key, result, 1200)
        return result
    
    def _get_search_index_optimized(self, video_ids=None):
        """검색 인덱스 최적화 조회"""
        cache_key = f"search_index:{hash(str(video_ids))}"
        cached_result = cache.get(cache_key)
        
        if cached_result:
            return cached_result
        
        # 기본 쿼리
        queryset = VideoSearchIndex.objects.select_related('video')
        
        if video_ids:
            queryset = queryset.filter(video_id__in=video_ids)
        
        # 결과 구성
        result = []
        for index in queryset:
            result.append({
                'video_id': index.video.id,
                'video_name': index.video.original_name,
                'dominant_objects': index.dominant_objects,
                'weather_tags': index.weather_tags,
                'time_tags': index.time_tags,
                'location_tags': index.location_tags,
                'clothing_colors': index.clothing_colors,
                'gender_distribution': index.gender_distribution,
                'scene_complexity_avg': index.scene_complexity_avg,
                'search_weights': index.search_weights
            })
        
        # 캐시에 저장 (10분)
        cache.set(cache_key, result, 600)
        return result
    
    def log_slow_query(self, query, execution_time):
        """느린 쿼리 로깅"""
        if execution_time > self.slow_query_threshold:
            self.query_stats['slow_queries'].append({
                'query': str(query)[:200],  # 처음 200자만
                'execution_time': execution_time,
                'timestamp': datetime.now().isoformat()
            })
            
            # 최대 100개만 유지
            if len(self.query_stats['slow_queries']) > 100:
                self.query_stats['slow_queries'] = self.query_stats['slow_queries'][-100:]
    
    def get_performance_stats(self):
        """성능 통계 반환"""
        return {
            'slow_queries_count': len(self.query_stats.get('slow_queries', [])),
            'slow_query_threshold': self.slow_query_threshold,
            'recent_slow_queries': self.query_stats.get('slow_queries', [])[-10:],
            'database_connections': len(connection.queries) if hasattr(connection, 'queries') else 0
        }


class SearchOptimizer:
    """검색 성능 최적화"""
    
    def __init__(self, cache_manager):
        self.cache_manager = cache_manager
        self.search_stats = defaultdict(int)
        
    def optimize_search_query(self, search_conditions, video_data):
        """검색 쿼리 최적화"""
        
        # 검색 조건 해시 생성
        conditions_hash = hashlib.md5(
            json.dumps(search_conditions, sort_keys=True).encode()
        ).hexdigest()[:16]
        
        cache_key = f"search_optimized:{conditions_hash}"
        
        # 캐시된 결과 확인
        cached_result = self.cache_manager.get(cache_key, 'search_results')
        if cached_result:
            self.search_stats['cache_hits'] += 1
            return cached_result
        
        # 검색 조건에 따른 최적화된 필터 적용
        optimized_results = self._apply_optimized_filters(search_conditions, video_data)
        
        # 결과 캐싱
        self.cache_manager.set(cache_key, optimized_results, 'search_results')
        self.search_stats['cache_misses'] += 1
        
        return optimized_results
    
    def _apply_optimized_filters(self, conditions, video_data):
        """최적화된 필터 적용"""
        
        # 조건별 가중치 설정
        condition_weights = {
            'weather': 0.4,
            'time_period': 0.3,
            'objects': 0.3,
            'location_type': 0.2
        }
        
        matching_videos = []
        
        for video in video_data:
            match_score = 0.0
            match_reasons = []
            
            # 날씨 조건 매칭 (최적화된 로직)
            if 'weather' in conditions:
                weather_score = self._calculate_weather_match(
                    conditions['weather'], 
                    video.get('metadata', {})
                )
                match_score += weather_score * condition_weights['weather']
                if weather_score > 0.5:
                    match_reasons.append(f"날씨: {conditions['weather']}")
            
            # 시간대 조건 매칭
            if 'time_period' in conditions:
                time_score = self._calculate_time_match(
                    conditions['time_period'],
                    video.get('metadata', {})
                )
                match_score += time_score * condition_weights['time_period']
                if time_score > 0.5:
                    match_reasons.append(f"시간대: {conditions['time_period']}")
            
            # 객체 조건 매칭 (최적화)
            if 'objects' in conditions:
                object_score = self._calculate_object_match(
                    conditions['objects'],
                    video.get('dominant_objects', [])
                )
                match_score += object_score * condition_weights['objects']
                if object_score > 0.3:
                    match_reasons.append("객체 매칭")
            
            # 장소 조건 매칭
            if 'location_type' in conditions:
                location_score = self._calculate_location_match(
                    conditions['location_type'],
                    video.get('metadata', {})
                )
                match_score += location_score * condition_weights['location_type']
                if location_score > 0.5:
                    match_reasons.append(f"장소: {conditions['location_type']}")
            
            # 임계값 확인
            match_threshold = conditions.get('match_threshold', 0.3)
            if match_score >= match_threshold:
                matching_videos.append({
                    'video_id': video['video_id'],
                    'video_name': video['video_name'],
                    'match_score': match_score,
                    'match_reasons': match_reasons
                })
        
        # 매칭 점수로 정렬
        matching_videos.sort(key=lambda x: x['match_score'], reverse=True)
        
        return matching_videos
    
    def _calculate_weather_match(self, target_weather, metadata):
        """날씨 매칭 점수 계산 (최적화)"""
        if not metadata:
            return 0.0
            
        detected_weather = metadata.get('dominant_weather', 'unknown')
        confidence = metadata.get('weather_confidence', 0.0)
        
        if detected_weather == target_weather:
            return confidence
        elif detected_weather == 'unknown':
            return 0.1  # 불확실한 경우 낮은 점수
        else:
            return 0.0
    
    def _calculate_time_match(self, target_time, metadata):
        """시간대 매칭 점수 계산"""
        if not metadata:
            return 0.0
            
        detected_time = metadata.get('dominant_time_period', 'unknown')
        confidence = metadata.get('time_confidence', 0.0)
        
        if detected_time == target_time:
            return confidence
        elif detected_time == 'unknown':
            return 0.1
        else:
            return 0.0
    
    def _calculate_object_match(self, target_objects, video_objects):
        """객체 매칭 점수 계산"""
        if not target_objects or not video_objects:
            return 0.0
        
        target_set = set(target_objects)
        video_set = set(video_objects)
        
        intersection = target_set & video_set
        union = target_set | video_set
        
        # Jaccard 유사도 계산
        jaccard_score = len(intersection) / len(union) if union else 0.0
        
        return jaccard_score
    
    def _calculate_location_match(self, target_location, metadata):
        """장소 매칭 점수 계산"""
        if not metadata:
            return 0.0
            
        detected_location = metadata.get('location_type', 'unknown')
        confidence = metadata.get('location_confidence', 0.0)
        
        if detected_location == target_location:
            return confidence
        elif detected_location == 'unknown':
            return 0.1
        else:
            return 0.0
    
    def get_search_stats(self):
        """검색 통계 반환"""
        total_searches = self.search_stats['cache_hits'] + self.search_stats['cache_misses']
        cache_hit_rate = (self.search_stats['cache_hits'] / total_searches * 100) if total_searches > 0 else 0
        
        return {
            'total_searches': total_searches,
            'cache_hits': self.search_stats['cache_hits'],
            'cache_misses': self.search_stats['cache_misses'],
            'cache_hit_rate_percentage': round(cache_hit_rate, 2)
        }


class PerformanceMonitor:
    """성능 모니터링 시스템"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        self.alerts = []
        self.thresholds = {
            'response_time_ms': 5000,    # 5초
            'memory_usage_mb': 1024,     # 1GB
            'cache_hit_rate': 70,        # 70%
            'error_rate': 5              # 5%
        }
    
    def record_metric(self, metric_name, value, timestamp=None):
        """메트릭 기록"""
        if timestamp is None:
            timestamp = time.time()
        
        self.metrics[metric_name].append({
            'value': value,
            'timestamp': timestamp
        })
        
        # 최대 1000개 데이터 포인트만 유지
        if len(self.metrics[metric_name]) > 1000:
            self.metrics[metric_name] = self.metrics[metric_name][-1000:]
        
        # 임계값 확인
        self._check_threshold(metric_name, value)
    
    def _check_threshold(self, metric_name, value):
        """임계값 확인 및 알림 생성"""
        if metric_name in self.thresholds:
            threshold = self.thresholds[metric_name]
            
            if (metric_name == 'cache_hit_rate' and value < threshold) or \
               (metric_name != 'cache_hit_rate' and value > threshold):
                
                alert = {
                    'metric': metric_name,
                    'value': value,
                    'threshold': threshold,
                    'timestamp': datetime.now().isoformat(),
                    'severity': 'warning' if value < threshold * 1.2 else 'critical'
                }
                
                self.alerts.append(alert)
                
                # 최대 50개 알림만 유지
                if len(self.alerts) > 50:
                    self.alerts = self.alerts[-50:]
    
    def get_performance_summary(self):
        """성능 요약 반환"""
        summary = {}
        
        for metric_name, data_points in self.metrics.items():
            if data_points:
                values = [dp['value'] for dp in data_points[-10:]]  # 최근 10개
                summary[metric_name] = {
                    'current': values[-1] if values else 0,
                    'average': sum(values) / len(values) if values else 0,
                    'min': min(values) if values else 0,
                    'max': max(values) if values else 0,
                    'trend': 'improving' if len(values) >= 2 and values[-1] < values[0] else 'stable'
                }
        
        return {
            'metrics': summary,
            'recent_alerts': self.alerts[-5:],  # 최근 5개 알림
            'total_alerts': len(self.alerts),
            'system_health': self._calculate_system_health()
        }
    
    def _calculate_system_health(self):
        """시스템 건강도 계산"""
        health_score = 100
        
        # 최근 알림 개수에 따른 점수 차감
        recent_alerts = len([a for a in self.alerts if 
                           datetime.fromisoformat(a['timestamp']) > datetime.now() - timedelta(hours=1)])
        
        health_score -= recent_alerts * 10
        
        # 캐시 히트율에 따른 점수 조정
        if 'cache_hit_rate' in self.metrics and self.metrics['cache_hit_rate']:
            cache_hit_rate = self.metrics['cache_hit_rate'][-1]['value']
            if cache_hit_rate < 50:
                health_score -= 20
            elif cache_hit_rate < 70:
                health_score -= 10
        
        return max(0, min(100, health_score))


# 전역 인스턴스들
cache_manager = AdvancedCacheManager()
db_optimizer = DatabaseOptimizer()
search_optimizer = SearchOptimizer(cache_manager)
performance_monitor = PerformanceMonitor()

# 성능 최적화 데코레이터
def cache_result(cache_type='default', ttl=None):
    """결과 캐싱 데코레이터"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # 캐시 키 생성
            cache_key = f"{func.__name__}:{hash(str(args) + str(kwargs))}"
            
            # 캐시된 결과 확인
            cached_result = cache_manager.get(cache_key, cache_type)
            if cached_result is not None:
                return cached_result
            
            # 함수 실행
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # 성능 메트릭 기록
            performance_monitor.record_metric(f"{func.__name__}_execution_time", execution_time * 1000)
            
            # 결과 캐싱
            cache_manager.set(cache_key, result, cache_type, ttl)
            
            return result
        return wrapper
    return decorator

def monitor_performance(metric_prefix=''):
    """성능 모니터링 데코레이터"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                execution_time = (time.time() - start_time) * 1000
                
                # 성공 메트릭 기록
                performance_monitor.record_metric(f"{metric_prefix}{func.__name__}_response_time", execution_time)
                performance_monitor.record_metric(f"{metric_prefix}{func.__name__}_success", 1)
                
                return result
                
            except Exception as e:
                execution_time = (time.time() - start_time) * 1000
                
                # 실패 메트릭 기록
                performance_monitor.record_metric(f"{metric_prefix}{func.__name__}_response_time", execution_time)
                performance_monitor.record_metric(f"{metric_prefix}{func.__name__}_error", 1)
                
                raise e
                
        return wrapper
    return decorator