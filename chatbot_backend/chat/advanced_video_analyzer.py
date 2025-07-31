# advanced_video_analyzer.py - 기존 video_analyzer.py 확장

import cv2
import numpy as np
from datetime import datetime, timedelta
from collections import Counter, defaultdict
import json
import torch
import torchvision.transforms as transforms
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import colorsys

class AdvancedVideoMetadataAnalyzer:
    """고급 비디오 메타데이터 분석기 - 날씨, 시간대, 장소 등 분석"""
    
    def __init__(self):
        self.weather_keywords = {
            'rainy': ['rain', 'wet', 'umbrella', 'puddle', 'dark_clouds'],
            'sunny': ['bright', 'clear_sky', 'shadow', 'sunlight'],
            'cloudy': ['overcast', 'gray_sky', 'dim'],
            'night': ['dark', 'street_light', 'artificial_light', 'low_visibility'],
            'foggy': ['fog', 'mist', 'low_visibility', 'hazy']
        }
        
        self.time_indicators = {
            'dawn': [4, 6],
            'morning': [6, 12],
            'afternoon': [12, 18],
            'evening': [18, 21],
            'night': [21, 4]
        }
        
    def analyze_weather_conditions(self, frame):
        """날씨 조건 분석"""
        try:
            # 밝기 분석
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            
            # 대비 분석 (비 오는 날은 대비가 낮음)
            contrast = np.std(gray)
            
            # 색상 분석
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # V 채널 (밝기) 분포
            v_hist = cv2.calcHist([hsv], [2], None, [256], [0, 256])
            dark_ratio = np.sum(v_hist[:85]) / np.sum(v_hist)  # 어두운 픽셀 비율
            
            # S 채널 (채도) 분석
            saturation_mean = np.mean(hsv[:, :, 1])
            
            # 날씨 조건 추론
            weather_scores = {
                'rainy': 0.0,
                'sunny': 0.0,
                'cloudy': 0.0,
                'night': 0.0,
                'foggy': 0.0
            }
            
            # 밤 판정
            if brightness < 80:
                weather_scores['night'] = 0.8 + (80 - brightness) / 80 * 0.2
            
            # 비 오는 날 판정 (어둡고 대비가 낮고 채도가 낮음)
            if brightness < 120 and contrast < 40 and saturation_mean < 100:
                weather_scores['rainy'] = 0.6 + (120 - brightness) / 120 * 0.4
            
            # 맑은 날 판정
            if brightness > 150 and contrast > 60 and saturation_mean > 120:
                weather_scores['sunny'] = 0.7 + (brightness - 150) / 105 * 0.3
            
            # 흐린 날 판정
            if 100 < brightness < 150 and contrast < 50:
                weather_scores['cloudy'] = 0.6 + (150 - brightness) / 50 * 0.4
            
            # 안개 판정 (밝지만 대비가 매우 낮음)
            if brightness > 120 and contrast < 30:
                weather_scores['foggy'] = 0.5 + (30 - contrast) / 30 * 0.5
            
            # 가장 높은 점수의 날씨 조건 선택
            predicted_weather = max(weather_scores, key=weather_scores.get)
            confidence = weather_scores[predicted_weather]
            
            return {
                'weather': predicted_weather,
                'confidence': confidence,
                'brightness': brightness,
                'contrast': contrast,
                'saturation': saturation_mean,
                'dark_ratio': dark_ratio,
                'all_scores': weather_scores
            }
            
        except Exception as e:
            print(f"⚠️ 날씨 분석 오류: {e}")
            return {
                'weather': 'unknown',
                'confidence': 0.0,
                'brightness': 128,
                'contrast': 50,
                'saturation': 128,
                'dark_ratio': 0.5,
                'all_scores': {}
            }
    
    def analyze_time_of_day(self, frame, timestamp=None):
        """시간대 분석"""
        try:
            weather_analysis = self.analyze_weather_conditions(frame)
            brightness = weather_analysis['brightness']
            
            # 밝기 기반 시간대 추정
            if brightness < 60:
                time_period = 'night'
                confidence = 0.9
            elif brightness < 100:
                time_period = 'evening'
                confidence = 0.7  
            elif brightness > 180:
                time_period = 'afternoon'
                confidence = 0.8
            elif brightness > 140:
                time_period = 'morning'
                confidence = 0.6
            else:
                time_period = 'cloudy_day'
                confidence = 0.5
            
            # 그림자 분석으로 정확도 향상
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            shadow_features = self._detect_shadows(frame)
            
            if shadow_features['strong_shadows'] and brightness > 120:
                time_period = 'afternoon'
                confidence = min(0.9, confidence + 0.2)
            
            return {
                'time_period': time_period,
                'confidence': confidence,
                'brightness_score': brightness,
                'shadow_features': shadow_features,
                'estimated_hour': self._estimate_hour(brightness, shadow_features)
            }
            
        except Exception as e:
            print(f"⚠️ 시간대 분석 오류: {e}")
            return {
                'time_period': 'unknown',
                'confidence': 0.0,
                'brightness_score': 128,
                'shadow_features': {},
                'estimated_hour': 12
            }
    
    def _detect_shadows(self, frame):
        """그림자 감지"""
        try:
            # LAB 색공간으로 변환
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            l_channel = lab[:, :, 0]
            
            # 어두운 영역 (그림자) 감지
            shadow_mask = l_channel < np.percentile(l_channel, 25)
            shadow_ratio = np.sum(shadow_mask) / (frame.shape[0] * frame.shape[1])
            
            # 그림자 연결성 분석 (긴 그림자는 태양이 낮을 때)
            contours, _ = cv2.findContours(shadow_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            long_shadows = 0
            for contour in contours:
                rect = cv2.minAreaRect(contour)
                width, height = rect[1]
                aspect_ratio = max(width, height) / (min(width, height) + 1e-6)
                
                if aspect_ratio > 3 and cv2.contourArea(contour) > 1000:
                    long_shadows += 1
            
            return {
                'shadow_ratio': shadow_ratio,
                'strong_shadows': shadow_ratio > 0.3,
                'long_shadows': long_shadows,
                'shadow_complexity': len(contours)
            }
            
        except Exception as e:
            print(f"⚠️ 그림자 분석 오류: {e}")
            return {
                'shadow_ratio': 0.0,
                'strong_shadows': False,
                'long_shadows': 0,
                'shadow_complexity': 0
            }
    
    def _estimate_hour(self, brightness, shadow_features):
        """시간 추정 (0-23)"""
        if brightness < 60:
            return np.random.choice([22, 23, 0, 1, 2, 3, 4, 5])  # 밤
        elif brightness < 100:
            return np.random.choice([18, 19, 20, 21])  # 저녁
        elif shadow_features.get('long_shadows', 0) > 2:
            return np.random.choice([7, 8, 16, 17])  # 그림자가 긴 시간
        elif brightness > 180:
            return np.random.choice([11, 12, 13, 14, 15])  # 한낮
        else:
            return np.random.choice([9, 10, 15, 16])  # 기타


class PersonAttributeAnalyzer:
    """사람 속성 분석기 - 성별, 나이, 의상 색상 등"""
    
    def __init__(self):
        self.color_ranges = {
            'red': [(0, 70, 50), (10, 255, 255), (170, 70, 50), (180, 255, 255)],
            'orange': [(11, 70, 50), (25, 255, 255)],
            'yellow': [(26, 70, 50), (35, 255, 255)],
            'green': [(36, 70, 50), (85, 255, 255)],
            'blue': [(86, 70, 50), (125, 255, 255)],
            'purple': [(126, 70, 50), (155, 255, 255)],
            'pink': [(156, 70, 50), (169, 255, 255)],
            'white': [(0, 0, 200), (180, 25, 255)],
            'black': [(0, 0, 0), (180, 255, 50)],
            'gray': [(0, 0, 51), (180, 25, 199)],
            'brown': [(8, 60, 20), (20, 255, 200)]
        }
    
    def analyze_person_attributes(self, frame, person_bbox):
        """사람 속성 분석"""
        try:
            h, w = frame.shape[:2]
            
            # 바운딩 박스를 실제 좌표로 변환
            x1 = max(0, int(person_bbox[0] * w))
            y1 = max(0, int(person_bbox[1] * h))
            x2 = min(w, int(person_bbox[2] * w))
            y2 = min(h, int(person_bbox[3] * h))
            
            person_roi = frame[y1:y2, x1:x2]
            
            if person_roi.size == 0:
                return self._get_default_attributes()
            
            # 상하체 분리 (상체 60%, 하체 40%)
            person_height = y2 - y1
            upper_body = person_roi[:int(person_height * 0.6), :]
            lower_body = person_roi[int(person_height * 0.6):, :]
            
            # 상의 색상 분석
            upper_color = self._analyze_dominant_clothing_color(upper_body)
            
            # 하의 색상 분석  
            lower_color = self._analyze_dominant_clothing_color(lower_body)
            
            # 성별 추정 (휴리스틱 기반)
            gender_estimation = self._estimate_gender(person_roi, person_bbox)
            
            # 나이대 추정 (휴리스틱 기반)
            age_estimation = self._estimate_age_group(person_roi, person_bbox)
            
            # 자세 분석
            posture_analysis = self._analyze_posture(person_bbox)
            
            return {
                'upper_body_color': upper_color,
                'lower_body_color': lower_color,
                'gender_estimation': gender_estimation,
                'age_estimation': age_estimation,
                'posture': posture_analysis,
                'bbox_info': {
                    'width': x2 - x1,
                    'height': y2 - y1,
                    'aspect_ratio': (y2 - y1) / (x2 - x1 + 1e-6)
                }
            }
            
        except Exception as e:
            print(f"⚠️ 사람 속성 분석 오류: {e}")
            return self._get_default_attributes()
    
    def _analyze_dominant_clothing_color(self, clothing_roi):
        """의상 주요 색상 분석"""
        if clothing_roi.size == 0:
            return {'color': 'unknown', 'confidence': 0.0}
        
        try:
            # 이미지 크기 조정
            if clothing_roi.shape[0] > 100 or clothing_roi.shape[1] > 100:
                clothing_roi = cv2.resize(clothing_roi, (100, 100))
            
            hsv = cv2.cvtColor(clothing_roi, cv2.COLOR_BGR2HSV)
            
            color_scores = {}
            
            for color_name, ranges in self.color_ranges.items():
                mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
                
                if len(ranges) == 4:  # red의 경우
                    mask1 = cv2.inRange(hsv, np.array(ranges[0]), np.array(ranges[1]))
                    mask2 = cv2.inRange(hsv, np.array(ranges[2]), np.array(ranges[3]))
                    mask = cv2.bitwise_or(mask1, mask2)
                else:
                    mask = cv2.inRange(hsv, np.array(ranges[0]), np.array(ranges[1]))
                
                percentage = (np.sum(mask > 0) / (hsv.shape[0] * hsv.shape[1])) * 100
                color_scores[color_name] = percentage
            
            # 가장 높은 점수의 색상 선택
            dominant_color = max(color_scores, key=color_scores.get)
            confidence = color_scores[dominant_color] / 100.0
            
            # 임계값 이하면 unknown
            if confidence < 0.15:
                return {'color': 'unknown', 'confidence': confidence}
            
            return {
                'color': dominant_color,
                'confidence': confidence,
                'all_scores': color_scores
            }
            
        except Exception as e:
            print(f"⚠️ 의상 색상 분석 오류: {e}")
            return {'color': 'unknown', 'confidence': 0.0}
    
    def _estimate_gender(self, person_roi, bbox):
        """성별 추정 (휴리스틱 기반)"""
        try:
            # 바운딩 박스 비율 분석
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            aspect_ratio = height / (width + 1e-6)
            
            # 색상 분석 (의상 선호도 기반)
            hsv = cv2.cvtColor(person_roi, cv2.COLOR_BGR2HSV)
            
            # 핑크/보라 계열 (여성 추정 요소)
            pink_purple_mask = cv2.inRange(hsv, np.array([140, 50, 50]), np.array([170, 255, 255]))
            pink_purple_ratio = np.sum(pink_purple_mask > 0) / (hsv.shape[0] * hsv.shape[1])
            
            # 파란/검정 계열 (남성 추정 요소)
            blue_black_mask1 = cv2.inRange(hsv, np.array([100, 50, 50]), np.array([130, 255, 255]))
            blue_black_mask2 = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 50]))
            blue_black_mask = cv2.bitwise_or(blue_black_mask1, blue_black_mask2)
            blue_black_ratio = np.sum(blue_black_mask > 0) / (hsv.shape[0] * hsv.shape[1])
            
            # 점수 계산
            female_score = 0.0
            male_score = 0.0
            
            # 색상 기반 점수
            female_score += pink_purple_ratio * 2
            male_score += blue_black_ratio * 1.5
            
            # 체형 기반 점수 (매우 단순한 휴리스틱)
            if aspect_ratio > 2.2:  # 키가 큰 편
                male_score += 0.3
            elif aspect_ratio < 1.8:  # 키가 작은 편
                female_score += 0.2
            
            # 결과 결정
            if female_score > male_score and female_score > 0.3:
                return {'gender': 'female', 'confidence': min(0.8, female_score)}
            elif male_score > female_score and male_score > 0.3:
                return {'gender': 'male', 'confidence': min(0.8, male_score)}
            else:
                return {'gender': 'unknown', 'confidence': 0.5}
                
        except Exception as e:
            print(f"⚠️ 성별 추정 오류: {e}")
            return {'gender': 'unknown', 'confidence': 0.5}
    
    def _estimate_age_group(self, person_roi, bbox):
        """나이대 추정 (휴리스틱 기반)"""
        try:
            # 바운딩 박스 크기 기반 추정
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            area = width * height
            
            # 크기 기반 나이 그룹 추정
            if area < 0.02:  # 작은 사람 (어린이 또는 멀리 있는 사람)
                age_group = 'child_or_distant'
                confidence = 0.4
            elif area > 0.08:  # 큰 사람 (가까이 있는 성인)
                age_group = 'adult_close'
                confidence = 0.6
            else:
                age_group = 'adult_medium'
                confidence = 0.5
            
            return {
                'age_group': age_group,
                'confidence': confidence,
                'bbox_area': area
            }
            
        except Exception as e:
            print(f"⚠️ 나이대 추정 오류: {e}")
            return {'age_group': 'unknown', 'confidence': 0.5}
    
    def _analyze_posture(self, bbox):
        """자세 분석"""
        try:
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            aspect_ratio = height / (width + 1e-6)
            
            if aspect_ratio > 2.5:
                posture = 'standing_upright'
            elif aspect_ratio < 1.5:
                posture = 'sitting_or_crouching'
            else:
                posture = 'normal_standing'
            
            return {
                'posture': posture,
                'aspect_ratio': aspect_ratio,
                'confidence': 0.6
            }
            
        except Exception as e:
            print(f"⚠️ 자세 분석 오류: {e}")
            return {'posture': 'unknown', 'aspect_ratio': 2.0, 'confidence': 0.3}
    
    def _get_default_attributes(self):
        """기본 속성 반환"""
        return {
            'upper_body_color': {'color': 'unknown', 'confidence': 0.0},
            'lower_body_color': {'color': 'unknown', 'confidence': 0.0},
            'gender_estimation': {'gender': 'unknown', 'confidence': 0.5},
            'age_estimation': {'age_group': 'unknown', 'confidence': 0.5},
            'posture': {'posture': 'unknown', 'aspect_ratio': 2.0, 'confidence': 0.3},
            'bbox_info': {'width': 0, 'height': 0, 'aspect_ratio': 0}
        }


class TemporalAnalyzer:
    """시간별 통계 분석기"""
    
    def __init__(self):
        self.person_tracks = defaultdict(list)  # 사람 추적 데이터
        self.temporal_data = defaultdict(list)  # 시간별 데이터
        
    def analyze_temporal_statistics(self, video_analysis_data, start_time_sec, end_time_sec):
        """특정 시간 구간의 통계 분석"""
        try:
            # 해당 시간 구간의 프레임들 필터링
            relevant_frames = []
            
            for frame_data in video_analysis_data.get('frame_results', []):
                timestamp = frame_data.get('timestamp', 0)
                if start_time_sec <= timestamp <= end_time_sec:
                    relevant_frames.append(frame_data)
            
            if not relevant_frames:
                return {
                    'time_range': f"{start_time_sec}-{end_time_sec}초",
                    'total_frames': 0,
                    'statistics': {},
                    'error': '해당 시간대에 분석된 프레임이 없습니다.'
                }
            
            # 사람 감지 및 속성 분석
            person_analyzer = PersonAttributeAnalyzer()
            all_person_data = []
            
            for frame_data in relevant_frames:
                timestamp = frame_data.get('timestamp', 0)
                detected_objects = frame_data.get('objects', [])
                
                # 사람 객체만 필터링
                persons = [obj for obj in detected_objects if obj.get('class') == 'person']
                
                for person in persons:
                    # 기존에 저장된 상세 정보가 있다면 사용
                    if 'person_attributes' in person:
                        person_data = person['person_attributes']
                    else:
                        # 간단한 분석 (실제로는 원본 프레임이 필요)
                        person_data = {
                            'timestamp': timestamp,
                            'confidence': person.get('confidence', 0.5),
                            'bbox': person.get('bbox', []),
                            'upper_body_color': {'color': 'unknown', 'confidence': 0.0},
                            'gender_estimation': {'gender': 'unknown', 'confidence': 0.5}
                        }
                    
                    person_data['timestamp'] = timestamp
                    all_person_data.append(person_data)
            
            # 통계 계산
            statistics = self._calculate_person_statistics(all_person_data, start_time_sec, end_time_sec)
            
            return {
                'time_range': f"{self._format_time(start_time_sec)} - {self._format_time(end_time_sec)}",
                'total_frames': len(relevant_frames),
                'total_persons_detected': len(all_person_data),
                'statistics': statistics,
                'detailed_data': all_person_data[:20]  # 최대 20개 상세 데이터
            }
            
        except Exception as e:
            print(f"⚠️ 시간별 통계 분석 오류: {e}")
            return {
                'time_range': f"{start_time_sec}-{end_time_sec}초",
                'error': str(e),
                'statistics': {}
            }
    
    def _calculate_person_statistics(self, person_data, start_time, end_time):
        """사람 관련 통계 계산"""
        if not person_data:
            return {}
        
        try:
            # 성별 분포
            gender_counts = Counter()
            gender_confidences = []
            
            # 색상 분포
            upper_colors = Counter()
            
            # 시간대별 분포
            time_distribution = defaultdict(int)
            
            for person in person_data:
                # 성별 통계
                gender_info = person.get('gender_estimation', {})
                gender = gender_info.get('gender', 'unknown')
                confidence = gender_info.get('confidence', 0.5)
                
                if confidence > 0.6:  # 신뢰도가 높은 경우만
                    gender_counts[gender] += 1
                    gender_confidences.append(confidence)
                
                # 상의 색상 통계
                upper_color_info = person.get('upper_body_color', {})
                color = upper_color_info.get('color', 'unknown')
                if color != 'unknown':
                    upper_colors[color] += 1
                
                # 시간대별 분포 (30초 단위)
                timestamp = person.get('timestamp', 0)
                time_bucket = int(timestamp // 30) * 30
                time_distribution[time_bucket] += 1
            
            # 성비 계산
            total_gendered = gender_counts['male'] + gender_counts['female']
            gender_ratio = {}
            
            if total_gendered > 0:
                gender_ratio = {
                    'male_percentage': (gender_counts['male'] / total_gendered) * 100,
                    'female_percentage': (gender_counts['female'] / total_gendered) * 100,
                    'total_identified': total_gendered,
                    'confidence_avg': np.mean(gender_confidences) if gender_confidences else 0
                }
            
            # 주요 색상 (상위 3개)
            top_colors = dict(upper_colors.most_common(3))
            
            # 시간별 밀도
            time_density = dict(sorted(time_distribution.items()))
            
            return {
                'gender_distribution': dict(gender_counts),
                'gender_ratio': gender_ratio,
                'top_upper_body_colors': top_colors,
                'time_density': time_density,
                'total_unique_persons': len(person_data),  # 중복 제거 로직 필요
                'analysis_quality': {
                    'total_detections': len(person_data),
                    'high_confidence_gender': len(gender_confidences),
                    'color_identified': len([p for p in person_data if p.get('upper_body_color', {}).get('color') != 'unknown'])
                }
            }
            
        except Exception as e:
            print(f"⚠️ 사람 통계 계산 오류: {e}")
            return {'error': str(e)}
    
    def _format_time(self, seconds):
        """초를 MM:SS 형태로 변환"""
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"


class VideoSearchEngine:
    """고급 비디오 검색 엔진"""
    
    def __init__(self):
        self.metadata_analyzer = AdvancedVideoMetadataAnalyzer()
        self.person_analyzer = PersonAttributeAnalyzer()
        self.temporal_analyzer = TemporalAnalyzer()
    
    def search_videos_by_conditions(self, videos_data, search_conditions):
        """조건별 비디오 검색"""
        try:
            matching_videos = []
            
            for video_data in videos_data:
                video_id = video_data.get('video_id')
                video_name = video_data.get('video_name', '')
                analysis_data = video_data.get('analysis_data', {})
                
                # 각 조건 체크
                matches = self._check_search_conditions(analysis_data, search_conditions)
                
                if matches['overall_match']:
                    matching_videos.append({
                        'video_id': video_id,
                        'video_name': video_name,
                        'match_score': matches['match_score'],
                        'match_reasons': matches['match_reasons'],
                        'relevant_frames': matches['relevant_frames']
                    })
            
            # 매치 점수로 정렬
            matching_videos.sort(key=lambda x: x['match_score'], reverse=True)
            
            return {
                'matching_videos': matching_videos,
                'search_conditions': search_conditions,
                'total_matches': len(matching_videos)
            }
            
        except Exception as e:
            print(f"⚠️ 비디오 검색 오류: {e}")
            return {
                'matching_videos': [],
                'error': str(e)
            }
    
    def _check_search_conditions(self, analysis_data, conditions):
        """검색 조건 체크"""
        match_score = 0.0
        match_reasons = []
        relevant_frames = []
        
        try:
            frame_results = analysis_data.get('frame_results', [])
            video_summary = analysis_data.get('video_summary', {})
            
            # 날씨 조건 체크
            if 'weather' in conditions:
                target_weather = conditions['weather'].lower()
                weather_matches = 0
                
                for frame_data in frame_results[:20]:  # 샘플 프레임만 체크
                    frame_weather = frame_data.get('weather_analysis', {}).get('weather', 'unknown')
                    if frame_weather == target_weather:
                        weather_matches += 1
                        if len(relevant_frames) < 5:
                            relevant_frames.append({
                                'frame_id': frame_data.get('image_id'),
                                'timestamp': frame_data.get('timestamp'),
                                'reason': f'{target_weather} 날씨'
                            })
                
                if weather_matches > 0:
                    weather_score = min(1.0, weather_matches / 10)
                    match_score += weather_score * 0.4
                    match_reasons.append(f"{target_weather} 날씨 ({weather_matches}개 프레임)")
            
            # 시간대 조건 체크
            if 'time_period' in conditions:
                target_time = conditions['time_period'].lower()
                time_matches = 0
                
                for frame_data in frame_results[:20]:
                    frame_time = frame_data.get('time_analysis', {}).get('time_period', 'unknown')
                    if frame_time == target_time:
                        time_matches += 1
                        if len(relevant_frames) < 10:
                            relevant_frames.append({
                                'frame_id': frame_data.get('image_id'),
                                'timestamp': frame_data.get('timestamp'),
                                'reason': f'{target_time} 시간대'
                            })
                
                if time_matches > 0:
                    time_score = min(1.0, time_matches / 10)
                    match_score += time_score * 0.3
                    match_reasons.append(f"{target_time} 시간대 ({time_matches}개 프레임)")
            
            # 객체 조건 체크
            if 'objects' in conditions:
                target_objects = [obj.lower() for obj in conditions['objects']]
                dominant_objects = [obj.lower() for obj in video_summary.get('dominant_objects', [])]
                
                matching_objects = set(target_objects) & set(dominant_objects)
                if matching_objects:
                    object_score = len(matching_objects) / len(target_objects)
                    match_score += object_score * 0.3
                    match_reasons.append(f"객체 매칭: {', '.join(matching_objects)}")
            
            # 임계값 체크
            threshold = conditions.get('match_threshold', 0.3)
            overall_match = match_score >= threshold
            
            return {
                'overall_match': overall_match,
                'match_score': match_score,
                'match_reasons': match_reasons,
                'relevant_frames': relevant_frames[:10]  # 최대 10개
            }
            
        except Exception as e:
            print(f"⚠️ 검색 조건 체크 오류: {e}")
            return {
                'overall_match': False,
                'match_score': 0.0,
                'match_reasons': [f'오류: {str(e)}'],
                'relevant_frames': []
            }