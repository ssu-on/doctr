#!/usr/bin/env python3
"""
자막 Detection 모델 테스트 스크립트
"""

import os
import sys
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

# 현재 디렉토리를 Python path에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from doctr.models.detection.differentiable_binarization.pytorch import db_mobilenet_v3_small


class SubtitleDetectionTester:
    """자막 Detection 모델 테스트 클래스"""
    
    def __init__(self, model_path: str, device: str = 'cpu'):
        self.device = torch.device(device)
        self.model = self._load_model(model_path)
        
    def _load_model(self, model_path: str):
        """모델 로드"""
        print(f"모델 로딩: {model_path}")
        
        # 모델 생성
        model = db_mobilenet_v3_small(pretrained=False)
        
        # 체크포인트 로드
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"체크포인트에서 로드 (에포크: {checkpoint.get('epoch', 'N/A')})")
            else:
                model.load_state_dict(checkpoint)
                print("모델 가중치에서 로드")
        else:
            print("체크포인트 파일이 없습니다. 랜덤 초기화된 모델을 사용합니다.")
        
        model.to(self.device)
        model.eval()
        
        return model
    
    def preprocess_image(self, image_path: str, input_size: int = 1024):
        """이미지 전처리"""
        # 이미지 로드
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"이미지를 로드할 수 없습니다: {image_path}")
        
        # BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_shape = image.shape[:2]
        
        # 리사이즈
        image = cv2.resize(image, (input_size, input_size))
        
        # 정규화
        image = image.astype(np.float32) / 255.0
        mean = np.array([0.798, 0.785, 0.772])
        std = np.array([0.264, 0.2749, 0.287])
        image = (image - mean) / std
        
        # 텐서로 변환
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        
        return image_tensor, original_shape
    
    def postprocess_predictions(self, predictions, original_shape, input_size: int = 1024):
        """예측 결과 후처리"""
        h_orig, w_orig = original_shape
        scale_x = w_orig / input_size
        scale_y = h_orig / input_size
        
        processed_boxes = []
        
        for box in predictions['text']:
            # 좌표 스케일링
            x1 = int(box[0] * scale_x)
            y1 = int(box[1] * scale_y)
            x2 = int(box[2] * scale_x)
            y2 = int(box[3] * scale_y)
            
            # 신뢰도 점수 (있는 경우)
            confidence = box[4] if len(box) > 4 else 1.0
            
            processed_boxes.append({
                'bbox': [x1, y1, x2, y2],
                'confidence': confidence
            })
        
        return processed_boxes
    
    def detect_subtitles(self, image_path: str, confidence_threshold: float = 0.5):
        """자막 감지"""
        print(f"자막 감지 중: {image_path}")
        
        # 이미지 전처리
        image_tensor, original_shape = self.preprocess_image(image_path)
        image_tensor = image_tensor.to(self.device)
        
        # 추론
        with torch.no_grad():
            outputs = self.model(image_tensor, return_preds=True)
            predictions = outputs['preds'][0]
        
        # 후처리
        boxes = self.postprocess_predictions(predictions, original_shape)
        
        # 신뢰도 필터링
        filtered_boxes = [box for box in boxes if box['confidence'] >= confidence_threshold]
        
        print(f"감지된 자막: {len(filtered_boxes)}개")
        
        return filtered_boxes
    
    def visualize_results(self, image_path: str, boxes: list, output_path: str = None):
        """결과 시각화"""
        # 이미지 로드
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 박스 그리기
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box['bbox']
            confidence = box['confidence']
            
            # 박스 그리기
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 신뢰도 텍스트
            label = f'Subtitle {i+1}: {confidence:.2f}'
            cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # 결과 표시
        plt.figure(figsize=(12, 8))
        plt.imshow(image)
        plt.title(f'자막 감지 결과 ({len(boxes)}개 감지)')
        plt.axis('off')
        
        if output_path:
            plt.savefig(output_path, bbox_inches='tight', dpi=150)
            print(f"결과 저장: {output_path}")
        
        plt.show()
    
    def test_on_directory(self, image_dir: str, output_dir: str = None, confidence_threshold: float = 0.5):
        """디렉토리의 모든 이미지에 대해 테스트"""
        image_dir = Path(image_dir)
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # 지원하는 이미지 확장자
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        # 이미지 파일 찾기
        image_files = [f for f in image_dir.iterdir() 
                      if f.suffix.lower() in image_extensions]
        
        print(f"테스트할 이미지: {len(image_files)}개")
        
        results = []
        
        for image_file in image_files:
            try:
                # 자막 감지
                boxes = self.detect_subtitles(str(image_file), confidence_threshold)
                
                # 결과 저장
                result = {
                    'image': str(image_file),
                    'boxes': boxes,
                    'count': len(boxes)
                }
                results.append(result)
                
                # 시각화 (선택적)
                if output_dir:
                    output_path = output_dir / f'result_{image_file.stem}.png'
                    self.visualize_results(str(image_file), boxes, str(output_path))
                
            except Exception as e:
                print(f"오류 발생 ({image_file}): {e}")
                continue
        
        # 결과 요약
        total_boxes = sum(r['count'] for r in results)
        avg_boxes = total_boxes / len(results) if results else 0
        
        print(f"\n=== 테스트 결과 요약 ===")
        print(f"처리된 이미지: {len(results)}개")
        print(f"총 감지된 자막: {total_boxes}개")
        print(f"평균 자막 수: {avg_boxes:.2f}개/이미지")
        
        return results


def main():
    parser = argparse.ArgumentParser(description='자막 Detection 모델 테스트')
    parser.add_argument('--model_path', type=str, required=True, help='모델 체크포인트 경로')
    parser.add_argument('--image_path', type=str, help='테스트할 이미지 경로')
    parser.add_argument('--image_dir', type=str, help='테스트할 이미지 디렉토리')
    parser.add_argument('--output_dir', type=str, help='결과 저장 디렉토리')
    parser.add_argument('--confidence', type=float, default=0.5, help='신뢰도 임계값')
    parser.add_argument('--device', type=str, default='cpu', help='사용할 디바이스 (cpu/cuda)')
    
    args = parser.parse_args()
    
    # 테스터 생성
    tester = SubtitleDetectionTester(args.model_path, args.device)
    
    if args.image_path:
        # 단일 이미지 테스트
        boxes = tester.detect_subtitles(args.image_path, args.confidence)
        tester.visualize_results(args.image_path, boxes, args.output_dir)
        
    elif args.image_dir:
        # 디렉토리 테스트
        tester.test_on_directory(args.image_dir, args.output_dir, args.confidence)
        
    else:
        print("--image_path 또는 --image_dir 중 하나를 지정해주세요.")


if __name__ == '__main__':
    main()
