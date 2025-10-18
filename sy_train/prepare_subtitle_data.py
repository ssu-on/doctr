#!/usr/bin/env python3
"""
자막 Detection을 위한 데이터 준비 스크립트
"""

import os
import json
import cv2
import numpy as np
from pathlib import Path
import argparse
from typing import List, Dict, Tuple
import xml.etree.ElementTree as ET


class SubtitleDataPreparer:
    """자막 데이터 준비 클래스"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # train/val 디렉토리 생성
        (self.output_dir / 'train').mkdir(exist_ok=True)
        (self.output_dir / 'val').mkdir(exist_ok=True)
    
    def create_sample_data(self, num_train: int = 100, num_val: int = 20):
        """샘플 데이터 생성 (실제 사용시에는 실제 자막 데이터로 교체)"""
        print("샘플 자막 데이터 생성 중...")
        
        # 학습 데이터 생성
        self._create_samples('train', num_train)
        
        # 검증 데이터 생성
        self._create_samples('val', num_val)
        
        print(f"데이터 생성 완료!")
        print(f"  - 학습 데이터: {num_train}개")
        print(f"  - 검증 데이터: {num_val}개")
        print(f"  - 저장 위치: {self.output_dir}")
    
    def _create_samples(self, split: str, num_samples: int):
        """샘플 데이터 생성"""
        split_dir = self.output_dir / split
        images_dir = split_dir / 'images'
        labels_dir = split_dir / 'labels'
        
        images_dir.mkdir(exist_ok=True)
        labels_dir.mkdir(exist_ok=True)
        
        for i in range(num_samples):
            # 랜덤 이미지 생성 (자막이 있는 영상 프레임 시뮬레이션)
            img, boxes = self._generate_sample_image()
            
            # 이미지 저장
            img_path = images_dir / f'sample_{i:04d}.jpg'
            cv2.imwrite(str(img_path), img)
            
            # 라벨 저장 (JSON 형식)
            label_path = labels_dir / f'sample_{i:04d}.json'
            self._save_label(label_path, boxes, img.shape)
    
    def _generate_sample_image(self) -> Tuple[np.ndarray, List[List[float]]]:
        """샘플 이미지와 자막 박스 생성"""
        # 배경 이미지 생성 (영상 프레임 시뮬레이션)
        height, width = 720, 1280
        img = np.random.randint(0, 50, (height, width, 3), dtype=np.uint8)
        
        # 그라데이션 배경 추가
        for y in range(height):
            img[y, :] = img[y, :] + int(30 * y / height)
        
        boxes = []
        
        # 1-3개의 자막 박스 생성
        num_subtitles = np.random.randint(1, 4)
        
        for _ in range(num_subtitles):
            # 자막 박스 크기와 위치 (정규화된 좌표)
            box_width = np.random.uniform(0.3, 0.8)  # 화면 너비의 30-80%
            box_height = np.random.uniform(0.05, 0.15)  # 화면 높이의 5-15%
            
            x1 = np.random.uniform(0.1, 1.0 - box_width)
            y1 = np.random.uniform(0.7, 0.9)  # 화면 하단에 자막 배치
            x2 = x1 + box_width
            y2 = y1 + box_height
            
            # 좌표 정규화
            x1, x2 = x1, x2
            y1, y2 = y1, y2
            
            boxes.append([x1, y1, x2, y2])
            
            # 자막 박스 그리기 (시각화용)
            x1_pixel = int(x1 * width)
            y1_pixel = int(y1 * height)
            x2_pixel = int(x2 * width)
            y2_pixel = int(y2 * height)
            
            # 반투명 검은색 배경
            overlay = img.copy()
            cv2.rectangle(overlay, (x1_pixel, y1_pixel), (x2_pixel, y2_pixel), (0, 0, 0), -1)
            img = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
            
            # 흰색 테두리
            cv2.rectangle(img, (x1_pixel, y1_pixel), (x2_pixel, y2_pixel), (255, 255, 255), 2)
        
        return img, boxes
    
    def _save_label(self, label_path: Path, boxes: List[List[float]], img_shape: Tuple[int, int, int]):
        """라벨 파일 저장 (doctr 형식)"""
        # doctr DetectionDataset 형식으로 저장
        label_data = {
            "boxes": boxes,
            "labels": ["text"] * len(boxes)  # 모든 박스는 텍스트로 라벨링
        }
        
        with open(label_path, 'w', encoding='utf-8') as f:
            json.dump(label_data, f, ensure_ascii=False, indent=2)
    
    def convert_from_other_format(self, input_dir: str, format_type: str = 'coco'):
        """다른 형식의 데이터를 doctr 형식으로 변환"""
        print(f"{format_type} 형식 데이터 변환 중...")
        
        if format_type == 'coco':
            self._convert_from_coco(input_dir)
        elif format_type == 'pascal_voc':
            self._convert_from_pascal_voc(input_dir)
        else:
            raise ValueError(f"지원하지 않는 형식: {format_type}")
    
    def _convert_from_coco(self, input_dir: str):
        """COCO 형식에서 변환"""
        # COCO 형식 변환 로직 구현
        pass
    
    def _convert_from_pascal_voc(self, input_dir: str):
        """Pascal VOC 형식에서 변환"""
        # Pascal VOC 형식 변환 로직 구현
        pass


def create_data_structure_example():
    """데이터 구조 예시 생성"""
    example_structure = """
자막 Detection 데이터 구조:

data/
├── train/
│   ├── images/
│   │   ├── sample_0000.jpg
│   │   ├── sample_0001.jpg
│   │   └── ...
│   └── labels/
│       ├── sample_0000.json
│       ├── sample_0001.json
│       └── ...
└── val/
    ├── images/
    │   ├── sample_0000.jpg
    │   ├── sample_0001.jpg
    │   └── ...
    └── labels/
        ├── sample_0000.json
        ├── sample_0001.json
        └── ...

각 JSON 파일 형식:
{
  "boxes": [
    [x1, y1, x2, y2],  // 정규화된 좌표 (0-1)
    [x1, y1, x2, y2]
  ],
  "labels": ["text", "text"]
}
"""
    print(example_structure)


def main():
    parser = argparse.ArgumentParser(description='자막 Detection 데이터 준비')
    parser.add_argument('--output_dir', type=str, default='./subtitle_data', help='출력 디렉토리')
    parser.add_argument('--num_train', type=int, default=100, help='학습 데이터 수')
    parser.add_argument('--num_val', type=int, default=20, help='검증 데이터 수')
    parser.add_argument('--create_sample', action='store_true', help='샘플 데이터 생성')
    parser.add_argument('--show_structure', action='store_true', help='데이터 구조 예시 보기')
    
    args = parser.parse_args()
    
    if args.show_structure:
        create_data_structure_example()
        return
    
    if args.create_sample:
        preparer = SubtitleDataPreparer(args.output_dir)
        preparer.create_sample_data(args.num_train, args.num_val)
        
        print("\n사용법:")
        print(f"python train_subtitle_detection.py --train_path {args.output_dir}/train --val_path {args.output_dir}/val")
    else:
        print("--create_sample 플래그를 사용하여 샘플 데이터를 생성하거나")
        print("--show_structure 플래그를 사용하여 데이터 구조를 확인하세요.")


if __name__ == '__main__':
    main()
