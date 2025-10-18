#!/usr/bin/env python3
"""
자막 Detection 전체 파이프라인 실행 스크립트
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path


def run_command(cmd, description):
    """명령어 실행"""
    print(f"\n{'='*50}")
    print(f"🚀 {description}")
    print(f"{'='*50}")
    print(f"실행 명령: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print("✅ 성공!")
        if result.stdout:
            print("출력:")
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 실패: {e}")
        if e.stderr:
            print("오류:")
            print(e.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description='자막 Detection 전체 파이프라인')
    parser.add_argument('--mode', type=str, required=True, 
                       choices=['prepare', 'train', 'test', 'full'],
                       help='실행할 모드')
    parser.add_argument('--data_dir', type=str, default='./subtitle_data',
                       help='데이터 디렉토리')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                       help='출력 디렉토리')
    parser.add_argument('--model_path', type=str, default='./outputs/best_model.pt',
                       help='모델 경로')
    parser.add_argument('--test_image', type=str, help='테스트할 이미지 경로')
    parser.add_argument('--epochs', type=int, default=50, help='학습 에포크 수')
    parser.add_argument('--batch_size', type=int, default=4, help='배치 크기')
    parser.add_argument('--device', type=int, default=0, help='GPU 디바이스 번호')
    
    args = parser.parse_args()
    
    print("🎬 자막 Detection 파이프라인 시작!")
    print(f"모드: {args.mode}")
    
    success = True
    
    if args.mode == 'prepare' or args.mode == 'full':
        # 1. 데이터 준비
        cmd = f"python prepare_subtitle_data.py --create_sample --output_dir {args.data_dir}"
        success &= run_command(cmd, "샘플 데이터 생성")
    
    if args.mode == 'train' or args.mode == 'full':
        # 2. 학습
        train_cmd = f"""python train_subtitle_detection.py \
            --train_path {args.data_dir}/train \
            --val_path {args.data_dir}/val \
            --output_dir {args.output_dir} \
            --epochs {args.epochs} \
            --batch_size {args.batch_size} \
            --device {args.device}"""
        
        success &= run_command(train_cmd, "모델 학습")
    
    if args.mode == 'test' or args.mode == 'full':
        # 3. 테스트
        if args.test_image:
            test_cmd = f"""python test_subtitle_detection.py \
                --model_path {args.model_path} \
                --image_path {args.test_image} \
                --device cuda:{args.device}"""
        else:
            # 샘플 이미지로 테스트
            sample_image = f"{args.data_dir}/val/images/sample_0000.jpg"
            if os.path.exists(sample_image):
                test_cmd = f"""python test_subtitle_detection.py \
                    --model_path {args.model_path} \
                    --image_path {sample_image} \
                    --device cuda:{args.device}"""
            else:
                print("❌ 테스트할 이미지를 찾을 수 없습니다.")
                success = False
                test_cmd = None
        
        if test_cmd:
            success &= run_command(test_cmd, "모델 테스트")
    
    # 결과 출력
    print(f"\n{'='*50}")
    if success:
        print("🎉 파이프라인 완료!")
        print(f"결과는 {args.output_dir}에 저장되었습니다.")
    else:
        print("❌ 파이프라인 실패!")
        print("오류를 확인하고 다시 시도해주세요.")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
