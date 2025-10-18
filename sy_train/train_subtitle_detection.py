#!/usr/bin/env python3
"""
자막 Detection을 위한 DBNet MobileNetV3-Small 학습 스크립트
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from tqdm import tqdm
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# 현재 디렉토리를 Python path에 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# doctr 모듈 import
from doctr.models.detection.differentiable_binarization.pytorch import db_mobilenet_v3_small
from doctr.datasets import DetectionDataset
from doctr import transforms as T


class SubtitleDetectionTrainer:
    """자막 Detection을 위한 학습 클래스"""
    
    def __init__(self, args):
        self.args = args
        self.device = torch.device(f'cuda:{args.device}' if torch.cuda.is_available() and args.device is not None else 'cpu')
        print(f"사용 디바이스: {self.device}")
        
        # 모델 생성
        self.model = db_mobilenet_v3_small(pretrained=False).to(self.device)
        print(f"모델 파라미터 수: {sum(p.numel() for p in self.model.parameters()):,}")
        
        # 옵티마이저 설정
        if args.optimizer == 'adam':
            self.optimizer = Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        else:
            self.optimizer = AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        # 스케줄러 설정
        if args.scheduler == 'cosine':
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=args.epochs)
        elif args.scheduler == 'onecycle':
            self.scheduler = OneCycleLR(self.optimizer, max_lr=args.lr, epochs=args.epochs, steps_per_epoch=1)
        else:
            self.scheduler = None
        
        # 데이터 변환 설정
        self.setup_transforms()
        
        # 학습 기록
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
    
    def setup_transforms(self):
        """데이터 변환 설정"""
        # 학습용 변환
        self.train_transforms = T.Compose([
            T.Resize((self.args.input_size, self.args.input_size)),
            T.Normalize(mean=(0.798, 0.785, 0.772), std=(0.264, 0.2749, 0.287)),
        ])
        
        # 검증용 변환
        self.val_transforms = T.Compose([
            T.Resize((self.args.input_size, self.args.input_size)),
            T.Normalize(mean=(0.798, 0.785, 0.772), std=(0.264, 0.2749, 0.287)),
        ])
    
    def create_datasets(self):
        """데이터셋 생성"""
        print("데이터셋 로딩 중...")
        
        # 학습 데이터셋
        self.train_dataset = DetectionDataset(
            img_folder=self.args.train_path,
            label_folder=self.args.train_path,
            img_transforms=self.train_transforms,
            use_polygons=False,  # 직사각형 박스 사용
        )
        
        # 검증 데이터셋
        self.val_dataset = DetectionDataset(
            img_folder=self.args.val_path,
            label_folder=self.args.val_path,
            img_transforms=self.val_transforms,
            use_polygons=False,
        )
        
        print(f"학습 데이터: {len(self.train_dataset)}개")
        print(f"검증 데이터: {len(self.val_dataset)}개")
        
        # 데이터로더 생성
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=self.args.workers,
            collate_fn=self.train_dataset.collate_fn,
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.args.batch_size,
            shuffle=False,
            num_workers=self.args.workers,
            collate_fn=self.val_dataset.collate_fn,
        )
    
    def train_epoch(self, epoch):
        """한 에포크 학습"""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.args.epochs}')
        
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(self.device)
            
            # 타겟을 디바이스로 이동
            targets = [target.to(self.device) for target in targets]
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(images, target=targets)
            loss = outputs['loss']
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # 진행률 업데이트
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss/(batch_idx+1):.4f}',
                'LR': f'{self.optimizer.param_groups[0]["lr"]:.6f}'
            })
        
        avg_loss = total_loss / num_batches
        self.train_losses.append(avg_loss)
        self.learning_rates.append(self.optimizer.param_groups[0]['lr'])
        
        return avg_loss
    
    def validate_epoch(self, epoch):
        """한 에포크 검증"""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Validation {epoch+1}/{self.args.epochs}')
            
            for batch_idx, (images, targets) in enumerate(pbar):
                images = images.to(self.device)
                targets = [target.to(self.device) for target in targets]
                
                outputs = self.model(images, target=targets)
                loss = outputs['loss']
                
                total_loss += loss.item()
                
                pbar.set_postfix({'Val Loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        self.val_losses.append(avg_loss)
        
        return avg_loss
    
    def save_checkpoint(self, epoch, val_loss, is_best=False):
        """체크포인트 저장"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'learning_rates': self.learning_rates,
        }
        
        if self.scheduler:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # 모델 저장
        model_path = Path(self.args.output_dir) / f'subtitle_detection_epoch_{epoch+1}.pt'
        torch.save(checkpoint, model_path)
        
        if is_best:
            best_path = Path(self.args.output_dir) / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"최고 성능 모델 저장: {best_path}")
        
        print(f"체크포인트 저장: {model_path}")
    
    def plot_training_curves(self):
        """학습 곡선 그리기"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 손실 곡선
        epochs = range(1, len(self.train_losses) + 1)
        ax1.plot(epochs, self.train_losses, 'b-', label='Training Loss')
        ax1.plot(epochs, self.val_losses, 'r-', label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 학습률 곡선
        ax2.plot(epochs, self.learning_rates, 'g-', label='Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Learning Rate')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(Path(self.args.output_dir) / 'training_curves.png')
        plt.show()
    
    def train(self):
        """전체 학습 과정"""
        print("자막 Detection 학습 시작!")
        print(f"학습 설정:")
        print(f"  - 에포크: {self.args.epochs}")
        print(f"  - 배치 크기: {self.args.batch_size}")
        print(f"  - 학습률: {self.args.lr}")
        print(f"  - 옵티마이저: {self.args.optimizer}")
        print(f"  - 스케줄러: {self.args.scheduler}")
        print(f"  - 입력 크기: {self.args.input_size}")
        
        # 데이터셋 생성
        self.create_datasets()
        
        # 출력 디렉토리 생성
        Path(self.args.output_dir).mkdir(parents=True, exist_ok=True)
        
        best_val_loss = float('inf')
        
        for epoch in range(self.args.epochs):
            print(f"\n=== Epoch {epoch+1}/{self.args.epochs} ===")
            
            # 학습
            train_loss = self.train_epoch(epoch)
            
            # 검증
            val_loss = self.validate_epoch(epoch)
            
            # 스케줄러 업데이트
            if self.scheduler:
                if isinstance(self.scheduler, OneCycleLR):
                    self.scheduler.step()
                else:
                    self.scheduler.step()
            
            # 결과 출력
            print(f"학습 손실: {train_loss:.4f}")
            print(f"검증 손실: {val_loss:.4f}")
            print(f"현재 학습률: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # 최고 성능 모델 저장
            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                print("🎉 새로운 최고 성능!")
            
            # 체크포인트 저장
            if (epoch + 1) % self.args.save_interval == 0 or is_best:
                self.save_checkpoint(epoch, val_loss, is_best)
        
        # 최종 모델 저장
        self.save_checkpoint(self.args.epochs - 1, best_val_loss, False)
        
        # 학습 곡선 그리기
        self.plot_training_curves()
        
        print(f"\n학습 완료! 최고 검증 손실: {best_val_loss:.4f}")


def parse_args():
    """명령행 인자 파싱"""
    parser = argparse.ArgumentParser(description='자막 Detection을 위한 DBNet MobileNetV3-Small 학습')
    
    # 데이터 관련
    parser.add_argument('--train_path', type=str, required=True, help='학습 데이터 경로')
    parser.add_argument('--val_path', type=str, required=True, help='검증 데이터 경로')
    parser.add_argument('--output_dir', type=str, default='./outputs', help='출력 디렉토리')
    
    # 모델 관련
    parser.add_argument('--input_size', type=int, default=1024, help='입력 이미지 크기')
    parser.add_argument('--device', type=int, default=0, help='사용할 GPU 디바이스 번호')
    
    # 학습 관련
    parser.add_argument('--epochs', type=int, default=50, help='학습 에포크 수')
    parser.add_argument('--batch_size', type=int, default=4, help='배치 크기')
    parser.add_argument('--lr', type=float, default=0.001, help='학습률')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='가중치 감쇠')
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw'], help='옵티마이저')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['cosine', 'onecycle', 'none'], help='학습률 스케줄러')
    
    # 기타
    parser.add_argument('--workers', type=int, default=4, help='데이터로더 워커 수')
    parser.add_argument('--save_interval', type=int, default=10, help='체크포인트 저장 간격')
    
    return parser.parse_args()


def main():
    """메인 함수"""
    args = parse_args()
    
    # 학습 시작
    trainer = SubtitleDetectionTrainer(args)
    trainer.train()


if __name__ == '__main__':
    main()
