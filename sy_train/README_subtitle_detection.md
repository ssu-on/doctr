# 자막 Detection을 위한 DBNet MobileNetV3-Small 학습

이 프로젝트는 DBNet with MobileNetV3-Small backbone을 사용하여 자막을 감지하는 모델을 학습하는 코드입니다.

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 의존성 설치
pip install torch torchvision
pip install opencv-python
pip install matplotlib
pip install tqdm
pip install numpy
```

### 2. 샘플 데이터 생성

```bash
# 샘플 데이터 생성 (테스트용)
python prepare_subtitle_data.py --create_sample --num_train 100 --num_val 20
```

### 3. 학습 실행

```bash
# 기본 학습
python train_subtitle_detection.py --train_path ./subtitle_data/train --val_path ./subtitle_data/val

# 고급 설정으로 학습
python train_subtitle_detection.py \
    --train_path ./subtitle_data/train \
    --val_path ./subtitle_data/val \
    --epochs 100 \
    --batch_size 8 \
    --lr 0.001 \
    --device 0 \
    --output_dir ./outputs
```

## 📁 데이터 구조

```
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
```

### JSON 라벨 형식

```json
{
  "boxes": [
    [x1, y1, x2, y2],  // 정규화된 좌표 (0-1)
    [x1, y1, x2, y2]
  ],
  "labels": ["text", "text"]
}
```

## ⚙️ 학습 설정

### 주요 하이퍼파라미터

| 파라미터 | 기본값 | 설명 |
|---------|--------|------|
| `--epochs` | 50 | 학습 에포크 수 |
| `--batch_size` | 4 | 배치 크기 |
| `--lr` | 0.001 | 학습률 |
| `--input_size` | 1024 | 입력 이미지 크기 |
| `--optimizer` | adamw | 옵티마이저 (adam/adamw) |
| `--scheduler` | cosine | 학습률 스케줄러 (cosine/onecycle/none) |

### GPU 사용

```bash
# GPU 0번 사용
python train_subtitle_detection.py --device 0 ...

# CPU 사용
python train_subtitle_detection.py --device -1 ...
```

## 📊 모니터링

학습 중 다음 정보가 출력됩니다:

- **학습 손실**: 각 배치의 평균 손실
- **검증 손실**: 각 에포크의 검증 손실
- **학습률**: 현재 학습률
- **진행률**: tqdm을 사용한 진행률 표시

### 학습 곡선

학습 완료 후 `outputs/training_curves.png`에 다음 그래프가 저장됩니다:

1. **손실 곡선**: 학습/검증 손실 변화
2. **학습률 곡선**: 학습률 스케줄 변화

## 💾 모델 저장

### 체크포인트

- `subtitle_detection_epoch_{epoch}.pt`: 각 에포크별 모델
- `best_model.pt`: 최고 성능 모델

### 체크포인트 내용

```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'val_loss': val_loss,
    'train_losses': train_losses,
    'val_losses': val_losses,
    'learning_rates': learning_rates,
}
```

## 🔧 모델 사용법

### 학습된 모델 로드

```python
import torch
from doctr.models.detection.differentiable_binarization.pytorch import db_mobilenet_v3_small

# 모델 생성
model = db_mobilenet_v3_small(pretrained=False)

# 체크포인트 로드
checkpoint = torch.load('best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# 추론 모드로 설정
model.eval()
```

### 추론 실행

```python
import torch
import cv2
import numpy as np

# 이미지 로드
image = cv2.imread('test_image.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 전처리
image = cv2.resize(image, (1024, 1024))
image = image.astype(np.float32) / 255.0
image = (image - np.array([0.798, 0.785, 0.772])) / np.array([0.264, 0.2749, 0.287])
image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)

# 추론
with torch.no_grad():
    outputs = model(image, return_preds=True)
    predictions = outputs['preds'][0]

# 결과 출력
for pred in predictions['text']:
    print(f"자막 박스: {pred}")
```

## 🎯 성능 최적화

### 배치 크기 조정

```bash
# GPU 메모리에 따라 조정
python train_subtitle_detection.py --batch_size 8 ...  # 더 큰 배치
python train_subtitle_detection.py --batch_size 2 ...  # 더 작은 배치
```

### 학습률 조정

```bash
# 더 높은 학습률
python train_subtitle_detection.py --lr 0.01 ...

# 더 낮은 학습률
python train_subtitle_detection.py --lr 0.0001 ...
```

### 입력 크기 조정

```bash
# 더 작은 입력 크기 (빠른 학습)
python train_subtitle_detection.py --input_size 512 ...

# 더 큰 입력 크기 (정확한 감지)
python train_subtitle_detection.py --input_size 1024 ...
```

## 🐛 문제 해결

### 메모리 부족

```bash
# 배치 크기 줄이기
python train_subtitle_detection.py --batch_size 1 ...

# 입력 크기 줄이기
python train_subtitle_detection.py --input_size 512 ...
```

### 학습이 느린 경우

```bash
# 워커 수 늘리기
python train_subtitle_detection.py --workers 8 ...

# 배치 크기 늘리기
python train_subtitle_detection.py --batch_size 8 ...
```

## 📈 실험 추적

### Weights & Biases 사용

```bash
# W&B 로깅 활성화 (추후 구현)
python train_subtitle_detection.py --use_wandb ...
```

### ClearML 사용

```bash
# ClearML 로깅 활성화 (추후 구현)
python train_subtitle_detection.py --use_clearml ...
```

## 🔄 데이터 증강

현재 구현된 변환:

- **Resize**: 입력 크기로 리사이즈
- **Normalize**: ImageNet 통계로 정규화

추가 가능한 변환:

- **RandomRotation**: 회전
- **RandomBrightness**: 밝기 조정
- **RandomContrast**: 대비 조정
- **RandomCrop**: 크롭

## 📝 주의사항

1. **데이터 형식**: JSON 라벨 파일은 정확한 형식을 따라야 합니다
2. **좌표 정규화**: 모든 박스 좌표는 0-1 범위로 정규화되어야 합니다
3. **이미지 크기**: 모든 이미지는 동일한 크기로 리사이즈됩니다
4. **메모리 사용량**: 배치 크기와 입력 크기에 따라 메모리 사용량이 달라집니다

## 🤝 기여하기

1. 이슈 리포트
2. 기능 제안
3. 코드 개선
4. 문서 업데이트

## 📄 라이선스

이 프로젝트는 Apache 2.0 라이선스를 따릅니다.
