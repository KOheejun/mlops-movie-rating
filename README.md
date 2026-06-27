# TMDB 영화 평점 예측 MLOps 파이프라인

TMDB 영화 메타데이터를 기반으로 사용자에게 영화를 추천하는 콘텐츠 기반 모델을 MLOps 파이프라인으로 구현한 프로젝트입니다.

## 프로젝트 개요

- **모델**: 2-layer MLP (numpy 직접 구현, PyTorch 미사용)
- **파이프라인**: 데이터 수집 → 전처리 → 학습 → 추론 → DB 저장 → API 서빙
- **실험 관리**: Weights & Biases (wandb)
- **서빙**: FastAPI

## 프로젝트 구조

```
src/
├── main.py              # CLI 진입점 (fire 기반: train / inference)
├── webapp.py            # FastAPI 추론 서버
├── dataset/
│   ├── data_loader.py   # 데이터 로더
│   └── watch_log.py     # 시청 로그 처리
├── model/
│   └── movie_predictor.py  # 2-layer MLP (numpy)
├── train/
│   └── train.py         # 학습 루프 + wandb 로깅
├── evaluate/
│   └── evaluate.py      # 평가
├── inference/
│   └── inference.py     # 추론 + 체크포인트 로드
├── postprocess/
│   └── postprocess.py   # DB 저장
└── utils/
    ├── utils.py
    └── enums.py
```

## 실행 방법

```bash
pip install -r requirements.txt

# 환경 변수 설정 (.env 파일)
WANDB_API_KEY=your_key_here

# 학습
python main.py train movie_predictor --num_epochs 10 --batch_size 64

# 추론
python main.py inference

# API 서버 실행
uvicorn webapp:app --reload
```

## 기술 스택

| 구분 | 내용 |
|------|------|
| 모델 | 2-layer MLP (numpy 직접 구현) |
| 실험 관리 | Weights & Biases |
| CLI | Python Fire |
| 서빙 | FastAPI + uvicorn |
| 데이터 | TMDB API |
