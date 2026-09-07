# TMDB 영화 평점 예측 MLOps 파이프라인

[![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Docker](https://img.shields.io/badge/Docker-20.10+-blue?style=flat-square&logo=docker&logoColor=white)](https://docker.com)
[![AWS](https://img.shields.io/badge/AWS-EC2-orange?style=flat-square&logo=amazon-aws&logoColor=white)](https://aws.amazon.com)
[![WandB](https://img.shields.io/badge/WandB-MLOps-yellow?style=flat-square&logo=weightsandbiases&logoColor=white)](https://wandb.ai)
[![BentoML](https://img.shields.io/badge/BentoML-Model%20Serving-green?style=flat-square)](https://bentoml.org)
[![Airflow](https://img.shields.io/badge/Apache-Airflow-red?style=flat-square&logo=apache-airflow&logoColor=white)](https://airflow.apache.org)

> 비전공자 2명(AI 부트캠프 22기 MLOps Team 2)이 MLOps 학습을 위해 10일간(2026.01.20~01.30) 만든 영화 평점 예측 미니 프로젝트입니다.

## 프로젝트 소개

TMDB API로 영화 데이터를 수집해 평점을 예측하는 모델을 만들고, 수집 → 전처리 → 학습 → 등록 → 서빙 → 자동화까지 MLOps의 주요 컴포넌트를 직접 구현해봤습니다. 모델 성능보다 "전체 파이프라인이 실제로 동작하게 만드는 것"에 집중했습니다.

## 시스템 구조

```
TMDB API → Data Pipeline → Feature Store → Model Training → Model Registry → API Serving
                                                  ↓
                                    Airflow가 수집·학습 단계를 매일 자동 실행
```

## 구현한 주요 기능

- TMDB API를 통한 영화 데이터 수집 (`data-pipeline/`)
- 데이터 전처리 및 피처 생성 (popularity, vote_count, release_year, genre_count, is_adult, is_english)
- numpy로 직접 구현한 2-layer 신경망 회귀 모델 학습 (`src/model/movie_predictor.py`)
- WandB를 통한 실험 추적 (실행 8회)
- BentoML Model Store 등록 + Staging → PROD 태그 승격 (`serving/save_model.py`)
- BentoML 기반 REST API 서빙 (`serving/service.py`) — TMDB 제목 검색 연동 포함
- FastAPI 기반 실시간 추론 서버도 별도로 구현 (`src/webapp.py`)
- Apache Airflow DAG로 "데이터 수집 → 피처 저장 → 모델 학습" 3단계 자동화 (`airflow/dags/movie_pipeline.py`, 매일 실행)
- Docker(python:3.11-bookworm) 컨테이너를 AWS EC2에 올려 구동 확인

## 리포 구조

```
mlops-movie-rating/
├── data-pipeline/       # TMDB 데이터 수집
│   ├── main.py
│   ├── crawler.py
│   └── preprocessing.py
├── src/                 # 모델 학습·평가·추론·FastAPI 서빙
│   ├── dataset/
│   ├── model/           # numpy 2-layer 신경망 (movie_predictor.py)
│   ├── train/
│   ├── evaluate/
│   ├── inference/
│   ├── postprocess/
│   ├── utils/
│   ├── main.py          # CLI 진입점 (fire 기반: train / inference)
│   └── webapp.py        # FastAPI 추론 서버
├── serving/             # BentoML 기반 API 서버
│   ├── service.py
│   ├── save_model.py    # BentoML Model Store 저장 + Staging→PROD 승격
│   ├── bentofile.yaml
│   └── bentoml_registry.json
└── airflow/
    └── dags/
        └── movie_pipeline.py   # 수집→피처저장→학습 3-task DAG (매일 실행)
```

## 실행 방법

### 1. 의존성 설치

```bash
pip install -r requirements.txt
pip install bentoml   # serving/ 실행 시 추가로 필요
```

### 2. 환경 변수

```bash
cp .env.example .env                       # WANDB_API_KEY 등
cp data-pipeline/.env.example data-pipeline/.env   # TMDB_API_KEY 등
```

### 3. 데이터 수집

```bash
cd data-pipeline
python main.py
```

### 4. 모델 학습

```bash
python -m src.main train --model_name movie_predictor --num_epochs 10
```

### 5. BentoML 등록 + 서빙

```bash
python serving/save_model.py save --stage staging
python serving/save_model.py promote --latest-staging
cd serving && bentoml serve service:MovieRatingService --reload
# http://localhost:3000 (Swagger UI)
```

### 6. (대안) FastAPI 서버로 직접 서빙

```bash
uvicorn src.webapp:app --reload
```

### 7. Airflow 파이프라인

`airflow/dags/movie_pipeline.py`를 Airflow의 `dags/` 폴더에 두면 매일 자동으로 수집→피처저장→학습이 실행됩니다. `PROJECT_DIR` 상수를 본인 배포 경로에 맞게 바꿔야 합니다.

## API 사용 예시

```bash
# 숫자 feature 직접 입력
curl -X POST http://localhost:3000/predict \
  -H "Content-Type: application/json" \
  -d '{"popularity": 100, "vote_count": 1000, "release_year": 2023, "genre_count": 2, "is_adult": 0, "is_english": 1}'

# 영화 제목만으로 예측 (TMDB 연동)
curl -X POST http://localhost:3000/predict_by_title \
  -H "Content-Type: application/json" \
  -d '{"title": "인셉션"}'
```

## 배운 점

- **MLOps 전체 흐름 이해**: 데이터 수집부터 서비스 배포까지 전체 라이프사이클 경험
- **학습과 추론 분리**: Airflow 배치 학습과 BentoML 실시간 추론의 구조 이해
- **협업 도구 활용**: Git 브랜치 전략, WandB 실험 공유 경험
- **문제 해결 능력**: 비전공자로서 SSH, Docker, Python 에러를 직접 해결하며 자신감 획득

## 아쉬운 점 & 개선 방향

- **모델 성능**: 파이프라인 구축에 집중하느라 정확도 향상 부족 → 하이퍼파라미터 튜닝, 피처 엔지니어링 개선 필요
- **자동 배포 미완성**: Airflow는 수집·학습까지만 자동화했고, 모델 검증 이후 PROD 승격·배포는 수동으로 진행했습니다
- **로컬 저장소 의존**: Feature Store와 모델을 로컬에만 저장 → AWS S3, Feast 같은 외부 스토리지 전환 검토
- **데이터 버전 관리**: 매번 덮어쓰기 방식 → 날짜별 누적 저장 구조로 개선
- **MLflow는 설치만 해봤습니다**: `mlflow` 자체를 실행해본 기록은 남아있지만(로컬 트래킹 스토어에 ad-hoc run 2개) 학습 코드에 직접 연동하지는 않았습니다. 실질적인 실험 추적은 WandB로 진행했습니다
- **Dockerfile 미보존**: Docker(python:3.11-bookworm)·EC2 환경에서 실제로 구동을 확인했지만, 재현 가능한 Dockerfile/배포 스크립트를 커밋해두지 않아 이 리포에는 포함하지 못했습니다

## 참고 자료

- [TMDB API 문서](https://developers.themoviedb.org/3)
- [BentoML 문서](https://docs.bentoml.org/)
- [WandB 가이드](https://docs.wandb.ai/)
- [Apache Airflow 문서](https://airflow.apache.org/docs/)

---

**팀**: 김원재, 고희준 (AI 부트캠프 22기 MLOps Team 2, 2026.01.20~01.30)
