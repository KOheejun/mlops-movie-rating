"""
영화 평점 예측 MLOps 파이프라인 DAG

이 DAG는 다음 작업들을 순서대로 자동 실행합니다:
1. TMDB에서 영화 데이터 수집 (Data Pipeline)
2. 수집한 데이터를 Feature Store로 복사
3. 모델 학습 (Automated Pipeline)
"""

from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

# ========== DAG 기본 설정 ==========
#
# default_args: 모든 Task에 공통으로 적용되는 설정
default_args = {
    'owner': 'team-2',                      # DAG 소유자
    'depends_on_past': False,               # 이전 실행 결과에 의존하지 않음
    'email_on_failure': False,              # 실패해도 이메일 안 보냄
    'email_on_retry': False,                # 재시도해도 이메일 안 보냄
    'retries': 1,                           # 실패하면 1번 재시도
    'retry_delay': timedelta(minutes=5),    # 재시도 전 5분 대기
}

# ========== DAG 정의 ==========
#
# DAG: Directed Acyclic Graph (방향성 비순환 그래프)
# 쉽게 말해 "작업 순서도"
dag = DAG(
    'movie_rating_pipeline',                # DAG 이름 (Airflow 웹에서 보임)
    default_args=default_args,
    description='TMDB 영화 데이터 수집 및 모델 학습 파이프라인',
    schedule_interval=timedelta(days=1),    # 매일 실행
    start_date=datetime(2026, 1, 1),        # 시작 날짜
    catchup=False,                          # 과거 날짜 실행 안 함
    tags=['mlops', 'movie', 'team-2'],      # 태그 (검색/필터용)
)

# ========== 프로젝트 경로 설정 ==========
PROJECT_DIR = '/opt/project-mlops-ai22-team-2-1'

# ========== Task 1: 데이터 수집 ==========
#
# data-pipeline/main.py 실행
# TMDB API에서 영화 데이터를 가져와서 CSV로 저장
task_collect_data = BashOperator(
    task_id='collect_movie_data',           # Task 이름
    bash_command=f'cd {PROJECT_DIR}/data-pipeline && python main.py',
    dag=dag,
)

# ========== Task 2: Feature Store로 복사 ==========
#
# 수집한 데이터를 mlops/dataset 폴더로 복사
task_copy_to_feature_store = BashOperator(
    task_id='copy_to_feature_store',
    bash_command=f'cp {PROJECT_DIR}/data-pipeline/result/movie_features.csv {PROJECT_DIR}/mlops/dataset/',
    dag=dag,
)

# ========== Task 3: 모델 학습 ==========
#
# mlops/src/main.py 실행
# 모델 학습 및 WandB에 기록
task_train_model = BashOperator(
    task_id='train_model',
    bash_command=f'cd {PROJECT_DIR}/mlops && python -m src.main train --model_name movie_predictor --num_epochs 10',
    dag=dag,
)

# ========== Task 순서 정의 ==========
#
# >> 는 "다음에 실행"이라는 뜻
# collect_data 끝나면 → copy_to_feature_store 실행 → train_model 실행
task_collect_data >> task_copy_to_feature_store >> task_train_model
