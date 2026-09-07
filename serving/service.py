"""
BentoML 기반 영화 평점 예측 API 서비스 (TMDB 연동 버전)

이 파일의 역할:
- 학습된 모델을 HTTP API로 제공하는 "서버"
- save_model.py에서 PROD로 승격된 모델만 로드하여 예측 수행

핵심 MLOps 포인트:
- service.py는 "판단"을 하지 않는다
- 어떤 모델이 PROD인지는 bentoml_registry.json에 이미 결정되어 있음
- service는 그 결정을 "읽어서" 그대로 서빙만 한다

실행 방법:
    (serving 폴더에서)
    bentoml serve service:MovieRatingService --reload

실행 후 접속:
    http://localhost:3000 (Swagger UI 제공)
"""

import os
import sys
import json
import numpy as np
import requests
import bentoml
from dotenv import load_dotenv

# ============================================================
# 프로젝트 경로 설정
# ------------------------------------------------------------
# 이 리포에서는 학습 코드가 <repo_root>/src/ 에 있으므로
# (원본 프로젝트에서는 <repo_root>/mlops/src/ 였음),
# repo 루트를 sys.path에 추가해 `from src...` import가 되게 한다.
# ============================================================
BASE_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
sys.path.insert(0, REPO_ROOT)

# ============================================================
# 환경변수 로드 (.env)
# ============================================================
# TMDB API 키는 코드에 하드코딩 ❌
# data-pipeline/.env 에서 로드
load_dotenv(os.path.join(BASE_DIR, "..", "data-pipeline", ".env"))

# ============================================================
# PROD 모델 레지스트리 경로
# ============================================================
# save_model.py에서 자동 관리되는 파일
REGISTRY_PATH = os.path.join(BASE_DIR, "bentoml_registry.json")


@bentoml.service
class MovieRatingService:
    """
    영화 평점 예측 BentoML 서비스
    """

    def __init__(self):
        """
        서버 시작 시 1회 실행

        하는 일:
        1) bentoml_registry.json에서 prod_tag 조회
        2) BentoML Model Store에서 해당 모델 로드
        3) scaler / scaler_y 함께 로드
        4) TMDB API 설정 로드
        """

        # ====================================================
        # 1. PROD 모델 태그 결정
        # ====================================================
        if not os.path.exists(REGISTRY_PATH):
            raise RuntimeError(
                "bentoml_registry.json not found. "
                "Run save_model.py save & promote first."
            )

        with open(REGISTRY_PATH, "r", encoding="utf-8") as f:
            registry = json.load(f)

        # PROD 우선, 없으면 STAGING fallback
        model_tag = registry.get("prod_tag") or registry.get("staging_tag")
        if not model_tag:
            raise RuntimeError("No prod/staging model tag found in registry")

        # ====================================================
        # 2. BentoML Model Store에서 모델 로드
        # ====================================================
        bento_model = bentoml.models.get(model_tag)

        # BentoML 1.4+ 권장 방식
        self.model = bento_model.load_model()

        # 학습 시 함께 저장한 객체들
        self.scaler = bento_model.custom_objects.get("scaler")
        self.scaler_y = bento_model.custom_objects.get("scaler_y")

        self.model_tag = model_tag
        print(f"✅ Loaded PROD model: {self.model_tag}")

        # ====================================================
        # 3. TMDB API 설정
        # ====================================================
        self.tmdb_base_url = os.environ.get("TMDB_BASE_URL")
        self.tmdb_api_key = os.environ.get("TMDB_API_KEY")

    # ============================================================
    # TMDB 검색 유틸
    # ============================================================
    def _search_movie(self, title: str) -> dict | None:
        """
        TMDB API로 영화 검색

        반환:
            - 영화 dict (성공)
            - None (실패)
        """
        search_url = self.tmdb_base_url.replace("/movie", "") + "/search/movie"

        params = {
            "api_key": self.tmdb_api_key,
            "query": title,
            "language": "ko-KR",
        }

        res = requests.get(search_url, params=params)
        if res.status_code != 200:
            return None

        results = res.json().get("results", [])
        return results[0] if results else None

    # ============================================================
    # API 1) 숫자 기반 예측
    # ============================================================
    @bentoml.api
    def predict(
        self,
        popularity: float = 100.0,
        vote_count: int = 1000,
        release_year: int = 2023,
        genre_count: int = 2,
        is_adult: int = 0,
        is_english: int = 1,
    ) -> dict:
        """
        숫자 feature 직접 입력 예측
        """

        features = np.array(
            [[popularity, vote_count, release_year, genre_count, is_adult, is_english]]
        )

        if self.scaler is not None:
            features = self.scaler.transform(features)

        # MoviePredictor는 forward 기반
        prediction = self.model.forward(features)

        if self.scaler_y is not None:
            prediction = self.scaler_y.inverse_transform(
                prediction.reshape(-1, 1)
            )

        return {
            "predicted_rating": round(float(prediction[0]), 2),
            "model_tag": self.model_tag,
            "input": {
                "popularity": popularity,
                "vote_count": vote_count,
                "release_year": release_year,
                "genre_count": genre_count,
                "is_adult": is_adult,
                "is_english": is_english,
            },
        }

    # ============================================================
    # API 2) 영화 제목 기반 예측 (TMDB 연동)
    # ============================================================
    @bentoml.api
    def predict_by_title(self, title: str = "Inception") -> dict:
        """
        영화 제목만으로 예측 (TMDB 연동)
        """

        movie = self._search_movie(title)
        if movie is None:
            return {"error": f"'{title}' 영화를 찾을 수 없습니다."}

        popularity = movie.get("popularity", 0)
        vote_count = movie.get("vote_count", 0)

        release_date = movie.get("release_date", "")
        release_year = int(release_date[:4]) if release_date else 2023

        genre_count = len(movie.get("genre_ids", []))
        is_adult = int(movie.get("adult", False))
        is_english = int(movie.get("original_language") == "en")

        features = np.array(
            [[popularity, vote_count, release_year, genre_count, is_adult, is_english]]
        )

        if self.scaler is not None:
            features = self.scaler.transform(features)

        prediction = self.model.forward(features)

        if self.scaler_y is not None:
            prediction = self.scaler_y.inverse_transform(
                prediction.reshape(-1, 1)
            )

        return {
            "predicted_rating": round(float(prediction[0]), 2),
            "actual_rating": movie.get("vote_average"),
            "model_tag": self.model_tag,
            "movie_info": {
                "title": movie.get("title"),
                "release_date": release_date,
                "popularity": popularity,
                "vote_count": vote_count,
                "genre_count": genre_count,
                "is_adult": is_adult,
                "is_english": is_english,
            },
        }

# ============================================================
# predict() vs predict_by_title() 비교
# ============================================================
#
#   predict()                      predict_by_title()
#   ─────────────────────         ─────────────────────
#   숫자 직접 입력                 영화 제목만 입력
#   TMDB 조회 안 함                TMDB에서 자동 조회
#   빠름 (외부 API 호출 없음)      느림 (API 호출 있음)
#   실제 평점 비교 불가            실제 평점 비교 가능
#
# ============================================================
# API 호출 예시
# ============================================================
#
#   # 숫자 기반 예측
#   curl -X POST http://localhost:3000/predict \
#     -H "Content-Type: application/json" \
#     -d '{}'
#
#   # 제목 기반 예측
#   curl -X POST http://localhost:3000/predict_by_title \
#     -H "Content-Type: application/json" \
#     -d '{"title": "인셉션"}'
#
# ============================================================