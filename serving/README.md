# Movie Rating Prediction API

BentoML 기반 영화 평점 예측 API 서비스

## 설치

```bash
pip install bentoml numpy scikit-learn
```

## 모델 저장

```bash
python save_model.py
```

## 서버 실행

```bash
bentoml serve service:MovieRatingService --reload
```

## API 사용법

### 직접 입력으로 예측

#### 요청

```bash
curl -X POST http://localhost:3000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "popularity": 100,
    "vote_count": 1000,
    "release_year": 2023,
    "genre_count": 2,
    "is_adult": 0,
    "is_english": 1
  }'
```

#### 응답

```json
{
  "predicted_rating": 6.5,
  "input": {
    "popularity": 100,
    "vote_count": 1000,
    "release_year": 2023,
    "genre_count": 2,
    "is_adult": 0,
    "is_english": 1
  }
}
```


### 영화 제목으로 예측

#### 요청

```bash
curl -X POST http://localhost:3000/predict_by_title \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Inception"
  }'
```

#### 응답

```json
{
  "predicted_rating": 0.34,
  "actual_rating": 8.4,
  "movie_info": {
    "title": "인셉션",
    "original_title": "Inception",
    "release_date": "2010-07-15",
    "popularity": 95.0,
    "vote_count": 38000
  }
}
```


## 입력 파라미터

| 파라미터 | 타입 | 설명 |
|---------|------|------|
| popularity | float | 인기도 점수 |
| vote_count | int | 투표(리뷰) 수 |
| release_year | int | 개봉 연도 |
| genre_count | int | 장르 개수 |
| is_adult | int | 성인 영화 여부 (0 또는 1) |
| is_english | int | 영어 영화 여부 (0 또는 1) |


##  demo_api.py는 발표용 API 동작 검증을 위한 임시 데모 파일입니다. 이후 제거 예정
