import pandas as pd
from dotenv import load_dotenv   # .env 파일에서 환경변수(API키 등)를 불러오는 라이브러리

from crawler import TMDBCrawler   # 같은 폴더의 crawler.py에서 TMDBCrawler 클래스 가져오기
from preprocessing import TMDBPreProcessor    # 같은 폴더의 preprocessing.py에서 가져오기

# .env 파일에 저장된 환경변수를 불러옴
# 예: TMDB API 키 같은 민감한 정보를 코드에 직접 쓰지 않고 .env 파일에 보관
load_dotenv()


def run_popular_movie_crawler():
    """
    영화 데이터를 수집하고 가공하는 전체 과정을 실행하는 함수
    
    실행 순서:
    1. TMDB에서 영화 데이터 크롤링(수집)
    2. 수집한 원본 데이터를 JSON 파일로 저장
    3. 데이터 전처리(가공)
    4. 가공된 데이터를 CSV 파일로 저장
    """
    
    # ========== 1단계: 데이터 수집 (크롤링) ==========
    
    # TMDB 크롤러 객체 생성
    tmdb_crawler = TMDBCrawler()

    # TMDB API에서 영화 목록 가져오기
    # - start_page=1, end_page=10: 1페이지부터 10페이지까지 수집
    # - 한 페이지당 20개 영화 → 총 200개 영화 수집
    result = tmdb_crawler.get_bulk_popular_movies(start_page=1, end_page=10)

    # 수집한 원본 데이터를 JSON 파일로 저장
    # - 저장 위치: ./result/popular.json
    # - 나중에 다시 크롤링하지 않고 이 파일을 재사용할 수 있음
    tmdb_crawler.save_movies_to_json_file(result, "./result", "popular")

    # ========== 2단계: 데이터 가공 (전처리) ==========

    # 전처리 객체 생성 (수집한 영화 데이터를 전달)
    tmdb_preprocessor = TMDBPreProcessor(result)

    # 전처리 실행 (필요한 정보 추출 및 변환)
    tmdb_preprocessor.run()

    # 가공된 데이터를 CSV 파일로 저장
    # - 저장 위치: ./result/movies.csv
    tmdb_preprocessor.save("movie_features")

    print(f"완료! 총 {len(result)}개 영화 데이터 수집")


# ========== 프로그램 시작점 ==========
if __name__ == '__main__':
    run_popular_movie_crawler()
