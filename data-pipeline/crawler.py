# 1. built-in libraries (파이썬 기본 제공 라이브러리)
import os      # 운영체제 관련 기능 (환경변수 읽기 등)
import json    # JSON 데이터 처리
import time    # 시간 관련 기능 (딜레이 주기)

# 2. external libraries (설치해야 하는 외부 라이브러리)
import requests  # HTTP 요청 보내기 (API 호출용)

# 3. in project


class TMDBCrawler:
    """
    TMDB API에서 영화 데이터를 가져오는 클래스
    
    사용 예시:
        crawler = TMDBCrawler()
        movies = crawler.get_bulk_popular_movies(start_page=1, end_page=3)
    """
    
    def __init__(
		    self,
            region="KR", 
		    language="ko-KR", 
		    request_interval_seconds=0.4 # API 호출 사이 대기 시간 (초)
		):
        """
        TMDBCrawler 초기화
        
        .env 파일에서 API 정보를 가져옴:
        - TMDB_BASE_URL: API 기본 주소
        - TMDB_API_KEY: 발급받은 API 키
        """
        self._base_url = os.environ.get("TMDB_BASE_URL") # .env에서 BASE_URL 읽기
        self._api_key = os.environ.get("TMDB_API_KEY") # .env에서 API_KEY 읽기
        self._region = region
        self._language = language
        self._request_interval_seconds = request_interval_seconds

    def get_popular_movies(self, page):
        """
        인기 영화 목록 1페이지 가져오기
        
        Args:
            page: 가져올 페이지 번호 (1페이지당 영화 20개)
        
        Returns:
            영화 정보 리스트 (딕셔너리들의 리스트)
            예: [{"id": 123, "title": "영화1", "vote_average": 7.5, ...}, ...]
        """
        # API에 보낼 파라미터 설정
        params = {
            "api_key": self._api_key,  # 인증용 API 키
            "language": self._language,  # 응답 언어
            "region": self._region,  # 지역
            "page": page  # 페이지 번호
        }

        # API 호출 (GET 요청)
        # 예: https://api.themoviedb.org/3/movie/popular?api_key=xxx&language=ko-KR&page=1
        response = requests.get(f"{self._base_url}/popular", params=params)

        # 응답 상태 확인 (200 = 성공)
        if not response.status_code == 200:   
            return    # 실패하면 None 반환
        
        # JSON 응답에서 "results" 부분만 추출해서 반환
        # response.text = {"page": 1, "results": [...], "total_pages": 500}
        # 우리가 필요한 건 "results" 안의 영화 목록
        return json.loads(response.text)["results"]

    def get_bulk_popular_movies(self, start_page, end_page):
        """
        여러 페이지의 인기 영화 목록 가져오기
        
        Args:
            start_page: 시작 페이지 번호
            end_page: 끝 페이지 번호
        
        Returns:
            모든 페이지의 영화를 합친 리스트
            
        예시:
            get_bulk_popular_movies(1, 3) → 1, 2, 3 페이지의 영화 총 60개
        """
        movies = []   # 모든 영화를 담을 빈 리스트

        # start_page부터 end_page까지 반복
        for page in range(start_page, end_page+1):
            movies.extend(self.get_popular_movies(page))   # 각 페이지 영화를 리스트에 추가
            time.sleep(self._request_interval_seconds)   # API 과부하 방지를 위해 잠시 대기

        return movies

    @staticmethod
    def save_movies_to_json_file(movies, dst="./result", filename="popular"):
        """
        영화 데이터를 JSON 파일로 저장
        
        Args:
            movies: 저장할 영화 리스트
            dst: 저장할 폴더 경로
            filename: 파일 이름 (확장자 제외)
        
        결과:
            ./result/popular.json 파일 생성
        """
        data = {"movies": movies}   # {"movies": [...]} 형태로 감싸기
        
        # 파일 열어서 JSON 형태로 저장
        with open(f"{os.path.join(dst, filename)}.json", "w", encoding='utf-8') as f:
            f.write(json.dumps(data))
