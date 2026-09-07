import pandas as pd  # 데이터를 표(테이블) 형태로 다루기 위한 라이브러리


class TMDBPreProcessor:
    """
    TMDB(영화 데이터베이스)에서 가져온 영화 데이터를 정리하여
    분석하기 좋은 형태로 가공하는 클래스입니다.   

    영화 자체의 정보를 깔끔하게 정리 (데이터 정제)
    """
        
    def __init__(self, movies: list):
        """
        초기 설정을 하는 함수 (클래스가 만들어질 때 자동 실행됨)
        
        매개변수:
            movies: 영화 정보가 담긴 리스트 (TMDB API에서 가져온 원본 데이터)
        """
        self._movies = movies  # 영화 목록 저장
        self._features = pd.DataFrame()  # 결과를 저장할 빈 테이블 생성

    def run(self):
        """
        영화 데이터를 가공하는 메인 함수
        
        TMDB 원본 데이터에서 필요한 정보만 뽑아서 새로운 형태로 정리
        """
        records = []  # 가공된 영화 정보를 담을 리스트
        
        # 각 영화에 대해 반복 처리
        for movie in self._movies:
            # 원본 데이터에서 필요한 정보만 추출하여 새로운 딕셔너리 생성
            record = {
                # === 기본 정보 (원본 그대로 가져오기) ===
                "id": movie["id"],  # 영화 고유 ID
                "title": movie["title"],  # 영화 제목
                "popularity": movie["popularity"],  # 인기도 점수
                "vote_count": movie["vote_count"],  # 투표(평가)한 사람 수
                "vote_average": movie["vote_average"],  # 평균 평점 (예: 7.5)
                
                # === 가공된 정보 (원본을 변환해서 만들기) ===
                
                # 개봉일에서 '연도'만 추출 (예: "2024-05-15" → 2024)
                "release_year": self._extract_year(movie.get("release_date", "")),
                
                # 장르 개수 (예: [액션, SF, 스릴러] → 3)
                # movie.get()은 해당 키가 없으면 기본값([])을 반환
                "genre_count": len(movie.get("genre_ids", [])),
                
                # 성인 영화 여부를 숫자로 변환 (True → 1, False → 0)
                # 머신러닝 모델은 True/False보다 숫자를 더 잘 처리함
                "is_adult": 1 if movie.get("adult", False) else 0,
                
                # 영어 영화 여부를 숫자로 변환 (영어 → 1, 그 외 → 0)
                "is_english": 1 if movie.get("original_language") == "en" else 0,
            }
            records.append(record)  # 가공된 영화 정보를 리스트에 추가

        # 리스트를 판다스 데이터프레임(표)으로 변환
        self._features = pd.DataFrame.from_records(records)

    def _extract_year(self, release_date: str) -> int:
        """
        개봉일 문자열에서 연도만 추출하는 함수
        
        매개변수:
            release_date: 개봉일 문자열 (예: "2024-05-15" 또는 빈 문자열)
            
        반환값:
            연도 (정수) 또는 0 (날짜 정보가 없는 경우)
            
        예시:
            "2024-05-15" → 2024
            "1999-12-31" → 1999
            "" → 0 (데이터 없음)
        """
        # 날짜가 있고, 최소 4글자 이상인 경우 (연도 추출 가능)
        if release_date and len(release_date) >= 4:
            return int(release_date[:4])  # 앞 4글자(연도)만 잘라서 숫자로 변환
        return 0  # 날짜 정보가 없으면 0 반환

    def save(self, filename):
        """
        생성된 데이터를 CSV 파일로 저장하는 함수
        
        매개변수:
            filename: 저장할 파일 이름 (확장자 제외)
        """
        if not self._features.empty:  # 데이터가 있을 때만 저장
            self._features.to_csv(f"./result/{filename}.csv", header=True, index=False)
            # header=True: 첫 줄에 컬럼명 포함 (id, title, popularity, ...)
            # index=False: 행 번호는 저장하지 않음

    @property
    def features(self):
        """
        생성된 데이터를 외부에서 가져갈 수 있게 해주는 속성(property)
        
        사용 예시: processor.features 로 접근 가능
        """
        return self._features


# ============================================================
# 이 클래스의 전체 흐름 요약:
# ============================================================
# 1. TMDB에서 가져온 영화 원본 데이터를 받음
# 2. 각 영화에서 필요한 정보만 추출
# 3. 일부 데이터는 분석하기 좋은 형태로 변환:
#    - 개봉일 → 연도만 추출
#    - 장르 목록 → 장르 개수
#    - True/False → 1/0 숫자로 변환
# 4. 정리된 데이터를 CSV 파일로 저장
#
# 결과물 컬럼:
# - id: 영화 고유 ID
# - title: 영화 제목
# - popularity: 인기도
# - vote_count: 평가 참여자 수
# - vote_average: 평균 평점
# - release_year: 개봉 연도
# - genre_count: 장르 개수
# - is_adult: 성인 영화 여부 (0 또는 1)
# - is_english: 영어 영화 여부 (0 또는 1)
# ============================================================