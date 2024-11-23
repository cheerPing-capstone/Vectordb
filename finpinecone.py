# !pip install pinecone-client
# !pip install sentence-transformers
# !pip install pinecone-client sentence-transformers
# 패키지 임포트
import json
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec

# SentenceTransformer 모델 로드
model_name = "jhgan/ko-sbert-multitask"
embedding_model = SentenceTransformer(model_name)

# Pinecone 초기화 (API 키 입력)
pc = Pinecone(
    api_key="b16c94ef-1e4b-4970-96c3-c2bd5e34447b"  # Pinecone API 키
)

# 인덱스 이름을 소문자 및 하이픈(-)으로만 구성
index_name = "company-infoall"

    # 인덱스가 존재하는 경우 제거(데이터 재 업로드를 위해)
if index_name in pc.list_indexes().names():
    pc.delete_index(index_name)  # 인덱스 삭제

# Pinecone 인덱스 생성 (이미 존재하는 경우 생략 가능)
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=768,  # jhgan/ko-sbert-multitask의 임베딩 차원
        metric='cosine',  # 거리 측정 방법 설정
        spec=ServerlessSpec(
            cloud='aws',  # 사용할 클라우드 제공자
            region='us-east-1'  # 사용할 리전
        )
    )

index = pc.Index(index_name)


def load_and_insert_data_minimal(file_path: str):
    with open(file_path, 'r') as file:
        company_data = json.load(file)

    for company, details in company_data.items():
        # 모든 정보를 하나의 텍스트로 결합
        doc_text = (
            f"{company}의 업종은 {details.get('업종', '정보 없음')}입니다. "
            f"{company}의 설립일은 {details.get('설립일', '정보 없음')}입니다. "
            f"{company}의 평균연봉은 {details.get('평균연봉', '정보 없음')}입니다. "
            f"{company}의 주소는 {details.get('주소', '정보 없음')}입니다. "
            f"{company}의 웹사이트는 {details.get('url', '정보 없음')}에서 확인 가능합니다. "
        )

        # 복리후생 정보 추가
        if "복리후생" in details:
            복리후생 = details["복리후생"]
            if isinstance(복리후생, dict):  # 사전인 경우
                복리후생_text = ", ".join([f"{key}: {', '.join(value)}" for key, value in 복리후생.items() if value])
                doc_text += f"{company}의 복리후생으로는 {복리후생_text} 등이 제공됩니다. "
            elif isinstance(복리후생, str):  # 문자열인 경우
                doc_text += f"{company}의 복리후생으로는 {복리후생}이(가) 제공됩니다. "

        # 채용정보 추가 및 근무지 제거
        채용_texts = []
        filtered_jobs = []  # 여기에서 초기화
        if details.get("채용정보"):
            for job in details["채용정보"]:
                # 근무지 필드 제거
                filtered_job = {k: v for k, v in job.items() if k != "근무지"}
                filtered_jobs.append(filtered_job)

                # 텍스트 생성 (근무지 제외)
                채용_texts.append(
                    f"채용 제목: {filtered_job.get('제목', '정보 없음')}, "
                    f"분야: {', '.join(filtered_job.get('분야', []))}, "
                    f"경력: {filtered_job.get('경력', '정보 없음')}, "
                    f"학력: {filtered_job.get('학력', '정보 없음')}, "
                    f"마감일: {filtered_job.get('마감일', '정보 없음')}, "
                    f"링크: {filtered_job.get('링크', '정보 없음')}"
                )
            doc_text += f"{company}의 채용 정보는 다음과 같습니다: {' | '.join(채용_texts)}."

        # 메타데이터 준비
        metadata = {
            "text": doc_text,
            "업종": details.get('업종', '정보 없음'),
            "설립일": details.get('설립일', '정보 없음'),
            "평균연봉": details.get('평균연봉', '정보 없음'),
            "주소": details.get('주소', '정보 없음'),
            "url": details.get('url', '정보 없음'),
            "복리후생": str(details.get('복리후생', '정보 없음')),  # 문자열로 변환
            "채용정보": str(filtered_jobs)  # 근무지 제거된 채용정보 저장
        }

        # 임베딩 생성
        embedding = embedding_model.encode(doc_text).tolist()

        # Pinecone에 추가
        index.upsert(
            vectors=[
                (details["id"], embedding, metadata)
            ]
        )

    print("데이터 삽입 완료")

        # 데이터 로드 및 삽입 호출
load_and_insert_data_minimal('company.json')


