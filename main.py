import streamlit as st
import pandas as pd
import numpy as np
import pytesseract
import cv2
from PIL import Image
from io import BytesIO
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as RLImage,
    PageBreak,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.enums import TA_CENTER
import os


# ============================================================
# 기본 설정
# ============================================================

st.set_page_config(
    page_title="MathSort AI",
    page_icon="📐",
    layout="wide"
)

st.title("📐 MathSort AI")
st.subheader("미적분Ⅰ 수학 문제 자동 분류 시스템")

st.write(
    "문제 이미지를 업로드하면 OCR로 문제를 읽고 "
    "코사인 유사도를 활용하여 미적분Ⅰ 단원과 난이도를 분류합니다."
)

st.info(
    "분류 대상 단원: 함수의 극한과 연속 / 미분 / 적분"
)


# ============================================================
# 단원 목록
# ============================================================

UNITS = [
    "함수의 극한과 연속",
    "미분",
    "적분"
]


# ============================================================
# 데이터 불러오기
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_csv("questions.csv")

    # 필요한 열이 존재하는지 확인
    required_columns = [
        "question",
        "unit",
        "difficulty"
    ]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(
                f"questions.csv에 '{col}' 열이 필요합니다."
            )

    # 문자열 처리
    df["question"] = df["question"].fillna("").astype(str)
    df["unit"] = df["unit"].fillna("").astype(str)

    # 난이도 숫자 처리
    df["difficulty"] = pd.to_numeric(
        df["difficulty"],
        errors="coerce"
    )

    df = df.dropna(subset=["difficulty"])

    df["difficulty"] = df["difficulty"].astype(int)

    return df


try:
    df = load_data()
except Exception as e:
    st.error(f"데이터를 불러오는 중 오류가 발생했습니다: {e}")
    st.stop()


# ============================================================
# TF-IDF + 코사인 유사도 모델
# ============================================================

@st.cache_resource
def create_model(df):

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(2, 5),
        min_df=1,
        sublinear_tf=True
    )

    X = vectorizer.fit_transform(
        df["question"]
    )

    return vectorizer, X


vectorizer, X = create_model(df)


# ============================================================
# OCR 함수
# ============================================================

def extract_text(image):

    # PIL → OpenCV
    img = np.array(image)

    # RGB → GRAY
    gray = cv2.cvtColor(
        img,
        cv2.COLOR_RGB2GRAY
    )

    # 이미지 크기 확대
    scale = 2

    gray = cv2.resize(
        gray,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC
    )

    # 노이즈 제거
    gray = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    # 이진화
    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # 한국어 + 영어 OCR
    text = pytesseract.image_to_string(
        binary,
        lang="kor+eng"
    )

    return text.strip()


# ============================================================
# 코사인 유사도 기반 분류
# ============================================================

def classify_question(text):

    if not text.strip():
        return None

    query_vector = vectorizer.transform([text])

    similarities = cosine_similarity(
        query_vector,
        X
    )[0]

    # 유사도가 높은 순서
    top_indices = np.argsort(
        similarities
    )[::-1]

    # 상위 K개 사용
    K = min(5, len(top_indices))

    top_indices = top_indices[:K]

    top_df = df.iloc[top_indices].copy()

    top_df["similarity"] = similarities[top_indices]

    # --------------------------------------------------------
    # 단원 결정
    # --------------------------------------------------------

    unit_scores = {}

    for unit in UNITS:

        unit_data = top_df[
            top_df["unit"] == unit
        ]

        if len(unit_data) > 0:

            weights = unit_data["similarity"].values

            score = np.sum(weights)

            unit_scores[unit] = score

    if len(unit_scores) == 0:

        predicted_unit = "분류 불가"

    else:

        predicted_unit = max(
            unit_scores,
            key=unit_scores.get
        )

    # --------------------------------------------------------
    # 난이도 결정
    # --------------------------------------------------------

    same_unit = top_df[
        top_df["unit"] == predicted_unit
    ]

    if len(same_unit) == 0:

        same_unit = top_df

    # 유사도를 가중치로 사용
    weights = same_unit["similarity"].values

    difficulties = same_unit["difficulty"].values

    if np.sum(weights) == 0:

        predicted_difficulty = int(
            round(np.mean(difficulties))
        )

    else:

        predicted_difficulty = int(
            round(
                np.average(
                    difficulties,
                    weights=weights
                )
            )
        )

    # 1~5 범위
    predicted_difficulty = max(
        1,
        min(5, predicted_difficulty)
    )

    # 가장 유사한 문제
    best_match = top_df.iloc[0]

    return {
        "unit": predicted_unit,
        "difficulty": predicted_difficulty,
        "similarity": float(
            best_match["similarity"]
        ),
        "best_match": best_match["question"],
        "top_matches": top_df
    }


# ============================================================
# 이미지 업로드
# ============================================================

st.header("1️⃣ 문제 이미지 업로드")

uploaded_files = st.file_uploader(
    "수학 문제 이미지를 여러 장 선택하세요.",
    type=["png", "jpg", "jpeg"],
    accept_multiple_files=True
)


# ============================================================
# 분석
# ============================================================

if uploaded_files:

    st.header("2️⃣ 문제 분석")

    results = []

    for index, uploaded_file in enumerate(
        uploaded_files,
        start=1
    ):

        image = Image.open(
            uploaded_file
        ).convert("RGB")

        with st.expander(
            f"문제 {index}",
            expanded=True
        ):

            col1, col2 = st.columns(2)

            # 이미지
            with col1:

                st.image(
                    image,
                    caption=f"문제 {index}",
                    use_container_width=True
                )

            # OCR
            text = extract_text(image)

            with col2:

                st.write("### 🔎 OCR 결과")

                if text:

                    st.text_area(
                        "추출된 문제",
                        text,
                        height=180,
                        key=f"text_{index}"
                    )

                else:

                    st.warning(
                        "문제 텍스트를 읽지 못했습니다."
                    )

            # 분류
            if text:

                result = classify_question(
                    text
                )

                if result:

                    results.append({
                        "number": index,
                        "image": image,
                        "text": text,
                        "unit": result["unit"],
                        "difficulty": result["difficulty"],
                        "similarity": result["similarity"]
                    })

                    st.success(
                        f"📚 단원: {result['unit']}"
                    )

                    st.info(
                        f"🎯 예상 난이도: "
                        f"{result['difficulty']} / 5"
                    )

                    st.caption(
                        f"코사인 유사도: "
                        f"{result['similarity']:.3f}"
                    )


    # ========================================================
    # 전체 결과
    # ========================================================

    if results:

        st.header("3️⃣ 전체 분류 결과")

        result_df = pd.DataFrame([
            {
                "문제 번호": r["number"],
                "단원": r["unit"],
                "난이도": r["difficulty"],
                "코사인 유사도": round(
                    r["similarity"],
                    3
                )
            }
            for r in results
        ])

        st.dataframe(
            result_df,
            use_container_width=True,
            hide_index=True
        )


        # ====================================================
        # 단원별 문제 묶기
        # ====================================================

        st.header("4️⃣ 단원별 문제")

        for unit in UNITS:

            unit_results = [
                r for r in results
                if r["unit"] == unit
            ]

            if unit_results:

                st.subheader(
                    f"📚 {unit}"
                )

                for r in unit_results:

                    st.write(
                        f"**문제 {r['number']}** "
                        f"— 난이도 {r['difficulty']}/5"
                    )

                    st.image(
                        r["image"],
                        width=500
                    )


        # ====================================================
        # PDF 생성
        # ====================================================

        st.header("5️⃣ A4 시험지 만들기")

        if st.button(
            "📄 단원별 A4 시험지 생성"
        ):

            pdf_buffer = BytesIO()

            doc = SimpleDocTemplate(
                pdf_buffer,
                pagesize=A4,
                rightMargin=45,
                leftMargin=45,
                topMargin=45,
                bottomMargin=45
            )

            styles = getSampleStyleSheet()

            title_style = ParagraphStyle(
                "Title",
                parent=styles["Title"],
                alignment=TA_CENTER,
                fontSize=20,
                spaceAfter=20
            )

            heading_style = ParagraphStyle(
                "Heading",
                parent=styles["Heading2"],
                fontSize=15,
                spaceBefore=15,
                spaceAfter=10
            )

            body_style = ParagraphStyle(
                "Body",
                parent=styles["BodyText"],
                fontSize=10,
                leading=15
            )

            story = []

            # 시험지 제목
            story.append(
                Paragraph(
                    "미적분Ⅰ 문제 시험지",
                    title_style
                )
            )

            story.append(
                Paragraph(
                    "MathSort AI 자동 분류 결과",
                    body_style
                )
            )

            story.append(
                Spacer(1, 20)
            )

            # 단원별 정렬
            question_number = 1

            for unit in UNITS:

                unit_results = [
                    r for r in results
                    if r["unit"] == unit
                ]

                if not unit_results:
                    continue

                # 단원 제목
                story.append(
                    Paragraph(
                        unit,
                        heading_style
                    )
                )

                # 난이도 순서대로 정렬
                unit_results = sorted(
                    unit_results,
                    key=lambda x: x["difficulty"]
                )

                for r in unit_results:

                    story.append(
                        Paragraph(
                            f"{question_number}. "
                            f"(난이도 {r['difficulty']}/5)",
                            body_style
                        )
                    )

                    story.append(
                        Spacer(1, 5)
                    )

                    # 이미지 임시 저장
                    temp_path = (
                        f"/tmp/question_"
                        f"{r['number']}.png"
                    )

                    r["image"].save(
                        temp_path
                    )

                    # 이미지 크기 조절
                    pdf_image = RLImage(
                        temp_path,
                        width=450,
                        height=300,
                        kind="proportional"
                    )

                    story.append(
                        pdf_image
                    )

                    story.append(
                        Spacer(1, 20)
                    )

                    question_number += 1

            doc.build(story)

            pdf_buffer.seek(0)

            st.success(
                "시험지가 완성되었습니다!"
            )

            st.download_button(
                label="⬇️ A4 시험지 PDF 다운로드",
                data=pdf_buffer,
                file_name="미적분Ⅰ_자동분류_시험지.pdf",
                mime="application/pdf"
            )


# ============================================================
# 사이드바
# ============================================================

with st.sidebar:

    st.header("🔬 탐구 정보")

    st.write(
        "**분석 방법**"
    )

    st.write(
        "TF-IDF + 코사인 유사도"
    )

    st.write(
        "**분류 범위**"
    )

    st.write(
        "미적분Ⅰ"
    )

    st.write(
        "- 함수의 극한과 연속"
    )

    st.write(
        "- 미분"
    )

    st.write(
        "- 적분"
    )

    st.write(
        "**난이도**"
    )

    st.write(
        "1 ~ 5"
    )

    st.caption(
        "※ 난이도 5가 가장 어려운 문제입니다."
    )
