import io
import json
import os
import time
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from google import genai
from google.genai import types

st.set_page_config(page_title="PrintReset", layout="wide")

st.title("PrintReset 📄✨")
st.write("AI (Gemini) が手書き位置と歪みを解析し、OpenCVでA4プリントにきれい復元します。")

# サイドバー
st.sidebar.header("設定")
api_key = st.sidebar.text_input(
    "Gemini API Key", 
    type="password",
    help="Google AI Studioで取得したAPIキーを入力してください。"
)

if not api_key:
    if "GEMINI_API_KEY" in st.secrets:
        api_key = st.secrets["GEMINI_API_KEY"]
    elif "GEMINI_API_KEY" in os.environ:
        api_key = os.environ["GEMINI_API_KEY"]

uploaded_file = st.file_uploader(
    "プリントの画像をアップロードしてください", type=["png", "jpg", "jpeg"]
)

ANALYSIS_PROMPT = """
Analyze this document image and return a JSON object with two fields:
1. "corners": Normalized coordinates [y, x] scaled from 0 to 1000 for the 4 corners of the main worksheet page in order: [top-left, top-right, bottom-right, bottom-left].
2. "handwriting_boxes": A list of bounding boxes [ymin, xmin, ymax, xmax] (scaled 0-1000) around ALL handwritten text, pencil marks, red/blue pen annotations, and manual drawings. Do NOT include original printed text, printed lines, or printed tables.

Return ONLY valid JSON matching the requested structure.
"""

def process_document(image_bytes, json_response):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    h, w, _ = img.shape

    data = json.loads(json_response)
    
    # 1. 手書き文字部分を白で塗りつぶし
    boxes = data.get("handwriting_boxes", [])
    for box in boxes:
        ymin, xmin, ymax, xmax = box
        pt1 = (int(xmin * w / 1000), int(ymin * h / 1000))
        pt2 = (int(xmax * w / 1000), int(ymax * h / 1000))
        cv2.rectangle(img, pt1, pt2, (255, 255, 255), -1)

    # 2. 台形補正 (Perspective Transform)
    corners = data.get("corners", [])
    if len(corners) == 4:
        pts1 = np.float32([[c[1] * w / 1000, c[0] * h / 1000] for c in corners])
        target_w, target_h = 1240, 1754
        pts2 = np.float32([[0, 0], [target_w, 0], [target_w, target_h], [0, target_h]])
        
        M = cv2.getPerspectiveTransform(pts1, pts2)
        img = cv2.warpPerspective(img, M, (target_w, target_h))

    # 3. 影の除去・背景純白化処理
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dilated = cv2.dilate(gray, np.ones((7,7), np.uint8))
    bg_img = cv2.medianBlur(dilated, 21)
    diff_img = 255 - cv2.absdiff(gray, bg_img)
    norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    
    result = cv2.cvtColor(norm_img, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(result)

if uploaded_file is not None:
    image_bytes = uploaded_file.read()
    image = Image.open(io.BytesIO(image_bytes))

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("元の画像")
        st.image(image, use_container_width=True)

    with col2:
        st.subheader("AIリセット後の画像")
        if not api_key:
            st.warning("⚠️ サイドバーに Gemini API Key を入力してください。")
        else:
            if st.button("AIでプリントをリセット実行", type="primary"):
                with st.spinner("AIが解析・画像処理中..."):
                    try:
                        client = genai.Client(api_key=api_key)
                        
                        # 503エラー（過負荷）対策の自動リトライ処理 (最大3回)
                        response = None
                        max_retries = 3
                        for attempt in range(max_retries):
                            try:
                                response = client.models.generate_content(
                                    model="gemini-3.6-flash",
                                    contents=[image, ANALYSIS_PROMPT],
                                    config=types.GenerateContentConfig(
                                        response_mime_type="application/json"
                                    )
                                )
                                break
                            except Exception as api_err:
                                if "503" in str(api_err) and attempt < max_retries - 1:
                                    time.sleep(2)  # 2秒待機してリトライ
                                    continue
                                else:
                                    raise api_err

                        # OpenCV加工処理
                        output_image = process_document(image_bytes, response.text)

                        st.image(output_image, use_container_width=True)
                        
                        # ダウンロード用バイト変換
                        buf = io.BytesIO()
                        output_image.save(buf, format="PNG")
                        byte_im = buf.getvalue()
                        
                        st.download_button(
                            label="処理後の画像をダウンロード",
                            data=byte_im,
                            file_name="print_reset_a4.png",
                            mime="image/png",
                        )

                    except Exception as e:
                        st.error(f"エラーが発生しました: {e}")
