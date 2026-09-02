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
st.write("1. 歪み補正 ➔ 2. 陰影除去 ➔ 3. 枠線を保護したピンポイント手書き消去")

# サイドバー設定
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
Analyze this Kanji worksheet image and return a JSON object with two fields:
1. "corners": Normalized coordinates [y, x] scaled from 0 to 1000 for the 4 outer corners of the paper in exact order: [top-left, top-right, bottom-right, bottom-left].
2. "handwriting_boxes": A list of bounding boxes [ymin, xmin, ymax, xmax] (scaled 0-1000) ONLY for regions that actually contain handwritten pencil answers, student writing, or red pen marks. Do NOT include unwritten blank answer boxes, printed text, or lines without handwriting.

Return ONLY valid JSON.
"""

def process_document(image_bytes, json_response):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    h, w, _ = img.shape

    data = json.loads(json_response)

    # ==========================================
    # Step 1: 画像の歪み補正 (Perspective Transform)
    # ==========================================
    corners = data.get("corners", [])
    if len(corners) == 4:
        pts1 = np.float32([[c[1] * w / 1000, c[0] * h / 1000] for c in corners])
        target_w, target_h = 1240, 1754  # A4比率
        pts2 = np.float32([[0, 0], [target_w, 0], [target_w, target_h], [0, target_h]])
        
        M = cv2.getPerspectiveTransform(pts1, pts2)
        img = cv2.warpPerspective(img, M, (target_w, target_h))
        h, w = target_h, target_w

    # ==========================================
    # Step 2: 陰影の除去と背景純白化・コントラスト補正
    # ==========================================
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 膨張・平滑化で照明ムラ（背景影）を抽出
    dilated = cv2.dilate(gray, np.ones((15, 15), np.uint8))
    bg = cv2.medianBlur(dilated, 21)
    
    # 背景差分による影の除去
    diff = cv2.absdiff(gray, bg)
    norm = 255 - diff
    
    # 背景を完全な白(255)へ持ち上げる処理
    norm = cv2.normalize(norm, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    _, cleaned_bg = cv2.threshold(norm, 240, 255, cv2.THRESH_TRUNC)
    cleaned_bg = cv2.normalize(cleaned_bg, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # カラー画像として復元
    result = cv2.cvtColor(cleaned_bg, cv2.COLOR_GRAY2BGR)

    # ==========================================
    # Step 3: 手書き箇所のみ枠線を保護しながら消去
    # ==========================================
    boxes = data.get("handwriting_boxes", [])
    
    for box in boxes:
        ymin, xmin, ymax, xmax = box
        
        # 座標をピクセルに変換
        y1 = max(0, int(ymin * h / 1000))
        x1 = max(0, int(xmin * w / 1000))
        y2 = min(h, int(ymax * h / 1000))
        x2 = min(w, int(xmax * w / 1000))
        
        roi = result[y1:y2, x1:x2]
        if roi.size == 0:
            continue
            
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # 濃い印刷枠線（とても暗い部分）と、薄い鉛筆跡（中間グレー）を分離
        # 濃い黒（印刷枠線・文字）: 0～120
        # 鉛筆手書き線: 121～220
        is_printed_line = roi_gray < 120
        is_pencil_mark = (roi_gray >= 120) & (roi_gray < 225)
        
        # 鉛筆手書き線と判定されたピクセルのみを「白(255)」に置き換え
        roi_gray[is_pencil_mark] = 255
        
        # 印刷線は元の濃さを維持
        roi_gray[is_printed_line] = roi_gray[is_printed_line]
        
        result[y1:y2, x1:x2] = cv2.cvtColor(roi_gray, cv2.COLOR_GRAY2BGR)

    final_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    return Image.fromarray(final_rgb)

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
                with st.spinner("歪み補正・影除去・手書き消去を実行中..."):
                    try:
                        client = genai.Client(api_key=api_key)
                        
                        response = None
                        max_retries = 5
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
                                    time.sleep((attempt + 1) * 2)
                                    continue
                                else:
                                    raise api_err

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
