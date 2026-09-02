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
st.write("AI (Gemini) が手書き位置と歪みを解析し、スキャナ品質のA4プリントにリセットします。")

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
Analyze this image of a school worksheet/test and return a JSON object with two fields:
1. "corners": Normalized coordinates [y, x] scaled from 0 to 1000 for the 4 outer corners of the main page in exact order: [top-left, top-right, bottom-right, bottom-left].
2. "handwriting_boxes": A list of bounding boxes [ymin, xmin, ymax, xmax] (scaled 0-1000) for ALL handwritten pencil marks, student answers, red/blue pen teacher corrections, and manual drawings. Ensure NOT to include printed questions, printed kanji, or printed borders.

Return ONLY valid JSON.
"""

def process_document(image_bytes, json_response):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    h, w, _ = img.shape

    data = json.loads(json_response)

    # --- 1. 手書き部分のインペイント（修復消去）マスク作成 ---
    inpaint_mask = np.zeros((h, w), dtype=np.uint8)
    boxes = data.get("handwriting_boxes", [])
    
    for box in boxes:
        ymin, xmin, ymax, xmax = box
        # 15%マージンを持たせてはみ出しを防止
        pad_y = int((ymax - ymin) * 0.15)
        pad_x = int((xmax - xmin) * 0.15)
        
        y1 = max(0, int((ymin - pad_y) * h / 1000))
        x1 = max(0, int((xmin - pad_x) * w / 1000))
        y2 = min(h, int((ymax + pad_y) * h / 1000))
        x2 = min(w, int((xmax + pad_x) * w / 1000))
        
        # 指定領域内の薄い鉛筆・カラー線（濃い黒の印刷線以外）をピンポイント抽出
        roi = img[y1:y2, x1:x2]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # 二値化で手書き跡を抽出 (閾値を調整して枠線破壊を防止)
        _, roi_mask = cv2.threshold(gray_roi, 190, 255, cv2.THRESH_BINARY_INV)
        inpaint_mask[y1:y2, x1:x2] = cv2.bitwise_or(inpaint_mask[y1:y2, x1:x2], roi_mask)

    # インペイントで背景になじませて消去
    cleaned = cv2.inpaint(img, inpaint_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

    # --- 2. 台形補正 (A4縦サイズ 1:1.414 にジャストフィット) ---
    corners = data.get("corners", [])
    if len(corners) == 4:
        pts1 = np.float32([[c[1] * w / 1000, c[0] * h / 1000] for c in corners])
        target_w, target_h = 1240, 1754  # A4標準ピクセルサイズ
        pts2 = np.float32([[0, 0], [target_w, 0], [target_w, target_h], [0, target_h]])
        
        M = cv2.getPerspectiveTransform(pts1, pts2)
        cleaned = cv2.warpPerspective(cleaned, M, (target_w, target_h))

    # --- 3. 純白化＆印刷クッキリ化 (スキャナ風処理) ---
    gray = cv2.cvtColor(cleaned, cv2.COLOR_BGR2GRAY)
    
    # アダプティブ処理で陰影を飛ばして背景を完全純白に
    bg = cv2.medianBlur(gray, 25)
    diff = cv2.absdiff(gray, bg)
    normalized = 255 - diff
    
    # コントラストを強調して印刷文字をクリアな黒にする
    _, result_thresh = cv2.threshold(normalized, 230, 255, cv2.THRESH_TRUNC)
    final_img = cv2.normalize(result_thresh, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    final_rgb = cv2.cvtColor(final_img, cv2.COLOR_GRAY2RGB)
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
                with st.spinner("スキャナ品質で手書き消去・レイアウト補正中..."):
                    try:
                        client = genai.Client(api_key=api_key)
                        
                        # 変更後（リトライ回数を5回、待機時間を指数関数的に増加）
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
                                    wait_time = (attempt + 1) * 3  # 3秒、6秒、9秒...と待機時間を延ばす
                                    time.sleep(wait_time)
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
