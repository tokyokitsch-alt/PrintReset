import io
import os
import numpy as np
import streamlit as st
from PIL import Image
from google import genai
from google.genai import types

st.set_page_config(page_title="PrintReset", layout="wide")

st.title("PrintReset 📄✨")
st.write(
    "AI (Gemini) がプリントの手書き文字や影を取り除き、A4サイズに補正してリセットします。"
)

# サイドバーにAPIキー設定
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

PROMPT = """
1. Image Clean-up:
- Remove all handwritten text, pencil marks, red pen annotations, and manual lines.
- Keep all original printed text, kanji grid boxes, background tables, lines, and numbers perfectly clear and intact.
- Ensure empty answer boxes (like [  ]) become completely blank and white inside.

2. Geometry & Layout Correction:
- Straighten the paper perspective, remove any tilt/skew, and warp-correct to a flat top-down view.
- Fit and adjust the document aspect ratio to standard A4 printable format with clean outer margins.

3. Image Quality Enhancement:
- Convert the paper background to a pure uniform white.
- Remove shadows, lighting unevenness, and paper wrinkles.
- Make all printed text and lines sharp, high-contrast, and dark grey/black for optimal printing.
"""

if uploaded_file is not None:
    image = Image.open(uploaded_file)

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
                with st.spinner("AIが処理中..."):
                    try:
                        client = genai.Client(api_key=api_key)
                        
                        # response_mime_type を指定せずに呼び出し
                        response = client.models.generate_content(
                            model="gemini-2.5-flash",
                            contents=[image, PROMPT]
                        )

                        # インラインデータ（画像データ）が含まれているか抽出
                        output_image = None
                        if response.candidates and response.candidates[0].content.parts:
                            for part in response.candidates[0].content.parts:
                                if part.inline_data:
                                    output_image = Image.open(io.BytesIO(part.inline_data.data))
                                    break
                        
                        if output_image:
                            st.image(output_image, use_container_width=True)
                            
                            buf = io.BytesIO()
                            output_image.save(buf, format="PNG")
                            byte_im = buf.getvalue()
                            
                            st.download_button(
                                label="処理後の画像をダウンロード",
                                data=byte_im,
                                file_name="print_reset_a4.png",
                                mime="image/png",
                            )
                        else:
                            st.warning("画像は出力されませんでした（テキスト応答のみ）。")
                            if response.text:
                                st.write(response.text)

                    except Exception as e:
                        st.error(f"エラーが発生しました: {e}")
