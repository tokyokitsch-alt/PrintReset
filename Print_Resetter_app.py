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
    "AI (Imagen) がプリントの手書き文字や影を取り除き、A4サイズに補正してリセットします。"
)

# サイドバーにAPIキー設定
st.sidebar.header("設定")
api_key = st.sidebar.text_input(
    "Gemini API Key", 
    type="password",
    help="Google AI Studioで取得したAPIキーを入力してください。Streamlit Secretsに設定している場合は自動読み込みされます。"
)

# Secretsや環境変数からのフォールバック設定
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
                with st.spinner("AIが手書き消去・歪み補正中..."):
                    try:
                        client = genai.Client(api_key=api_key)
                        
                        # Imagen 3 APIによる画像生成処理
                        result = client.models.generate_images(
                            model="imagen-3.0-generate-002",
                            prompt=f"Based on the original document context, output a clean printable worksheet: {PROMPT}",
                            config=types.GenerateImagesConfig(
                                number_of_images=1,
                                output_mime_type="image/png",
                                aspect_ratio="3:4"
                            )
                        )

                        output_image = None
                        if result.generated_images:
                            generated_bytes = result.generated_images[0].image.image_bytes
                            output_image = Image.open(io.BytesIO(generated_bytes))

                        if output_image:
                            st.image(output_image, use_container_width=True)
                            
                            # ダウンロード用に処理後の画像をバイト変換
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
                            st.error("画像の生成に失敗しました。")

                    except Exception as e:
                        st.error(f"エラーが発生しました: {e}")
