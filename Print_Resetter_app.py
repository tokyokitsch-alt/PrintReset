import cv2
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(page_title="PrintReset", layout="wide")

st.title("PrintReset 📄✨")
st.write(
    "プリントやドリルの手書き文字を取り除き、何度でも解き直せる状態にリセットします。"
)

uploaded_file = st.file_uploader(
    "プリントの画像をアップロードしてください", type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:
    # 画像の読み込み
    image = Image.open(uploaded_file)
    img_array = np.array(image)

    # 2カラムレイアウトで比較表示
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("元の画像")
        st.image(image, use_container_width=True)

    # 画像処理ルーチン（グレースケール化 ＞ 二値化 ＞ メディアンフィルタ）
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

    # アダプティブ閾値処理による文字・線の抽出
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
    )

    # ノイズ除去
    processed_img = cv2.medianBlur(binary, 3)

    with col2:
        st.subheader("文字消去後の画像")
        st.image(processed_img, use_container_width=True, channels="GRAY")

    # ダウンロードボタン
    result_image = Image.fromarray(processed_img)
    st.download_button(
        label="処理後の画像をダウンロード",
        data=uploaded_file,
        file_name="print_reset.png",
        mime="image/png",
    )
