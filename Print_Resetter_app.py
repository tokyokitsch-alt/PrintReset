def process_document(image_bytes, json_response):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    h, w, _ = img.shape

    data = json.loads(json_response)

    # 1. 歪み補正 (Perspective Transform)
    corners = data.get("corners", [])
    if len(corners) == 4:
        pts1 = np.float32([[c[1] * w / 1000, c[0] * h / 1000] for c in corners])
        target_w, target_h = 1240, 1754
        pts2 = np.float32([[0, 0], [target_w, 0], [target_w, target_h], [0, target_h]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        img = cv2.warpPerspective(img, M, (target_w, target_h))
        h, w = target_h, target_w

    # 2. 背景の照明ムラ除去・純白化
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dilated = cv2.dilate(gray, np.ones((15, 15), np.uint8))
    bg = cv2.medianBlur(dilated, 21)
    diff = cv2.absdiff(gray, bg)
    norm = 255 - diff
    norm = cv2.normalize(norm, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # 3. 手書き領域のみ Inpainting（修復）処理
    # 印刷線と手書きを二値化でマスク化し、周囲の白背景で補間・除去
    mask = np.zeros((h, w), dtype=np.uint8)
    boxes = data.get("handwriting_boxes", [])

    for box in boxes:
        ymin, xmin, ymax, xmax = box
        
        # ボックスに少しのパディング（余白）を加えて消し残しを防ぐ
        pad = 5
        y1 = max(0, int(ymin * h / 1000) - pad)
        x1 = max(0, int(xmin * w / 1000) - pad)
        y2 = min(h, int(ymax * h / 1000) + pad)
        x2 = min(w, int(xmax * w / 1000) + pad)

        roi_norm = norm[y1:y2, x1:x2]
        if roi_norm.size == 0:
            continue

        # 適応閾値処理で局所的な手書き線・ストロークを抽出
        local_mask = cv2.adaptiveThreshold(
            roi_norm, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 15, 8
        )
        
        # 膨張処理で線の輪郭をカバー
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        local_mask = cv2.dilate(local_mask, kernel, iterations=1)
        
        mask[y1:y2, x1:x2] = cv2.bitwise_or(mask[y1:y2, x1:x2], local_mask)

    # Inpainting による手書き線の除去・背景復元
    cleaned_gray = cv2.inpaint(norm, mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

    # 最終的なコントラスト調整（背景＝白、印刷線・文字＝黒）
    _, final_thresh = cv2.threshold(cleaned_gray, 235, 255, cv2.THRESH_TRUNC)
    final_img = cv2.normalize(final_thresh, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    final_rgb = cv2.cvtColor(final_img, cv2.COLOR_GRAY2RGB)
    return Image.fromarray(final_rgb)
