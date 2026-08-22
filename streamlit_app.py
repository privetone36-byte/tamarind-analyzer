import streamlit as st
import numpy as np
from PIL import Image

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="วิเคราะห์มะขาม",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@300;400;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Sarabun', sans-serif !important; }
.stApp { background-color: #1a1208; }
.stButton>button {
    background-color: #ffcc80 !important;
    color: #3e2723 !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    padding: 14px !important;
    width: 100% !important;
    font-size: 1rem !important;
}
.stButton>button:hover { background-color: #ffb74d !important; }
.stSlider > div > div > div { color: #ffcc80 !important; }
.stSlider > div > div > div > div { background: #ffcc80 !important; }
[data-testid="stMetricValue"] { color: #ffcc80 !important; font-size: 2rem !important; font-weight: 800 !important; }
[data-testid="stMetricLabel"] { color: #a1887f !important; }
h1, h2, h3, h4 { color: #ffcc80 !important; }
.stSelectbox > div > div { background-color: #2d1f16 !important; color: #efebe9 !important; border-color: #5d4037 !important; }
.stNumberInput > div > div > input { background-color: #2d1f16 !important; color: #efebe9 !important; border-color: #5d4037 !important; }
.stFileUploader > div > div { background-color: #2d1f16 !important; border-color: #5d4037 !important; color: #efebe9 !important; }
.stCameraInput > div > div > div > div { background-color: #2d1f16 !important; border-color: #5d4037 !important; }
.stCameraInput > div > button { background-color: #6d4c41 !important; color: white !important; border-radius: 12px !important; }
.stProgress > div > div > div > div { background-color: #ffcc80 !important; }
.css-1dp5vir, .css-1y4p8pa { background-color: #2d1f16 !important; }
</style>
""", unsafe_allow_html=True)

# ==================== ENGINE ====================
W = {"size": 15, "color": 15, "cleanliness": 20, "moisture": 15, "pulpRatio": 15, "taste": 15, "shape": 5}
SC = {"excellent": 100, "good": 80, "fair": 60, "poor": 30, "high": 100, "medium": 70, "low": 40}

def size_score(t, s):
    s = float(s)
    if t == "sweet":
        if s >= 3.0: return 100
        if s >= 2.5: return 85
        if s >= 2.0: return 70
        if s >= 1.5: return 50
        return 30
    elif t == "sour":
        if s >= 2.5: return 100
        if s >= 2.0: return 85
        if s >= 1.5: return 70
        if s >= 1.0: return 50
        return 30
    else:
        if s >= 2.0: return 100
        if s >= 1.5: return 80
        if s >= 1.0: return 60
        return 40

def moisture_score(m):
    m = float(m)
    if m <= 12: return 100
    if m <= 15: return 90
    if m <= 18: return 75
    if m <= 22: return 55
    if m <= 25: return 35
    return 20

def get_grade(total):
    if total >= 85: return ("A", "พรีเมียม / ส่งออก", "#4caf50")
    if total >= 70: return ("B", "ดี / ตลาดทั่วไป", "#8bc34a")
    if total >= 55: return ("C", "พอใช้ / แปรรูป", "#ffc107")
    return ("D", "ต่ำ / ไม่ผ่านเกณฑ์", "#f44336")

def get_market(gr, t):
    prices = {
        "sweet": {"A": "80-120", "B": "50-75", "C": "30-45", "D": "15-25"},
        "sour":  {"A": "40-60",  "B": "25-40", "C": "15-25", "D": "8-15"},
        "pulp":  {"A": "150-250", "B": "100-150", "C": "60-90", "D": "30-50"}
    }
    markets = {
        "A": "ส่งออก (ญี่ปุ่น, จีน, ยุโรป), ซูเปอร์มาร์เก็ตพรีเมียม, ออนไลน์",
        "B": "ตลาดสดทั่วไป, ร้านค้าออนไลน์, แปรรูประดับกลาง",
        "C": "โรงงานแปรรูป, สกัดน้ำมะขาม, ทำมะขามอบแห้ง",
        "D": "ควรแปรรูปเป็นน้ำมะขามเข้มข้น หรือใช้เป็นวัตถุดิบอาหารสัตว์"
    }
    return f"**ตลาด:** {markets[gr]}\n**ราคาแนะนำ:** {prices[t][gr]} บาท/กก."

def get_suggestions(scores):
    sugg = []
    if scores.get("size", 100) < 70: sugg.append("ฝักมีขนาดเล็ก ควรคัดแยกขนาดก่อนส่งตลาด")
    if scores.get("color", 100) < 70: sugg.append("สีไม่สม่ำเสมอ ควรจัดเรียงสีก่อนบรรจุ")
    if scores.get("clean", 100) < 70: sugg.append("มีคราบ/รา/แมลง ควรทำความสะอาดก่อนจำหน่าย")
    if scores.get("moisture", 100) < 70: sugg.append("ความชื้นสูงเกินไป ควรอบแห้งเพิ่มเติม")
    if scores.get("pulp", 100) < 70: sugg.append("เนื้อน้อย ควรคัดเลือกพันธุ์ที่ให้เนื้อมากขึ้น")
    if scores.get("taste", 100) < 70: sugg.append("รสชาติไม่ดี ควรตรวจสอบวิธีการเก็บรักษา")
    if scores.get("shape", 100) < 70: sugg.append("รูปทรงไม่ดี ควรระวังการขนส่ง")
    if not sugg: sugg.append("คุณภาพดีเยี่ยม! พร้อมส่งตลาดพรีเมียมได้เลย")
    return sugg

def analyze_image_colors(img_array):
    h, w = img_array.shape[:2]
    cx, cy = int(w * 0.2), int(h * 0.2)
    cw, ch = int(w * 0.6), int(h * 0.6)
    crop = img_array[cy:cy + ch, cx:cx + cw]
    rgb = crop.astype(float) / 255.0
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    delta = max_c - min_c
    v = max_c
    s = np.where(max_c == 0, 0, delta / max_c)
    h_val = np.zeros_like(v)
    mask = delta != 0
    hr = np.where(mask & (max_c == r), ((g - b) / delta) % 6, 0)
    hg = np.where(mask & (max_c == g), ((b - r) / delta) + 2, 0)
    hb = np.where(mask & (max_c == b), ((r - g) / delta) + 4, 0)
    h_val = hr + hg + hb
    h_val = (h_val / 6.0) * 360
    avg_bri = np.mean(v) * 100
    warm = np.sum(((h_val >= 15) & (h_val <= 45)) | ((h_val >= 0) & (h_val <= 15) & (s * 100 > 30)))
    dark = np.sum(v * 100 < 25)
    total = h_val.size
    wR = warm / total
    dR = dark / total
    if wR > 0.4:
        if avg_bri > 40 and avg_bri < 70:
            cSc, cLab = 95, "สีสวย น้ำตาลแดงสม่ำเสมอ"
        elif avg_bri >= 30 and avg_bri <= 80:
            cSc, cLab = 80, "สีดี น้ำตาลเข้ม"
        else:
            cSc, cLab = 65, "สีพอใช้"
    elif wR > 0.2:
        cSc, cLab = 60, "สีไม่สม่ำเสมอ"
    else:
        cSc, cLab = 35, "สีไม่ดี (เขียว/ดำ)"
    eSize = 3.0 if (wR > 0.5 and avg_bri > 50) else (2.2 if (wR > 0.3 and avg_bri > 40) else 1.8)
    eMoist = 12 if (avg_bri > 60 and dR < 0.1) else (16 if (avg_bri > 45 and dR < 0.2) else 22)
    if dR > 0.15:
        clSc, clLab = 40, "poor"
    elif dR > 0.08:
        clSc, clLab = 60, "fair"
    elif dR < 0.03:
        clSc, clLab = 100, "excellent"
    else:
        clSc, clLab = 80, "good"
    return {
        "colorScore": cSc, "colorLabel": cLab,
        "estimatedSize": eSize, "estimatedMoisture": eMoist,
        "cleanScore": clSc, "cleanLabel": clLab,
        "shapeScore": 75, "shapeLabel": "good",
        "avgBrightness": avg_bri, "warmRatio": wR
    }

def calculate_grade(type_val, size, color, clean, moisture, pulp, taste, shape):
    sSc = size_score(type_val, size)
    cSc = color if isinstance(color, (int, float)) else SC[color]
    clSc = SC[clean]
    mSc = moisture_score(moisture)
    pSc = SC[pulp]
    tSc = SC[taste]
    shSc = SC[shape]
    total = round(
        sSc * W["size"] / 100 + cSc * W["color"] / 100 + clSc * W["cleanliness"] / 100 +
        mSc * W["moisture"] / 100 + pSc * W["pulpRatio"] / 100 + tSc * W["taste"] / 100 + shSc * W["shape"] / 100
    )
    return total, {"size": sSc, "color": cSc, "clean": clSc, "moisture": mSc, "pulp": pSc, "taste": tSc, "shape": shSc}

def display_result(t_val, total, scores, detected=None):
    gr, label, color = get_grade(total)
    st.markdown("---")
    st.markdown(f"<h2 style='text-align:center;color:{color};'>🌿 เกรด {gr}</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align:center;color:#ffcc80;font-size:1.2rem;font-weight:700;'>{label}</p>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("คะแนนรวม", f"{total}/100")
    c2.metric("เกรด", gr)
    c3.metric("ตลาด", "พรีเมียม" if gr == "A" else "ทั่วไป" if gr == "B" else "แปรรูป" if gr == "C" else "ไม่ผ่าน")
    st.progress(total / 100)
    st.markdown("#### 📊 คะแนนแต่ละหัวข้อ")
    name_map = {"size": "ขนาดฝัก", "color": "สี", "clean": "ความสะอาด", "moisture": "ความชื้น",
                "pulp": "อัตราส่วนเนื้อ/เมล็ด", "taste": "รสชาติ", "shape": "รูปทรง"}
    detail_data = [{"หัวข้อ": name_map.get(k, k), "คะแนน": f"{v} คะแนน"} for k, v in scores.items()]
    st.table(detail_data)
    st.markdown("#### 💰 แนะนำตลาด/ราคา")
    st.info(get_market(gr, t_val))
    st.markdown("#### 📝 ข้อเสนอแนะ")
    for s in get_suggestions(scores):
        st.write(f"• {s}")
    if detected:
        st.caption(f"📸 วิเคราะห์จากภาพ: ความสว่างเฉลี่ย {detected['avgBrightness']:.1f}% | อัตราส่วนสีอุ่น {detected['warmRatio'] * 100:.1f}%")

# ==================== UI ====================
st.markdown("<h1 style='text-align:center;'>🌿 วิเคราะห์มะขาม</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#a1887f;'>Tamarind Quality Analyzer | ตามมาตรฐานสินค้าเกษตร</p>", unsafe_allow_html=True)

mode = st.radio("เลือกโหมด", ["📸 ถ่ายด้วยกล้อง", "📤 อัปโหลดภาพ", "📝 กรอกข้อมูล"], horizontal=True)

type_map = {"มะขามหวาน": "sweet", "มะขามเปรี้ยว": "sour", "เนื้อมะขาม": "pulp"}

# ---------- CAMERA MODE ----------
if mode == "📸 ถ่ายด้วยกล้อง":
    st.info("กดปุ่มด้านล่างเพื่อเปิดกล้อง → ถ่ายภาพมะขาม → ระบบวิเคราะห์อัตโนมัติ")
    cam_photo = st.camera_input("วางมะขามในกรอบแล้วกดถ่าย", key="cam1")
    if cam_photo is not None:
        img = Image.open(cam_photo)
        img_array = np.array(img)
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(img, caption="ภาพที่ถ่าย", use_container_width=True)
        detected = analyze_image_colors(img_array)
        with col2:
            st.markdown("#### 🔍 ผลการตรวจจับ")
            st.write(f"🎨 **สี:** {detected['colorLabel']} ({detected['colorScore']} คะแนน)")
            st.write(f"📏 **ขนาดโดยประมาณ:** {detected['estimatedSize']:.1f} cm")
            st.write(f"💧 **ความชื้นโดยประมาณ:** {detected['estimatedMoisture']:.1f}%")
            st.write(f"✨ **ความสะอาด:** {detected['cleanLabel']} ({detected['cleanScore']} คะแนน)")
        st.markdown("---")
        st.markdown("#### ✏️ ปรับแต่งค่า (ถ้าจำเป็น)")
        c1, c2 = st.columns(2)
        with c1:
            up_type = st.selectbox("ประเภท", ["มะขามหวาน", "มะขามเปรี้ยว", "เนื้อมะขาม"], key="cam_type")
            t_val = type_map[up_type]
            up_size = st.slider("ขนาดฝัก (cm)", 1.0, 5.0, detected['estimatedSize'], 0.1, key="cam_size")
            up_moisture = st.slider("ความชื้น (%)", 5.0, 30.0, detected['estimatedMoisture'], 0.5, key="cam_moist")
        with c2:
            clean_opts = ["excellent", "good", "fair", "poor"]
            clean_labels = ["สะอาดมาก", "สะอาด", "มีตำหนิเล็กน้อย", "มีคราบ/รา/แมลง"]
            clean_index = clean_opts.index(detected['cleanLabel']) if detected['cleanLabel'] in clean_opts else 1
            up_clean = st.selectbox("ความสะอาด", clean_opts, format_func=lambda x: dict(zip(clean_opts, clean_labels))[x], index=clean_index, key="cam_clean")
            up_taste = st.selectbox("รสชาติ", clean_opts, format_func=lambda x: {"excellent": "รสชาติดีมาก", "good": "รสชาติดี", "fair": "รสชาติพอใช้", "poor": "รสชาติไม่ดี"}[x], key="cam_taste")
            up_shape = st.selectbox("รูปทรง", clean_opts, format_func=lambda x: {"excellent": "สวย สมมาตร", "good": "ดี มีตำหนิเล็กน้อย", "fair": "พอใช้", "poor": "ไม่ดี บุบ/แตก"}[x], key="cam_shape")
        if st.button("คำนวณเกรด", key="cam_calc"):
            total, scores = calculate_grade(t_val, up_size, detected['colorScore'], up_clean, up_moisture, "medium", up_taste, up_shape)
            display_result(t_val, total, scores, detected)

# ---------- UPLOAD MODE ----------
elif mode == "📤 อัปโหลดภาพ":
    uploaded = st.file_uploader("เลือกภาพมะขาม (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        img = Image.open(uploaded)
        img_array = np.array(img)
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(img, caption="ภาพที่อัปโหลด", use_container_width=True)
        detected = analyze_image_colors(img_array)
        with col2:
            st.markdown("#### 🔍 ผลการตรวจจับ")
            st.write(f"🎨 **สี:** {detected['colorLabel']} ({detected['colorScore']} คะแนน)")
            st.write(f"📏 **ขนาดโดยประมาณ:** {detected['estimatedSize']:.1f} cm")
            st.write(f"💧 **ความชื้นโดยประมาณ:** {detected['estimatedMoisture']:.1f}%")
            st.write(f"✨ **ความสะอาด:** {detected['cleanLabel']} ({detected['cleanScore']} คะแนน)")
        st.markdown("---")
        st.markdown("#### ✏️ ปรับแต่งค่า (ถ้าจำเป็น)")
        c1, c2 = st.columns(2)
        with c1:
            up_type = st.selectbox("ประเภท", ["มะขามหวาน", "มะขามเปรี้ยว", "เนื้อมะขาม"], key="up_type")
            t_val = type_map[up_type]
            up_size = st.slider("ขนาดฝัก (cm)", 1.0, 5.0, detected['estimatedSize'], 0.1, key="up_size")
            up_moisture = st.slider("ความชื้น (%)", 5.0, 30.0, detected['estimatedMoisture'], 0.5, key="up_moist")
        with c2:
            clean_opts = ["excellent", "good", "fair", "poor"]
            clean_labels = ["สะอาดมาก", "สะอาด", "มีตำหนิเล็กน้อย", "มีคราบ/รา/แมลง"]
            clean_index = clean_opts.index(detected['cleanLabel']) if detected['cleanLabel'] in clean_opts else 1
            up_clean = st.selectbox("ความสะอาด", clean_opts, format_func=lambda x: dict(zip(clean_opts, clean_labels))[x], index=clean_index, key="up_clean")
            up_taste = st.selectbox("รสชาติ", clean_opts, format_func=lambda x: {"excellent": "รสชาติดีมาก", "good": "รสชาติดี", "fair": "รสชาติพอใช้", "poor": "รสชาติไม่ดี"}[x], key="up_taste")
            up_shape = st.selectbox("รูปทรง", clean_opts, format_func=lambda x: {"excellent": "สวย สมมาตร", "good": "ดี มีตำหนิเล็กน้อย", "fair": "พอใช้", "poor": "ไม่ดี บุบ/แตก"}[x], key="up_shape")
        if st.button("คำนวณเกรด", key="up_calc"):
            total, scores = calculate_grade(t_val, up_size, detected['colorScore'], up_clean, up_moisture, "medium", up_taste, up_shape)
            display_result(t_val, total, scores, detected)

# ---------- MANUAL MODE ----------
else:
    st.markdown("#### 📝 กรอกข้อมูลมะขาม")
    c1, c2 = st.columns(2)
    with c1:
        mn_type = st.selectbox("ประเภท", ["มะขามหวาน", "มะขามเปรี้ยว", "เนื้อมะขาม"], key="mn_type")
        t_val = type_map[mn_type]
        mn_size = st.number_input("ขนาดฝัก (cm)", min_value=0.5, max_value=10.0, value=2.5, step=0.1, key="mn_size")
        mn_color = st.selectbox("สี", ["excellent", "good", "fair", "poor"],
                                format_func=lambda x: {"excellent": "สีสวย สม่ำเสมอ", "good": "สีดี", "fair": "สีพอใช้", "poor": "สีไม่ดี"}[x], key="mn_color")
        mn_clean = st.selectbox("ความสะอาด", ["excellent", "good", "fair", "poor"],
                                format_func=lambda x: {"excellent": "สะอาดมาก", "good": "สะอาด", "fair": "มีตำหนิเล็กน้อย", "poor": "มีคราบ/รา/แมลง"}[x], key="mn_clean")
    with c2:
        mn_moisture = st.number_input("ความชื้น (%)", min_value=0.0, max_value=50.0, value=15.0, step=0.5, key="mn_moist")
        st.caption("มาตรฐานมะขามแห้ง ≤ 18%")
        mn_pulp = st.selectbox("อัตราส่วนเนื้อ/เมล็ด", ["high", "medium", "low"],
                                 format_func=lambda x: {"high": "เนื้อมาก (≥60%)", "medium": "เนื้อปานกลาง (40-59%)", "low": "เนื้อน้อย (<40%)"}[x], key="mn_pulp")
        mn_taste = st.selectbox("รสชาติ", ["excellent", "good", "fair", "poor"],
                                format_func=lambda x: {"excellent": "รสชาติดีมาก", "good": "รสชาติดี", "fair": "รสชาติพอใช้", "poor": "รสชาติไม่ดี"}[x], key="mn_taste")
        mn_shape = st.selectbox("รูปทรง", ["excellent", "good", "fair", "poor"],
                                format_func=lambda x: {"excellent": "สวย สมมาตร", "good": "ดี มีตำหนิเล็กน้อย", "fair": "พอใช้", "poor": "ไม่ดี บุบ/แตก"}[x], key="mn_shape")
    if st.button("วิเคราะห์คุณภาพ", key="mn_calc"):
        total, scores = calculate_grade(t_val, mn_size, mn_color, mn_clean, mn_moisture, mn_pulp, mn_taste, mn_shape)
        display_result(t_val, total, scores)

st.markdown("---")
st.caption("อ้างอิง: มาตรฐานสินค้าเกษตรมะขามหวาน (ราชกิจจานุเบกษา) และเกณฑ์ตลาดมะขามทั่วไป")
