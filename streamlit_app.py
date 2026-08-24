import streamlit as st
import numpy as np
from PIL import Image
import cv2
import pandas as pd

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="วิเคราะห์มะขาม - Multi Detector v2",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
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
[data-testid="stMetricValue"] { color: #ffcc80 !important; font-size: 1.5rem !important; font-weight: 800 !important; }
[data-testid="stMetricLabel"] { color: #a1887f !important; }
h1, h2, h3, h4 { color: #ffcc80 !important; }
.stSelectbox > div > div { background-color: #2d1f16 !important; color: #efebe9 !important; border-color: #5d4037 !important; }
.stNumberInput > div > div > input { background-color: #2d1f16 !important; color: #efebe9 !important; border-color: #5d4037 !important; }
.stFileUploader > div > div { background-color: #2d1f16 !important; border-color: #5d4037 !important; color: #efebe9 !important; }
.stCameraInput > div > div > div > div { background-color: #2d1f16 !important; border-color: #5d4037 !important; }
.stCameraInput > div > button { background-color: #6d4c41 !important; color: white !important; border-radius: 12px !important; }
.stProgress > div > div > div > div { background-color: #ffcc80 !important; }
.css-1dp5vir, .css-1y4p8pa { background-color: #2d1f16 !important; }
.stDataFrame { background-color: #2d1f16 !important; }
</style>
""", unsafe_allow_html=True)

# ==================== TAMARIND TYPES ====================
TAMARIND_TYPES = {
    "มะขามหวานสด": {"code": "sweet_fresh", "size_min": 2.0, "size_opt": 3.5, "moisture_max": 25, "price_A": "80-120", "price_B": "50-75", "price_C": "30-45", "price_D": "15-25"},
    "มะขามหวานแกะเมล็ด": {"code": "sweet_seedless", "size_min": 1.5, "size_opt": 3.0, "moisture_max": 20, "price_A": "120-180", "price_B": "80-120", "price_C": "50-80", "price_D": "25-40"},
    "มะขามหวานอบแห้ง": {"code": "sweet_dried", "size_min": 1.0, "size_opt": 2.5, "moisture_max": 18, "price_A": "150-220", "price_B": "100-150", "price_C": "60-100", "price_D": "30-50"},
    "มะขามเปรี้ยวสด": {"code": "sour_fresh", "size_min": 1.5, "size_opt": 3.0, "moisture_max": 30, "price_A": "40-60", "price_B": "25-40", "price_C": "15-25", "price_D": "8-15"},
    "มะขามเปรี้ยวแกะเมล็ด": {"code": "sour_seedless", "size_min": 1.0, "size_opt": 2.5, "moisture_max": 22, "price_A": "60-90", "price_B": "40-60", "price_C": "25-40", "price_D": "12-20"},
    "มะขามเปรี้ยวอบแห้ง": {"code": "sour_dried", "size_min": 0.8, "size_opt": 2.0, "moisture_max": 18, "price_A": "100-150", "price_B": "70-100", "price_C": "40-70", "price_D": "20-35"},
    "มะขามคั่ว": {"code": "roasted", "size_min": 0.5, "size_opt": 1.5, "moisture_max": 10, "price_A": "200-300", "price_B": "150-200", "price_C": "80-150", "price_D": "40-80"},
    "มะขามอัดเม็ด": {"code": "compressed", "size_min": 0.3, "size_opt": 1.0, "moisture_max": 15, "price_A": "180-280", "price_B": "120-180", "price_C": "70-120", "price_D": "35-70"},
    "มะขามแช่อิ่ม": {"code": "candied", "size_min": 0.5, "size_opt": 1.5, "moisture_max": 35, "price_A": "150-250", "price_B": "100-150", "price_C": "60-100", "price_D": "30-60"},
    "เนื้อมะขาม": {"code": "pulp", "size_min": 0.5, "size_opt": 2.0, "moisture_max": 25, "price_A": "150-250", "price_B": "100-150", "price_C": "60-90", "price_D": "30-50"},
}

W = {"size": 15, "color": 15, "cleanliness": 20, "moisture": 15, "pulpRatio": 15, "taste": 15, "shape": 5}
SC = {"excellent": 100, "good": 80, "fair": 60, "poor": 30, "high": 100, "medium": 70, "low": 40}

def size_score(type_info, s):
    s = float(s)
    opt = type_info["size_opt"]
    min_s = type_info["size_min"]
    if s >= opt: return 100
    if s >= opt * 0.85: return 85
    if s >= opt * 0.7: return 70
    if s >= min_s: return 50
    return 30

def moisture_score(type_info, m):
    m = float(m)
    max_m = type_info["moisture_max"]
    if m <= max_m * 0.5: return 100
    if m <= max_m * 0.7: return 85
    if m <= max_m * 0.9: return 70
    if m <= max_m: return 55
    if m <= max_m * 1.2: return 35
    return 20

def get_grade(total):
    if total >= 85: return ("A", "พรีเมียม / ส่งออก", "#4caf50")
    if total >= 70: return ("B", "ดี / ตลาดทั่วไป", "#8bc34a")
    if total >= 55: return ("C", "พอใช้ / แปรรูป", "#ffc107")
    return ("D", "ต่ำ / ไม่ผ่านเกณฑ์", "#f44336")

def get_market(gr, type_info):
    markets = {
        "A": "ส่งออก, ซูเปอร์มาร์เก็ตพรีเมียม, ออนไลน์",
        "B": "ตลาดสดทั่วไป, ร้านค้าออนไลน์",
        "C": "โรงงานแปรรูป, สกัดน้ำมะขาม",
        "D": "แปรรูปเป็นน้ำมะขามเข้มข้น หรืออาหารสัตว์"
    }
    price = type_info.get(f"price_{gr}", "N/A")
    return f"**ตลาด:** {markets[gr]}\n**ราคา:** {price} บาท/กก."

def get_suggestions(scores):
    sugg = []
    if scores.get("size", 100) < 70: sugg.append("ขนาดเล็ก ควรคัดแยก")
    if scores.get("color", 100) < 70: sugg.append("สีไม่สม่ำเสมอ")
    if scores.get("clean", 100) < 70: sugg.append("มีคราบ/รา/แมลง")
    if scores.get("moisture", 100) < 70: sugg.append("ความชื้นสูง ควรอบแห้ง")
    if scores.get("pulp", 100) < 70: sugg.append("เนื้อน้อย")
    if scores.get("taste", 100) < 70: sugg.append("รสชาติไม่ดี")
    if scores.get("shape", 100) < 70: sugg.append("รูปทรงไม่ดี")
    if not sugg: sugg.append("คุณภาพดีเยี่ยม!")
    return sugg

def calculate_grade(type_info, size, color, clean, moisture, pulp, taste, shape):
    sSc = size_score(type_info, size)
    cSc = color if isinstance(color, (int, float)) else SC[color]
    clSc = SC[clean]
    mSc = moisture_score(type_info, moisture)
    pSc = SC[pulp]
    tSc = SC[taste]
    shSc = SC[shape]
    total = round(
        sSc * W["size"] / 100 + cSc * W["color"] / 100 + clSc * W["cleanliness"] / 100 +
        mSc * W["moisture"] / 100 + pSc * W["pulpRatio"] / 100 + tSc * W["taste"] / 100 + shSc * W["shape"] / 100
    )
    return total, {"size": sSc, "color": cSc, "clean": clSc, "moisture": mSc, "pulp": pSc, "taste": tSc, "shape": shSc}

# ==================== ADVANCED DETECTION ENGINE v2 ====================
def detect_tamarinds_advanced(img_array, params):
    """
    ตรวจจับมะขามแต่ละผลด้วย HSV + Morphology + minAreaRect + Multi-criteria Filtering
    params: dict ของพารามิเตอร์ที่ปรับได้
    """
    img_rgb = img_array.copy()
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    h, w = img_rgb.shape[:2]

    h_low = params.get("hsv_h_low", 0)
    h_high = params.get("hsv_h_high", 40)
    s_low = params.get("hsv_s_low", 8)
    s_high = params.get("hsv_s_high", 255)
    v_low = params.get("hsv_v_low", 10)
    v_high = params.get("hsv_v_high", 250)
    morph_close = params.get("morph_close", 20)
    min_area = params.get("min_area", 300)
    max_area = params.get("max_area", 50000)
    min_ar = params.get("min_aspect", 0.8)
    max_ar = params.get("max_aspect", 20.0)
    min_sol = params.get("min_solidity", 0.25)
    min_ext = params.get("min_extent", 0.15)

    # ========== STEP 1: HSV Color Masking ==========
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    lower = np.array([h_low, s_low, v_low])
    upper = np.array([h_high, s_high, v_high])
    color_mask = cv2.inRange(hsv, lower, upper)

    # กรองสีเขียว (ใบไม้)
    green_mask = cv2.inRange(hsv, np.array([35, 20, 20]), np.array([90, 255, 255]))
    color_mask = cv2.bitwise_and(color_mask, cv2.bitwise_not(green_mask))

    # กรองสีน้ำเงิน/ม่วง
    blue_mask = cv2.inRange(hsv, np.array([90, 20, 20]), np.array([140, 255, 255]))
    color_mask = cv2.bitwise_and(color_mask, cv2.bitwise_not(blue_mask))

    # กรองขาว/ดำสนิท
    white_mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 30, 255]))
    black_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([180, 255, 20]))
    color_mask = cv2.bitwise_and(color_mask, cv2.bitwise_not(white_mask))
    color_mask = cv2.bitwise_and(color_mask, cv2.bitwise_not(black_mask))

    # ========== STEP 2: Morphological Operations ==========
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_close, morph_close))

    mask_clean = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel_open)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel_close)

    # ========== STEP 3: Find Contours ==========
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_objects = []
    rejected_log = []

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            if area > 0:
                rejected_log.append(f"#{i} area={area:.0f}")
            continue

        # ใช้ minAreaRect แทน boundingRect เพื่อรองรับการเอียง
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int32(box)  # แก้ไขจาก np.int0

        # คำนวณขนาดจาก minAreaRect
        (cx, cy), (rw, rh), angle = rect
        rw, rh = max(rw, 1), max(rh, 1)
        aspect = max(rw, rh) / min(rw, rh)

        # Bounding box สำหรับ crop ROI
        x, y, bw, bh = cv2.boundingRect(cnt)

        # Convex hull
        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        solidity = float(area) / hull_area if hull_area > 0 else 0

        # Extent (ใช้ minAreaRect area แทน)
        rect_area = rw * rh
        extent = float(area) / rect_area if rect_area > 0 else 0

        # Shape filtering
        if aspect < min_ar:
            rejected_log.append(f"#{i} AR={aspect:.2f} < {min_ar}")
            continue
        if aspect > max_ar:
            rejected_log.append(f"#{i} AR={aspect:.2f} > {max_ar}")
            continue
        if solidity < min_sol:
            rejected_log.append(f"#{i} solidity={solidity:.2f} < {min_sol}")
            continue
        if extent < min_ext:
            rejected_log.append(f"#{i} extent={extent:.2f} < {min_ext}")
            continue

        # Position filtering (ignore edges)
        center_y = y + bh / 2
        if center_y < h * 0.05 or center_y > h * 0.95:
            rejected_log.append(f"#{i} at edge y={center_y:.0f}")
            continue

        # Color validation
        margin = 3
        x1, y1 = max(0, x-margin), max(0, y-margin)
        x2, y2 = min(w, x+bw+margin), min(h, y+bh+margin)
        roi = img_rgb[y1:y2, x1:x2]
        if roi.size == 0:
            continue

        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        mean_h = np.mean(roi_hsv[:,:,0])
        mean_s = np.mean(roi_hsv[:,:,1])
        mean_v = np.mean(roi_hsv[:,:,2])

        # Reject non-tamarind colors
        if 40 < mean_h < 85 and mean_s > 20:
            rejected_log.append(f"#{i} green h={mean_h:.1f}")
            continue
        if 85 <= mean_h <= 140 and mean_s > 25:
            rejected_log.append(f"#{i} blue h={mean_h:.1f}")
            continue
        if mean_s < 5 and mean_v > 195:
            rejected_log.append(f"#{i} white")
            continue
        if mean_v < 10:
            rejected_log.append(f"#{i} black")
            continue

        # Color scoring
        if (0 <= mean_h <= 35 or mean_h >= 160) and mean_s > 15 and 20 <= mean_v <= 230:
            cSc, cLab = 95, "สีสวย"
        elif mean_s > 8 and 15 <= mean_v <= 240:
            cSc, cLab = 80, "สีดี"
        else:
            cSc, cLab = 60, "สีพอใช้"

        # Size estimation
        px_per_cm = 45
        est_size = np.sqrt(area) / px_per_cm

        detected_objects.append({
            "id": 0,
            "x": x, "y": y, "w": bw, "h": bh,
            "area": int(area),
            "estimated_size": round(est_size, 1),
            "color_score": int(cSc),
            "color_label": cLab,
            "avg_brightness": round(mean_v / 2.55, 1),
            "aspect_ratio": round(aspect, 2),
            "solidity": round(solidity, 2),
            "extent": round(extent, 2),
            "angle": round(angle, 1),
            "roi": roi,
            "box": box
        })

    detected_objects.sort(key=lambda x: x["area"], reverse=True)
    for i, obj in enumerate(detected_objects):
        obj["id"] = i + 1

    return detected_objects, mask_clean, color_mask, rejected_log


def draw_detection_advanced(img_array, objects):
    """วาดกรอบครอบมะขามแต่ละผลด้วย minAreaRect"""
    img_copy = img_array.copy()
    colors = [
        (0, 200, 0), (0, 140, 255), (255, 0, 255),
        (255, 255, 0), (0, 255, 255), (255, 0, 0),
        (200, 100, 0), (100, 0, 200), (0, 100, 200), (200, 0, 100)
    ]

    for obj in objects:
        color = colors[(obj["id"] - 1) % len(colors)]

        # วาด minAreaRect (กรอบเอียง)
        if "box" in obj:
            cv2.drawContours(img_copy, [obj["box"]], 0, color, 3)
        else:
            x, y, bw, h = obj["x"], obj["y"], obj["w"], obj["h"]
            cv2.rectangle(img_copy, (x, y), (x+bw, y+h), color, 3)

        # Label
        x, y = obj["x"], obj["y"]
        label = f"#{obj['id']} {obj['color_label']} {obj['estimated_size']}cm"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(img_copy, (x, y-th-10), (x+tw+4, y), color, -1)
        cv2.putText(img_copy, label, (x+2, y-3), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)

    return img_copy


# ==================== SESSION STATE ====================
if "batch_results" not in st.session_state:
    st.session_state.batch_results = []
if "detected_objects" not in st.session_state:
    st.session_state.detected_objects = []
if "detection_params" not in st.session_state:
    st.session_state.detection_params = {
        "hsv_h_low": 0, "hsv_h_high": 45,
        "hsv_s_low": 5, "hsv_s_high": 255,
        "hsv_v_low": 10, "hsv_v_high": 250,
        "morph_close": 15,
        "min_area": 200, "max_area": 40000,
        "min_aspect": 0.5, "max_aspect": 15.0,
        "min_solidity": 0.2, "min_extent": 0.1,
    }

# ==================== UI ====================
st.markdown("<h1 style='text-align:center;'>🌿 วิเคราะห์มะขาม - หลายผลในภาพเดียว (v2)</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#a1887f;'>ปรับพารามิเตอร์ให้เหมาะสม → AI แยกผล → วิเคราะห์แยกต่อผล</p>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["📸 ถ่าย/อัปโหลด", "⚙️ ตั้งค่าการตรวจจับ", "🔍 แยกผล & วิเคราะห์", "📊 สรุปผล"])

# ==================== TAB 1: CAPTURE ====================
with tab1:
    col1, col2 = st.columns([1, 1])
    with col1:
        selected_type = st.selectbox("เลือกชนิดมะขาม", list(TAMARIND_TYPES.keys()), key="main_type")
        type_info = TAMARIND_TYPES[selected_type]
    with col2:
        st.info(f"ขนาดเหมาะสม: ≥{type_info['size_opt']}cm | ความชื้น: ≤{type_info['moisture_max']}%")

    st.markdown("---")

    cam_photo = st.camera_input("📷 ถ่ายภาพกองมะขาม", key="multi_cam")
    uploaded = st.file_uploader("หรืออัปโหลดภาพ", type=["jpg", "jpeg", "png"], key="multi_upload")

    img_source = None
    if cam_photo is not None:
        img_source = Image.open(cam_photo)
    elif uploaded is not None:
        img_source = Image.open(uploaded)

    if img_source is not None:
        img_array = np.array(img_source)
        st.session_state["source_image"] = img_array
        st.image(img_source, caption="ภาพต้นฉบับ", use_container_width=True)

        st.success("✅ ภาพพร้อม! ไปที่แท็บ '⚙️ ตั้งค่าการตรวจจับ' เพื่อปรับพารามิเตอร์และตรวจจับ")

# ==================== TAB 2: DETECTION SETTINGS ====================
with tab2:
    st.markdown("### ⚙️ ตั้งค่าการตรวจจับมะขาม")
    st.info("💡 ปรับค่าต่างๆ ให้เหมาะกับสภาพภาพ แล้วกด 'ตรวจจับ' ดูผลลัพธ์ ถ้ายังไม่ดีให้กลับมาปรับใหม่")

    if "source_image" not in st.session_state:
        st.warning("กรุณาถ่าย/อัปโหลดภาพในแท็บแรกก่อน")
    else:
        img_array = st.session_state["source_image"]

        p = st.session_state.detection_params

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**🎨 สี (HSV)**")
            h_low = st.slider("Hue ต่ำสุด", 0, 180, p["hsv_h_low"], key="h_low")
            h_high = st.slider("Hue สูงสุด", 0, 180, p["hsv_h_high"], key="h_high")
            s_low = st.slider("Saturation ต่ำสุด", 0, 255, p["hsv_s_low"], key="s_low")
            s_high = st.slider("Saturation สูงสุด", 0, 255, p["hsv_s_high"], key="s_high")
            v_low = st.slider("Value ต่ำสุด", 0, 255, p["hsv_v_low"], key="v_low")
            v_high = st.slider("Value สูงสุด", 0, 255, p["hsv_v_high"], key="v_high")

        with col2:
            st.markdown("**🔧 Morphology**")
            morph_close = st.slider("Closing Kernel", 3, 50, p["morph_close"], 1, key="morph_close")

            st.markdown("**📐 ขนาด**")
            min_area = st.slider("พื้นที่ต่ำสุด (px)", 50, 5000, p["min_area"], 50, key="min_area")
            max_area = st.slider("พื้นที่สูงสุด (px)", 5000, 150000, p["max_area"], 1000, key="max_area")

            st.markdown("**📏 รูปทรง**")
            min_ar = st.slider("Aspect Ratio ต่ำสุด", 0.1, 5.0, p["min_aspect"], 0.1, key="min_ar")
            max_ar = st.slider("Aspect Ratio สูงสุด", 1.0, 50.0, p["max_aspect"], 1.0, key="max_ar")

        with col3:
            st.markdown("**✅ เกณฑ์คุณภาพ**")
            min_sol = st.slider("Solidity ต่ำสุด", 0.0, 1.0, p["min_solidity"], 0.05, key="min_sol")
            min_ext = st.slider("Extent ต่ำสุด", 0.0, 1.0, p["min_extent"], 0.05, key="min_ext")

            st.markdown("**💡 คำแนะนำ**")
            st.caption("กองมะขาม: Close=25-35, Min Area=200")
            st.caption("พื้นขาว: Close=8-15, Min Area=100")
            st.caption("บนต้น: Close=10-15, Min Area=80")

        # Save params
        current_params = {
            "hsv_h_low": h_low, "hsv_h_high": h_high,
            "hsv_s_low": s_low, "hsv_s_high": s_high,
            "hsv_v_low": v_low, "hsv_v_high": v_high,
            "morph_close": morph_close,
            "min_area": min_area, "max_area": max_area,
            "min_aspect": min_ar, "max_aspect": max_ar,
            "min_solidity": min_sol, "min_extent": min_ext,
        }
        st.session_state.detection_params = current_params

        st.markdown("---")

        if st.button("🔍 ตรวจจับมะขามในภาพ", key="detect_btn"):
            with st.spinner("กำลังวิเคราะห์ภาพ..."):
                objects, final_mask, color_mask, rejected = detect_tamarinds_advanced(img_array, current_params)
                st.session_state.detected_objects = objects

                if len(objects) == 0:
                    st.warning(f"ไม่พบมะขามที่ผ่านเกณฑ์ ลองปรับค่าให้ผ่อนปรนลง (Rejected: {len(rejected)})")
                else:
                    st.success(f"พบมะขาม {len(objects)} ผลในภาพ!")

                col_r1, col_r2, col_r3 = st.columns(3)

                with col_r1:
                    st.image(img_array, caption="ภาพต้นฉบับ", use_container_width=True)

                with col_r2:
                    st.image(color_mask, caption="Color Mask (สีขาว = พื้นที่สีมะขาม)", use_container_width=True)

                with col_r3:
                    boxed_img = draw_detection_advanced(img_array, objects)
                    st.image(boxed_img, caption=f"ผลการตรวจจับ ({len(objects)} ผล)", use_container_width=True)

                with st.expander("🔧 Debug Info (คลิกดูรายละเอียด)"):
                    st.write(f"Rejected contours: {len(rejected)}")
                    if rejected:
                        st.text("\n".join(rejected[:50]))
                    if len(rejected) > 50:
                        st.text(f"... และอีก {len(rejected)-50} รายการ")

# ==================== TAB 3: ANALYZE EACH ====================
with tab3:
    if not st.session_state.detected_objects:
        st.info("กรุณาถ่าย/อัปโหลดภาพและกด 'ตรวจจับมะขามในภาพ' ในแท็บที่ 2 ก่อน")
    else:
        objects = st.session_state.detected_objects
        st.markdown(f"### พบ {len(objects)} ผล วิเคราะห์แยกต่อผลได้เลย")

        for obj in objects:
            with st.expander(f"🌿 ผลที่ {obj['id']} | {obj['color_label']} | ประมาณ {obj['estimated_size']}cm"):
                col_img, col_form = st.columns([1, 2])

                with col_img:
                    st.image(obj["roi"], caption=f"ผลที่ {obj['id']}", use_container_width=True)
                    st.caption(f"พื้นที่: {obj['area']} px\nความสว่าง: {obj['avg_brightness']}%")

                with col_form:
                    c1, c2 = st.columns(2)
                    with c1:
                        size = st.number_input(f"ขนาด(cm) #{obj['id']}", 0.1, 10.0, obj['estimated_size'], 0.1, key=f"size_{obj['id']}")
                        moisture = st.number_input(f"ความชื้น(%) #{obj['id']}", 0.0, 50.0, 15.0, 0.5, key=f"moist_{obj['id']}")
                        clean = st.selectbox(f"ความสะอาด #{obj['id']}", ["excellent", "good", "fair", "poor"],
                                             format_func=lambda x: {"excellent": "สะอาดมาก", "good": "สะอาด", "fair": "มีตำหนิ", "poor": "มีคราบ/รา"}[x],
                                             key=f"clean_{obj['id']}")
                    with c2:
                        pulp = st.selectbox(f"เนื้อ/เมล็ด #{obj['id']}", ["high", "medium", "low"],
                                          format_func=lambda x: {"high": "เนื้อมาก", "medium": "ปานกลาง", "low": "เนื้อน้อย"}[x],
                                          key=f"pulp_{obj['id']}")
                        taste = st.selectbox(f"รสชาติ #{obj['id']}", ["excellent", "good", "fair", "poor"],
                                             format_func=lambda x: {"excellent": "ดีมาก", "good": "ดี", "fair": "พอใช้", "poor": "ไม่ดี"}[x],
                                             key=f"taste_{obj['id']}")
                        shape = st.selectbox(f"รูปทรง #{obj['id']}", ["excellent", "good", "fair", "poor"],
                                             format_func=lambda x: {"excellent": "สวย", "good": "ดี", "fair": "พอใช้", "poor": "ไม่ดี"}[x],
                                             key=f"shape_{obj['id']}")

                    if st.button(f"➕ เพิ่มผลที่ {obj['id']} เข้าชุด", key=f"add_{obj['id']}"):
                        total, scores = calculate_grade(type_info, size, obj['color_score'], clean, moisture, pulp, taste, shape)
                        gr, label, gr_color = get_grade(total)

                        result = {
                            "ลำดับ": len(st.session_state.batch_results) + 1,
                            "ผลที่": obj['id'],
                            "ชนิด": selected_type,
                            "ขนาด(cm)": size,
                            "ความชื้น(%)": moisture,
                            "สี": obj['color_score'],
                            "ความสะอาด": scores["clean"],
                            "รสชาติ": scores["taste"],
                            "รูปทรง": scores["shape"],
                            "คะแนนรวม": total,
                            "เกรด": gr,
                            "ตลาด": label,
                            "ราคา": type_info.get(f"price_{gr}", "N/A"),
                            "ข้อเสนอแนะ": " | ".join(get_suggestions(scores))
                        }
                        st.session_state.batch_results.append(result)
                        st.success(f"✅ เพิ่มผลที่ {obj['id']} เกรด {gr} ({total}/100) แล้ว!")

# ==================== TAB 4: SUMMARY ====================
with tab4:
    if not st.session_state.batch_results:
        st.warning("ยังไม่มีข้อมูล")
    else:
        df = pd.DataFrame(st.session_state.batch_results)

        total_items = len(df)
        avg_score = df["คะแนนรวม"].mean()
        grade_counts = df["เกรด"].value_counts().to_dict()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("จำนวนผล", total_items)
        m2.metric("คะแนนเฉลี่ย", f"{avg_score:.1f}")
        m3.metric("เกรด A", grade_counts.get("A", 0))
        m4.metric("เกรด D", grade_counts.get("D", 0))

        st.markdown("#### การกระจายตัวของเกรด")
        st.bar_chart(df["เกรด"].value_counts().sort_index())

        st.markdown("#### ตารางผลการวิเคราะห์")
        display_df = df[["ลำดับ", "ผลที่", "ชนิด", "ขนาด(cm)", "ความชื้น(%)", "คะแนนรวม", "เกรด", "ราคา"]].copy()

        def color_grade(val):
            colors = {"A": "background-color: #4caf50; color: white;",
                      "B": "background-color: #8bc34a; color: white;",
                      "C": "background-color: #ffc107; color: black;",
                      "D": "background-color: #f44336; color: white;"}
            return colors.get(val, "")

        styled_df = display_df.style.applymap(color_grade, subset=["เกรด"])
        st.dataframe(styled_df, use_container_width=True)

        csv = df.to_csv(index=False).encode('utf-8-sig')
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button("📥 ดาวน์โหลด CSV", csv, "tamarind_multi_analysis.csv", "text/csv")
        with col_dl2:
            if st.button("🗑️ ล้างข้อมูล"):
                st.session_state.batch_results = []
                st.session_state.detected_objects = []
                st.rerun()

        st.markdown("#### สรุปตามชนิด")
        type_summary = df.groupby("ชนิด").agg({"คะแนนรวม": ["mean", "min", "max", "count"]}).round(1)
        type_summary.columns = ["คะแนนเฉลี่ย", "ต่ำสุด", "สูงสุด", "จำนวน"]
        st.dataframe(type_summary, use_container_width=True)

st.markdown("---")
st.caption("อ้างอิง: มาตรฐานสินค้าเกษตรมะขามหวาน (ราชกิจจานุเบกษา) | Algorithm v2 - minAreaRect + Multi-criteria Filtering")
