import streamlit as st
import numpy as np
from PIL import Image, ImageDraw
import cv2
import pandas as pd

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="วิเคราะห์มะขาม - Multi Detector",
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

# ==================== MULTI-DETECT ENGINE ====================
def detect_tamarinds_in_image(img_array, min_area=500, max_area=50000):
    """
    ตรวจจับมะขามหลายลูกในภาพเดียวด้วย OpenCV
    ใช้ Color Masking (HSV) + Contour Detection
    """
    img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    # สร้าง Mask สำหรับสีน้ำตาล/เหลือง/แดงของมะขาม
    # Range 1: สีน้ำตาลแดง (ripe tamarind)
    lower1 = np.array([0, 40, 20])
    upper1 = np.array([30, 255, 200])
    mask1 = cv2.inRange(hsv, lower1, upper1)

    # Range 2: สีเหลืองอมน้ำตาล
    lower2 = np.array([10, 30, 30])
    upper2 = np.array([40, 255, 220])
    mask2 = cv2.inRange(hsv, lower2, upper2)

    # รวม Mask
    mask = cv2.bitwise_or(mask1, mask2)

    # Morphological operations ลบ noise
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # หา Contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detected_objects = []
    h_img, w_img = img_array.shape[:2]

    for i, cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        if area < min_area or area > max_area:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        # ป้องกัน out of bounds
        x = max(0, x)
        y = max(0, y)
        w = min(w, w_img - x)
        h = min(h, h_img - y)

        # Crop ROI
        roi = img_array[y:y+h, x:x+w]
        if roi.size == 0:
            continue

        # วิเคราะห์สี ROI
        roi_rgb = roi.astype(float) / 255.0
        r, g, b = roi_rgb[:,:,0], roi_rgb[:,:,1], roi_rgb[:,:,2]
        max_c = np.maximum(np.maximum(r, g), b)
        min_c = np.minimum(np.minimum(r, g), b)
        delta = max_c - min_c
        v = max_c
        s = np.where(max_c == 0, 0, delta / max_c)

        avg_bri = float(np.mean(v) * 100)
        warm = np.sum(((delta != 0) & (((np.arctan2(np.sqrt(3)*(g-b), 2*r-g-b) * 180/np.pi + 360) % 360) >= 15) & 
                       (((np.arctan2(np.sqrt(3)*(g-b), 2*r-g-b) * 180/np.pi + 360) % 360) <= 45)))
        total_px = roi_rgb.shape[0] * roi_rgb.shape[1]
        wR = float(warm / total_px) if total_px > 0 else 0

        if wR > 0.4:
            if avg_bri > 40 and avg_bri < 70:
                cSc, cLab = 95, "สีสวย"
            elif avg_bri >= 30 and avg_bri <= 80:
                cSc, cLab = 80, "สีดี"
            else:
                cSc, cLab = 65, "สีพอใช้"
        elif wR > 0.2:
            cSc, cLab = 60, "สีไม่สม่ำเสมอ"
        else:
            cSc, cLab = 35, "สีไม่ดี"

        # ประมาณขนาดจาก pixel area (สมมติ 1cm = 50 pixels)
        px_per_cm = 50
        est_size = np.sqrt(area) / px_per_cm

        detected_objects.append({
            "id": i + 1,
            "x": x, "y": y, "w": w, "h": h,
            "area": int(area),
            "estimated_size": round(est_size, 1),
            "color_score": int(cSc),
            "color_label": cLab,
            "avg_brightness": round(avg_bri, 1),
            "roi": roi
        })

    # เรียงตามพื้นที่จากมากไปน้อย
    detected_objects.sort(key=lambda x: x["area"], reverse=True)
    return detected_objects, mask

def draw_detection_boxes(img_array, objects):
    """วาดกรอบครอบมะขามแต่ละลูก"""
    img_copy = img_array.copy()
    colors = [(0, 255, 0), (255, 165, 0), (0, 165, 255), (255, 0, 255), (0, 255, 255)]

    for obj in objects:
        color = colors[(obj["id"] - 1) % len(colors)]
        x, y, w, h = obj["x"], obj["y"], obj["w"], obj["h"]
        cv2.rectangle(img_copy, (x, y), (x+w, y+h), color, 3)
        label = f"#{obj['id']} {obj['color_label']}"
        cv2.putText(img_copy, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return img_copy

# ==================== SESSION STATE ====================
if "batch_results" not in st.session_state:
    st.session_state.batch_results = []
if "detected_objects" not in st.session_state:
    st.session_state.detected_objects = []

# ==================== UI ====================
st.markdown("<h1 style='text-align:center;'>🌿 วิเคราะห์มะขาม - หลายลูกในภาพเดียว</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:#a1887f;'>ถ่ายภาพกองมะขาม → AI แยกลูก → วิเคราะห์แยกต่อลูก</p>", unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["📸 ถ่าย/อัปโหลด", "🔍 แยกลูก & วิเคราะห์", "📊 สรุปผล"])

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

        st.markdown("#### ⚙️ ตั้งค่าการตรวจจับ")
        c1, c2 = st.columns(2)
        with c1:
            min_area = st.slider("พื้นที่ต่ำสุด (pixels)", 100, 5000, 800, 100, key="min_area")
        with c2:
            max_area = st.slider("พื้นที่สูงสุด (pixels)", 5000, 100000, 50000, 1000, key="max_area")

        if st.button("🔍 ตรวจจับมะขามในภาพ", key="detect_btn"):
            with st.spinner("กำลังวิเคราะห์ภาพ..."):
                objects, mask = detect_tamarinds_in_image(img_array, min_area, max_area)
                st.session_state.detected_objects = objects

                if len(objects) == 0:
                    st.warning("ไม่พบมะขามในภาพ ลองปรับค่าพื้นที่ต่ำสุด/สูงสุด หรือถ่ายภาพใหม่")
                else:
                    st.success(f"พบมะขาม {len(objects)} ลูกในภาพ!")

                    # แสดงภาพพร้อมกรอบ
                    boxed_img = draw_detection_boxes(img_array, objects)
                    st.image(boxed_img, caption="ผลการตรวจจับ (เลข = ลำดับลูก)", use_container_width=True)

                    # แสดง mask
                    st.image(mask, caption="AI Mask (สีขาว = พื้นที่ที่ตรวจพบ)", use_container_width=True)

# ==================== TAB 2: ANALYZE EACH ====================
with tab2:
    if not st.session_state.detected_objects:
        st.info("กรุณาถ่าย/อัปโหลดภาพและกด 'ตรวจจับมะขามในภาพ' ในแท็บแรกก่อน")
    else:
        objects = st.session_state.detected_objects
        st.markdown(f"### พบ {len(objects)} ลูก วิเคราะห์แยกต่อลูกได้เลย")

        for obj in objects:
            with st.expander(f"🌿 ลูกที่ {obj['id']} | {obj['color_label']} | ประมาณ {obj['estimated_size']}cm"):
                col_img, col_form = st.columns([1, 2])

                with col_img:
                    st.image(obj["roi"], caption=f"ลูกที่ {obj['id']}", use_container_width=True)
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

                    if st.button(f"➕ เพิ่มลูกที่ {obj['id']} เข้าชุด", key=f"add_{obj['id']}"):
                        total, scores = calculate_grade(type_info, size, obj['color_score'], clean, moisture, pulp, taste, shape)
                        gr, label, gr_color = get_grade(total)

                        result = {
                            "ลำดับ": len(st.session_state.batch_results) + 1,
                            "ลูกที่": obj['id'],
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
                        st.success(f"✅ เพิ่มลูกที่ {obj['id']} เกรด {gr} ({total}/100) แล้ว!")

# ==================== TAB 3: SUMMARY ====================
with tab3:
    if not st.session_state.batch_results:
        st.warning("ยังไม่มีข้อมูล")
    else:
        df = pd.DataFrame(st.session_state.batch_results)

        total_items = len(df)
        avg_score = df["คะแนนรวม"].mean()
        grade_counts = df["เกรด"].value_counts().to_dict()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("จำนวนลูก", total_items)
        m2.metric("คะแนนเฉลี่ย", f"{avg_score:.1f}")
        m3.metric("เกรด A", grade_counts.get("A", 0))
        m4.metric("เกรด D", grade_counts.get("D", 0))

        st.markdown("#### การกระจายตัวของเกรด")
        st.bar_chart(df["เกรด"].value_counts().sort_index())

        st.markdown("#### ตารางผลการวิเคราะห์")
        display_df = df[["ลำดับ", "ลูกที่", "ชนิด", "ขนาด(cm)", "ความชื้น(%)", "คะแนนรวม", "เกรด", "ราคา"]].copy()

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
st.caption("อ้างอิง: มาตรฐานสินค้าเกษตรมะขามหวาน (ราชกิจจานุเบกษา)")

        st.dataframe(type_summary, use_container_width=True)

st.markdown("---")
st.caption("อ้างอิง: มาตรฐานสินค้าเกษตรมะขามหวาน (ราชกิจจานุเบกษา)")
