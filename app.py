import streamlit as st
from transformers import pipeline
import sqlite3
import datetime
import pandas as pd

# --- CẤU HÌNH TRANG ---
st.set_page_config(
    page_title="Đồ án: Trợ lý Cảm xúc",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- TÙY CHỈNH GIAO DIỆN (CSS) ---
st.markdown("""
    <style>
    h1, h2, h3 {
        color: white !important;
    }
    .stButton>button {
        background-color: #0056b3;
        color: white;
        border-radius: 10px;
        height: 3em;
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 1. SIDEBAR (THÔNG TIN SINH VIÊN) ---
with st.sidebar:
    
    st.title("SEMINAR CHUYÊN ĐỀ")
    st.info("ĐỀ TÀI: XÂY DỰNG TRỢ LÝ PHÂN LOẠI CẢM XÚC TIẾNG VIỆT SỬ DỤNG TRANSFORMER")
    
    st.markdown("---")
    st.subheader("Thông tin sinh viên")
    st.markdown("**Họ tên:** Trần Hữu Minh")
    st.markdown("**MSSV:** 3121410323")
    st.markdown("**Lớp:** DCT1214")

# --- 2. LOAD MODEL ---
@st.cache_resource
def load_sentiment_pipeline():
    model_name = "wonrax/phobert-base-vietnamese-sentiment"
    nlp = pipeline("sentiment-analysis", model=model_name, tokenizer=model_name)
    return nlp

try:
    classifier = load_sentiment_pipeline()
except Exception as e:
    st.error(f"Lỗi tải model: {e}")

# --- 3. DATABASE ---
def init_db():
    conn = sqlite3.connect('sentiment_history.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS sentiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT,
            label TEXT,
            score REAL,
            timestamp DATETIME
        )
    ''')
    conn.commit()
    conn.close()

def save_to_db(text, label, score):
    conn = sqlite3.connect('sentiment_history.db')
    c = conn.cursor()
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    c.execute('INSERT INTO sentiments (text, label, score, timestamp) VALUES (?, ?, ?, ?)',
              (text, label, score, current_time))
    conn.commit()
    conn.close()

def get_history():
    conn = sqlite3.connect('sentiment_history.db')
    query = "SELECT id, text, label, score, timestamp FROM sentiments ORDER BY timestamp DESC LIMIT 50"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

init_db()

# --- 4. LOGIC XỬ LÝ ---
teencode_dict = {
    "bt": "bình thường", "dc": "được", "hok": "không", "ko": "không", 
    "k": "không", "uh": "ừ", "uk": "ừ", "oke": "ok",
    "bun": "buồn", "chan": "chán", "vui": "vui", "hp": "hạnh phúc",
    "suong": "sướng", "phe": "phê", "tuyet": "tuyệt", "tot": "tốt",
    "xau": "xấu", "te": "tệ", "do": "dở", "met": "mệt", "duoi": "đuối",
    "so": "sợ", "lo": "lo", "yeu": "yếu", "khoe": "khỏe",
    "wa": "quá", "qua": "quá", "rat": "rất", "lam": "lắm",
    "hom": "hôm", "nay": "nay", "toi": "tôi", "t": "tôi", "minh": "mình",
    "cam": "cảm", "thay": "thấy", "bùn": "buồn", "tui": "tôi"
}

def preprocess_text(text):
    if not text: return ""
    text = text.lower().strip()
    words = text.split()
    corrected_words = [teencode_dict.get(word, word) for word in words]
    return " ".join(corrected_words)

def map_label(model_label):
    label = model_label.upper()
    if label in ["POS", "POSITIVE", "LABEL_1", "LABEL_2"]: return "POSITIVE"
    if label in ["NEG", "NEGATIVE", "LABEL_0"]: return "NEGATIVE"
    return "NEUTRAL"

# --- 5. GIAO DIỆN CHÍNH (MAIN AREA) ---
st.title("Dự báo Cảm xúc Văn bản")

col_input, col_result = st.columns([2, 1])

with col_input:
    user_input = st.text_area("Nhập câu tiếng Việt cần phân tích:", height=150, placeholder="Ví dụ: Hôm nay tôi rất vui")
    analyze_btn = st.button("Thực hiện Phân tích")

# Biến để lưu kết quả hiển thị
result_container = col_result.empty()

if analyze_btn:
    if len(user_input.strip()) < 5:
        st.toast("Câu quá ngắn, vui lòng nhập lại!")
    else:
        # Tiền xử lý & Model
        clean_text = preprocess_text(user_input)
        with st.spinner('Đang xử lý...'):
            result = classifier(clean_text)[0]
            raw_label = result['label']
            score = result['score']
            final_label = map_label(raw_label)
            
            # Threshold Logic
            if score < 0.60:
                final_label = "NEUTRAL"
            
            save_to_db(user_input, final_label, score)

        with col_result:
            st.subheader("Kết quả dự báo:")
            
            # Logic hiển thị màu sắc
            if final_label == "POSITIVE":
                st.success(f"#### {final_label} (TÍCH CỰC)")
            elif final_label == "NEGATIVE":
                st.error(f"#### {final_label} (TIÊU CỰC)")
            else:
                st.info(f"#### {final_label} (TRUNG TÍNH)")
                
            st.write(f"Độ tin cậy: **{score:.2%}**")

            with st.expander("Xem chi tiết kỹ thuật (Debug)"):
                st.json({
                    "text": user_input,
                    "sentiment": final_label,
                    "raw_score": score,
                    "model_processed": clean_text
                })

st.divider()

# --- 6. LỊCH SỬ (DƯỚI CÙNG) ---
st.subheader("Nhật ký phân tích")
if st.button("Cập nhật danh sách"):
    st.rerun()

df_history = get_history()
if not df_history.empty:
    df_history.columns = ["ID", "Nội dung", "Cảm xúc", "Độ tin cậy", "Thời gian"]
    st.dataframe(df_history.set_index("ID"), use_container_width=True)
else:
    st.info("Chưa có dữ liệu lịch sử.")