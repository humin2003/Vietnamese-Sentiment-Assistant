Trợ Lý Phân Loại Cảm Xúc Tiếng Việt (Vietnamese Sentiment Analysis)

Môn học: Seminar Chuyên đề

Sinh viên thực hiện: Trần Hữu Minh

MSSV: 3121410323

Thời gian: 11/2025

📖 Giới thiệu (Introduction)

Dự án xây dựng một ứng dụng web (Web App) đơn giản sử dụng mô hình Transformer (PhoBERT) để phân tích cảm xúc của các câu văn tiếng Việt. Ứng dụng có khả năng nhận diện 3 trạng thái cảm xúc:

🟢 TÍCH CỰC (Positive)

🔴 TIÊU CỰC (Negative)

⚪ TRUNG TÍNH (Neutral)

Ứng dụng được tối ưu hóa để hiểu cả những câu viết tắt, không dấu (teencode) và có cơ chế lưu trữ lịch sử phân tích.

🚀 Tính năng nổi bật (Key Features)

Mô hình AI mạnh mẽ: Sử dụng wonrax/phobert-base-vietnamese-sentiment (dựa trên PhoBERT) đạt độ chính xác cao cho tiếng Việt.

Xử lý ngôn ngữ tự nhiên (NLP Pipeline):

Chuẩn hóa: Tự động sửa lỗi chính tả, map các từ viết tắt (ko -> không, bun -> buồn...) thông qua bộ từ điển tùy chỉnh.

Ngưỡng tin cậy (Confidence Threshold): Nếu mô hình không chắc chắn (độ tin cậy < 60%), hệ thống sẽ tự động gán nhãn "Trung tính" để đảm bảo an toàn.

Giao diện thân thiện: Xây dựng bằng Streamlit, trực quan, dễ sử dụng, hỗ trợ Dark Mode.

Lưu trữ cục bộ: Tích hợp SQLite để lưu lại toàn bộ lịch sử các câu đã phân tích (Text, Label, Score, Timestamp).

🛠️ Công nghệ sử dụng (Tech Stack)

Ngôn ngữ: Python 3.10+

Giao diện (Frontend): Streamlit

AI Core: Hugging Face Transformers, PyTorch

Database: SQLite3 (Built-in)

Xử lý dữ liệu: Pandas

⚙️ Hướng dẫn cài đặt (Installation)

Do sự xung đột giữa phiên bản NumPy 2.0 mới và các thư viện Deep Learning cũ, vui lòng tuân thủ các bước cài đặt sau để đảm bảo ứng dụng chạy ổn định.

Bước 1: Clone dự án hoặc tải về máy

Giải nén thư mục dự án.

Bước 2: Cài đặt thư viện

Mở Terminal tại thư mục dự án và chạy lệnh sau (đảm bảo đã cài Python):

pip install -r requirements.txt


Lưu ý kỹ thuật: Nếu gặp lỗi liên quan đến numpy.dtype size changed, hãy chạy lệnh sau để hạ cấp NumPy:
pip install "numpy<2.0"

Bước 3: Chạy ứng dụng

streamlit run app.py


Ứng dụng sẽ tự động mở trên trình duyệt tại địa chỉ: http://localhost:8501

📂 Cấu trúc dự án

📁 VietnameseSentimentAssistant/
├── 📄 app.py                 # Mã nguồn chính (Giao diện + Logic AI + DB)
├── 📄 requirements.txt       # Danh sách thư viện cần cài đặt
├── 📄 README.md              # Tài liệu hướng dẫn sử dụng
└── 🗄️ sentiment_history.db   # Database SQLite (Tự động tạo khi chạy app)


🧪 Kết quả thử nghiệm (Test Cases)

Hệ thống đã được kiểm thử với các trường hợp sau:

## 🧪 Kết quả thử nghiệm (Test Cases)

| STT | Đầu vào (Input) | Tiền xử lý | Model Output | Kết quả hiển thị | Đánh giá |
| :---: | :--- | :---: | :---: | :---: | :---: |
| 1 | Hôm nay tôi rất vui | (Giữ nguyên) | POSITIVE | POSITIVE | Đúng |
| 2 | Món ăn này dở quá | (Giữ nguyên) | NEGATIVE | NEGATIVE | Đúng |
| 3 | Thời tiết bình thường | (Giữ nguyên) | NEUTRAL | NEUTRAL | Đúng |
| 4 | Rat vui hom nay | Rất vui hôm nay | POSITIVE | POSITIVE | Đúng (Nhờ Dict) |
| 5 | Công việc ổn định | (Giữ nguyên) | NEUTRAL | NEUTRAL | Đúng |
| 6 | Phim này hay lắm | (Giữ nguyên) | POSITIVE | POSITIVE | Đúng |
| 7 | Tôi buồn vì thất bại | (Giữ nguyên) | NEGATIVE | NEGATIVE | Đúng |
| 8 | Ngày mai đi học | (Giữ nguyên) | NEUTRAL | NEUTRAL | Đúng |
| 9 | Cảm ơn bạn rất nhiều | (Giữ nguyên) | POSITIVE | POSITIVE | Đúng |
| 10 | Mệt mỏi quá hôm nay | (Giữ nguyên) | NEGATIVE | NEGATIVE | Đúng |
| 11 | Tui thấy bt | Tôi thấy bình thường | NEUTRAL | NEUTRAL | Đúng |
| 12 | Tui thấy hok vui | Tôi thấy không vui | NEGATIVE | NEGATIVE | Đúng |

🐛 Khắc phục sự cố (Troubleshooting)

Trong quá trình phát triển, nhóm đã xử lý các vấn đề sau:

Lỗi xung đột NumPy 2.x: Đã cố định version numpy<2.0.

Lỗi Model không hiểu Tiếng Việt không dấu: Đã xây dựng hàm preprocess_text với từ điển teencode_dict để dịch sang tiếng Việt chuẩn trước khi đưa vào AI.

Lỗi hiển thị sai màu: Đã chuẩn hóa nhãn đầu ra của Model (POS/NEG/NEU) về định dạng thống nhất Tiếng Việt.



