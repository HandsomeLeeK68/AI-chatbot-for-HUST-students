#.\venv\Scripts\activate
# streamlit run app.py

import streamlit as st
import time
from QA_Chatbot import load_llm, read_vector_db, create_qa_chain

st.set_page_config(page_title="HUST Assistant Chatbot", page_icon="🎓", layout="centered")

# --- LOAD RESOURCES (Chỉ load 1 lần) ---
@st.cache_resource
def load_resources():
    db = read_vector_db()
    model_file = "models/vinallama-7b-chat_q5_0.gguf"
    llm = load_llm(model_file)
    return llm, db

try:
    with st.spinner("Đang khởi động hệ thống..."):
        llm, db = load_resources()
        
    # --- KHỞI TẠO CHAIN & MEMORY TRONG SESSION STATE ---
    # Lưu chain vào session_state để nó ghi nhớ qua các lần rerun của Streamlit
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = create_qa_chain(llm, db)
        
    st.success("Hệ thống đã sẵn sàng!", icon="✅")
    time.sleep(1)
    st.empty()
except Exception as e:
    st.error(f"Lỗi khởi động: {e}")
    st.stop()

# --- QUẢN LÝ GIAO DIỆN ---
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.image("hust_logo.png", width=250)
    st.title("🎓 HUST Bot")
    st.markdown("---")
    st.markdown("**Các nội dung hỗ trợ trả lời:**")
    st.markdown("""
    - Quy chế đào tạo đại học
    - Quy định về học phí, học bổng
    - Quy trình đăng ký môn học, thi cử
    - Các chính sách hỗ trợ sinh viên
    - Thông tin liên hệ các phòng ban
    - Các câu hỏi thường gặp khác về HUST...
    """)    
    if st.button("🗑️ Xóa lịch sử chat"):
        st.session_state.messages = []
        # Quan trọng: Phải clear cả memory trong chain
        st.session_state.qa_chain.memory.clear() 
        st.rerun()

st.title("🎓 Trợ lý Quy chế AI - HUST")
st.caption("🚀 Đặt câu hỏi về bất kỳ vấn đề nào liên quan tới đại học Bách khoa Hà Nội - HUST")
st.markdown("---")

# --- THÊM CSS ĐỂ HIỂN THỊ TEXT DƯỚI CHAT INPUT (FIX) ---
st.markdown("""
    <style>
    /* 1. Ẩn footer mặc định */
    footer {visibility: hidden;}
    
    /* 2. Thay đổi style của container chat input */
    [data-testid="stChatInput"] {
        padding-bottom: 25px !important; /* Tạo khoảng trống phía dưới box chat để chứa chữ */
        position: relative; /* Để làm mốc tọa độ cho dòng chữ */
    }

    /* 3. Chèn dòng chữ vào vị trí mong muốn */
    [data-testid="stChatInput"]::after {
        content: "AI có thể mắc lỗi, vui lòng kiểm tra lại";
        position: absolute;   /* Tách ra khỏi dòng chảy flexbox bình thường */
        bottom: 0px;          /* Đặt nằm sát đáy của container cha (đã có padding ở trên) */
        left: 0;
        width: 100%;          /* Chiếm toàn bộ chiều rộng để căn giữa */
        text-align: center;
        font-size: 11px;
        color: #888;          /* Màu xám nhạt dịu mắt */
        font-style: italic;
        pointer-events: none; /* Đảm bảo chuột không bấm nhầm vào chữ */
    }
    </style>
""", unsafe_allow_html=True)

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Nhập câu hỏi của bạn..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        with st.spinner("Đang suy nghĩ..."):
            try:
                # ConversationalRetrievalChain nhận input là 'question'
                response = st.session_state.qa_chain.invoke({"question": prompt})
                
                result_text = response.get('answer', "Xin lỗi, không có câu trả lời.")
                
                # Hiệu ứng gõ chữ
                import re
                
                # Dùng regex để tách từ nhưng GIỮ LẠI khoảng trắng và xuống dòng
                # split() cũ sẽ xóa mất \n khiến văn bản bị dồn cục
                tokens = re.split(r'(\s+)', result_text) 
                
                for token in tokens:
                    full_response += token
                    # Giảm thời gian sleep xuống một chút để chat mượt hơn
                    time.sleep(0.01) 
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
                
            except Exception as e:
                full_response = f"⚠️ Lỗi: {str(e)}"
                message_placeholder.markdown(full_response)
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# git init
# git remote add origin link_to_your_repo
# git add .
# git commit -m "Initial commit"
# git branch -M main   # Đặt tên nhánh là main (tuỳ repo)
# git push -u origin main

# mỗi lần sửa code xong, chạy lệnh này để push lên repo
# git add .
# git commit -m "Mô tả thay đổi"
# git push