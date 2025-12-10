import os
import re
import pymupdf4llm
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

pdf_data_path = "data"
vector_db_path = "vectorstores/db_faiss"

def clean_and_format_markdown(md_text):
    """
    Hàm chuẩn hóa Markdown, đảm bảo Splitter cắt đúng Chương/Điều
    """
    # Regex này tìm các dòng bắt đầu bằng "Điều" + số, và thêm "## " vào trước
    md_text = re.sub(r'\n(Điều \d+)', r'\n## \1', md_text)
    # 2. Thêm dấu # vào trước chữ "Chương"
    md_text = re.sub(r'\n(Chương [IVX]+)', r'\n# \1', md_text)
    # 3. Xử lý trường hợp in đậm sai: **Điều 1** -> ## Điều 1
    md_text = re.sub(r'\n\*\*(Điều \d+.*?)\*\*', r'\n## \1', md_text)
    
    return md_text

def create_db_from_pdf_via_markdown():
    print("🚀 Bắt đầu quy trình tự động: PDF -> Markdown -> Vector DB...")
    
    all_splits = []
    
    if not os.path.exists(pdf_data_path):
        print(f"❌ Lỗi: Không tìm thấy thư mục '{pdf_data_path}'")
        return

    # Duyệt file và chuyển đổi
    for filename in os.listdir(pdf_data_path):
        if filename.lower().endswith(".pdf"):
            file_path = os.path.join(pdf_data_path, filename)
            print(f"📄 Đang xử lý: {filename}...")
            
            try:
                # PDF -> Markdown
                md_text = pymupdf4llm.to_markdown(file_path)
                
                # BƯỚC TỰ ĐỘNG SỬA LỖI HEADER
                md_text = clean_and_format_markdown(md_text)
                
                # (Tùy chọn) Lưu file .md ra máy để bạn kiểm tra
                # with open(f"{file_path}.md", "w", encoding="utf-8") as f:
                #     f.write(md_text)

                # Cắt theo cấu trúc văn bản hành chính (Header)
                headers_to_split_on = [
                    ("#", "Header 1"),      # Chương
                    ("##", "Header 2"),     # Điều
                    ("###", "Header 3"),    # Khoản / Mục lục nhỏ
                ]
                
                markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
                md_header_splits = markdown_splitter.split_text(md_text)
                
                # Gán metadata tên file nguồn
                for split in md_header_splits:
                    split.metadata["source"] = filename
                
                all_splits.extend(md_header_splits)
                
            except Exception as e:
                print(f"⚠️ Lỗi khi đọc file {filename}: {e}")

    if not all_splits:
        print("⚠️ Không tìm thấy dữ liệu nào để xử lý.")
        return

    print(f"✅ Đã tách sơ bộ thành {len(all_splits)} khối theo Chương/Điều.")

    # 3. Cắt nhỏ tiếp nếu một "Điều" quá dài (để vừa context của LLM)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1024,
        chunk_overlap=100
    )
    final_chunks = text_splitter.split_documents(all_splits)
    
    print(f"✂️ Tổng số chunks cuối cùng: {len(final_chunks)}")

    # 4. Tạo Vector DB
    print("🧠 Đang tạo Embeddings & lưu vào FAISS (bước này mất vài phút)...")
    embedding_model = HuggingFaceEmbeddings(model_name="bkai-foundation-models/vietnamese-bi-encoder")
    
    db = FAISS.from_documents(final_chunks, embedding=embedding_model)
    db.save_local(vector_db_path)
    print(f"🎉 Xong! Database đã lưu tại: {vector_db_path}")

if __name__ == "__main__":
    create_db_from_pdf_via_markdown()