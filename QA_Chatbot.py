from langchain_community.llms import LlamaCpp
from langchain_classic.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- THƯ VIỆN MỚI CHO RE-RANKING ---
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

model_file = "models/vinallama-7b-chat_q5_0.gguf"
vector_db_path = "vectorstores/db_faiss"

# Load LLM
def load_llm(model_file):
    llm = LlamaCpp(
        model_path=model_file,
        temperature=0.1,
        n_ctx=2048,                # Độ dài cửa sổ ngữ cảnh
        max_tokens=1024,
        top_p=1,
        
        # Cấu hình GPU
        n_gpu_layers=-1,
        n_batch=512,               # Số token xử lý song song
        f16_kv=True,               # Tiết kiệm VRAM
        verbose=True,              # Hiện log chi tiết
    )
    return llm

# Hàm cấu hình Re-ranker Retriever (MỚI)
def create_reranker_retriever(vector_db):
    # 1. Base Retriever: Lấy nhiều document hơn (k=15) để lọc lại sau
    base_retriever = vector_db.as_retriever(
        search_type="similarity", 
        search_kwargs={"k": 15} 
    )
    
    # 2. Khởi tạo mô hình Cross-Encoder để Re-ranking
    # Model này rất mạnh cho đa ngôn ngữ (bao gồm tiếng Việt)
    model_name = "BAAI/bge-reranker-v2-m3" 
    
    print(f"Đang tải model Re-ranking: {model_name}...")
    model = HuggingFaceCrossEncoder(model_name=model_name)
    
    # 3. Tạo Compressor: Chỉ lấy top 3 kết quả tốt nhất sau khi chấm điểm
    compressor = CrossEncoderReranker(model=model, top_n=3)
    
    # 4. Tạo Retriever tích hợp Re-ranking
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=base_retriever
    )
    
    return compression_retriever

# Tạo prompt template
def create_prompt(template):
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    return prompt

# Tạo chain RetrievalQA
def create_qa_chain(llm, vector_db):
    #Condense Question
    condense_template = """<|im_start|>system
    Bạn là một trợ lý AI hữu ích. Nhiệm vụ của bạn là viết lại câu hỏi tiếp theo của người dùng thành một câu hỏi độc lập, đầy đủ ý nghĩa dựa trên lịch sử trò chuyện.
    Lịch sử trò chuyện: {chat_history}
    Câu hỏi mới: {question}
    <|im_end|>
    <|im_start|>assistant
    Câu hỏi được viết lại:"""
    condense_question_prompt = PromptTemplate.from_template(condense_template)

    # 2. Prompt để trả lời câu hỏi (QA Prompt)
    qa_template = """<|im_start|>system
    Bạn là trợ lý ảo AI hữu ích của Đại học Bách Khoa Hà Nội (HUST). Nhiệm vụ của bạn là trả lời câu hỏi dựa trên thông tin được cung cấp.

    Yêu cầu:
    1. Chỉ trả lời dựa trên "Ngữ cảnh được cung cấp".
    2. Nếu không tìm thấy thông tin trong ngữ cảnh, hãy trả lời: "Xin lỗi, tôi không tìm thấy thông tin này trong bộ quy chế hiện tại."
    3. Trả lời đúng trọng tâm, văn phong lịch sự, trang trọng.

    Ngữ cảnh:
    {context}
    <|im_end|>
    <|im_start|>user
    Câu hỏi: {question}
    <|im_end|>
    <|im_start|>assistant
    """
    qa_prompt = PromptTemplate(template=qa_template, input_variables=["context", "question"])

    # 3. Tạo Memory
    # memory_key="chat_history" phải khớp với input của condense_template
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key='answer' # Chỉ lưu câu trả lời vào lịch sử, bỏ qua source
    )

    # --- SỬ DỤNG RETRIEVER MỚI ---
    reranker_retriever = create_reranker_retriever(vector_db)

    # 4. Tạo Chain Conversational
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=reranker_retriever, # Thay thế retriever cũ bằng reranker
        memory=memory,
        condense_question_prompt=condense_question_prompt, # Bước 1: Viết lại câu hỏi
        combine_docs_chain_kwargs={"prompt": qa_prompt},   # Bước 2: Trả lời
        return_source_documents=True,
        verbose=True # Bật lên để debug xem nó viết lại câu hỏi thế nào
    )
    return qa_chain

# Read Vector DB
def read_vector_db():
    # Embedding model
    embedding_model = HuggingFaceEmbeddings(model_name="bkai-foundation-models/vietnamese-bi-encoder")
    # Load vector db
    db = FAISS.load_local(vector_db_path, embedding_model, allow_dangerous_deserialization=True)
    return db

#test
if __name__ == "__main__":
    db = read_vector_db()
    llm = load_llm(model_file)

    llm_chain = create_qa_chain(llm, db)

    def returnAnswer(question):
        result = llm_chain.invoke({"question": question})
        return result
    question = "mục điểm học tập trong khung đánh giá kết quả rèn luyện có tối đa bao nhiêu điểm?"
    print(f"Test câu hỏi: {question}")
    print(returnAnswer(question)['answer'])