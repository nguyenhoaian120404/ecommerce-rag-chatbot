"""
RAG Chatbot — Pháp luật TMĐT & Quy định Shopee / TikTok Shop
Streamlit app | BGE-M3 Fine-tuned | FAISS | Gemini Flash
"""
import logging
import streamlit as st
import markdown as md
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import vertexai
from vertexai.generative_models import GenerativeModel
import google.generativeai as genai
# =============================================
# LOGGING — in ra terminal nơi chạy `streamlit run`
# =============================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
    force=True,
)
log = logging.getLogger("rag")
# =============================================
# CONFIG
# =============================================
MODEL_PATH = f"hoaian12/bge-m3-legal-vn"
FAISS_INDEX = f"step6_faiss_final/index.faiss"
FAISS_META  = f"step6_faiss_final/chunks_meta.pkl"
GCP_PROJECT  = "project-9fc99aa7-4b1f-4118-9ee"
GCP_LOCATION = "us-central1"
MAX_SEQ_LEN  = 4096
TOP_K        = 10
MAX_CHARS_PER_CHUNK = 1500   # tăng lên để không mất thông tin
MAX_TOTAL_CHARS     = 12000  # giới hạn tổng context

# =============================================
# LOAD RESOURCES (cache)
# =============================================
@st.cache_resource(show_spinner="⏳ Đang tải model và index...")
def load_resources():
    model = SentenceTransformer(MODEL_PATH)
    model.max_seq_length = MAX_SEQ_LEN
    index = faiss.read_index(FAISS_INDEX)
    with open(FAISS_META, "rb") as f:
        meta = pickle.load(f)
    # vertexai.init(project=GCP_PROJECT, location=GCP_LOCATION) # mơi chỉnh cho deploy
    
    # gemini = GenerativeModel("gemini-2.5-flash")
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    gemini = genai.GenerativeModel("gemini-2.5-flash")
    return model, index, meta, gemini


# =============================================
# RETRIEVE — balanced + dedup, không có domain filter
# =============================================
def retrieve(query, model, index, meta, top_k=TOP_K):
    q_emb = model.encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    )
    scores, indices = index.search(q_emb.astype(np.float32), top_k * 4)

    # Dedup theo doc_id + section
    seen, all_hits = set(), []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk = meta[idx].copy()
        chunk["score"] = float(score)
        key = (chunk["doc_id"],
               chunk.get("section_h2") or str(chunk.get("dieu_so")))
        if key in seen:
            continue
        seen.add(key)
        all_hits.append(chunk)

    # Balanced: ít nhất 3 legal + 2 shopee + 1 tiktok, còn lại top sim
    legal  = [c for c in all_hits if c["doc_type"] == "legal"]
    shopee = [c for c in all_hits if c.get("platform") == "shopee"]
    tiktok = [c for c in all_hits if c.get("platform") == "tiktok"]

    result = legal[:3] + shopee[:2] + tiktok[:1]
    added  = {c["chunk_id"] for c in result}
    for c in all_hits:
        if len(result) >= top_k:
            break
        if c["chunk_id"] not in added:
            result.append(c)
            added.add(c["chunk_id"])

    result.sort(key=lambda x: -x["score"])
    return result[:top_k]


# =============================================
# BUILD CITATION — trích dẫn chi tiết cho Gemini
# =============================================
def build_citation_label(c: dict) -> str:
    """
    Tạo nhãn trích dẫn đầy đủ để Gemini dùng trong câu trả lời.
    Legal:    "Luật Thuế TNCN — Chương III, Điều 7. Thuế suất"
    Platform: "Shopee — Quản lý Shop > Thiết lập Shop — Câu hỏi thường gặp"
    """
    if c["doc_type"] == "legal":
        label = c["doc_name"]
        if c.get("chuong"):
            label += f" — {c['chuong']}"
        if c.get("dieu_so"):
            label += f", Điều {c['dieu_so']}"
            if c.get("dieu_ten"):
                label += f". {c['dieu_ten']}"
    else:
        platform = (c.get("platform") or "platform").capitalize()
        label    = platform

        if c.get("category_main"):
            label += f" — {c['category_main']}"
            if (c.get("category_sub")
                    and c["category_sub"] != c["category_main"]):
                label += f" > {c['category_sub']}"

        if c.get("doc_name"):
            label += f" — {c['doc_name']}"

        if c.get("section_h2"):
            label += f" | {c['section_h2']}"

        if c.get("section_h3"):
            label += f" > {c['section_h3']}"

    return label


# =============================================
# FORMAT CONTEXT — dùng citation label chi tiết
# =============================================
def format_context(chunks,
                   max_per_chunk=MAX_CHARS_PER_CHUNK,
                   max_total=MAX_TOTAL_CHARS):
    parts = []
    total = 0

    for i, c in enumerate(chunks, 1):
        text = c["text"]
        if len(text) > max_per_chunk:
            text = text[:max_per_chunk] + "...[đã cắt bớt]"

        if total + len(text) > max_total:
            break
        total += len(text)

        citation = build_citation_label(c)
        # parts.append(f"[{i}] {citation}\n{text}")
        parts.append(f"Nguồn: {citation}\n{text}")

    return "\n\n---\n\n".join(parts)


# =============================================
# GENERATE
# =============================================
# SYSTEM_PROMPT = """Bạn là chuyên gia tư vấn pháp lý về thuế và thương mại điện tử tại Việt Nam.

# Dựa vào các đoạn văn bản pháp luật và quy định sàn TMĐT được cung cấp, hãy trả lời câu hỏi chính xác, đầy đủ và dễ hiểu.

# Yêu cầu:
# - Chỉ trả lời dựa trên thông tin trong context
# - Khi trích dẫn nguồn, dùng ĐÚNG nhãn trong dấu ngoặc vuông [i], ví dụ:
#   [Shopee — Tài chính > Phí bán hàng — Phí dịch vụ là gì?, 1]
#   [Luật Thuế TNCN — Chương III, Điều 7. Thuế suất, 2]
# - Nếu không đủ thông tin → nói rõ "Tôi không tìm thấy thông tin về vấn đề này"
# - Trả lời bằng tiếng Việt, rõ ràng, có cấu trúc"""
SYSTEM_PROMPT = """Bạn là chuyên gia tư vấn pháp lý về thuế và thương mại điện tử tại Việt Nam.

Dựa vào các đoạn văn bản pháp luật và quy định sàn TMĐT được cung cấp, hãy trả lời câu hỏi chính xác, đầy đủ và dễ hiểu.

Yêu cầu:
- Chỉ trả lời dựa trên thông tin trong context
- Khi trích dẫn nguồn, PHẢI viết đầy đủ theo format:
  Văn bản pháp luật: [Tên luật/nghị định, Điều X. Tên điều]
  Ví dụ: [Nghị định thuế hộ kinh doanh, Điều 5. Phương pháp tính thuế]
  
  Quy định sàn: [Tên sàn — Tên mục > Tên tiểu mục]
  Ví dụ: [Shopee — Tài chính > Phí bán hàng | Phí dịch vụ là gì?]

- KHÔNG dùng số thứ tự [1], [2]... để trích dẫn
- Nếu không đủ thông tin → nói rõ "Tôi không tìm thấy thông tin về vấn đề này"
- Trả lời bằng tiếng Việt, rõ ràng, có cấu trúc
- Trả lời ngắn gọn, súc tích nếu câu hỏi đơn giản
- Trả lời chi tiết, có cấu trúc nếu câu hỏi phức tạp
- Không liệt kê thừa, không lặp lại thông tin"""

def generate_answer(query, context, gemini):
    prompt = f"""{SYSTEM_PROMPT}

===== CÁC ĐOẠN VĂN BẢN LIÊN QUAN =====
{context}
========================================

Câu hỏi: {query}

Trả lời:"""
    resp = gemini.generate_content(
        prompt,
        generation_config={"temperature": 0.1, "max_output_tokens": 8192},
        stream=True,
    )
    return resp.text


# =============================================
# UI
# =============================================
st.set_page_config(
    page_title="RAG Chatbot — Pháp luật TMĐT",
    page_icon="⚖️",
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lexend:wght@300;400;500;600;700&family=Noto+Serif+Display:wght@400;600&display=swap');
html, body, [class*="css"] { font-family: 'Lexend', sans-serif; }

.main-header {
    background: linear-gradient(135deg, #0a0a0f 0%, #0d1f3c 60%, #1a0a2e 100%);
    padding: 2.5rem 2rem; border-radius: 16px; margin-bottom: 1.5rem;
    text-align: center; border: 1px solid rgba(255,255,255,0.08);
    position: relative; overflow: hidden;
}
.main-header::before {
    content: ''; position: absolute; inset: 0;
    background: radial-gradient(ellipse at 30% 50%, rgba(99,102,241,0.15) 0%, transparent 60%),
                radial-gradient(ellipse at 70% 50%, rgba(236,72,153,0.1) 0%, transparent 60%);
}
.main-header h1 {
    font-family: 'Noto Serif Display', serif; font-size: 1.9rem; font-weight: 600;
    color: #f1f5f9; margin: 0; position: relative; z-index: 1;
}
.main-header p {
    font-size: 0.85rem; color: rgba(241,245,249,0.6);
    margin: 0.5rem 0 0; position: relative; z-index: 1;
}

.chat-user {
    background: linear-gradient(135deg, #1e3a5f, #1a2744);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 12px 12px 4px 12px;
    padding: 0.9rem 1.1rem; margin: 0.6rem 0; color: #e2e8f0;
}
.chat-bot {
    background: #f8fafc;
    color: #1e293b;
    border-left: 3px solid #6366f1;
}
            

.badge {
    display: inline-block; font-size: 0.7rem; font-weight: 600;
    padding: 2px 10px; border-radius: 20px; margin: 2px;
}
.badge-legal  { background: rgba(99,102,241,0.2); color: #a5b4fc; border: 1px solid rgba(99,102,241,0.3); }
.badge-shopee { background: rgba(238,77,45,0.2);  color: #fca5a5; border: 1px solid rgba(238,77,45,0.3); }
.badge-tiktok { background: rgba(20,20,20,0.5);   color: #94a3b8; border: 1px solid rgba(255,255,255,0.15); }
.src-body {
    margin-top: 8px; padding: 12px 14px;
    background: #f1f5f9;
    border-left: 3px solid #6366f1;
    border-radius: 6px;
    font-size: 0.82rem; line-height: 1.6;
    color: #1e293b;
    max-height: 360px; overflow-y: auto;
}
.src-body p { margin: 0.4em 0; }
.src-body h4 { color: #374151; margin: 0.6em 0 0.3em; font-weight: 600; }
.src-body ul, .src-body ol { margin: 0.3em 0 0.3em 1.2em; }
.src-body li { margin: 0.2em 0; }
.src-body strong { color: #111827; }

.src-row { padding: 4px 0; border-bottom: 1px solid rgba(255,255,255,0.04); }
.src-path { font-size: 0.75rem; color: #64748b; margin-top: 2px; }
.score-pill {
    font-size: 0.65rem; color: #64748b;
    background: rgba(255,255,255,0.04);
    padding: 1px 6px; border-radius: 8px; margin-left: 6px;
}

.stButton > button {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    color: white; border: none; border-radius: 8px; font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
  <h1>⚖️ Tư vấn Thuế, Pháp luật TMĐT & Quy định Sàn</h1>
  <p>Luật Thương mại điện tử · Thuế · Shopee · TikTok Shop · BGE-M3 Fine-tuned</p>
</div>
""", unsafe_allow_html=True)

# Load
try:
    model, index, meta, gemini = load_resources()
except Exception as e:
    st.error(f"❌ Lỗi: {e}")
    st.stop()

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Cài đặt")
    top_k     = st.slider("Số chunks tham chiếu", 3, 15, TOP_K)
    max_chars = st.slider("Ký tự tối đa/chunk", 500, 2000, MAX_CHARS_PER_CHUNK, step=100)

    st.markdown("---")
    st.markdown("### 📚 Nguồn dữ liệu")
    st.markdown("""
**Pháp luật:**
- Luật Thương mại điện tử 122/2025
- Luật Quản lý thuế 108/2025
- Luật Thuế TNCN 109/2025
- NĐ 68/2026, NĐ 117/2025
- TT 18/2026 + văn bản hợp nhất

**Shopee :** 40 tài liệu

**TikTok Shop:** 17 tài liệu
    """)

    st.markdown("---")
    st.markdown("### 💡 Câu hỏi gợi ý")
    suggestions = [
        "Hộ kinh doanh bán Shopee nộp thuế không?",
        "Mức thuế GTGT hàng hóa bán online?",
        "Livestream bán hàng cần tuân thủ gì?",
        "Shopee thu phí dịch vụ như thế nào?",
        "TikTok Shop rút tiền về ngân hàng thế nào?",
        "Nền tảng TMĐT lưu dữ liệu bao lâu?",
    ]
    for s in suggestions:
        if st.button(s, key=f"s_{s[:12]}", use_container_width=True):
            st.session_state["pending_query"] = s
            st.rerun()

# Session state
if "messages"      not in st.session_state: st.session_state.messages = []
if "pending_query" not in st.session_state: st.session_state.pending_query = ""

# Lịch sử chat
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(
            f'<div class="chat-user">🧑 <b>Bạn:</b> {msg["content"]}</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="chat-bot">🤖 <b>Trợ lý:</b><br>{msg["content"]}</div>',
            unsafe_allow_html=True
        )
        if msg.get("sources"):
            with st.expander(f"📎 {len(msg['sources'])} nguồn tham chiếu"):
                for i, src in enumerate(msg["sources"], 1):
                    # ── Fix badge: luôn check doc_type trước ──
                    if src["doc_type"] == "legal":
                        bcls, blabel = "badge-legal", "⚖️ Pháp luật"
                    elif src.get("platform") == "shopee":
                        bcls, blabel = "badge-shopee", "🛒 Shopee"
                    elif src.get("platform") == "tiktok":
                        bcls, blabel = "badge-tiktok", "🎵 TikTok Shop"
                    else:
                        bcls, blabel = "badge-legal", "⚖️ Pháp luật"

                    # Citation path đầy đủ
                    citation = build_citation_label(src)

                    st.markdown(
                        f'<div class="src-row">'
                        f'<span class="badge {bcls}">{blabel}</span>'
                        f'<span class="score-pill">[{i}] sim {src["score"]:.3f}</span>'
                        f'<div class="src-path">{citation}</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )
                    # THÊM NGAY SAU ĐÓ:
                    text = src.get("text", "").strip()
                    if text:
                        html_body = md.markdown(
                            text,
                            extensions=["extra", "sane_lists", "nl2br"],
                        )
                        st.markdown(
                            f'<div class="src-body">{html_body}</div>',
                            unsafe_allow_html=True,
                        )

# # ── Input form — không auto-submit khi gõ ──
# with st.form("chat_form", clear_on_submit=True):
#     col1, col2 = st.columns([6, 1])
#     with col1:
#         user_input = st.text_input(
#             "q",
#             value=st.session_state.pending_query,
#             placeholder="Nhập câu hỏi và nhấn Enter hoặc Gửi...",
#             label_visibility="collapsed",
#         )
#     with col2:
#         submitted = st.form_submit_button("Gửi ➤", use_container_width=True)

# if st.session_state.pending_query:
#     st.session_state.pending_query = ""

# # Xử lý
# if submitted and user_input.strip():
#     query = user_input.strip()
#     st.session_state.messages.append({"role": "user", "content": query})

#     with st.spinner("🔍 Đang tìm kiếm và tổng hợp..."):
#         try:
#             chunks  = retrieve(query, model, index, meta, top_k=top_k)
#             context = format_context(chunks, max_per_chunk=max_chars)
#             answer  = generate_answer(query, context, gemini)
#             st.session_state.messages.append({
#                 "role": "assistant", "content": answer, "sources": chunks
#             })
#         except Exception as e:
#             st.session_state.messages.append({
#                 "role": "assistant", "content": f"❌ Lỗi: {e}", "sources": []
#             })
#     st.rerun()
# ── Lấy query đang chờ từ suggestion (nếu có), pop ra luôn để không bị reset/ghi đè ──
pending = st.session_state.pop("pending_query", "") or ""

# ── Input form — không auto-submit khi gõ ──
with st.form("chat_form", clear_on_submit=True):
    col1, col2 = st.columns([6, 1])
    with col1:
        user_input = st.text_input(
            "q",
            placeholder="Nhập câu hỏi và nhấn Enter hoặc Gửi...",
            label_visibility="collapsed",
        )
    with col2:
        submitted = st.form_submit_button("Gửi ➤", use_container_width=True)

# Ưu tiên: suggestion click → auto-submit luôn. Nếu không, dùng input người dùng gõ.
query = None
if pending.strip():
    query = pending.strip()
elif submitted and user_input.strip():
    query = user_input.strip()

# Xử lý
if query:
    st.session_state.messages.append({"role": "user", "content": query})

    # Render lại ngay câu hỏi của user trước khi stream
    st.markdown(
        f'<div class="chat-user">🧑 <b>Bạn:</b> {query}</div>',
        unsafe_allow_html=True
    )

    # Status panel — luôn hiển thị câu đang xử lý + bước hiện tại
    status_box = st.status(f"🤔 Đang xử lý: **{query}**", expanded=True)
    t0 = time.time()
    log.info(f"========== NEW QUERY ==========")
    log.info(f"Q: {query}")

    try:
        # ---- 1) RETRIEVAL ----
        with status_box:
            st.write("🔍 **Bước 1/3:** Mã hoá câu hỏi + tìm trong FAISS...")
        t1 = time.time()
        chunks  = retrieve(query, model, index, meta, top_k=top_k)
        context = format_context(chunks, max_per_chunk=max_chars)
        retr_ms = (time.time() - t1) * 1000
        log.info(f"⏱  Retrieval: {retr_ms:.0f}ms | {len(chunks)} chunks | context={len(context)} chars")
        with status_box:
            st.write(f"✅ Lấy {len(chunks)} đoạn ({retr_ms:.0f}ms, {len(context)} ký tự context)")

        # ---- 2) GỌI OPENAI (chờ token đầu tiên) ----
        with status_box:
            st.write(f"🧠 **Bước 2/3:** Gọi AI Gemini 2.5 Flash... (chờ token đầu tiên)")
        log.info(f"→ Calling OpenAI {OPENAI_MODEL}...")

        placeholder = st.empty()
        answer_parts = []
        first_token_time = None
        t2 = time.time()

        for token in generate_answer_stream(query, context, client):
            if first_token_time is None:
                first_token_time = time.time()
                ttfb = (first_token_time - t2) * 1000
                log.info(f"⏱  Time-to-first-token: {ttfb:.0f}ms")
                with status_box:
                    st.write(f"✍️  **Bước 3/3:** Đang sinh câu trả lời... (TTFT {ttfb:.0f}ms)")
            answer_parts.append(token)
            placeholder.markdown(
                f'<div class="chat-bot">🤖 <b>Trợ lý:</b><br>{"".join(answer_parts)}▌</div>',
                unsafe_allow_html=True
            )

        answer = "".join(answer_parts)
        gen_ms = (time.time() - t2) * 1000
        total_ms = (time.time() - t0) * 1000
        log.info(f"⏱  Generation: {gen_ms:.0f}ms | answer={len(answer)} chars")
        log.info(f"⏱  TOTAL: {total_ms:.0f}ms")

        placeholder.empty()
        status_box.update(
            label=f"✅ Hoàn tất ({total_ms/1000:.1f}s) — {query}",
            state="complete",
            expanded=False,
        )

        st.session_state.messages.append({
            "role": "assistant", "content": answer, "sources": chunks
        })
    except Exception as e:
        log.exception(f"❌ Error: {e}")
        status_box.update(label=f"❌ Lỗi: {e}", state="error", expanded=True)
        st.session_state.messages.append({
            "role": "assistant", "content": f"❌ Lỗi: {e}", "sources": []
        })
    st.rerun()

# Clear
if st.session_state.messages:
    if st.button("🗑️ Xóa lịch sử", type="secondary"):
        st.session_state.messages = []
        st.rerun()

st.markdown("""
<div style="text-align:center;color:#334155;font-size:0.72rem;
            margin-top:2rem;padding-top:1rem;border-top:1px solid #1e293b">
  BGE-M3 Fine-tuned · FAISS IndexFlatIP · Gemini 2.5 Flash · Khoá luận tốt nghiệp NEU
</div>
""", unsafe_allow_html=True)
