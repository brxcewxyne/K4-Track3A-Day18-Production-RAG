# Individual Reflection — Lab 18: Production RAG

**Họ và tên:** DinhTuanLong  
**Khóa:** K4 - Track 3A  
**Ngày hoàn thành:** 04/10/2026

---

## Phần 1: Mapping bài giảng (Lecture Mapping)

| Lecture Concept | Module | Hàm cụ thể | Observation & Phân tích |
|----------------|--------|-------------|--------------------------|
| Semantic chunking | M1 | `chunk_semantic()` | Threshold mặc định 0.85 từ `config.SEMANTIC_THRESHOLD`; cắt câu bằng regex `(?<=[.!?])\s+|\n\n`, encode `all-MiniLM-L6-v2`, gộp khi cosine ≥ threshold. Model chỉ load 1 lần nhờ cache module-level (`_SEMANTIC_MODEL`). |
| Hierarchical parent-child | M1 | `chunk_hierarchical()` | Parent 2048 / child 256 ký tự; production tạo **26 parents / 104 children** từ 26 tài liệu, ID ổn định `parent_0…`. Child mang cả `parent_id` attr lẫn metadata. Quan sát từ failure analysis: pipeline chỉ dùng child text, chưa resolve parent → mất các ý nằm ở child anh em (case #2, #4, #5). |
| Structure-aware chunking | M1 | `chunk_structure_aware()` | Metadata `section` lấy từ heading `#{1,3}`; giữ nguyên list/table theo section. Pipeline production hiện dùng hierarchical nên strategy này mới chỉ pass test, chưa dùng trong eval. |
| Vietnamese segmentation + BM25 | M2 | `segment_vietnamese()`, `BM25Search` | `underthesea.word_tokenize(format="text")` nối từ ghép bằng `_` (`nghỉ_phép`); phải `.replace("_", " ")` nếu không query 2 token sẽ không khớp index 1 token. BM25 chỉ trả doc có score > 0. |
| BM25 + Dense fusion (RRF) | M2 | `reciprocal_rank_fusion()` | `score(d) = Σ 1/(60 + rank + 1)`, không trộn raw score BM25/dense; key theo `result.text`. Production index 104 chunks (BM25 + bge-m3 vào Qdrant) mất 72.3s. |
| Dense retrieval | M2 | `DenseSearch.index()/search()` | bge-m3 1024 chiều, cosine, `recreate_collection` + `upsert`; search dùng `query_points()` (không dùng `search()` deprecated), payload giữ nguyên metadata + text. |
| Cross-encoder reranking | M3 | `CrossEncoderReranker.rerank()` | `sentence_transformers.CrossEncoder("BAAI/bge-reranker-v2-m3")`, batch `predict` cho toàn bộ cặp `(query, doc)`, sort giảm dần, top-3, rank zero-based. Đo thật trên CPU: **~539ms cho 5 docs** — cao hơn mục tiêu <150ms của lab (không fake số liệu). |
| RAGAS 4 metrics | M4 | `evaluate_ragas()` | RAGAS **0.1.22** (đúng pin `ragas>=0.1.10,<0.2`): dataset cần `ground_truth` số ít và `contexts` dạng `Sequence[string]`; aggregate lấy mean bỏ qua NaN, per-question ép về `float` Python. Kết quả production thật: F 0.8167 / AR 0.7775 / CP 0.9375 / CR 0.8583. |
| Diagnostic failure analysis | M4 | `failure_analysis()` | Mean 4 metric → `worst_metric` → diagnostic tree (faithfulness/context_recall/context_precision/answer_relevancy). Bottom-5 thực tế: lỗi generation dù context đúng (#1, #3) và lỗi thiếu evidence đa bước (#2, #4, #5). |
| Contextual embeddings / enrichment | M5 | `_enrich_single_call()`, `contextual_prepend()` | 1 API call/chunk trả `summary + questions + context + metadata`; production enrich **104/104 chunks trong 326.5s, 0 fallback**. `enriched_text = context + "\n\n" + raw text`, metadata gốc (`source`, `parent_id`) được merge giữ nguyên. |

---

## Phần 2: Khó khăn & Cách giải quyết (Challenges & Debugging)

- **Môi trường Python bị nhiễm từ lab cũ:**
  - *Lỗi:* ban đầu `.venv` resolve nhầm package của lab trước, import sai phiên bản thư viện.
  - *Cách xử lý:* xóa và tạo lại `.venv` riêng cho Day 18 bằng **Python 3.11**, cài `pip install -r requirements.txt`, và xác minh đúng interpreter bằng đường dẫn `sys.executable` (`.venv\Scripts\python.exe`) trước khi chạy test.
- **Windows Application Control chặn sklearn native `.pyd`:**
  - *Lỗi:* DLL/native extension của sklearn (dependency của RAGAS) bị chặn load.
  - *Cách xử lý:* cho phép thư viện trong môi trường tin cậy và xác minh lại bằng `import sklearn` + `from sklearn.metrics.pairwise import cosine_similarity` — hiện chạy tốt (sklearn 1.9.1).
- **Lỗi encoding tiếng Việt trên Windows PowerShell:**
  - *Lỗi:* `UnicodeEncodeError: 'charmap' codec can't encode character '\u1ef9' in position ...` khi in tiếng Việt ra console.
  - *Cách xử lý:* các module đã có `sys.stdout.reconfigure(encoding="utf-8")`; khi chạy script audit tạm thì ghi file UTF-8 và chạy `python -X utf8`.
- **`ModuleNotFoundError: No module named 'src'` khi chạy script audit từ thư mục tạm:**
  - *Cách xử lý:* đặt `PYTHONPATH` về repo root (hoặc chạy trong repo), không sửa code sản phẩm.
- **`load_dotenv()` báo thiếu API key (dương tính giả):**
  - *Lỗi:* script audit đặt trong `%TEMP%` gọi `load_dotenv()` nên tìm `.env` sai thư mục → `OPENAI_API_KEY present: False`.
  - *Cách xử lý:* kiểm tra qua `import config; bool(config.OPENAI_API_KEY)` (chỉ in boolean, **không in key**); `main.py` load đúng qua `config.py`.
- **Khác biệt API RAGAS 0.1.x so với tài liệu mới:**
  - *Cách xử lý:* đọc trực tiếp `ragas/validation.py` trong `.venv` để xác nhận schema (`ground_truth` số ít, `contexts` là `Sequence[string]`) thay vì copy ví dụ từ docs mới.
- **Cross-Encoder chậm trên CPU:**
  - *Đo thật:* benchmark `n_runs=5` trên 5 docs cho avg ~539ms (min 530ms, max 550ms), vượt mục tiêu <150ms của lab.
  - *Cách xử lý:* ghi nhận trung thực, không chỉnh số; giữ `FlashrankReranker` làm phương án nhẹ nếu cần tối ưu sau này.
- **Từ ghép tiếng Việt không khớp BM25:**
  - *Lỗi:* `word_tokenize` trả `nghỉ_phép` (1 token) trong khi query thường tách 2 token.
  - *Cách xử lý:* chuẩn hóa `.replace("_", " ")` cho cả index lẫn query.

---

## Phần 3: Action Plan cho Project cá nhân (Application Plan)

Dựa trên những kỹ thuật đã học và thực hành, lập kế hoạch cụ thể áp dụng vào project của bạn:

### Project: Trợ lý RAG tra cứu chính sách nội bộ doanh nghiệp (HR/IT)

#### 1. Hiện trạng
- **Pipeline hiện tại:** Basic RAG (paragraph chunking + dense-only top-3, chưa hybrid, chưa rerank, chưa enrichment).
- **Vấn đề / Bottlenecks đang gặp:** Tài liệu chính sách có nhiều phiên bản (v1.0/v2.0, 2023/2024) dễ trích nhầm bản hết hiệu lực; câu hỏi đa bước (số liệu + điều kiện) thiếu evidence; câu trả lời đôi khi từ chối oan dù context có đáp án.

#### 2. Kế hoạch cải tiến
1. **Chunking strategy:** Hierarchical parent-child (2048/256) làm mặc định kết hợp structure-aware cho bảng biểu; bổ sung **parent resolution sau rerank** — đây là bài học trực tiếp từ failure #2/#4/#5.
2. **Search retrieval:** Hybrid BM25 (underthesea, `_`→space) + Dense bge-m3 + RRF k=60; thêm metadata filter theo `version`/`effective_date` để ưu tiên bản mới (bài học từ failure #1).
3. **Reranking:** Giữ bge-reranker-v2-m3 khi có GPU; nếu chỉ CPU thì dùng FlashRank/ONNX vì đo thực tế ~539ms cho 5 docs; thêm ngưỡng score tối thiểu để loại context nhiễu (<0.1).
4. **Evaluation:** RAGAS 4 metrics (đúng bản 0.1.22) trên bộ 20 câu có cả câu đa bước và câu xung đột phiên bản; theo dõi riêng Faithfulness và Context Recall.
5. **Enrichment:** Combined mode 1 call/chunk (`_enrich_single_call`) để tiết kiệm chi phí (đo thực tế: 104 chunks/326.5s, 0 fallback); thêm field `version`/`effective_date` vào metadata.

#### 3. Timeline triển khai
- **Tuần 1:** Dựng chunking phân cấp + hybrid search + metadata phiên bản; xây bộ test 20 câu theo dạng (tra cứu, số liệu, đa bước, xung đột phiên bản).
- **Tuần 2:** Nối parent resolution vào pipeline, bật reranker + ngưỡng score, chạy RAGAS, tinh chỉnh prompt synthesis (tránh "Không tìm thấy" oan) và viết failure analysis định kỳ.
