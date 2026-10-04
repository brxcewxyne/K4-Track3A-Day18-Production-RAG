# Failure Analysis — Lab 18: Production RAG

**Họ và tên học viên:** DinhTuanLong  
**Khóa:** K4 - Track 3A  
**Nguồn số liệu:** `reports/ragas_report.json` + `reports/naive_baseline_report.json` (lần chạy `python main.py`, tổng 890.8s).  
**Phương pháp audit Bottom-5:** nạp lại 104 enriched chunks từ Qdrant `lab18_production`, dựng lại BM25 + Dense + RRF + Cross-Encoder (retrieval tất định nên context trùng khớp), sinh lại câu trả lời bằng đúng prompt của `pipeline.py`, và chấm lại 4 metric cho 5 câu.

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.8238 | 0.8167 | -0.0071 |
| Answer Relevancy | 0.8040 | 0.7775 | -0.0265 |
| Context Precision | 0.9250 | 0.9375 | +0.0125 |
| Context Recall | 0.9250 | 0.8583 | -0.0667 |

**Nhận xét:** Production đạt ngưỡng 0.75+ ở cả 4 metric (rubric: ≥3 metric ≥0.70 → 10 điểm; cả 4 metric ≥0.75 → +3 bonus). Tuy nhiên chưa vượt baseline: baseline là dense-only top-3 trên 57 paragraph chunks nên "dễ thở" với bộ test nhỏ này, còn production dùng child chunks 256 ký tự + context prepend. Điểm giảm chủ yếu ở Context Recall (0.9250 → 0.8583), đúng với phân tích bên dưới: nhiều câu hỏi đa bước/đa điều kiện cần 2 mảnh bằng chứng nằm ở các child khác nhau.

## Bottom-5 Failures

### #1
- **Question:** Bao lâu phải đổi mật khẩu một lần?
- **Expected:** Theo chính sách hiện hành (v2.0), mật khẩu phải được thay đổi mỗi **120 ngày**. Chính sách cũ yêu cầu 90 ngày nhưng **đã bị thay thế**.
- **Got:** `"Không tìm thấy."` (từ chối trả lời dù bằng chứng đã được retrieve)
- **Worst metric:** faithfulness (report: score trung bình 0.4583; audit: faith 0.0, relevancy 0.0, precision 0.83, recall 1.00)
- **Contexts (rerank):** `mat_khau_v2.md` 0.9695 (chu kỳ 120 ngày) → `mat_khau_v1.md` 0.9640 (chu kỳ 90 ngày, hết hiệu lực) → `mat_khau_v2.md` 0.0941 ("thay thế Chính sách mật khẩu v1.0")
- **Error Tree:** Output sai (từ chối) → Context đúng? **CÓ** (recall = 1.0, v2 nằm ngay top-1) → Query OK? **CÓ** → Lỗi ở bước **generation + xử lý xung đột phiên bản**.
- **Root cause:** Xung đột phiên bản kinh điển: chunk v1 (90 ngày) và v2 (120 ngày) có rerank score gần bằng nhau (0.9695 vs 0.9640) và cả hai đều được đưa vào context. Pipeline không có metadata `version`/`effective_date` để lọc/ưu tiên bản mới, LLM nhận 2 số liệu mâu thuẫn nên chọn phương án an toàn "Không tìm thấy." Hệ quả: faithfulness = 0 vì câu khẳng định từ chối không được context support.
- **Suggested fix:** (1) M5 bổ sung `version`/`effective_date` vào metadata và (2)/hoặc M2 lọc/tăng trọng số bản mới trước RRF; (3) sửa prompt synthesis trong `pipeline.py`: yêu cầu ưu tiên tài liệu có ngày hiệu lực mới nhất và nói rõ khi tài liệu cũ đã bị thay thế.

### #2
- **Question:** Một nhân viên Senior có 9 năm thâm niên được nghỉ bao nhiêu ngày phép năm và lương trong khoảng nào?
- **Expected:** Theo chính sách v2024: 15 ngày cơ bản + 3 ngày thâm niên (9÷3=3) = **18 ngày phép**. Lương Senior (P3-P4): **20-35 triệu VNĐ/tháng**.
- **Got:** `"Nhân viên 9 năm thâm niên được 18 ngày phép năm (15 + 3). Mỗi nhân viên chính thức được hưởng 12 ngày phép năm có lương. Do đó, nhân viên này có 18 ngày phép, trong đó 12 ngày là có lương."` — đúng phần ngày phép nhưng **thiếu hoàn toàn phần lương** và thêm chi tiết "12 ngày có lương" gây nhiễu.
- **Worst metric:** context_recall (report: 0.6444; audit: faith 1.0, relevancy 0.85, precision 1.0, recall 0.0)
- **Contexts (rerank):** `nghi_phep_nam_v2024.md` 0.9809 → `nghi_phep_nam_v2023.md` 0.6611 → `nghi_phep_nam_v2023.md` 0.4028. **Không có chunk lương nào** (`bang_luong_2024.md` không lọt top-3 sau rerank).
- **Error Tree:** Output thiếu ý → Context đúng? **THIẾU** (recall = 0 cho vế lương) → Query OK? **CÓ về mặt ngữ nghĩa nhưng là câu hỏi đa bước (multi-hop)** → Lỗi ở **retrieval/reranking cho multi-hop**.
- **Root cause:** Câu hỏi cần 2 tài liệu (nghỉ phép + bảng lương). Hybrid top-20 có thể chứa chunk lương nhưng Cross-Encoder chấm từng child chunk 256 ký tự riêng lẻ nên không "nhìn thấy" rằng query cần cả 2 nguồn; các slot 2-3 bị chiếm bởi v2023 nhiễu. Thêm nữa, pipeline không resolve `parent_id` để lấy thêm ngữ cảnh anh em cùng parent.
- **Suggested fix:** (1) Parent resolution sau rerank: retrieve child → kéo parent (2048 ký tự) làm context; (2) HyQA của M5 sinh câu hỏi kiểu "lương Senior P3-P4 bao nhiêu" để bridge vocabulary; (3) tăng diversity khi chọn top-k sau rerank.

### #3
- **Question:** Nhân viên thử việc có được hưởng bảo hiểm sức khỏe PVI không?
- **Expected:** **KHÔNG.** Nhân viên thử việc chưa được hưởng gói bảo hiểm sức khỏe PVI. Chỉ được tham gia bảo hiểm xã hội bắt buộc.
- **Got:** `"Không tìm thấy."` (từ chối dù bằng chứng nằm ngay top-1)
- **Worst metric:** answer_relevancy (report: 0.75; audit: faith 1.0, relevancy 0.0, precision 1.0, recall 1.0)
- **Contexts (rerank):** `thu_viec.md` 0.9984 ("...chưa được hưởng gói bảo hiểm sức khỏe") → `bao_hiem_suc_khoe.md` 0.7841 → `bao_hiem_suc_khoe.md` 0.5066
- **Error Tree:** Output sai (từ chối) → Context đúng? **CÓ** (recall/precision = 1.0, câu trả lời nằm trong top-1) → Query OK? **CÓ** → Lỗi ở **generation (prompt synthesis)**.
- **Root cause:** Prompt hệ thống quá bảo thủ: "Trả lời CHỈ dựa trên context. Nếu không có → nói 'Không tìm thấy.'". Child chunk ngắn + phần contextual prepend của M5 khiến mô hình không tự tin, nên từ chối thay vì trả lời "Không". answer_relevancy = 0 vì câu trả lời từ chối không giải quyết câu hỏi.
- **Suggested fix:** Viết lại prompt synthesis: bắt buộc trả lời trực tiếp nếu context có chứa câu trả lời (kể cả câu trả lời phủ định), chỉ dùng "Không tìm thấy" khi context thực sự trống; cân nhắc truyền parent chunk thay vì child chunk.

### #4
- **Question:** Nghỉ phép không lương 20 ngày cần ai phê duyệt?
- **Expected:** Nghỉ 16-30 ngày cần phê duyệt của **Giám đốc điều hành (CEO)**. Lưu ý: nghỉ trên 14 ngày không lương, nhân viên **phải tự đóng phần bảo hiểm của mình**.
- **Got:** `"Nghỉ phép không lương 20 ngày cần phê duyệt của Giám đốc điều hành (CEO)."` — đúng ý chính, **thiếu ý bảo hiểm**.
- **Worst metric:** context_recall (report: 0.7792; audit: faith 0.5, relevancy 0.78, precision 0.83, recall 0.5)
- **Contexts (rerank):** `nghi_phep_khong_luong.md` 0.9352 (bảng phân cấp phê duyệt 1-5/6-15/16-30 ngày) → `nghi_phep_nam_v2023.md` 0.6657 (nhiễu) → `nghi_phep_khong_luong.md` 0.5200 (tối đa 30 ngày/năm).
- **Error Tree:** Output thiếu 1 ý → Context đúng? **THIẾU** (recall = 0.5; chi tiết "trên 14 ngày tự đóng bảo hiểm" nằm ở child khác cùng tài liệu) → Query OK? **CÓ** → Lỗi ở **M1/pipeline: không hợp nhất các child cùng parent**.
- **Root cause:** Hierarchical chunking cắt tài liệu thành các child 256 ký tự; hai dữ kiện của cùng một câu trả lời nằm ở 2 child anh em. Pipeline hiện chỉ dùng text của child trúng top-3 mà không resolve `parent_id` về parent 2048 ký tự, nên phần bảo hiểm bị mất.
- **Suggested fix:** Sau rerank, resolve `child.parent_id` → lấy parent text (hoặc merge các child cùng parent) trước khi đưa vào LLM. Đây chính là mục đích thiết kế của M1 parent-child nhưng chưa được nối vào `pipeline.py`.

### #5
- **Question:** Nếu cần mua một chiếc laptop 30 triệu cho nhân viên mới, ai phê duyệt và cần gì từ phòng CNTT?
- **Expected:** Laptop 30 triệu nằm trong khoảng **5-50 triệu nên cần Giám đốc phòng ban (Director) phê duyệt**. Mua sắm thiết bị CNTT cần **xác nhận cấu hình kỹ thuật từ phòng CNTT**; đính kèm **≥3 báo giá** vì trên 10 triệu.
- **Got:** `"Để mua một chiếc laptop 30 triệu cho nhân viên mới, cần có xác nhận của phòng CNTT về cấu hình kỹ thuật trước khi đề xuất. Về phê duyệt, do đơn hàng trên 5.000.000 VNĐ, nên cần phê duyệt của Kế toán trưởng."` — đúng phần CNTT, **sai người phê duyệt** (Kế toán trưởng thay vì Director) và thiếu yêu cầu 3 báo giá.
- **Worst metric:** faithfulness (report: 0.7812; audit: faith 0.75, relevancy 0.90, precision 1.0, recall 0.667)
- **Contexts (rerank):** `mua_sam.md` 0.8325 (xác nhận CNTT + báo giá) → `tam_ung.md` 0.0524 (ngưỡng tạm ứng 5 triệu — **nhiễu**) → `hoan_chi_dao_tao.md` 0.0326 (nhiễu).
- **Error Tree:** Output sai số liệu → Context đúng? **THIẾU** bảng ngưỡng phê duyệt mua sắm, lại **có chunk nhiễu** `tam_ung.md` → Query OK? **CÓ** (2 điều kiện rõ ràng) → Lỗi ở **retrieval/rerank + generation**.
- **Root cause:** Bảng ngưỡng phê duyệt (5-50 triệu → Director) không nằm trong top-1 chunk `mua_sam.md` được chọn (bị cắt sang child khác); rerank score của 2 slot cuối sụp xuống ~0.05/0.03 (không có evidence liên quan) nhưng vẫn được đưa vào vì top_k=3. LLM thấy ngưỡng "5.000.000" trong chunk tạm ứng nên suy diễn nhầm sang Kế toán trưởng → faithfulness giảm.
- **Suggested fix:** Parent resolution để lấy trọn bảng ngưỡng mua sắm trong parent; ngưỡng score tối thiểu cho context đưa vào LLM (loại chunk có rerank score quá thấp); prompt synthesis yêu cầu bỏ qua context không liên quan và không suy diễn chéo tài liệu.

## Case Study (cho presentation)

**Question chọn phân tích:** #1 — "Bao lâu phải đổi mật khẩu một lần?" (xung đột phiên bản v1.0 90 ngày vs v2.0 120 ngày).

**Error Tree walkthrough:**
1. Output đúng? → **Không** ("Không tìm thấy." dù tài liệu hiện hành ghi rõ 120 ngày).
2. Context đúng? → **Có** — `mat_khau_v2.md` được rerank 0.9695 (top-1) và recall = 1.0; vấn đề là `mat_khau_v1.md` (90 ngày, hết hiệu lực) bám sát ngay sau với 0.9640.
3. Query rewrite OK? → **Có** — câu hỏi ngắn, rõ; không cần viết lại.
4. Fix ở bước: **M5/M2 metadata phiên bản + prompt synthesis** — thêm `version`/`effective_date`, ưu tiên bản mới, và dạy LLM rằng v1.0 đã bị v2.0 thay thế thay vì từ chối trả lời.

**Nếu có thêm 1 giờ, sẽ optimize:**
- Version-aware retrieval: gắn `effective_date`/`version` trong `_enrich_single_call()`, filter trước rerank (hoặc boost bản mới trong RRF).
- Parent resolution sau rerank: `parent_id` đang được lưu nhưng chưa dùng — nối vào `pipeline.py` để trả parent 2048 ký tự.
- Prompt synthesis chống từ chối quá mức + yêu cầu bỏ qua context nhiễu.
- Ngưỡng rerank score tối thiểu (ví dụ loại context < 0.1) để tránh đưa chunk nhiễu vào LLM.
