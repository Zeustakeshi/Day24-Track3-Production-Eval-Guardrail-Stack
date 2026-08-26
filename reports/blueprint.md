# Báo Cáo CI/CD Blueprint: RAG Evaluation và Guardrail Stack

**Sinh viên:** Phạm Minh Hiếu - 2A202601562  
**Ngày thực hiện:** 2026-08-27  
**Bài lab:** Lab 24 - Production Eval + Guardrail Stack

---

## 1. Mục Tiêu

Báo cáo này trình bày thiết kế CI/CD cho hệ thống RAG có tích hợp đánh giá chất lượng và guardrail ở mức production. Mục tiêu chính là kiểm soát ba nhóm rủi ro: chất lượng câu trả lời của RAG, độ tin cậy của LLM-as-Judge, và khả năng chặn các đầu vào/đầu ra không an toàn trước khi triển khai.

---

## 2. Kiến Trúc Guardrail Stack

```text
User Input
    |
    v  P95 = 16.71 ms
[Presidio PII Scan]
    |  Chặn nếu phát hiện VN_CCCD, VN_PHONE hoặc EMAIL
    |  Hành động: từ chối yêu cầu và ghi log
    v  P95 = 2314.13 ms
[LLM Input Rail]
    |  Chặn off-topic, jailbreak, prompt injection, yêu cầu PII
    |  Hành động: trả lời từ chối và ghi nhận lý do
    v
[RAG Pipeline]
    |  M1 Chunking -> M2 Search -> M3 Rerank -> LLM Answer
    v
[LLM Output Rail]
    |  Kiểm tra PII, dữ liệu nhạy cảm hoặc nội dung ngoài phạm vi
    |  Hành động: thay bằng câu trả lời an toàn nếu vi phạm
    v
User Response
```

Thiết kế trên đặt Presidio ở lớp đầu tiên vì đây là bước kiểm tra cục bộ, phù hợp để phát hiện thông tin định danh cá nhân trước khi dữ liệu được gửi sang các thành phần LLM. Sau đó, LLM Input Rail đóng vai trò phân loại ý định người dùng và chặn các mẫu tấn công phức tạp hơn như prompt injection hoặc jailbreak. LLM Output Rail được đặt ở cuối pipeline để giảm rủi ro phản hồi cuối cùng chứa thông tin nhạy cảm.

---

## 3. Ngân Sách Độ Trễ

| Thành phần | P50 (ms) | P95 (ms) | P99 (ms) | Ngân sách |
|---|---:|---:|---:|---:|
| Presidio PII Scan | 9.59 | 16.71 | 16.71 | < 10 |
| LLM Input Rail | 0.01 | 2314.13 | 2314.13 | < 300 |
| RAG Pipeline | chưa đo | chưa đo | chưa đo | < 2000 |
| LLM Output Rail | chưa đo | chưa đo | chưa đo | < 300 |
| **Tổng Guard Stack** | **14.82** | **2323.72** | **2323.72** | **< 500** |

**Kết luận về latency:** Chưa đạt ngân sách P95. Tổng P95 của guard stack là 2323.72 ms, cao hơn ngưỡng mục tiêu 500 ms. Nguyên nhân chính nằm ở lớp LLM Input Rail do phải gọi mô hình LLM qua API. Nếu triển khai production, cần bổ sung timeout, cache kết quả phân loại cho các mẫu lặp lại, batching nếu phù hợp, hoặc thay thế bằng mô hình guard chuyên dụng có độ trễ thấp hơn.

---

## 4. CI/CD Quality Gates

Các gate dưới đây được đề xuất để chạy trước khi merge vào nhánh chính:

```yaml
- name: RAGAS Quality Gate
  run: python src/phase_a_ragas.py
  env:
    MIN_FAITHFULNESS: 0.75
    MIN_AVG_SCORE: 0.65

- name: Guardrail Adversarial Gate
  run: pytest tests/test_phase_c.py -k "test_adversarial_suite_pass_rate"
  # Điều kiện: adversarial suite pass >= 15/20
  # Kết quả hiện tại: 20/20

- name: Guardrail Latency Gate
  run: python -c "from src.phase_c_guard import measure_p95_latency; ..."
  # Điều kiện: P95 total guard latency < 500 ms
  # Kết quả hiện tại: 2323.72 ms
```

Đối với lần chạy hiện tại, gate về chất lượng guardrail đã đạt yêu cầu vì adversarial pass rate là 20/20. Tuy nhiên, gate về latency chưa đạt và cần được xem là một vấn đề cần tối ưu trước khi triển khai thực tế.

---

## 5. Monitoring Khi Triển Khai

| Chỉ số theo dõi | Ngưỡng cảnh báo | Hành động đề xuất |
|---|---|---|
| RAGAS faithfulness trên mẫu hằng ngày | < 0.70 | Kiểm tra prompt, context retrieval và các câu trả lời hallucination |
| RAGAS average score | < 0.65 | Dừng release, phân tích bottom-10 và failure clusters |
| Adversarial pass rate | < 90% | Bổ sung mẫu tấn công mới và cập nhật guardrail |
| Guardrail P95 latency | > 500 ms | Profile lớp LLM guard, thêm timeout/cache hoặc đổi mô hình |
| Số lượng PII bị phát hiện | tăng đột biến > 10/giờ | Kiểm tra nguồn request và thông báo nhóm bảo mật |

---

## 6. Kết Quả Thực Nghiệm

| Hạng mục | Kết quả |
|---|---:|
| RAGAS average score trên 50 câu | 0.7738 |
| Metric yếu nhất | faithfulness |
| Distribution có nhiều failure nhất | factual |
| Cohen's kappa của LLM Judge | 0.1379 |
| Adversarial suite pass rate | 20/20 |
| Guardrail P95 latency | 2323.72 ms |

---

## 7. Nhận Xét

Hệ thống RAG đạt mức chất lượng tổng thể tương đối tốt với average score 0.7738, nhưng metric faithfulness vẫn là điểm yếu cần ưu tiên cải thiện. LLM Judge có position bias thấp trong lần chạy này, tuy nhiên Cohen's kappa chỉ đạt 0.1379, thể hiện mức đồng thuận thấp với nhãn của con người. Guardrail stack chặn thành công toàn bộ 20 mẫu adversarial, cho thấy hiệu quả tốt về mặt an toàn. Điểm hạn chế lớn nhất là độ trễ của lớp LLM Input Rail, vượt đáng kể ngân sách production và cần được tối ưu trước khi đưa vào môi trường thật.
