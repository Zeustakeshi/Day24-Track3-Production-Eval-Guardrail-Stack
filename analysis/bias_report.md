# Báo Cáo Phân Tích LLM-as-Judge - Phase B

**Sinh viên:** Phạm Minh Hiếu - 2A202601562  
**Ngày thực hiện:** 2026-08-27  
**Nội dung:** Đánh giá pairwise judge, swap-and-average, Cohen's kappa và bias

---

## 1. Mục Tiêu

Phase B được thực hiện nhằm đánh giá mức độ tin cậy của LLM-as-Judge khi so sánh chất lượng câu trả lời trong RAG pipeline. Ba nội dung chính được kiểm tra gồm khả năng chọn câu trả lời tốt hơn trong pairwise comparison, độ ổn định khi đổi vị trí câu trả lời bằng swap-and-average, và mức độ đồng thuận với nhãn con người thông qua Cohen's kappa.

---

## 2. Kết Quả Pairwise Judge

| # | Câu hỏi | Winner | Nhận xét |
|---:|---|---|---|
| 1 | Nhân viên được nghỉ bao nhiêu ngày phép năm? | A | Câu trả lời A được chọn trong ví dụ demo pairwise. Kết quả này nhất quán sau khi hoán đổi vị trí hai câu trả lời. |

Trong ví dụ kiểm thử, LLM Judge chọn đáp án A là câu trả lời tốt hơn. Sau khi thực hiện swap-and-average, kết quả cuối cùng vẫn là A, cho thấy judge không bị ảnh hưởng bởi vị trí trong trường hợp này.

---

## 3. Kết Quả Swap-and-Average

| # | Pass 1 Winner | Pass 2 Winner sau khi quy đổi | Final Winner | Position Consistent |
|---:|---|---|---|---|
| 1 | A | A | A | True |

**Position bias rate:** 0.0%  
**Position bias count:** 0/10

Kết quả cho thấy không phát hiện position bias trong tập 10 câu đã được đánh giá. Hai lượt đánh giá sau khi đổi vị trí câu trả lời vẫn nhất quán, do đó swap-and-average giúp tăng độ tin cậy cho kết quả cuối cùng.

---

## 4. Cohen's Kappa Với Human Labels

**Human labels:** `human_labels_10q.json`  
**Judge labels:** `[0, 0, 0, 0, 0, 0, 0, 0, 1, 0]`

| Question ID | Human Label | Judge Label | Kết quả |
|---:|---:|---:|---|
| 1 | 1 | 0 | Không khớp |
| 5 | 0 | 0 | Khớp |
| 12 | 1 | 0 | Không khớp |
| 21 | 1 | 0 | Không khớp |
| 23 | 1 | 0 | Không khớp |
| 29 | 0 | 0 | Khớp |
| 33 | 1 | 0 | Không khớp |
| 41 | 0 | 0 | Khớp |
| 46 | 1 | 1 | Khớp |
| 50 | 0 | 0 | Khớp |

**Cohen's kappa:** 0.1379  
**Mức diễn giải:** slight agreement

Giá trị kappa 0.1379 cho thấy mức đồng thuận giữa LLM Judge và nhãn con người còn thấp. Kết quả này chưa đủ để xem LLM Judge là nguồn đánh giá độc lập đáng tin cậy trong production. LLM Judge nên được sử dụng như một tín hiệu hỗ trợ, kết hợp với RAGAS metrics và kiểm duyệt thủ công ở các mẫu quan trọng.

---

## 5. Phân Tích Verbosity Bias

| Chỉ số | Kết quả |
|---|---:|
| A thắng và A dài hơn B | 0/10 |
| B thắng và B dài hơn A | 9/10 |
| Verbosity bias rate | 90.0% |

Verbosity bias rate đạt 90.0%, cho thấy judge có xu hướng chọn câu trả lời dài hơn trong nhiều trường hợp có winner rõ ràng. Đây là một rủi ro khi dùng LLM-as-Judge vì câu trả lời dài hơn không đồng nghĩa với chính xác hơn. Trong các lần đánh giá sau, rubric nên tách riêng độ chính xác, mức độ bám context và độ đầy đủ, đồng thời tránh để độ dài câu trả lời chi phối quyết định cuối cùng.

---

## 6. Nhận Xét Tổng Kết

LLM Judge hoạt động đúng về mặt cấu trúc đầu ra và không thể hiện position bias trong lần chạy hiện tại. Tuy nhiên, Cohen's kappa thấp cho thấy mức độ đồng thuận với human labels chưa cao. Ngoài ra, verbosity bias là vấn đề đáng chú ý vì có thể làm lệch kết quả đánh giá về phía các câu trả lời dài. Khi triển khai thực tế, nên sử dụng LLM Judge như một thành phần phụ trong hệ thống evaluation, đồng thời tiếp tục hiệu chỉnh prompt, rubric và bộ nhãn chuẩn của con người.
