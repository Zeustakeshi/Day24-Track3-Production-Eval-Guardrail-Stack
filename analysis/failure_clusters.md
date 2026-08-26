# Báo Cáo Phân Tích Failure Clusters - Phase A

**Sinh viên:** Phạm Minh Hiếu - 2A202601562  
**Ngày thực hiện:** 2026-08-27  
**Nội dung:** Đánh giá RAGAS trên bộ kiểm thử 50 câu hỏi

---

## 1. Mục Tiêu Phân Tích

Phase A được thực hiện nhằm đánh giá chất lượng RAG pipeline trên ba nhóm câu hỏi: factual, multi_hop và adversarial. Các chỉ số RAGAS được sử dụng gồm faithfulness, answer_relevancy, context_precision và context_recall. Mục tiêu của phần phân tích là xác định nhóm câu hỏi có hiệu năng thấp, metric thường xuyên gây lỗi, và đề xuất hướng cải thiện cho pipeline.

---

## 2. Kết Quả Theo Distribution

| Metric | factual | multi_hop | adversarial |
|---|---:|---:|---:|
| faithfulness | 0.8333 | 0.4442 | 0.9000 |
| answer_relevancy | 0.7309 | 0.5893 | 0.6805 |
| context_precision | 0.9917 | 0.9833 | 0.8750 |
| context_recall | 0.8500 | 0.7625 | 0.6500 |
| **avg_score** | **0.8515** | **0.6948** | **0.7764** |

Nhóm factual đạt điểm trung bình cao nhất với 0.8515, cho thấy pipeline xử lý tốt các câu hỏi tra cứu trực tiếp. Nhóm multi_hop có điểm trung bình thấp nhất với 0.6948, phản ánh khó khăn khi cần kết hợp nhiều nguồn thông tin hoặc thực hiện suy luận nhiều bước. Nhóm adversarial đạt 0.7764, thấp hơn factual nhưng cao hơn multi_hop.

---

## 3. Bottom 10 Câu Hỏi Có Điểm Thấp Nhất

| Rank | Distribution | Question ID | avg_score | worst_metric |
|---:|---|---:|---:|---|
| 1 | multi_hop | 39 | 0.2500 | faithfulness |
| 2 | factual | 7 | 0.3333 | faithfulness |
| 3 | multi_hop | 21 | 0.3750 | faithfulness |
| 4 | multi_hop | 30 | 0.3750 | faithfulness |
| 5 | multi_hop | 33 | 0.3750 | faithfulness |
| 6 | factual | 5 | 0.4152 | faithfulness |
| 7 | adversarial | 50 | 0.4167 | faithfulness |
| 8 | multi_hop | 22 | 0.4792 | answer_relevancy |
| 9 | factual | 9 | 0.5000 | faithfulness |
| 10 | multi_hop | 38 | 0.6517 | faithfulness |

Trong bottom 10, nhóm multi_hop chiếm 6/10 câu. Điều này cho thấy pipeline còn yếu ở các truy vấn yêu cầu tổng hợp nhiều tài liệu hoặc so sánh nhiều chính sách. Phần lớn các câu trong nhóm này có worst_metric là faithfulness, tức câu trả lời có xu hướng suy diễn vượt quá hoặc không bám sát context.

---

## 4. Failure Cluster Matrix

| Worst metric | factual | multi_hop | adversarial | Tổng |
|---|---:|---:|---:|---:|
| faithfulness | 4 | 13 | 1 | 18 |
| answer_relevancy | 13 | 3 | 1 | 17 |
| context_precision | 0 | 0 | 2 | 2 |
| context_recall | 3 | 4 | 6 | 13 |

Metric faithfulness xuất hiện nhiều nhất với 18 trường hợp, tiếp theo là answer_relevancy với 17 trường hợp. Nhóm factual có nhiều câu bị xếp worst_metric là answer_relevancy, trong khi nhóm multi_hop tập trung lỗi ở faithfulness. Với nhóm adversarial, context_recall là vấn đề nổi bật hơn, cho thấy pipeline chưa luôn truy xuất đủ ngữ cảnh cần thiết khi câu hỏi có bẫy hoặc mâu thuẫn phiên bản.

---

## 5. Phân Tích Nguyên Nhân

Distribution có tổng số failure cao nhất là factual, tuy nhiên xét theo điểm trung bình thì multi_hop mới là nhóm yếu nhất. Sự khác biệt này đến từ việc factual có số lượng câu hỏi lớn và nhiều câu yêu cầu trả lời đúng chi tiết cụ thể như cấp phê duyệt, số ngày nghỉ hoặc ngưỡng chi phí. Đối với multi_hop, nguyên nhân chính là pipeline phải kết hợp nhiều tài liệu và nhiều điều kiện, khiến mô hình dễ tạo câu trả lời không đủ căn cứ.

Faithfulness là metric cần ưu tiên cải thiện vì xuất hiện nhiều nhất trong failure matrix và chiếm phần lớn bottom 10. Điều này cho thấy retrieval không phải lúc nào cũng là vấn đề chính; nhiều trường hợp context có thể đã được lấy đúng nhưng bước sinh câu trả lời chưa bám sát bằng chứng.

---

## 6. Đề Xuất Cải Thiện

| Metric cần cải thiện | Nguyên nhân chính | Hướng cải thiện |
|---|---|---|
| faithfulness | Mô hình suy diễn vượt quá context hoặc tổng hợp sai điều kiện | Siết prompt theo hướng chỉ trả lời dựa trên context, yêu cầu nêu bằng chứng, giảm temperature |
| context_recall | Chưa lấy đủ chunk liên quan, đặc biệt ở câu hỏi adversarial | Tăng top-k trước rerank, cải thiện chunking, bổ sung metadata về phiên bản tài liệu |
| context_precision | Một số context chưa đủ tập trung vào câu hỏi | Áp dụng reranker mạnh hơn và filter theo domain/version |
| answer_relevancy | Câu trả lời chưa đi thẳng vào yêu cầu của câu hỏi | Thêm bước query rewriting và định dạng trả lời theo từng ý hỏi |

---

## 7. Nhận Xét Về Nhóm Adversarial

Nhóm adversarial có avg_score 0.7764, thấp hơn factual nhưng cao hơn multi_hop. Kết quả này cho thấy pipeline nhận diện được một phần các tình huống có bẫy về phiên bản hoặc chính sách mâu thuẫn, nhưng vẫn gặp vấn đề khi cần truy xuất đầy đủ context. Trong bottom 10 chỉ có một câu adversarial, liên quan đến việc sử dụng VPN cá nhân khi làm việc từ xa. Đây là dạng câu hỏi dễ gây nhầm vì có yếu tố bảo mật nhưng câu trả lời cần tuân theo chính sách truy cập nội bộ cụ thể.
