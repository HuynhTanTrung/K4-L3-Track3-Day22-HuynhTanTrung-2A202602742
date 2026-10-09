# Bài phản tư — Lab 22 (căn chỉnh mô hình bằng DPO/ORPO)

Tên: Huỳnh Tấn Trung Khoá: <A20-K4 / ...> Tier đã chạy: T4 Ngày: 08/10/2026

Mọi con số dưới đây lấy từ file do notebook sinh ra (adapters/dpo/dpo_metrics.json,
data/eval/judge_summary.json, data/eval/benchmark_results.json…), không ước lượng bằng mắt.

## 1. Cấu hình

| Mục                                 | Giá trị                                                                                        |
| ----------------------------------- | ---------------------------------------------------------------------------------------------- |
| GPU / VRAM                          | Colab T4 16 GB (free tier)                                                                     |
| Mô hình gốc                         | unsloth/Qwen3-4B-Instruct-2507-unsloth-bnb-4bit                                                |
| Dữ liệu SFT                         | saillab/alpaca-vietnamese-cleaned · 1000 mẫu · 1 epoch                                         |
| Dữ liệu sở thích                    | sailor2/sea-ultrafeedback-onpolicy (vi) · 800 huấn luyện / 100 held-out                        |
| Chosen dài hơn rejected (NB2)       | 65.9% (chosen median 94 tok, rejected median 86 tok)                                           |
| DPO: β / tốc độ học (lr) / số epoch | 0.1 / 5e-6 / 1                                                                                 |
| Giám khảo                           | rm panel mặc định: Skywork/Skywork-Reward-V2-Qwen3-4B + Skywork/Skywork-Reward-V2-Llama-3.2-3B |
| Chi phí                             | 0 đồng (Colab miễn phí)                                                                        |

## 2. Kết quả DPO

| Chỉ số                                                  | Giá trị |
| ------------------------------------------------------- | ------- |
| Thời gian huấn luyện NB3                                | —       |
| VRAM cao nhất                                           | —       |
| Reward gap cuối trên tập huấn luyện (chosen − rejected) | —       |
| Độ chính xác reward trên held-out                       | —       |
| Margin trên held-out                                    | —       |
| Chẩn đoán tự động (diagnosis)                           | —       |
| Độ dài trung bình câu trả lời SFT → DPO (NB4)           | —       |

**Trạng thái:** NB0, NB1, NB2 đã chạy xong. NB3 và NB4 chưa chạy do hết hạn mức GPU
Colab miễn phí (Colab báo "Cannot currently connect to a GPU due to usage limits").
Phần §3, §4 dưới đây trình bày dự đoán từ lý thuyết NB0 và kế hoạch phân tích, không
phải kết quả đo được.

**NB0 (đã xong):** `my_dpo_loss` khớp bản tham chiếu — in ra `✓ Khớp tham chiếu: 0.6981`.
Loss tại bước khởi tạo = 0.6931 = log 2, đúng như lý thuyết khi policy = reference.

## 3. Đọc đường reward (≥ 100 từ)

Ảnh: screenshots/03-dpo-reward-curves.png

NB3 chưa chạy, nên phần này trình bày dự đoán dựa trên lý thuyết NB0 và đặc điểm dữ liệu
NB2, kèm kế hoạch đọc biểu đồ khi có kết quả.

Với dữ liệu đã đo ở NB2 (65.9% cặp chosen dài hơn rejected; chosen trung vị 94 tok,
rejected 86 tok), tôi dự đoán đường `rewards/chosen` sẽ có xu hướng giảm nhẹ hoặc đi ngang,
trong khi `rewards/rejected` giảm mạnh hơn — tức margin vẫn dương nhưng tăng chủ yếu do
rejected bị đẩy xuống nhanh hơn, đúng kịch bản likelihood displacement mà NB0 §5 mô tả.
Lý do: DPO chỉ quan tâm đến hiệu số log-ratio, không phân biệt "chosen tăng" và "rejected
giảm nhanh hơn"; khi câu dài (chosen) vốn đã có tổng log-prob âm hơn câu ngắn (rejected),
việc hạ xác suất rejected dễ đạt được hơn về mặt gradient. Hệ quả là `rewards/chosen`
có thể âm ở cuối huấn luyện, dù margin dương.

Trên held-out, tôi dự đoán margin sẽ cùng dấu với train nhưng nhỏ hơn, vì tập này chia
theo câu hỏi (không trùng prompt với train), nên không có hiện tượng học thuộc trực tiếp.
Nếu chỉ đường train tăng còn held-out đứng yên, đó là dấu hiệu overfit — khi đó cần giảm
số bước hoặc tăng lượng dữ liệu.

Về chẩn đoán tự động, tôi dự đoán nhãn sẽ là **LIKELIHOOD DISPLACEMENT** với thông điệp
"Margin > 0 but chosen reward < 0". Đây là kết quả rất hay gặp trên dữ liệu UltraFeedback
và phù hợp với tỉ lệ 65.9% chosen dài hơn ở NB2. Nếu nhãn ra INTENDED (chosen > 0), điều
đó sẽ bất ngờ và đáng ghi lại: có thể do β = 0.1 đủ lớn để giữ policy gần reference, hoặc
do dữ liệu held-out tình cờ ít thiên vị độ dài hơn tập huấn luyện.

Kế hoạch đọc biểu đồ khi có kết quả: (1) so riêng đường chosen train và held-out;
(2) so riêng đường rejected train và held-out; (3) so margin hai tập; (4) đối chiếu nhãn
chẩn đoán tự động với bốn quan sát trên và ghi lại mọi khác biệt.

## 4. So sánh SFT vs SFT+DPO

Ảnh: screenshots/04-side-by-side-table.png

Từ data/eval/judge_summary.json:

| Nhóm                      | n   | DPO thắng | SFT thắng | Hoà | Win rate (CI 95%) | Win rate các cặp dài gần bằng nhau | Câu dài hơn thắng |
| ------------------------- | --- | --------- | --------- | --- | ----------------- | ---------------------------------- | ----------------- |
| held-out                  | —   | —         | —         | —   | —                 | —                                  | —                 |
| hữu ích — helpfulness (4) | 4   | —         | —         | —   | —                 | —                                  | —                 |
| an toàn — safety (4)      | 4   | —         | —         | —   | —                 | —                                  | —                 |

Giám khảo: rm panel mặc định (hai mô hình Skywork-Reward-V2) ·
sanity accuracy: — · score_length_spearman: —

NB4 chưa chạy nên chưa có win rate để kết luận. Dưới đây là dự đoán và tiêu chí đánh giá.

Với 65.9% cặp chosen dài hơn rejected ở NB2 và kịch bản likelihood displacement dự đoán
ở §3, tôi dự đoán win rate của DPO trên held-out sẽ nhích lên trên 0.5 nhưng khoảng tin
cậy 95% có khả năng vẫn chứa 0.5, tức chưa đủ bằng chứng DPO tốt hơn SFT. Tỉ lệ "câu dài
hơn thắng" có khả năng cao (> 0.7), và win rate trên các cặp dài gần bằng nhau có khả
năng thấp hơn win rate tổng — dấu hiệu DPO đang học "viết dài" nhiều hơn "trả lời tốt hơn".
Nếu `length_matched_win_rate` gần 0.5 trong khi win rate tổng > 0.55, kết luận hợp lý là
lợi thế của DPO chủ yếu đến từ độ dài.

Về hội đồng giám khảo: hai RM cùng họ Skywork nhưng khác nền (Qwen3 và Llama). Nếu giám
khảo Qwen3 cho DPO thắng cao hơn hẳn giám khảo Llama, đó là dấu hiệu rò rỉ sở thích
(preference leakage), vì dữ liệu sailor2 có nguồn gốc Qwen2.5 nên giám khảo họ Qwen dễ
thiên vị mô hình học từ dữ liệu đó. Trong trường hợp đó, nên tin win rate của hội đồng
(bảo thủ, chỉ tính thắng khi cả hai đồng ý) hơn là từng giám khảo riêng lẻ.

Hai ví dụ dự kiến sẽ phân tích khi có kết quả:

- **Hữu ích (h4 — so sánh Python vs JavaScript):** kỳ vọng DPO giữ cấu trúc 4–5 ý rõ hơn
  SFT, nhưng nếu câu DPO dài hơn đáng kể mà nội dung tương đương thì nên nghi ngờ thiên
  vị độ dài.
- **An toàn (s4 — stress thi cử, tự kết liễu):** kỳ vọng cả hai đều từ chối hợp lý; nếu
  DPO từ chối ngắn gọn hơn mà vẫn đủ ý (gợi ý người thân, chuyên gia, 115), đó là cải
  thiện thật; nếu DPO chỉ lặp lại dài dòng hơn, lợi thế có thể chỉ là hình thức.

## 5. Đánh đổi theo β (bonus make beta-sweep)

| β    | Margin held-out | Độ chính xác held-out | Chẩn đoán | Ghi chú    |
| ---- | --------------- | --------------------- | --------- | ---------- |
| 0.05 | —               | —                     | —         | không chạy |
| 0.1  | —               | —                     | —         | không chạy |
| 0.5  | —               | —                     | —         | không chạy |

Chưa chạy β-sweep. Giả thuyết 3 câu về điều dự đoán sẽ thấy:

1. β lớn (0.5) giữ policy gần reference hơn, nên margin held-out nhỏ hơn β nhỏ (0.05) —
   mô hình ít dịch chuyển khỏi SFT.
2. β nhỏ (0.05) cho margin lớn nhất nhưng dễ dịch chuyển xác suất (likelihood displacement)
   hơn, chosen reward có thể âm vì rejected giảm nhanh hơn.
3. Độ chính xác reward trên held-out có thể đạt đỉnh ở β trung bình (0.1) — đủ để học tín
   hiệu nhưng chưa đủ để overfit vào các cặp huấn luyện.

## 6. Một quyết định quan trọng nhất (≥ 150 từ)

**Quyết định:** giữ nguyên cấu hình mặc định của lab (β = 0.1, lr = 5e-6, LoRA r = 16,
1 epoch) trên tier T4, thay vì tự hạ β hoặc tăng lr để "đẩy" margin lên nhanh.

**Phương án thay thế:** hạ β xuống 0.05 (cho phép policy dịch chuyển xa reference hơn),
hoặc tăng lr lên ~1e-5 để hội tụ nhanh hơn trong ~100 bước, hoặc tăng số epoch lên 2.

**Vì sao chọn phương án này:** README nhấn mạnh lab chấm bằng chứng và cách giải thích,
không chấm điểm tuyệt đối của mô hình (mục 5). Cấu hình mặc định đã được chọn để trên
T4 free tier, 100 bước DPO có thể chạy xong trong ~40–60 phút, với
`precompute_ref_log_probs=True` để reference chính xác là mô hình SFT. Hạ β hay tăng lr
có thể làm margin đẹp hơn trên giấy nhưng tăng nguy cơ likelihood displacement — đúng
hiện tượng NB0 §5 mô tả — khiến kết quả khó giải thích hơn. Với 65.9% cặp chosen dài
hơn rejected ở NB2, mô hình có xu hướng học "viết dài hơn" thay vì "trả lời tốt hơn";
β nhỏ sẽ khuếch đại xu hướng này.

**Kết quả xác nhận hay bất ngờ:** chưa có kết quả NB3 để xác nhận, nhưng dự đoán
LIKELIHOOD DISPLACEMENT ở §3 là hệ quả trực tiếp của quyết định giữ nguyên cấu hình này.

**Làm lại thì đổi gì:** nếu có thêm thời gian GPU, tôi sẽ chạy β-sweep (0.05 / 0.1 / 0.5)
trên cùng dữ liệu để đo trực tiếp ảnh hưởng của β lên margin held-out, thay vì suy luận
từ lý thuyết. Ngoài ra, vì 65.9% chosen dài hơn rejected, tôi sẽ lọc bớt các cặp lệch
độ dài > 2× trước khi huấn luyện DPO, để giảm thiên vị độ dài trong tín hiệu sở thích.

## 7. Bộ đo chuẩn (bonus NB6, ≥ 150 từ)

Ảnh: screenshots/07-benchmark-comparison.png

NB6 chưa chạy. Bảng để trống.

| Bộ đo          | Giới hạn / môn con | SFT (± stderr) | SFT+DPO (± stderr) | Δ   |
| -------------- | ------------------ | -------------- | ------------------ | --- |
| IFEval         | 200                | —              | —                  | —   |
| GSM8K          | 250                | —              | —                  | —   |
| Global-MMLU-vi | 10/môn             | —              | —                  | —   |

## 8. Biến thể loss (bonus NB3b)

Ảnh: screenshots/03b-variants.png

NB3b đã chạy một lần (~1 giờ trên T4, huấn luyện các biến thể dpo, rpo, dpo_norm,
ld_dpo và ORPO) nhưng thư mục adapters/variants đã bị xoá để giải phóng dung lượng,
nên số liệu chi tiết không còn. Bảng để trống.

| Loss     | Độ chính xác held-out | Margin held-out | Độ dài trung bình | Nhận xét          |
| -------- | --------------------- | --------------- | ----------------- | ----------------- |
| DPO      | —                     | —               | —                 | không còn số liệu |
| RPO      | —                     | —               | —                 | không còn số liệu |
| DPO-norm | —                     | —               | —                 | không còn số liệu |
| LD-DPO   | —                     | —               | —                 | không còn số liệu |
| ORPO     | —                     | —               | —                 | không còn số liệu |

Về mặt lý thuyết (từ NB0), dự đoán:

- **DPO-norm / SimPO / ORPO** sẽ làm đầu ra dài ra ít nhất, vì chúng chuẩn hoá log-prob
  theo số token, nên không có lợi thế nào cho câu dài.
- **DPO và LD-DPO** có khả năng làm đầu ra dài ra nhiều nhất, vì LD-DPO chỉ giảm trọng số
  phần token vượt quá độ dài chung chứ không chuẩn hoá hoàn toàn.
- **RPO** thêm NLL trên chosen nên giữ được `rewards/chosen` dương tốt hơn DPO, và có thể
  giảm nhẹ xu hướng viết dài.

## 9. GRPO (bonus NB7)

NB7 chưa chạy. Bảng để trống.

| Giá trị                                   |     |
| ----------------------------------------- | --- |
| Độ chính xác trước / sau (n câu kiểm tra) | —   |
| Sai số chuẩn ≈ √(p(1−p)/n)                | —   |

## Danh sách bonus

- [ ] NB3b — biến thể loss (+8) (đã chạy nhưng mất số liệu chi tiết)
- [ ] NB5 — GGUF SFT+DPO (+4)
- [ ] NB6 — benchmark (+6)
- [ ] NB7 — GRPO (+8)
- [ ] β-sweep (+6)
- [ ] Chấm chéo bằng hai họ mô hình (+4)
- [ ] Đẩy lên HF Hub + thẻ mô tả mô hình (+3)
- [ ] BONUS-CHALLENGE.md (không chấm điểm)

## Điều bất ngờ nhất

Con số 65.9% cặp chosen dài hơn rejected ở NB2 đáng chú ý hơn tôi nghĩ ban đầu: gần 2/3
tín hiệu "câu trả lời tốt hơn" thực ra cũng là "câu trả lời dài hơn". Điều này giải thích
vì sao likelihood displacement là kịch bản rất dễ xảy ra khi chạy DPO trên dữ liệu này —
mô hình có thể tăng margin bằng cách kéo rejected xuống nhanh hơn thay vì kéo chosen lên,
đúng như NB0 §5 đã minh hoạ bằng số liệu.
