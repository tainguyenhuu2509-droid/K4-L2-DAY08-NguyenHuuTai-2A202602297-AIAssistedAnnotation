# Báo cáo Lab Ngày 08: Học chủ động cho bộ phát hiện xe

Họ và tên: Nguyễn Hữu Tài

Công cụ gán nhãn đã dùng: CVAT (AnyLabeling, CVAT, SAM hoặc sửa trực tiếp file nhãn)

Sao chép file này thành `reports/REPORT.md` rồi điền vào các chỗ ĐIỀN. Mọi con số phải truy được
từ `reports/rounds_table.md`, `outputs/selection_round1.csv`, `outputs/metrics_round*.json` hoặc
`outputs/round*_diff.md`. Không coi nhãn test do mô hình tạo là chân lý tuyệt đối.

## 1. Dữ liệu và cách chia tập

Tập chưa gán nhãn (pool) và tập kiểm thử (test set) được tách theo trục thời gian và có vùng đệm ở giữa để hạn chế việc hai tập chứa những khung hình gần như giống nhau của cùng một cảnh hoặc cùng một chiếc xe. Với video cố định, một chiếc xe có thể xuất hiện trong nhiều khung hình liên tiếp; nếu chia ngẫu nhiên, các ảnh rất giống nhau có thể bị phân vào cả tập train/pool và test. Khi đó mô hình có thể gặp lại những cảnh gần như đã thấy trước khi được đánh giá, làm số đo trên test trở nên lạc quan hơn và không phản ánh tốt khả năng xử lý ảnh thực sự chưa từng gặp.

Nói cách khác, vấn đề không chỉ là “trùng ảnh” theo nghĩa hai file giống hệt nhau, mà còn là tương quan theo thời gian giữa các frame. Cách chia theo thời gian có vùng đệm giúp giảm nguy cơ này và làm cho kết quả test khó bị nâng lên chỉ vì các frame quá gần nhau.

## 2. Mô hình khởi đầu lạnh (cold start)

Dòng vòng 0 trong `rounds_table.md` là:

| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |

Trên cùng tập kiểm thử 20 ảnh có 403 box tham chiếu, trong đó bỏ qua 14 box có chiều cao dưới 16 pixel, cold start đạt AP50 = 0.771. Ở ngưỡng conf = 0.25, precision là 0.925 nhưng recall chỉ là 0.489, cho thấy mô hình khá ít dự đoán sai nhưng vẫn bỏ sót một lượng đáng kể xe. Recall theo kích thước là 0.182 với xe nhỏ, 0.547 với xe vừa và 0.561 với xe lớn. Điều này cho thấy xe nhỏ hoặc ở xa khó được tìm thấy hơn rõ rệt so với xe vừa và lớn. :contentReference[oaicite:1]{index=1}

Ảnh so sánh cold start cho thấy hiện tượng này rõ qua số TP, FP và FN. Ví dụ, `frame_0050` có 19 box tham chiếu nhưng cold start chỉ khớp 11 box, với 2 FP và 7 FN. `frame_0150` có 20 box tham chiếu, cold start khớp 10 box, với 2 FP và 10 FN. `frame_0250` có 17 box tham chiếu, chỉ khớp 6 box, với 2 FP và 9 FN. `frame_0350` có 23 box tham chiếu, khớp 9 box, với 2 FP và 14 FN. Như vậy, lỗi nổi bật trong ảnh ban đêm là bỏ sót xe, đặc biệt ở những xe nhỏ và ở xa; đồng thời vẫn xuất hiện một số dự đoán sai.

Tuy nhiên, không nên kết luận tất cả các FN đều chứng minh mô hình sai. Một trường hợp cần rà lại nhãn tham chiếu là các xe rất xa chỉ còn một hoặc hai điểm sáng. Guideline quy định với xe ở rất xa, chỉ còn hai chấm đèn và box cao dưới khoảng 16 pixel thì việc gán hoặc không gán đều được, đồng thời các box cỡ này bị bỏ qua khi chấm điểm. Vì vậy, một box tham chiếu cực nhỏ cần được kiểm tra lại trước khi dùng nó làm bằng chứng rằng model đã bỏ sót một chiếc xe chắc chắn. :contentReference[oaicite:2]{index=2}

## 3. Chiến lược chọn mẫu

Giải thích bằng lời công thức `score = W_U·U + W_A·A + W_D·D` và vai trò của `MIN_GAP_S`.
Dẫn ba frame trong `reports/SELECTION.md` và một frame khác để chứng minh cách bạn cân nhắc
độ bất định, ảnh gần trùng và công gán nhãn. Điểm bất định có chứng minh ảnh đó sẽ cải thiện
mô hình không? Vì sao?

Công cụ tính điểm theo công thức `score = W_U·U + W_A·A + W_D·D`, trong đó `U` là mức độ bất định của model, `A` là tỷ lệ số khung có confidence từ 0.15 đến dưới 0.50, còn `D` là khoảng cách thời gian tới ảnh đã được gán nhãn gần nhất. Theo trọng số mặc định, `U` đóng góp 0.5, `A` đóng góp 0.3 và `D` đóng góp 0.2 vào score. `MIN_GAP_S` dùng để hạn chế chọn những ảnh quá gần nhau về thời gian, vì các frame gần nhau trong video thường có nội dung rất giống nhau.

Ba frame trong lô 12 ảnh là `frame_0182.jpg` (hạng 1, score 0.9591, thời điểm 72.8 s), `frame_0369.jpg` (hạng 2, score 0.9324, thời điểm 147.6 s) và `frame_0392.jpg` (score 0.8874, thời điểm 156.8 s). `frame_0182.jpg` có `U = 0.9182`, `frame_0369.jpg` có `U = 0.9315`, còn `frame_0392.jpg` có `U = 0.9747`, cho thấy cả ba đều có mức bất định đáng chú ý.

Một frame khác là `frame_0372.jpg`, đứng hạng 6 với score 0.9101 và thời điểm 148.8 s, nhưng không được chọn vì chỉ cách `frame_0369.jpg` 1.2 giây, nên hai ảnh có nguy cơ gần trùng nhau. Điều này cho thấy không chỉ nhìn vào score mà còn phải xét độ đa dạng của ảnh và công sức sửa nhãn; tuy nhiên, công sửa nhãn không phải là một thành phần trực tiếp trong công thức score mà được cân nhắc khi quyết định cuối cùng.

Điểm bất định không chứng minh rằng ảnh đó chắc chắn sẽ cải thiện mô hình. Nó chỉ cho biết model đang chưa chắc chắn theo tiêu chí chọn mẫu; một ảnh điểm cao vẫn có thể quá mờ, khó gán nhãn hoặc gần trùng với ảnh khác, nên lợi ích thực tế chỉ có thể đánh giá sau khi sửa nhãn và huấn luyện lại.

## 4. Các vòng học chủ động (active learning)

Chép bảng từ `rounds_table.md`. Với mỗi vòng, trình bày:

- mức độ bạn đã sửa nhãn gợi ý (số box giữ nguyên, chỉnh sửa, xoá, thêm mới, lấy từ
  `outputs/round*_diff.md`);
- AP50 thay đổi bao nhiêu so với khởi đầu lạnh và so với vòng trước;
- nhóm xe nào tốt lên hoặc xấu đi theo số đo trên cùng tập test.

Dựa vào các ảnh `compare_round*.jpg`, chỉ ra một ca kết quả đổi sau fine-tune (tốt hơn hoặc xấu
đi), cùng lý do có thể kiểm. Dùng `BLIND_SCAN.md`, `REVIEW_LOG.csv` và `round1_diff.md` phân biệt
quan sát độc lập, lỗi pre-label đã sửa và kết quả mô hình sau train. Mô tả một ca khó theo guideline.

| vòng | model | ảnh train | box train | AP50 | Δ AP50 so cold start | P@0.25 | R@0.25 | F1 | R small | R medium | R large |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | yolov8n cold start (COCO car+bus+truck) | 0 | 0 | 0.771 | — | 0.925 | 0.489 | 0.640 | 0.182 | 0.547 | 0.561 |
| 1 | yolov8n fine-tune vong 1..1 | 12 | 340 | 0.818 | +0.047 | 0.985 | 0.323 | 0.486 | 0.000 | 0.328 | 0.805 |

Ở vòng 1, model ban đầu đề xuất 169 box trên 12 ảnh. Sau khi kiểm tra và sửa nhãn, số box tăng lên 340. Có 137 box được giữ nguyên (`accepted`), 17 box được chỉnh sửa (`edited`), 15 box bị xoá (`deleted`) và 186 box được thêm mới (`added`), với accept rate là 81%. Số box được thêm mới lớn hơn số box bị chỉnh sửa và xoá, cho thấy lỗi bỏ sót xe là vấn đề nổi bật trong các pre-label của vòng này.

AP50 tăng từ 0.771 ở cold start lên 0.818 ở vòng 1, tức tăng 0.047 so với cold start và cũng tăng 0.047 so với vòng trước. Tuy nhiên, tại ngưỡng conf = 0.25, precision tăng từ 0.925 lên 0.985 nhưng recall giảm từ 0.489 xuống 0.323 và F1 giảm từ 0.640 xuống 0.486. Điều này cho thấy AP50 tăng nhưng khả năng thu hồi xe tại ngưỡng confidence cố định lại giảm, nên không thể kết luận model tốt hơn trên mọi khía cạnh chỉ từ AP50.

Theo kích thước, recall xe nhỏ giảm từ 0.182 xuống 0.000, recall xe vừa giảm từ 0.547 xuống 0.328, trong khi recall xe lớn tăng từ 0.561 lên 0.805. Như vậy, nhóm xe lớn có kết quả tốt hơn, còn nhóm xe nhỏ và vừa có kết quả kém hơn sau vòng 1.

Về ca thay đổi sau fine-tune, hai file `compare_round0.jpg` và `compare_round1.jpg` mà tôi đang sử dụng có cùng nội dung, nên từ các bằng chứng hiện có tôi không thể xác nhận một chiếc xe cụ thể đã được phát hiện tốt hơn hoặc xấu đi sau fine-tune. Vì vậy, tôi không gán một ca cụ thể khi chưa có ảnh so sánh vòng 1 đúng phiên bản.

Trong `BLIND_SCAN.md`, tôi đã quan sát độc lập `frame_0392.jpg` và tự đếm 25 xe trước khi xem pre-label; hai vị trí khó là một xe mờ ở góc trái dưới và một xe mờ gần cuối làn bên phải. Đây là quan sát độc lập của người kiểm tra, không phải kết quả của model. `round1_diff.md` sau đó cho thấy ở `frame_0392.jpg` có 10 box được giữ nguyên, 2 box được chỉnh sửa, 1 box bị xoá và 15 box được thêm mới. Như vậy, việc tôi quan sát thấy nhiều xe trong Blind Scan phù hợp với thực tế rằng pre-label đã bỏ sót khá nhiều đối tượng, nhưng hai nguồn bằng chứng vẫn cần được phân biệt: Blind Scan là quan sát trước khi xem AI, còn diff là kết quả so sánh nhãn trước và sau khi tôi sửa.

Một tình huống khó theo guideline là xe rất xa chỉ còn hai chấm đèn hoặc xe có thân rất tối. Nếu xe ở rất xa, chỉ còn hai chấm đèn và box cao dưới khoảng 16 pixel thì có thể gán hoặc bỏ và loại box này được bỏ qua khi chấm điểm. Nếu vẫn có thể suy ra đường viền thân xe thì phải vẽ box theo phần thân xe có thể xác định được, không chỉ khoanh hai điểm sáng; đồng thời không được đưa vệt sáng của đèn chiếu trên mặt đường vào box.

## 5. Kết luận và giới hạn

Kết quả vòng này so với cold start ra sao? Vì sao bạn dừng hoặc tiếp tục? Đề xuất hai ca còn yếu
hoặc bất định cho vòng sau, kèm chi phí rà nhãn và nguy cơ ảnh gần trùng. Tập kiểm thử chỉ 20 ảnh,
có luật bỏ qua xe quá nhỏ và nhãn tham chiếu do mô hình tạo chưa được rà thủ công; các giới hạn đó
ảnh hưởng thế nào đến kết luận? Nếu AP50 giảm, bạn sẽ kiểm tra điều gì trước khi train thêm?

So với cold start, AP50 của vòng 1 tăng từ 0.771 lên 0.818, tương ứng tăng 0.047. Tuy nhiên, recall tại conf = 0.25 lại giảm từ 0.489 xuống 0.323, F1 giảm từ 0.640 xuống 0.486, recall xe nhỏ giảm từ 0.182 xuống 0.000 và recall xe vừa giảm từ 0.547 xuống 0.328, trong khi recall xe lớn tăng từ 0.561 lên 0.805. Vì vậy, tôi chọn dừng ở vòng hiện tại để kiểm tra lại chất lượng nhãn và kết quả dự đoán trước khi quyết định có tiếp tục học thêm hay không.

Hai nhóm trường hợp còn yếu hoặc bất định cho vòng sau là xe rất xa hoặc rất nhỏ, và xe bị che hoặc bị cắt ở mép ảnh. Xe rất xa có chi phí rà nhãn cao vì cần phân biệt giữa xe thực sự và các điểm sáng của đèn; xe bị che hoặc cắt mép cũng cần kiểm tra kỹ phần thân xe còn nhìn thấy để đặt box đúng. Ngoài ra, cần tránh chọn nhiều frame quá gần nhau về thời gian vì chúng có nguy cơ gần trùng nội dung, khiến công sức sửa nhãn tăng nhưng thông tin mới thu được ít.

Tập kiểm thử chỉ có 20 ảnh nên kết quả chưa chắc đại diện tốt cho toàn bộ dữ liệu. Bên cạnh đó, các box cao dưới khoảng 16 pixel được bỏ qua khi chấm điểm, nên khả năng phát hiện các xe cực nhỏ không được phản ánh đầy đủ. Nhãn tham chiếu dùng để đánh giá cũng chưa được người kiểm tra lại từng box, vì vậy AP50 phản ánh mức độ khớp với bộ nhãn tham chiếu hiện có chứ chưa đủ để khẳng định chất lượng nhận diện xe ngoài thực tế.

Nếu AP50 giảm ở vòng tiếp theo, trước tiên tôi sẽ kiểm tra lại các box đã thêm (`added`) và chỉnh sửa (`edited), đối chiếu với `REVIEW_LOG.csv` và `round1_diff.md`, sau đó xem lại các ảnh so sánh để xác định nguyên nhân. Chỉ sau khi loại trừ lỗi gán nhãn hoặc vấn đề trong dữ liệu tôi mới quyết định có tiếp tục cho model học thêm hay không.
