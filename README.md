
> 
> To do: **Rinn**
> 
> Last update: **06/08/2024**
> 
##### Tổng quan về bài toán, các module sử dụng
> 
> Thuộc nhóm bài toán nhận diện(~~~update diagram~~~)
> 
> Backbone: desenet121
> 
> Model: deepLabV3 trainning và dự đoán.
> 

##### Kiến trúc mô hình deepLabV3
Ở DeepLabV3 tác giả đưa vào 2 cải tiến chỉnh là:

- Tiến hành tích chập song song ASPP tại nhiều scale khác nhau và đưa thêm batch normalization, kế thừa ý tưởng từ mạng Inception.

- Bỏ Fully Connected CRF tại bước xử lý sau cùng giúp gia tăng tốc độ tính toán.

Một trong những thách thức của segmentation sử dụng mạng học sâu nhiều tầng DCNN là feature map ngày càng nhỏ hơn sau mỗi layer tích chập. Giảm độ phân giải có thể sẽ dẫn tới mất mát thông tin về vị trí và độ chi tiết của các đối tượng dự báo.

<img width="500" alt="image" src="https://github.com/user-attachments/assets/88f1c436-9442-4bc1-bdbf-9ec93296075a">

"Output là đặc trưng ảnh. Sau đó các output này được concanate lại với nhau và truyền qua một tích chập kích thước 1 x 1.

ASPP đã được giới thiệu trong DeepLabv2. Ở thời điểm này, batch normalization (BN) từ Inception-V2 được thêm vào ASPP. Tác dụng của batch normalization đó là giúp cho mô hình hội tụ nhanh hơn.

Lý do của việc sử dụng ASPP đó là thực tế cho thấy khi lấy mẫu rate lớn hơn, số lượng các bộ lọc hợp lệ giảm. Bộ lọc hợp lệ là bộ lọc có khả năng áp dụng cho các vùng feature hợp lệ, mà không phải padding thêm các giá trị 0. Sử dụng ASPP giúp làm đa dạng các bộ lọc với nhiều kích thước khác nhau và số lượng bộ lọc hợp lệ cũng nhiều hơn.

Áp dụng một tích chập 1 x 1 và 3 tích chập 3 x 3 với atrous rates = 
 khi output stride = 16.

Image pooling hoặc image-level feature cũng được thêm vào để ghi nhận bối cảnh toàn bộ (global context). Image pooling sẽ được tạo ra bằng cách global average pooling của toàn bộ layer trước đó. Global context đã được chứng minh là giúp làm rõ hơn sự nhầm lẫn cục bộ (local confusion). Đây là ý tưởng được kế thừa từ ParseNet.

Ápp dụng 256 bộ lọc và batch normalization ở các nhánh của mỗi biến đổi trong ASPP tại các block từ 4 đến 7. Ý tưởng batch normalization kết thừa từ Inception.

Kết quả các đặc trưng từ toàn bộ các nhánh được concatenate và truyền qua một tích chập 1 x 1 trước khi áp dụng tích chập 1 x 1 một lần nữa để tạo ra các giá trị logits từ hàm activation sigmoid."

##### Ứng dụng deepLab vào bài toán nhận diện

<img width="500" alt="image" src="https://github.com/user-attachments/assets/aadbc12e-5cdd-4a3a-a9cf-d4e80857a746">

Với 1 ảnh đầu vào là H x W x C thì cho ra kết quả là 1 feature với số kênh màu tăng lên và kích thước giảm đi. Đầu vào và ra được mô tả như sau:

<img width="505" alt="image" src="https://github.com/user-attachments/assets/7b16816e-09ba-46a6-b5ee-0985e2e7426e">

- Không áp dụng Atrous Convolution: Dòng đầu tiên, chúng ta chỉ áp dụng tích chập thông thường và max-pooling. Chúng ta thấy output stride gia tăng một cách đáng kể và khiến cho feature map nhỏ dần theo độ sâu của mô hình. Điều này gây hại cho segmentation bởi vì thông tin sẽ bị mất khi độ phân giải giảm tại những layers sâu hơn.

- Áp dụng Atrous Convolution: Dòng thứ 2, chúng ta có thể giữ cho độ phân giải của các block là ổn định và đồng thời gia tăng tầm nhìn (field-of-view) mà không cần gia tăng số lượng tham số và số lượng tính toán. Cuối cùng chúng ta thu được một feature map có kích thước lớn hơn và bảo toàn được thông tin về vị trí và không gian. Đây là một yếu tố có lợi cho segmentation.


Như vậy, ta có đầu ra là 1 feature có kích thước ảnh giảm so với hxw đầu vào và tăng số kênh màu lên theo tích chập được cấu hình:

<img width="353" alt="image" src="https://github.com/user-attachments/assets/e84c1f92-dd55-47e4-83b8-cdad56252a45">

Và kết quả từ việc trích xuất đặc trưng ảnh đưa ra cac dự đoán:

<img width="426" alt="image" src="https://github.com/user-attachments/assets/f87591ab-bd24-4e97-9d91-c465cf18177e">

...................

##### Hàm mất mát

Loss function kí hiệu là L, là thành phần cốt lõi của evaluation function và objective function.
hân lớp nhị phân là bài toán mà biến đầu ra (y) chỉ nhận một trong hai giá trị là 1 trong 2 nhãn.

Bài toán thường dưới dạng bài toán dự đoán giá trị 0 hoặc 1 cho lớp đầu tiên hoặc lớp thứ hai và thường được phát biểu như dự đoán xác suất của đầu vào thuộc giá trị lớp 1.

Trong phần này chúng ta sẽ khảo sát các hàm loss cho bài toán phân lớp nhị phân.

<img width="447" alt="image" src="https://github.com/user-attachments/assets/13d12228-c1b1-4528-90cf-b9b54c6abd11">



