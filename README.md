# Hướng dẫn chạy project

1. Cài các thư viện:
   pip install -r requirements.txt

2. Cách chạy:

   chạy đơn giản: 
   python test.py, thay đổi dòng 66 thay thế parth ảnh

   chạy server API để load Mobile với WEB
   python server.py 
   sau khi teminal đã chạy server đã run
   thì nhập lệnh sau: curl.exe -X POST -F "image=@path của ảnh" http://ip server:5000/predict


3. Cấu trúc thư mục:
   - model/: chứa mã nguồn
   - dataset/: chứa dữ liệu ảnh chim (https://github.com/professorNhan-2824/apipython)

4. Dataset:
   Link Git: https://github.com/professorNhan-2824/apipython
   Đã đính kèm trong thư mục dataset/ của link git trên

5. Mô hình sử dụng:
   MobileNetV2, huấn luyện 50 epochs trên tập ảnh chim phân loại nhiều loài.

6. Liên hệ:
   - Đỗ Hữu Nhân- 22IT.B151
   - Nguyễn Đình Quan - 22IT.B164
   - Nguyễn Thị Ngọc - 22IT.B146
   - Phạm Ngọc Thiên Ân - 22IT.B011
