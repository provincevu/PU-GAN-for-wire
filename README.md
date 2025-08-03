# PU-GAN cho hệ thống đường dây truyền tải

> **Lưu ý**: dự án này được phát triển dựa trên [PU-GAN](https://github.com/liruihui/PU-GAN)

## thay đổi chính
Tôi đã bổ sung một vài tính năng để phù hợp hơn với việc xử lý dữ liệu đường dây truyền tải
Các tính năng được bổ sung gồm:
- **Sửa đổi các thư viện**: do mã nguồn đã được công khai từ năm 2019 nên nhiều thư viện đã không còn hỗ trợ, tôi đã chỉnh sửa lại phiên bản của các thư viện cũng như mã nguồn khi sử dụng các thư viện đó cho phù hợp
- **Tự động nhận diện dữ liệu đường dây truyền tải**
- **Tự động tạo dữ liệu huấn luyện và kiểm thử cho mô hình**

## Trích dẫn
@inproceedings{li2019pugan,
     title={PU-GAN: a Point Cloud Upsampling Adversarial Network},
     author={Li, Ruihui and Li, Xianzhi and Fu, Chi-Wing and Cohen-Or, Daniel and Heng, Pheng-Ann},
     booktitle = {{IEEE} International Conference on Computer Vision ({ICCV})},
     year = {2019}
 }
