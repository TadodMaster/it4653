# IT4653 - Học sâu và ứng dụng

## Đề tài 9. Mô hình sinh: từ Autoencoder tới VAE và GAN

| Mức độ | GAN có thể huấn luyện thất bại; hãy làm AE/VAE trước để chắc chắn có kết quả |
|---|---|
| Tài nguyên tính toán | Cần GPU. MNIST: ~15 phút/mô hình. CelebA 64×64: ~1 giờ cho DCGAN trên T4. |

**Mục tiêu.** So sánh hai họ mô hình sinh trên cùng dữ liệu và cùng ngân sách tính toán, làm rõ đánh đổi giữa độ nét ảnh và tính ổn định khi huấn luyện.

**Dữ liệu.** MNIST hoặc Fashion-MNIST cho phần cơ sở; CelebA cắt 64×64 (lấy 20–30k ảnh) cho phần nâng cao.

### Yêu cầu bắt buộc

- Nắm rõ được các hyperparameter trong quá trình xây dựng mô hình, cần thử nghiệm sự ảnh hưởng, tác dụng của mỗi tham số đóng góp trong quá trình xây dựng mô hình
- Cài Autoencoder thường và Variational Autoencoder: viết rõ hàm mất mát gồm reconstruction + KL, giải thích và cài đặt reparameterization trick (nêu rõ vì sao không lan truyền ngược qua phép lấy mẫu được).
- Khảo sát số chiều không gian ẩn (2, 8, 32, 128) tới chất lượng tái tạo; với chiều ẩn = 2, vẽ bản đồ không gian 2D.
- Nội suy tuyến tính giữa hai điểm trong không gian ẩn của AE và của VAE, đặt cạnh nhau để cho thấy VAE cho không gian ẩn liên tục hơn.
- Cài DCGAN: mô tả generator/discriminator, ghi lại đường loss của cả hai và bình luận về tính bất ổn định; ghi nhận mode collapse nếu xảy ra.
- So sánh AE / VAE / GAN bằng cả chỉ số định lượng (FID hoặc Inception Score) và lưới ảnh sinh ra ở cùng số epoch.

### Yêu cầu khác

- Cài Conditional VAE hoặc Conditional GAN để sinh ảnh theo nhãn cho trước.
- Ứng dụng phát hiện bất thường bằng sai số tái tạo của autoencoder (huấn luyện trên 9 chữ số, kiểm thử trên chữ số còn lại).

### Sản phẩm

Mã 3 mô hình + lưới ảnh sinh theo epoch + bảng FID + hình nội suy không gian ẩn + phân tích sự bất ổn định của GAN.