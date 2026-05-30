# Automated_Abdominal_Multi-Organ_Segmentation_via_Contrastive_Learning_and_Attention_Mechanisms
*(Đồ án cuối kỳ: Phân vùng đa cơ quan vùng bụng trên ảnh CT sử dụng Học đối chiếu và Kiến trúc lai CNN - Transformer)*

## 1. Project Overview (Tổng quan dự án)
Dự án này tập trung giải quyết bài toán Medical Image Segmentation (Phân vùng ảnh y tế) trên bộ dữ liệu Synapse Multi-organ CT. Nhằm vượt qua rào cản về dữ liệu và nâng cao độ chính xác, dự án triển khai phương pháp huấn luyện tiên tiến **Contrastive Learning** (Học đối chiếu) áp dụng cho hai kiến trúc mô hình tiêu biểu:
* **Model 1 (CNN-based):** ResNet50-UNet
* **Model 2 (Attention-based):** TransUNet (Kết hợp CNN và Transformer)

Mô hình được huấn luyện end-to-end từ bước tiền xử lý khối 3D NIfTI thành ảnh 2D, data augmentation, đến việc tính toán loss và đánh giá đa chỉ số. Kết quả của mô hình được triển khai trực quan thông qua ứng dụng Web UI sử dụng Gradio.

## 2. Team Members
| STT | Họ và Tên | Lớp | MSSV | GitHub Account |
| :--: | :-------- | :--: | :--: | :------------- |
| 1 | Trần Viết Gia Huy | CS0001 | 31231027056 | [@Tommyhuy1705](https://github.com/Tommyhuy1705) |
| 2 | Nguyễn Minh Nhựt | CS0001 | 31231022656 | [@Sura3607](https://github.com/Sura3607) |
| 3 | Nguyễn Trọng Hưởng | CS0001 | 31231023691 | [@trongjhuongwr](https://github.com/trongjhuongwr) |
| 4 | Tô Xuân Đông | CS0001 | 31231025345 | [@xuandongg1801](https://github.com/xuandongg1801) |

## 3. Directory Structure
Dự án được tổ chức theo cấu trúc module hóa chuẩn mực:
```text
synapse-multiorgan-segmentation/
├── configs/               # Chứa requirements.txt, .gitignore
├── data/                  # (Not tracked) /raw (.nii.gz) và /processed (.npy slices)
├── preprocessing/         # Mã nguồn xử lý 3D->2D, HU Normalization, dataset.py
├── models/                # Kiến trúc resnet_unet.py, transunet.py, losses.py (Contrastive, DiceCE)
├── training/              # Vòng lặp train.py, config.py, augmentation.py (Albumentations)
├── evaluation/            # Tính toán metrics (evaluate.py) và vẽ biểu đồ (visualize.py)
├── checkpoints/           # (Not tracked) Lưu trữ weights mô hình tốt nhất (.pth)
├── notebooks/             # 01_EDA.ipynb, 02_results_comparison.ipynb
└── app/                   # Triển khai ứng dụng app.py (Gradio UI), inference.py
```
4. Methodology Summary
- Tiền xử lý (Preprocessing): Trích xuất ảnh cắt ngang (axial slices) từ khối 3D; Kẹp cửa sổ Hounsfield Unit (HU windowing) trong khoảng [-125, 275] để làm nổi bật mô mềm; Resize ảnh về kích thước 224x224.
- Data Augmentation: Áp dụng các kỹ thuật augmentation đặc thù cho ảnh y tế bằng thư viện Albumentations (ElasticTransform, GridDistortion, Random Rotation).
- Mô hình & Huấn luyện (Models & Training):
  Sử dụng kiến trúc lai kết hợp sức mạnh trích xuất đặc trưng cục bộ của ResNet50 và khả năng nắm bắt ngữ cảnh toàn cục của Transformer (TransUNet).
  Ứng dụng Pixel-level Contrastive Learning làm auxiliary loss kết hợp với DiceCE Loss để tăng khả năng phân biệt ranh giới giữa các cơ quan nội tạng. Optimizer: AdamW; Scheduler: Cosine.
- Đánh giá (Evaluation): Benchmark hiệu năng thông qua các chỉ số: Mean DSC (Dice Score), IoU, HD95 (95th-percentile Hausdorff Distance), và số lượng Parameters.
- Ứng dụng (Deployment): Xây dựng luồng dữ liệu (Data flow) thông qua Gradio UI: Upload ảnh CT -> Chọn Mô hình -> Tiền xử lý & Suy luận -> Overlay Mask phân vùng lên ảnh gốc.

5. Dataset Details
- Cấu trúc chia tập: 18 bệnh nhân cho tập Train, 12 bệnh nhân cho tập Test để tránh rò rỉ dữ liệu (Patient-wise split).
- Số lượng nhãn (Classes): 9 classes (Bao gồm Nền (0) và 8 cơ quan nội tạng: Lách, Thận phải, Thận trái, Túi mật, Gan, Dạ dày, Động mạch chủ, và Tuyến tụy).

## 6. Streamlit Web Demo

Ứng dụng demo nằm trong `app/streamlit_app.py`. App cho phép upload một CT slice hoặc file `.npy/.npz`, chọn biến thể model trên Kaggle Model, chạy inference và hiển thị ảnh đầu vào, mask dự đoán, overlay cùng legend 9 lớp.

Các biến thể model:

- `resnet-unet`: baseline, `contrastive_weight = 0.0`
- `resnet-unet-cw001`: `contrastive_weight = 0.01`
- `resnet-unet-cw003`: `contrastive_weight = 0.03`
- `resnet-unet-cw005`: `contrastive_weight = 0.05`
- `transunet`: baseline, `contrastive_weight = 0.0`
- `transunet-cw001`: `contrastive_weight = 0.01`
- `transunet-cw003`: `contrastive_weight = 0.03`
- `transunet-cw005`: `contrastive_weight = 0.05`

Chạy local:

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Deploy lên Streamlit Community Cloud:

1. Push repo lên GitHub.
2. Vào `https://share.streamlit.io`, chọn **Create app**.
3. Chọn repository, branch, và main file path: `app/streamlit_app.py`.
4. Trong **Advanced settings**, chọn Python version tương thích với PyTorch, khuyến nghị Python 3.11 hoặc 3.12.
5. Nếu Kaggle Model ở chế độ private, dán secrets bên dưới vào phần **Secrets**.
6. Deploy và theo dõi build log trong Streamlit Cloud.

Nếu Kaggle Model ở chế độ private, thêm secrets trên Streamlit Cloud:

```toml
KAGGLE_USERNAME = "your-kaggle-username"
KAGGLE_KEY = "your-kaggle-api-key"
```

Không cần API inference riêng trong phiên bản này: Streamlit tải checkpoint bằng `kagglehub`, load PyTorch model và chạy inference trực tiếp. Nếu muốn dùng checkpoint local thay vì tải từ Kaggle, set biến môi trường như `TRANSUNET_CW001_MODEL_DIR` hoặc `RESNET_UNET_MODEL_DIR` trỏ tới thư mục chứa file `.pt/.pth`.
