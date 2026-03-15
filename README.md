# Search Project - Introduction to AI

## 1. Giới thiệu
Đây là đồ án môn Nhập môn Trí tuệ Nhân tạo, tập trung vào việc:
- Cài đặt và so sánh nhiều thuật toán tìm kiếm/tối ưu.
- Thử nghiệm trên các bài toán liên tục, rời rạc và tìm đường.
- Đánh giá theo nhiều tiêu chí: chất lượng nghiệm, thời gian, bộ nhớ, độ hội tụ.
- Xuất báo cáo trực quan dưới dạng HTML trong thư mục `results/`.

## 2. Thành viên nhóm

| STT | MSSV     | Họ và tên            |
|-----|----------|----------------------|
| 1   | 24127025 | Lê Minh Đức          |
| 2   | 24127244 | Phạm Tấn Nhật Thịnh  |
| 3   | 24127258 | Diệp Minh Khánh Tuân |
| 4   | 24127396 | Lê Trần Quang Huy    |

## 3. Thuật toán đã triển khai

### 3.1. Nhóm thuật toán cổ điển
- Hill Climbing
- Tabu Search
- BFS
- DFS
- UCS
- Greedy Best-First Search
- A*

### 3.2. Nhóm thuật toán lấy cảm hứng vật lý
- Simulated Annealing
- Gravitational Search Algorithm
- Harmony Search

### 3.3. Nhóm thuật toán tiến hóa
- Genetic Algorithm
- Differential Evolution
- Evolution Strategy (có cài nhưng không trình bày trong video)

### 3.4. Nhóm thuật toán lấy cảm hứng sinh học
- PSO
- ACO
- Cuckoo Optimization
- Artificial Bee Colony
- Firefly Algorithm

### 3.5. Nhóm thuật toán lấy cảm hứng con người
- TLBO

## 4. Bài toán thử nghiệm

### 4.1. Bài toán liên tục
- Sphere
- Rastrigin
- Ackley
- Rosenbrock
- Griewank

### 4.2. Bài toán rời rạc
- Shortest Path
- Traveling Salesman
- Knapsack
- Graph Coloring

## 5. Cấu trúc thư mục

```text
.
|-- algorithms/
|   |-- classic/
|   |-- biology/
|   |-- evolutionary/
|   |-- physic_based/
|   `-- human_based/
|-- core/
|-- problems/
|-- results/
|-- tests/
|-- utils/
|-- main.py
|-- visual.py
`-- requirements.txt
```

## 6. Hướng dẫn cài đặt và chạy

### 6.1. Tạo môi trường và cài thư viện
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Nếu gặp lỗi với một số package chuẩn của Python trong `requirements.txt` (ví dụ `random`, `abc`, `ast`), có thể cài thủ công các thư viện chính:
```bash
pip install plotly matplotlib Pillow numpy pandas scipy streamlit
```

### 6.2. Chạy benchmark tổng hợp
```bash
python main.py
```
Kết quả sẽ được xuất dưới dạng các file HTML trong thư mục `results/`.

### 6.3. Chạy giao diện trực quan
```bash
streamlit run visual.py
```

## 7. Đầu ra và báo cáo
- Các biểu đồ hội tụ, boxplot, biểu đồ 3D và so sánh thuật toán được lưu trong `results/`.
- Có thể mở trực tiếp các file `.html` bằng trình duyệt để xem tương tác.

## 8. Ghi chú
- Dự án thiên về mục tiêu học thuật: minh họa nguyên lý thuật toán và so sánh thực nghiệm.
- Các tham số trong `main.py` có thể thay đổi để cân bằng giữa thời gian chạy và chất lượng kết quả.
