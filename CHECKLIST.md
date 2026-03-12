# Checklist Đồ Án 1 — Search & Nature-Inspired Algorithms
> Đối chiếu **yêu cầu PDF** vs **trạng thái repo hiện tại**
> Cập nhật lần cuối: 12/03/2026

Ký hiệu: ✅ Hoàn thành · ⚠️ Có nhưng bị lỗi / chưa đủ · ❌ Chưa làm

---

## 1. THUẬT TOÁN CẦN CÀI ĐẶT

### 1.1 Classical Graph Search (bắt buộc để so sánh)
| # | Thuật toán | Trạng thái | Ghi chú |
|---|-----------|-----------|---------|
| 1 | Breadth-First Search (BFS) | ✅ | `algorithms/classic/bfs.py` |
| 2 | Depth-First Search (DFS) | ✅ | `algorithms/classic/dfs.py` |
| 3 | Uniform Cost Search (UCS) | ✅ | `algorithms/classic/ucs.py` |
| 4 | Greedy Best-First Search | ✅ | `algorithms/classic/greedy.py` |
| 5 | A* Search | ✅ | `algorithms/classic/astar.py` |
| 6 | Hill Climbing (Steepest Ascent) | ✅ | `algorithms/classic/hill_climbing.py` |
| 7 | Simulated Annealing (local search) | ✅ | `algorithms/physic_based/simulated_annealing.py` |

> Lưu ý: BFS/DFS/UCS/Greedy/A* áp dụng cho bài toán đồ thị (ShortestPath, TSP). PDF yêu cầu **so sánh** nature-inspired với ít nhất **4 trong số** các thuật toán truyền thống này.

### 1.2 Evolution-Based (bắt buộc)
| # | Thuật toán | Trạng thái | Ghi chú |
|---|-----------|-----------|---------|
| 1 | Genetic Algorithm (GA) | ✅ | `algorithms/evolutionary/genetic_algorithm.py` |
| 2 | Differential Evolution (DE) | ✅ | `algorithms/evolutionary/differential_evolution.py` |
| 3 | Evolution Strategies (ES) | ❌ | PDF liệt kê nhưng không bắt buộc trong bảng implementation |

### 1.3 Physics-Based (bắt buộc)
| # | Thuật toán | Trạng thái | Ghi chú |
|---|-----------|-----------|---------|
| 1 | Simulated Annealing (SA) | ✅ | Đã có, dùng chung với phần classical |

### 1.4 Biology-Based (bắt buộc)
| # | Thuật toán | Trạng thái | Ghi chú |
|---|-----------|-----------|---------|
| 1 | Ant Colony Optimization (ACO) | ✅ | `algorithms/biology/aco.py` |
| 2 | Particle Swarm Optimization (PSO) | ✅ | `algorithms/biology/pso.py` |
| 3 | Artificial Bee Colony (ABC) | ✅ | `algorithms/biology/artificial_bee_colony.py` |
| 4 | Firefly Algorithm (FA) | ✅ | `algorithms/biology/firefly.py` |
| 5 | Cuckoo Search (CS) | ✅ | `algorithms/biology/co.py` |

### 1.5 Human Behavior-Based (tùy chọn — bonus)
| # | Thuật toán | Trạng thái | Ghi chú |
|---|-----------|-----------|---------|
| 1 | TLBO | ✅ | `algorithms/human_based/TLBO.py` |

---

## 2. BÀI TOÁN THỬ NGHIỆM

### 2.1 Continuous Optimization
| # | Bài toán | Trạng thái | Ghi chú |
|---|---------|-----------|---------|
| 1 | Sphere (unimodal) | ✅ | `problems/continuous/sphere.py` |
| 2 | Rastrigin (multimodal) | ✅ | `problems/continuous/rastrigin.py` |
| 3 | Rosenbrock (narrow valley) | ⚠️ | Bug: `for i,x in solution[:-1]` → phải là `enumerate(solution[:-1])` |
| 4 | Griewank | ✅ | `problems/continuous/griewank.py` |
| 5 | Ackley (many local optima) | ✅ | `problems/continuous/ackley.py` |

### 2.2 Discrete Optimization
| # | Bài toán | Trạng thái | Ghi chú |
|---|---------|-----------|---------|
| 1 | Traveling Salesman (TSP) với time & cost constraints | ⚠️ | Bug: `for i in range(solution)` → `range(len(solution)-1)` |
| 2 | Knapsack (KP) | ⚠️ | Bug: `for _ in self.size` → `for _ in range(self.size)` |
| 3 | Graph Coloring (GC) | ⚠️ | Bug: `get_bounds` dùng `self.size` thay vì `range(self.size)` |
| 4 | Shortest Path | ✅ | Đã viết lại: graph có trọng số, đảm bảo connected, có heuristic, hỗ trợ cả graph-search lẫn metaheuristic |

---

## 3. LỖI KỸ THUẬT CẦN SỬA NGAY

| # | File | Lỗi | Mức độ |
|---|-----|-----|--------|
| 1 | `utils/evaluator.py:20` | `print(f"  ▶ Running...")` — ký tự Unicode crash Windows terminal | 🔴 Nghiêm trọng |
| 2 | `problems/discrete/knapsack.py:14` | `for _ in self.size` → `for _ in range(self.size)` | 🔴 |
| 3 | `problems/discrete/traveling_salesman.py:46` | `for i in range(solution)` → `for i in range(len(solution)-1)` | 🔴 |
| 4 | `problems/discrete/shortest_path.py:22` | `return 0` trong `__init__` → xoá đi | 🔴 |
| 5 | `problems/discrete/shortest_path.py:32` | `random_solution_generate` chỉ có `pass` → cần trả về path từ node 0 | 🔴 |
| 6 | `problems/discrete/shortest_path.py:60` | `for _ in self.size` → `for _ in range(self.size)` | 🔴 |
| 7 | `problems/discrete/graph_coloring.py:66` | `for _ in self.size` → `for _ in range(self.size)` | 🔴 |
| 8 | `problems/continuous/rosenbrock.py:22` | `for i,x in solution[:-1]` → `for i,x in enumerate(solution[:-1])` | 🔴 |
| 9 | `problems/discrete/knapsack.py` | `is_goal` chỉ có `pass` → cần return boolean | 🟡 |
| 10 | `problems/discrete/traveling_salesman.py` | `is_goal` chỉ có `pass` → cần return boolean | 🟡 |
| 11 | `algorithms/evolutionary/differential_evolution.py` | Chỉ hỗ trợ minimization (hardcode `argmin`) — không dùng được cho Knapsack (max) | 🟡 |

---

## 4. VISUALIZATION

| # | Yêu cầu PDF | Trạng thái | Ghi chú |
|---|------------|-----------|---------|
| 1 | Convergence curves | ✅ | `convergence_*.html` — dùng Plotly |
| 2 | Comparative performance bar chart | ✅ | `bar_*.html` |
| 3 | 3D surface plot (continuous) | ✅ | `3d_*.html` — có trajectory |
| 4 | Heatmap tổng hợp | ✅ | `heatmap_*.html` |
| 5 | Parameter sensitivity analysis | ❌ | Chưa có — thay đổi 1 param, giữ nguyên phần còn lại, vẽ ảnh hưởng |
| 6 | Thư viện: Matplotlib / Seaborn | ⚠️ | PDF recommend Matplotlib; repo đang dùng Plotly (vẫn chấp nhận được) |

---

## 5. METRICS SO SÁNH

| # | Metric | Trạng thái | Ghi chú |
|---|--------|-----------|---------|
| 1 | Convergence speed | ✅ | Có qua đồ thị convergence |
| 2 | Best solution quality | ✅ | In ra trong evaluator |
| 3 | Average solution quality | ⚠️ | `num_runs=1` — cần chạy nhiều lần (khuyến nghị ≥ 10) |
| 4 | Computational time | ✅ | Đo time trong evaluator |
| 5 | Robustness (mean ± std) | ❌ | Cần gom kết quả nhiều lần chạy và tính mean/std |
| 6 | Scalability (tăng kích thước) | ❌ | Chưa có — cần test nhiều `size` khác nhau |
| 7 | Exploration vs Exploitation | ❌ | Chưa có phân tích (ví dụ: đa dạng quần thể theo vòng lặp) |

---

## 6. YÊU CẦU CHUNG VỀ CODE

| # | Yêu cầu | Trạng thái | Ghi chú |
|---|---------|-----------|---------|
| 1 | Chỉ dùng NumPy (không scikit-learn, scipy.optimize) | ✅ | Đang tuân thủ |
| 2 | Code modular, có tham số cấu hình | ✅ | Kiến trúc tốt |
| 3 | Xử lý cả continuous & discrete | ✅ | Có, nhưng một số thuật toán chỉ continuous |
| 4 | Python best practices, well-documented | ⚠️ | Thiếu docstring, một số hàm `pass` |
| 5 | README với hướng dẫn setup và ví dụ chạy | ❌ | Chưa có |
| 6 | Upload lên GitHub | ⚠️ | Cần kiểm tra |

---

## 7. NỘP BÀI

| # | Hạng mục | Trạng thái | Ghi chú |
|---|---------|-----------|---------|
| 1 | Báo cáo PDF (≥ 25 trang, tiếng Việt hoặc Anh) | ❌ | Chưa có |
| 2 | Source code có README | ❌ | Chưa có README |
| 3 | Video demo YouTube (≥ 5 phút) | ❌ | Chưa có |
| 4 | File .zip đặt tên `<Group_ID>.zip` | ❌ | Chưa có |

---

## 8. CẤU TRÚC BÁO CÁO (theo PDF)

| # | Chương | Trạng thái |
|---|-------|-----------|
| 1 | Chapter 1 — Introduction & Algorithm Taxonomy | ❌ |
| 2 | Chapter 2 — Classical Graph Search Algorithms (BFS, DFS, UCS, Greedy, A*) | ❌ |
| 3 | Chapter 3 — Local Search & Physics-Based (Hill Climbing, SA) | ❌ |
| 4 | Chapter 4 — Evolution-Based (GA, DE) | ❌ |
| 5 | Chapter 5 — Swarm & Biology-Based (ACO, PSO, ABC, FA, CS) | ❌ |
| 6 | Chapter 6 — Human-Inspired (TLBO) | ❌ |
| 7 | Chapter 7 — Discussion & Insights | ❌ |
| 8 | Chapter 8 — Conclusion & Future Work | ❌ |

---

## 9. ƯU TIÊN CÔNG VIỆC CÒN LẠI

### � Đã hoàn thành
1. ~~Sửa bug `TravelingSalesman` class header thiếu~~ ✅
2. ~~Implement BFS, DFS, UCS, Greedy Best-First, A* (từng file riêng)~~ ✅
3. ~~Viết lại `ShortestPath` (weighted, connected, heuristic, dual-mode)~~ ✅
4. ~~3 benchmark độc lập trong `main.py`~~ ✅

### 🔴 Làm tiếp

### 🟡 Làm sau khi code chạy ổn định
4. Tăng `num_runs` lên ≥ 10, thêm tính mean/std vào `BenchmarkEngine`
5. Thêm parameter sensitivity analysis
6. Viết README.md

### 🟢 Làm song song với báo cáo
7. Phân tích scalability
8. Viết báo cáo theo 8 chương
9. Quay video demo
