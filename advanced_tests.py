import os
import time
import pickle
from algorithms import PSO
from problems import Griewank

def run_advanced_tests():
    os.makedirs("results", exist_ok=True)
    
    print("1. Đang chạy Parameter Sensitivity Test (Đổi số lượng hạt)...")
    sens_results = []
    for particles in [10, 20, 30, 40, 50]:
        algo = PSO(num_particles=particles, max_iters=200)
        prob = Griewank(dim=2, bound=5.0)
        
        final_score = float('inf')
        for state in algo.run(prob):
            final_score = state['best_score']
        sens_results.append({'param': particles, 'score': final_score})
        print(f"  - Particles: {particles} -> Loss: {final_score:.4f}")
        
    with open("results/sensitivity_results.pkl", "wb") as f:
        pickle.dump(sens_results, f)

    print("\n2. Đang chạy Scalability Test (Tăng số chiều của bài toán)...")
    scale_results = []
    for d in [2, 5, 10, 20, 30]:
        algo = PSO(num_particles=20, max_iters=200)
        prob = Griewank(dim=d, bound=5.0)
        
        start = time.time()
        for state in algo.run(prob): pass
        exec_time = time.time() - start
        scale_results.append({'dim': d, 'time': exec_time})
        print(f"  - Dimension: {d} -> Time: {exec_time:.3f}s")
        
    with open("results/scalability_results.pkl", "wb") as f:
        pickle.dump(scale_results, f)
        
    print("\n[+] Đã tạo xong dữ liệu! Bây giờ hãy chạy file report.py.")

if __name__ == "__main__":
    run_advanced_tests()