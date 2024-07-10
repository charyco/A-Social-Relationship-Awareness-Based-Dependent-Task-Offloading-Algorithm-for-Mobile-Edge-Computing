import numpy as np
import matplotlib.pyplot as plt

# 定义参数
P_trans = 0.1
P_d = 500
B_d2d = 20
B_up = 10
B_down = 10
T_d2d_trans = 0.05
T_d2d_recv = 0.05
T_cell_trans = 0.2
T_cell_recv = 0.1
T_edge_proc = 0.5
C_t = 100
R_t = 500
mu_values = [0.1, 0.5, 0.9]  # 不同的mu值

# 数据大小范围
data_sizes = np.linspace(200, 2000, 10)

# 创建一个图形框，在其中可以放置多个轴
fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharey=True)

for idx, mu in enumerate(mu_values):
    costs = {'Local': [], 'D2D': [], 'Edge': [], 'D2D-Edge': []}

    for D_t in data_sizes:
        # 本地执行成本
        E_local = P_trans * C_t
        T_local = D_t / P_d
        costs['Local'].append(mu * T_local + (1 - mu) * E_local)

        # D2D执行成本
        E_D2D = (T_d2d_trans + T_d2d_recv) * (D_t / B_d2d) + P_trans * C_t
        T_D2D = (D_t / B_d2d) + (D_t / P_d)
        costs['D2D'].append(mu * T_D2D + (1 - mu) * E_D2D)

        # 边缘执行成本
        E_edge = T_cell_trans * (D_t / B_up) + T_cell_recv * (R_t / B_down)
        T_edge = (D_t / B_up) + T_edge_proc
        costs['Edge'].append(mu * T_edge + (1 - mu) * E_edge)

        # D2D-边缘混合执行成本
        E_D2D_edge = (T_d2d_trans + T_cell_recv) * (D_t / B_d2d) + T_cell_trans * (D_t / B_up) + T_cell_recv * (
                    R_t / B_down)
        T_D2D_edge = (D_t / B_d2d) + (D_t / B_up) + T_edge_proc
        costs['D2D-Edge'].append(mu * T_D2D_edge + (1 - mu) * E_D2D_edge)

    # 绘制每个mu值的成本
    ax = axs[idx]
    for offloading_type, cost in costs.items():
        ax.plot(data_sizes, cost, label=f'{offloading_type} Cost')

    ax.set_title(f'Impact of Data Size on Offloading Cost (mu={mu})')
    ax.set_xlabel('Data Size (MB)')
    if idx == 0:
        ax.set_ylabel('Total Cost (Weighted Sum of Energy and Time)')
    ax.legend()
    ax.grid(True)

# 显示整个图表
plt.tight_layout()
plt.show()
