import numpy as np
import matplotlib.pyplot as plt

'''
# 场景 2: 物联网设备（如传感器网络）
P_trans = 0.01
P_d = 50
B_d2d = 1
B_up = 2
B_down = 2
T_d2d_trans = 0.02
T_d2d_recv = 0.02
T_cell_trans = 0.1
T_cell_recv = 0.05
T_edge_proc = 0.1
C_t = 10
R_t = 50
mu = 0.2  # 能耗是关键考虑因素


# 场景 3: 数据中心或服务器
P_trans = 0.5
P_d = 1000
B_d2d = 100
B_up = 100
B_down = 100
T_d2d_trans = 0.5
T_d2d_recv = 0.5
T_cell_trans = 0.5
T_cell_recv = 0.5
T_edge_proc = 0.05
C_t = 1000
R_t = 5000
mu = 0.9  # 时延是主要关注点
'''

# 场景 4: 智能车辆
P_trans = 0.3
P_d = 800
B_d2d = 50
B_up = 50
B_down = 50
T_d2d_trans = 0.1
T_d2d_recv = 0.1
T_cell_trans = 0.3
T_cell_recv = 0.2
T_edge_proc = 0.2
C_t = 500
R_t = 2500
mu = 0.7  # 响应时间和能耗都很重要

# 数据大小范围
data_sizes = np.linspace(200, 2000, 10)

# 计算各种卸载策略的成本
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

# 绘图
plt.figure(figsize=(10, 6))
for offloading_type, cost in costs.items():
    plt.plot(data_sizes, cost, label=f'{offloading_type} Cost')

plt.title('Impact of Data Size on Offloading Cost')
plt.xlabel('Data Size (MB)')
plt.ylabel('Total Cost (Weighted Sum of Energy and Time)')
plt.legend()
plt.grid(True)
plt.show()
