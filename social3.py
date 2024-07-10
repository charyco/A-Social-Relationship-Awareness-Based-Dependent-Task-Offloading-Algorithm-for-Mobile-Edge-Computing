import networkx as nx
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment


class Task:
    def __init__(self, task_id, computation_cost, data_size, social_group, dependencies=[], result_size=0):
        self.task_id = task_id
        self.computation_cost = computation_cost
        self.data_size = data_size
        self.social_group = social_group
        self.dependencies = dependencies
        self.result_size = result_size
        self.is_completed = False  # 默认任务未完成


class Device:
    def __init__(self, device_id, capacity, transmission_power, social_group,
                 d2d_transmit_power, d2d_receive_power, cellular_transmit_power,
                 cellular_receive_power, upload_bandwidth, download_bandwidth, d2d_bandwidth):
        self.device_id = device_id
        self.capacity = capacity
        self.transmission_power = transmission_power
        self.social_group = social_group
        self.d2d_transmit_power = d2d_transmit_power
        self.d2d_receive_power = d2d_receive_power
        self.cellular_transmit_power = cellular_transmit_power
        self.cellular_receive_power = cellular_receive_power
        self.upload_bandwidth = upload_bandwidth
        self.download_bandwidth = download_bandwidth
        self.d2d_bandwidth = d2d_bandwidth


class SocialGraph:
    def __init__(self):
        self.relationships = {}  # This now represents device-task type relationships

    def add_relationship(self, device_id, task_type, strength):
        self.relationships[(device_id, task_type)] = strength

    def get_strength(self, device_id, task_type):
        return self.relationships.get((device_id, task_type), 0)



class TaskDependencyGraph:
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_task(self, task):
        self.graph.add_node(task.task_id, task=task)

    def add_dependency(self, task_id, dependent_task_id):
        self.graph.add_edge(dependent_task_id, task_id)

    def display_graph(self):
        pos = nx.spring_layout(self.graph)
        labels = {node: f'Task {node}' for node in self.graph.nodes()}
        nx.draw(self.graph, pos, labels=labels, with_labels=True, node_color='skyblue')
        plt.show()

    def get_dependencies(self, task_id):
        return list(self.graph.predecessors(task_id))

    def is_executable(self, task_id):
        if not self.get_dependencies(task_id):  # 如果没有依赖项，立即可执行
            return True
        return all(self.graph.nodes[dep]['task'].is_completed for dep in self.get_dependencies(task_id))


class TaskScheduler:
    def __init__(self, tasks, devices, social_graph):
        self.tasks = tasks
        self.devices = devices
        self.social_graph = social_graph
        self.dependency_graph = TaskDependencyGraph()
        for task in self.tasks:
            self.dependency_graph.add_task(task)
            for dep in task.dependencies:
                self.dependency_graph.add_dependency(task.task_id, dep)

    def get_device_by_id(self, device_id):
        """通过设备ID获取设备对象"""
        return next((device for device in self.devices if device.device_id == device_id), None)

    def is_task_executable(self, task_id):
        """检查任务是否可以执行，即所有依赖任务都已完成"""
        task = self.get_task_by_id(task_id)
        if not task.dependencies:
            return True  # 如果没有依赖，任务可以立即执行
        return all(self.get_task_by_id(dep).is_completed for dep in task.dependencies)

    def get_task_by_id(self, task_id):
        """通过任务ID获取任务对象"""
        return next((task for task in self.tasks if task.task_id == task_id), None)

    def calculate_overhead_U_matrix(self):
        U_matrix = {}
        for task in self.tasks:
            if self.is_task_executable(task.task_id):
                U_matrix[task.task_id] = {}
                for device in self.devices:
                    _, U_value = self.calculate_overhead_U(task, device)
                    U_matrix[task.task_id][device.device_id] = U_value
        return U_matrix

    def calculate_overhead_U(self, task, device):
        overheads = {}
        mu = 0.5  # 假设时间和能耗的权重为0.5，可根据需求调整
        for offloading_type in ['local', 'D2D', 'edge', 'D2D-edge']:
            energy = self.calculate_energy_consumption(task, device, offloading_type)
            time = self.calculate_time_delay(task, device, offloading_type)
            if offloading_type in ['D2D', 'D2D-edge']:
                social_factor = self.calculate_social_factor(task, device)
            else:
                social_factor = 1  # 对于 'local' 和 'edge' 不考虑社会因素
            U = mu * time + (1 - mu) * energy * social_factor
            overheads[offloading_type] = U
        best_type = min(overheads, key=overheads.get)
        return best_type, overheads[best_type]

    def calculate_energy_consumption(self, task, device, offloading_type):
        if offloading_type == 'local':
            return device.transmission_power * task.computation_cost
        elif offloading_type == 'D2D':
            return (
                           device.d2d_transmit_power + device.d2d_receive_power) * task.data_size / device.d2d_bandwidth + device.transmission_power * task.computation_cost
        elif offloading_type == 'edge':
            return device.cellular_transmit_power * task.data_size / device.upload_bandwidth + device.cellular_receive_power * task.result_size / device.download_bandwidth
        elif offloading_type == 'D2D-edge':
            return (
                           device.d2d_transmit_power + device.cellular_receive_power) * task.data_size / device.d2d_bandwidth + device.cellular_transmit_power * task.data_size / device.upload_bandwidth + device.cellular_receive_power * task.result_size / device.download_bandwidth
        else:
            raise ValueError("Unsupported offloading type")

    def calculate_time_delay(self, task, device, offloading_type):
        edge_processing_time = 0.5  # 假设的云处理时间
        if offloading_type == 'local':
            return task.data_size / device.capacity
        elif offloading_type == 'D2D':
            return task.data_size / device.d2d_bandwidth + task.data_size / device.capacity
        elif offloading_type == 'edge':
            return task.data_size / device.upload_bandwidth + edge_processing_time
        elif offloading_type == 'D2D-edge':
            return task.data_size / device.d2d_bandwidth + task.data_size / device.upload_bandwidth + edge_processing_time
        else:
            raise ValueError("Unsupported offloading type")

    def calculate_social_factor(self, task, device):
        # 计算社会因素对能耗和时间的加权影响
        task_type = task.social_group  # 从任务对象中获取任务类型
        device_id = device.device_id  # 从设备对象中获取设备ID
        w_ij = self.social_graph.get_strength(device_id, task_type)
        alpha = 0.5  # 示例: 调节因子
        social_cost = 1 / (1 + alpha * w_ij)
        return social_cost

    def global_optimization(self, U_matrix):
        task_ids = list(U_matrix.keys())
        device_ids = [device.device_id for device in self.devices]
        cost_matrix = [[U_matrix[task_id][device_id] for device_id in device_ids] for task_id in task_ids]
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        return [(task_ids[row], device_ids[col]) for row, col in zip(row_indices, col_indices)]

    def optimize_task_offloading(self):
        """优化任务卸载，直到没有更多任务可以执行为止"""
        changes = True
        while changes:
            changes = False
            U_matrix = self.calculate_overhead_U_matrix()
            if not U_matrix:
                break  # 如果没有任务可以执行，退出循环

            assignments = self.global_optimization(U_matrix)
            for task_id, device_id in assignments:
                task = self.get_task_by_id(task_id)
                device = self.get_device_by_id(device_id)
                if not task.is_completed:  # 确保任务尚未完成
                    task.is_completed = True  # 标记任务为完成
                    changes = True  # 标记有任务完成，可能触发新的任务可以执行
                    offloading_type, _ = self.calculate_overhead_U(task, device)
            self.visualize_assignments(assignments)

    def visualize_assignments(self, assignments):
        G = nx.DiGraph()
        pos = {}
        labels = {}

        for task_id, device_id in assignments:
            G.add_node(task_id, label=f'Task {task_id}', color='lightblue')
            G.add_node(device_id, label=f'Device {device_id}', color='lightgreen')
            G.add_edge(task_id, device_id)
            pos[task_id] = (1, -int(task_id))
            pos[device_id] = (2, -int(device_id))
            labels[task_id] = f'Task {task_id}'
            labels[device_id] = f'Device {device_id}'

        colors = [G.nodes[n]['color'] for n in G.nodes()]

        nx.draw(G, pos, labels=labels, node_color=colors, with_labels=True, edge_color='black', width=2)
        plt.title('Task to Device Assignments')
        plt.show()


def initialize_system():
    tasks = [
        Task(1, 100, 1000, 'A', [], 500),
        Task(2, 150, 1200, 'B', [1], 600),
        Task(3, 200, 800, 'C', [2], 400),
        Task(4, 175, 950, 'A', [1, 3], 475),
        Task(5, 120, 700, 'B', [1, 2, 4], 350),
        Task(6, 130, 1100, 'C', [5], 300),
        Task(7, 160, 500, 'A', [6], 550),
        Task(8, 140, 1200, 'B', [6, 7], 600),
        Task(9, 180, 1000, 'C', [1, 5, 8], 500)
    ]
    tasks[0].is_completed = True
    tasks[1].is_completed = True
    tasks[8].is_completed = True
    devices = [
        Device(1, 500, 0.1, 'A', 0.05, 0.05, 0.2, 0.1, 500, 500, 10),
        Device(2, 1000, 0.2, 'B', 0.05, 0.05, 0.2, 0.1, 500, 500, 20),
        Device(3, 750, 0.15, 'C', 0.04, 0.04, 0.15, 0.12, 400, 400, 15),
        Device(4, 800, 0.12, 'A', 0.03, 0.03, 0.18, 0.11, 450, 450, 18),
        Device(5, 900, 0.13, 'B', 0.06, 0.06, 0.22, 0.12, 550, 550, 25)
    ]
    social_graph = SocialGraph()
    # Assuming 'A', 'B', 'C' are the identifiers for task types directly
    social_graph.add_relationship(1, 'A', 0.9)  # Device 1 has high affinity for Task type 'A'
    social_graph.add_relationship(2, 'B', 0.9)  # Device 2 for Task type 'B'
    social_graph.add_relationship(3, 'C', 0.9)  # Device 3 for Task type 'C'
    social_graph.add_relationship(4, 'A', 0.8)  # Device 4 also supports Task type 'A' well
    social_graph.add_relationship(5, 'B', 0.8)  # Device 5 supports Task type 'B'
    return tasks, devices, social_graph



if __name__ == '__main__':
    tasks, devices, social_graph = initialize_system()
    scheduler = TaskScheduler(tasks, devices, social_graph)
    scheduler.optimize_task_offloading()
