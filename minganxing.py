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
        self.is_completed = False
        self.device_id = None

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
        self.cost_data = {'local': [], 'D2D': [], 'edge': [], 'D2D-edge': []}  # 初始化存储成本数据的字典


    def get_device_by_id(self, device_id):
        """通过设备ID获取设备对象"""
        return next((device for device in self.devices if device.device_id == device_id), None)

    def is_task_executable(self, task_id):
        task = self.get_task_by_id(task_id)
        if not task.dependencies:
            return True  # If there are no dependencies, the task can be executed immediately
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
        if task.data_size > device.capacity:
            print(
                f"Task {task.task_id} with data size {task.data_size} exceeds Device {device.device_id} capacity {device.capacity}. Returning inf.")
            return None, float('inf')  # 如果任务数据大小超过设备容量，则返回无穷大

        overheads = {}
        mu = 0.1  # 时间和能耗的权重因子
        for offloading_type in ['local', 'D2D', 'edge', 'D2D-edge']:
            energy = self.calculate_energy_consumption(task, device, offloading_type)
            time = self.calculate_time_delay(task, device, offloading_type)
            social_factor = self.calculate_social_factor(task, device) if offloading_type in ['D2D', 'D2D-edge'] else 1
            U = mu * time + (1 - mu) * energy * social_factor
            overheads[offloading_type] = U
            print(f"Offloading {offloading_type}: Energy={energy}, Time={time}, Social={social_factor}, U={U}")

        best_type = min(overheads, key=overheads.get)
        best_cost = overheads[best_type]
        print(f"Best offloading option: {best_type} with cost {best_cost}")
        return best_type, best_cost

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
        changes = True
        while changes:
            changes = False
            U_matrix = self.calculate_overhead_U_matrix()
            if not U_matrix:
                break  # 如果没有可执行的任务，则中断循环

            assignments = self.global_optimization(U_matrix)
            for task_id, device_id in assignments:
                task = self.get_task_by_id(task_id)
                device = self.get_device_by_id(device_id)
                if not task.is_completed:
                    task.is_completed = True
                    task.device_id = device_id  # 确保分配了设备ID
                    changes = True
                    _, _ = self.calculate_overhead_U(task, device)

    def sensitivity_analysis_data_size(self, task_id, data_sizes):
        original_data_size = self.get_task_by_id(task_id).data_size
        for size in data_sizes:
            self.get_task_by_id(task_id).data_size = size
            self.optimize_task_offloading()
            task = self.get_task_by_id(task_id)
            if task.device_id is not None:
                device = self.get_device_by_id(task.device_id)
                offloading_type, cost = self.calculate_overhead_U(task, device)
                self.cost_data[offloading_type].append((size, cost))
                print(f"Data size {size} results in {offloading_type} with cost {cost}")
            else:
                # 如果没有设备可以处理任务，确保记录这一信息
                for key in self.cost_data.keys():
                    self.cost_data[key].append((size, float('inf')))
                    print(f"No feasible offloading found for data size {size}")
        self.get_task_by_id(task_id).data_size = original_data_size


def initialize_system():
    tasks = [
        Task(1, 100, 500, 'A'),
        Task(2, 150, 600, 'B', [1]),
        Task(3, 200, 400, 'C'),
        Task(4, 175, 450, 'A', [1, 3])
    ]
    tasks[0].is_completed = True  # 假设任务 1 已完成

    devices = [
        Device(1, 5000, 0.1, 'A', 0.05, 0.05, 0.2, 0.1, 1000, 1000, 20),
        Device(2, 10000, 0.2, 'B', 0.05, 0.05, 0.2, 0.1, 1000, 1000, 20),
        Device(3, 7500, 0.15, 'C', 0.04, 0.04, 0.15, 0.12, 800, 800, 30),
        Device(4, 10000, 0.1, 'A', 0.05, 0.05, 0.2, 0.1, 2000, 2000, 50)  # 增加一个高容量设备
    ]
    social_graph = SocialGraph()
    # Assuming 'A', 'B', 'C' are the identifiers for task types directly
    social_graph.add_relationship(1, 'A', 0.9)
    social_graph.add_relationship(2, 'B', 0.9)
    social_graph.add_relationship(3, 'C', 0.9)
    social_graph.add_relationship(4, 'A', 0.8)
    return tasks, devices, social_graph





if __name__ == '__main__':
    tasks, devices, social_graph = initialize_system()
    scheduler = TaskScheduler(tasks, devices, social_graph)
    data_sizes = range(500, 2500, 200)  # 定义数据大小范围
    scheduler.sensitivity_analysis_data_size(1, data_sizes)  # 进行敏感性分析

    print("Cost data collected:", scheduler.cost_data)

    plt.figure(figsize=(10, 6))
    for offload_type, data in scheduler.cost_data.items():
        if data:
            sizes, costs = zip(*data)
            plt.plot(sizes, costs, marker='o', label=offload_type)
    plt.title('Cost vs Data Size for Different Offloading Types')
    plt.xlabel('Data Size (MB)')
    plt.ylabel('Cost')
    plt.legend()
    plt.grid(True)
    plt.show()

