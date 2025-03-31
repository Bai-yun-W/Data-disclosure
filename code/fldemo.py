import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error

# 检查 GPU 可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 加载 CSV 数据
def load_data(csv_file):
    df = pd.read_csv(csv_file)
    # 只保留 VN5 列作为特征
    vn5_data = df["VN5"].values.reshape(-1, 1)
    
    # 归一化处理 VN5 数据
    scaler = MinMaxScaler(feature_range=(0, 1))
    vn5_data = scaler.fit_transform(vn5_data)
    
    return vn5_data, scaler

# 创建时间序列数据窗口
def create_sequences(data, seq_length):
    x = []
    y = []
    for i in range(len(data) - seq_length):
        x.append(data[i:i+seq_length])  # 当前时间步的过去 seq_length 数据
        y.append(data[i+seq_length])    # 预测下一个时间步的 VN5 值
    return np.array(x), np.array(y)

# 将数据转换为 PyTorch 数据加载器
def create_dataloader(x, y, batch_size=16):
    tensor_x = torch.tensor(x, dtype=torch.float32)
    tensor_y = torch.tensor(y, dtype=torch.float32).view(-1, 1)  # 目标值需要调整形状
    dataset = TensorDataset(tensor_x, tensor_y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 定义 GRU 模型
class GRUNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRUNet, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h, _ = self.gru(x)
        out = self.fc(h[:, -1, :])  # 使用最后时刻的隐藏状态
        return out

# 创建训练函数
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in dataloader:
        # 将数据移动到指定设备
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

# 创建评估函数
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for inputs, labels in dataloader:
            # 将数据移动到指定设备
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    return total_loss / len(dataloader)

# 模拟客户端训练
def client_train(model, data, labels, optimizer, criterion, device):
    model.train()
    data_loader = create_dataloader(data, labels, batch_size=16)
    running_loss = 0.0
    for inputs, labels in data_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(data_loader)

# 模拟客户端聚合
def aggregate_models(models, client_data_sizes):
    # 根据每个客户端的数据大小进行加权平均
    global_model = models[0]
    total_data_size = sum(client_data_sizes)
    
    with torch.no_grad():
        for param_name, param_tensor in global_model.state_dict().items():
            weighted_sum = sum([model.state_dict()[param_name] * (client_data_sizes[i] / total_data_size) for i, model in enumerate(models)])
            param_tensor.copy_(weighted_sum)


# 主函数
if __name__ == "__main__":
    # 1. 加载数据
    csv_file = 'data/4_mm.csv'  # 请替换为您的 CSV 文件路径
    vn5_data, scaler = load_data(csv_file)

    # 2. 创建时间序列窗口数据
    seq_length = 10  # 使用过去 10 个时间步的数据预测下一个时间步
    x, y = create_sequences(vn5_data, seq_length)

    # 3. 数据拆分
    split_index = int(len(x) * 0.8)  # 80% 的数据用于训练，20% 用于测试
    x_train, x_test = x[:split_index], x[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # 4. 模拟多个客户端（假设有 3 个客户端）
    num_clients = 1
    client_data = [x_train[i::num_clients] for i in range(num_clients)]
    client_labels = [y_train[i::num_clients] for i in range(num_clients)]

    # 5. 创建多个客户端的模型
    input_size = 1  # 每个时间步只有一个特征 VN5
    hidden_size = 64
    output_size = 1  # 预测下一个时间步的 VN5 值
    client_models = [GRUNet(input_size=input_size, hidden_size=hidden_size, output_size=output_size).to(device) for _ in range(num_clients)]
    client_optimizers = [torch.optim.Adam(model.parameters(), lr=0.001) for model in client_models]
    criterion = nn.MSELoss()

    # 计算每个客户端的数据大小
    client_data_sizes = [len(client_data[i]) for i in range(num_clients)]

    # 6. 客户端训练并聚合模型
    epochs = 10
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        for client_id in range(num_clients):
            print(f"Training client {client_id+1}...")
            loss = client_train(client_models[client_id], client_data[client_id], client_labels[client_id], client_optimizers[client_id], criterion, device)
            print(f"Client {client_id+1} Loss: {loss}")
        
        # 聚合客户端模型，传递数据大小列表
        aggregate_models(client_models, client_data_sizes)

    # 7. 使用全局模型进行评估
    model = client_models[0]  # 使用任何一个客户端的模型作为全局模型
    model.eval()
    predictions = []
    actuals = []
    with torch.no_grad():
        for inputs, labels in create_dataloader(x_test, y_test, batch_size=16):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs).squeeze()
            predictions.extend(outputs.cpu().numpy())
            actuals.extend(labels.cpu().numpy())

    # 将预测值和实际值从归一化范围还原到原始值
    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    actuals = scaler.inverse_transform(np.array(actuals).reshape(-1, 1))

    # 计算 MAE
    mae = mean_absolute_error(actuals, predictions)
    print(f"Mean Absolute Error (MAE): {mae}")

