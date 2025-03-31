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
def load_data(csv_files):
    IN11_data = []
    scalers = []
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        # 只保留 IN11 列作为特征
        IN11 = df["IN11"].values.reshape(-1, 1)
        
        # 归一化处理 IN11 数据
        scaler = MinMaxScaler(feature_range=(0, 1))
        IN11_scaled = scaler.fit_transform(IN11)
        
        IN11_data.append(IN11_scaled)
        scalers.append(scaler)
    
    # 合并所有 CSV 文件的 IN11 数据
    IN11_data = np.vstack(IN11_data)
    
    return IN11_data, scalers

# 创建时间序列数据窗口
def create_sequences(data, seq_length):
    x = []
    y = []
    for i in range(len(data) - seq_length):
        x.append(data[i:i+seq_length])  # 当前时间步的过去 seq_length 数据
        y.append(data[i+seq_length])    # 预测下一个时间步的 IN11 值
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

# 修改客户端训练函数，支持多轮训练
def client_train_multiple_rounds(model, data, labels, optimizer, criterion, device, rounds=5):
    for round_num in range(rounds):
        loss = client_train(model, data, labels, optimizer, criterion, device)
        print(f"Round {round_num+1}/{rounds}, Loss: {loss}")
    return loss

# 模拟客户端聚合
def aggregate_models(models, client_data_sizes):
    # 根据每个客户端的数据大小进行加权平均
    global_model = models[0]
    total_data_size = sum(client_data_sizes)
    
    with torch.no_grad():
        for param_name, param_tensor in global_model.state_dict().items():
            weighted_sum = sum([model.state_dict()[param_name] * (client_data_sizes[i] / total_data_size) for i, model in enumerate(models)])
            param_tensor.copy_(weighted_sum)

# 保存结果到 CSV 文件
def save_to_csv(filename, content):
    # 如果文件不存在，创建并写入标题
    try:
        df = pd.read_csv(filename)
    except FileNotFoundError:
        df = pd.DataFrame(columns=["Epoch", "Client", "Loss"])
        df.to_csv(filename, index=False)
    
    # 追加新行数据
    df = pd.read_csv(filename)
    df = df.append(content, ignore_index=True)
    df.to_csv(filename, index=False)

# 主函数
if __name__ == "__main__":
    # 1. 加载多个 CSV 文件的数据
    csv_files = ['data/4_mm.csv', 'data/5_mm.csv', 'data/6_mm.csv']  # 替换为您的多个 CSV 文件路径
    IN11_data, scalers = load_data(csv_files)

    # 2. 创建时间序列窗口数据
    seq_length = 15  # 使用过去 10 个时间步的数据预测下一个时间步
    x, y = create_sequences(IN11_data, seq_length)

    # 3. 数据拆分
    split_index = int(len(x) * 0.8)  # 80% 的数据用于训练，20% 用于测试
    x_train, x_test = x[:split_index], x[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]

    # 4. 模拟多个客户端
    num_clients = 10
    client_data = [x_train[i::num_clients] for i in range(num_clients)]
    client_labels = [y_train[i::num_clients] for i in range(num_clients)]

    # 5. 创建多个客户端的模型
    input_size = 1  # 每个时间步只有一个特征 IN11
    hidden_size = 64
    output_size = 1  # 预测下一个时间步的 IN11 值
    client_models = [GRUNet(input_size=input_size, hidden_size=hidden_size, output_size=output_size).to(device) for _ in range(num_clients)]
    client_optimizers = [torch.optim.Adam(model.parameters(), lr=0.001) for model in client_models]
    criterion = nn.MSELoss()

    # 计算每个客户端的数据大小
    client_data_sizes = [len(client_data[i]) for i in range(num_clients)]

    # 6. 客户端训练并聚合模型
    epochs = 20
    rounds_per_client = 5  # 每个客户端的训练轮次
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        save_to_csv('result/training_results_3dataset.csv', [{"Epoch": epoch+1, "Client": "All", "Loss": "Start"}])

        for client_id in range(num_clients):
            print(f"Training client {client_id+1}...")
            loss = client_train_multiple_rounds(
                client_models[client_id],
                client_data[client_id],
                client_labels[client_id],
                client_optimizers[client_id],
                criterion,
                device,
                rounds=rounds_per_client
            )
            print(f"Client {client_id+1} Final Loss: {loss}")
            save_to_csv('result/training_results_3dataset.csv', [{"Epoch": epoch+1, "Client": client_id+1, "Loss": loss}])

        # 聚合客户端模型
        aggregate_models(client_models, client_data_sizes)

        # 7. 计算并保存全局模型的 MAE
        model = client_models[9]  # 使用第一个客户端的模型作为全局模型
        model.eval()
        epoch_predictions = []
        epoch_actuals = []

        with torch.no_grad():
            for inputs, labels in create_dataloader(x_test, y_test, batch_size=16):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs).squeeze()
                epoch_predictions.extend(outputs.cpu().numpy())
                epoch_actuals.extend(labels.cpu().numpy())

        # 将预测值和实际值从归一化范围还原到原始值
        epoch_predictions = scalers[0].inverse_transform(np.array(epoch_predictions).reshape(-1, 1))
        epoch_actuals = scalers[0].inverse_transform(np.array(epoch_actuals).reshape(-1, 1))

        # 计算 MAE
        mae = mean_absolute_error(epoch_actuals, epoch_predictions)
        print(f"Epoch {epoch+1} Mean Absolute Error (MAE): {mae}")
        save_to_csv('result/training_results_3dataset.csv', [{"Epoch": epoch+1, "Client": "All", "Loss": mae}])