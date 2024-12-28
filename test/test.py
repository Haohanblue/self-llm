import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# 定义通用神经网络框架
class SimpleNN(nn.Module):
    def __init__(self, input_dim, hidden_layers=[64, 32], activation_fn=nn.ReLU(), output_activation_fn=nn.Identity(), output_dim=1):
        super(SimpleNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.activation_fn = activation_fn
        self.output_activation_fn = output_activation_fn

        # 构建神经网络
        layers = []
        # 输入层到第一个隐藏层
        layers.append(nn.Linear(input_dim, hidden_layers[0]))
        layers.append(self.activation_fn)
        
        # 隐藏层
        for i in range(1, len(hidden_layers)):
            layers.append(nn.Linear(hidden_layers[i-1], hidden_layers[i]))
            layers.append(self.activation_fn)
        
        # 输出层
        layers.append(nn.Linear(hidden_layers[-1], output_dim))
        layers.append(self.output_activation_fn)

        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

# 通用训练函数
def train_model(model, train_loader, optimizer, criterion, epochs=100, device='cpu'):
    model.to(device)
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss / len(train_loader)}")

# 通用评估函数
def evaluate_model(model, test_loader, criterion, device='cpu'):
    model.to(device)
    model.eval()
    with torch.no_grad():
        total_loss = 0
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
    return total_loss / len(test_loader)

# 入口函数：用户可以传递自定义的网络配置和训练参数
def main(input_dim=1, hidden_layers=[64, 32], activation_fn=nn.ReLU(), output_activation_fn=nn.Identity(), 
         output_dim=1, learning_rate=0.001, batch_size=32, epochs=100, device='cpu'):
    # 创建一些模拟数据
    X_train = np.random.rand(100, input_dim).astype(np.float32)  # 输入数据：100个样本
    y_train = X_train * 2 + 1  # 标签数据：y = 2x + 1

    X_test = np.random.rand(20, input_dim).astype(np.float32)  # 测试数据
    y_test = X_test * 2 + 1  # 测试标签数据

    # 将数据转换为 PyTorch 的 Tensor 并创建 DataLoader
    train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
    test_dataset = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 获取设备（GPU/CPU）
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 初始化模型
    model = SimpleNN(input_dim=input_dim, hidden_layers=hidden_layers, activation_fn=activation_fn, 
                     output_activation_fn=output_activation_fn, output_dim=output_dim)
    
    # 设置优化器和损失函数
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()  # 默认使用均方误差作为损失函数（适用于回归任务）

    # 训练模型
    train_model(model, train_loader, optimizer, criterion, epochs=epochs, device=device)

    # 评估模型
    loss = evaluate_model(model, test_loader, criterion, device=device)
    print(f"Test Loss: {loss}")
    
    # 进行预测
    with torch.no_grad():
        predictions = model(torch.tensor(X_test).to(device))
        print("Predictions:", predictions.cpu().numpy())  # 移回CPU进行输出

if __name__ == "__main__":
    # 用户可以在这里指定自己的参数
    main(input_dim=1, 
         hidden_layers=[128, 64, 32],  # 3层隐藏层
         activation_fn=nn.ReLU(), 
         output_activation_fn=nn.Identity(),  # 输出层没有激活函数，适合回归任务
         output_dim=1,  # 输出一个数值
         learning_rate=0.001, 
         batch_size=32, 
         epochs=200, 
         device='cuda')  # 如果有GPU可以用'cuda'，否则使用'cpu'
