import torch
import torch.optim as optim

from train_config import TrainConfig

class TextToSpeechTrainer:
    def __init__(self, model, train_loader, val_loader, train_config: TrainConfig):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.learning_rate = train_config.learning_rate
        self.num_epochs = train_config.num_epochs

        # 定义优化器和损失函数
        self.optimizer = optim.Adam(model.parameters(), lr=train_config.learning_rate)
        # self.criterion = nn.MSELoss()  # 可根据任务选择合适的损失函数
        print(f"初始化完成 TextToSpeechTrainer model {self.model} train_loader {self.train_loader} val_loader {self.val_loader} learning_rate {self.learning_rate} num_epochs {self.num_epochs}")

    def train(self):
        print("开始训练")
        for epoch in range(self.num_epochs):
            print(f"训练 epoch={epoch}")
            self.model.train()
            total_loss = 0.0

            for inputs, targets in self.train_loader:
                print(f"训练 epoch={epoch} 归一化")
                self.optimizer.zero_grad()
                print(f"训练 epoch={epoch} 模型执行")
                outputs = self.model(inputs)
                print(f"训练 epoch={epoch} loss计算")
                loss = self.model.criterion(outputs, targets)
                print(f"训练 epoch={epoch} loss回归")
                loss.backward()
                print(f"训练 epoch={epoch} 优化器执行")
                self.optimizer.step()
                print(f"训练 epoch={epoch} 损失汇总")
                total_loss += loss.item()

            print(f"Epoch {epoch + 1}/{self.num_epochs}, Loss: {total_loss / len(self.train_loader)}")

    def evaluate(self):
        if self.val_loader is None:
            print("No validation set provided.")
            return

        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for inputs, targets in self.val_loader:
                outputs = self.model(inputs)
                loss = self.model.criterion(outputs, targets)
                total_loss += loss.item()

        print(f"Validation Loss: {total_loss / len(self.val_loader)}")

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))