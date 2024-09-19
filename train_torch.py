import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms


# PyTorchモデル定義
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 8 * 8, 64)
        self.fc2 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# モデルのトレーニング
def train_model():
    # Transformations for CIFAR-10 dataset
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    # CIFAR-10データセットのダウンロードと準備
    trainset = datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=64, shuffle=True, num_workers=2
    )

    testset = datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=64, shuffle=False, num_workers=2
    )

    # デバイスの設定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # モデルのインスタンス化
    model = CNNModel().to(device)

    # 損失関数とオプティマイザー
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # MLflowのExperiment設定
    mlflow.set_experiment("pytorch_cifar10_experiment")

    with mlflow.start_run():
        # パラメータをログ
        mlflow.log_param("optimizer", "adam")
        mlflow.log_param("epochs", 10)
        mlflow.log_param("batch_size", 64)

        # トレーニング
        num_epochs = 10
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)

                # 勾配の初期化
                optimizer.zero_grad()

                # 順伝播 + 損失計算 + 逆伝播 + 最適化
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 100 == 99:  # ログ
                    print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}")
                    running_loss = 0.0

            # エポックごとにメトリクスをログ
            mlflow.log_metric("loss", loss.item(), step=epoch)

        # モデルの評価
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f"Accuracy of the network on the test images: {accuracy:.2f}%")

        # 精度をログ
        mlflow.log_metric("accuracy", accuracy)

        # モデルをMLflowに保存
        mlflow.pytorch.log_model(model, "model")


if __name__ == "__main__":
    mlflow.set_tracking_uri("http://127.0.0.1:5012")
    train_model()
