import os
from datetime import datetime
from sklearn.metrics import classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class BaseModel(nn.Module):
    def __init__(self):
        # use a simple CNN model
        super(BaseModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7) # flatten
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def load_mnist_dataset():
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    return train_dataset, test_dataset


def load_data_loader():
    train_dataset, test_dataset = load_mnist_dataset()

    train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1000, shuffle=False)
    return train_loader, test_loader


device = torch.device("mps")


def train_model(model, criterion, optimizer, train_loader, test_loader, num_epochs, model_save_path, device):
    for epoch in range(num_epochs):
        model.train()
        correct_train = 0
        total_train = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(output.data, 1)
            total_train += target.size(0)
            correct_train += (predicted == target).sum().item()
        
        train_accuracy = 100 * correct_train / total_train
        
        # evaluate on test data
        model.eval()
        correct_test = 0
        total_test = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = torch.max(output.data, 1)
                total_test += target.size(0)
                correct_test += (predicted == target).sum().item()
        
        test_accuracy = 100 * correct_test / total_test
        print(f'Epoch: {epoch}\ttrain_acc = {train_accuracy:.2f}%, test_acc = {test_accuracy:.2f}%')

    if not os.path.exists('models'):
        os.makedirs('models')

    # save model
    torch.save(model.state_dict(), model_save_path)
    print(f'Model saved to {model_save_path}')

    return model


def load_model(path, device):
    model = BaseModel()
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(torch.device('mps'))
    model.eval()
    return model


def test_model(model, test_loader, device):
    # evaluate on test data
    model.eval()
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            total_test += target.size(0)
            correct_test += (predicted == target).sum().item()
    
    test_accuracy = 100 * correct_test / total_test
    return test_accuracy


def main():
    torch.manual_seed(42)
    timestamp = datetime.today().strftime('%Y%m%d')
    # load data
    train_loader, test_loader = load_data_loader()

    # device
    device = torch.device('mps')

    # initialize model, loss function, and optimizer
    model = BaseModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model_save_path = f'models/model_{timestamp}.pth'
    NUM_EPOCHS = 10

    # train model
    model = train_model(model, criterion, optimizer, train_loader, test_loader, num_epochs=NUM_EPOCHS, model_save_path=model_save_path, device=device)

    # one last classification report
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            y_true.extend(target.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    print("Final classification report on test data:")
    print(classification_report(y_true, y_pred))


if __name__ == "__main__":
    main()
