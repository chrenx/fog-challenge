import torch
import torch.nn as nn
import torch.optim as optim

class CNN1D(nn.Module):
    def __init__(self, seq_len, num_features):
        super(CNN1D, self).__init__()
        
        self.conv1 = nn.Conv1d(num_features, 32, kernel_size=11, padding=5)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 32, kernel_size=11, padding=5)
        self.bn2 = nn.BatchNorm1d(32)
        
        self.conv3 = nn.Conv1d(32, 64, kernel_size=11, padding=5)
        self.bn3 = nn.BatchNorm1d(64)
        self.conv4 = nn.Conv1d(64, 64, kernel_size=11, padding=5)
        self.bn4 = nn.BatchNorm1d(64)
        
        self.conv5 = nn.Conv1d(64, 128, kernel_size=11, padding=5)
        self.bn5 = nn.BatchNorm1d(128)
        self.conv6 = nn.Conv1d(128, 128, kernel_size=11, padding=5)
        self.bn6 = nn.BatchNorm1d(128)
        
        self.conv7 = nn.Conv1d(128, 256, kernel_size=11, padding=5)
        self.bn7 = nn.BatchNorm1d(256)
        self.conv8 = nn.Conv1d(256, 256, kernel_size=11, padding=5)
        self.bn8 = nn.BatchNorm1d(256)
        
        self.conv9 = nn.Conv1d(256, 512, kernel_size=11, padding=5)
        self.bn9 = nn.BatchNorm1d(512)
        self.conv10 = nn.Conv1d(512, 512, kernel_size=11, padding=5)
        self.bn10 = nn.BatchNorm1d(512)
        
        self.pool = nn.MaxPool1d(2)
        self.leaky_relu = nn.LeakyReLU()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(512 * (seq_len // 32), 21)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.leaky_relu(self.bn1(self.conv1(x)))
        x = self.leaky_relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        
        x = self.leaky_relu(self.bn3(self.conv3(x)))
        x = self.leaky_relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        
        x = self.leaky_relu(self.bn5(self.conv5(x)))
        x = self.leaky_relu(self.bn6(self.conv6(x)))
        x = self.pool(x)
        
        x = self.leaky_relu(self.bn7(self.conv7(x)))
        x = self.leaky_relu(self.bn8(self.conv8(x)))
        x = self.pool(x)
        
        x = self.leaky_relu(self.bn9(self.conv9(x)))
        x = self.leaky_relu(self.bn10(self.conv10(x)))
        x = self.pool(x)
        
        x = self.flatten(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

seq_len = 1024
num_features = 5  
model = CNN1D(seq_len, num_features)
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=3e-5, betas=(0.9, 0.999))

x = torch.randn(1, num_features, seq_len)  # Example input tensor
output = model(x)
print(output)
