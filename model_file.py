from torch import nn
import torch

class MNIST_3D(nn.Module):
    def __init__(self, num_classes):
        super(MNIST_3D, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv3d(in_channels=3, out_channels=16, kernel_size=(3,3,3), stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm3d(16)
        self.average_pool1 = nn.AvgPool3d(kernel_size=(3,3,3), stride=1)
        self.dropout1 = nn.Dropout3d(0.3)

        self.conv2 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(3,3,3), stride=1, padding=0)
        self.batch_norm2 = nn.BatchNorm3d(32)
        self.dropout2 = nn.Dropout3d(0.3)

        self.conv3 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(3,3,3), stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm3d(64)
        self.average_pool3 = nn.AvgPool3d(kernel_size=(3,3,3), stride=1)
        self.dropout3 = nn.Dropout3d(0.3)

        self.conv4 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(3,3,3), stride=1, padding=0)
        self.batch_norm4 = nn.BatchNorm3d(64)
        self.dropout4 = nn.Dropout3d(0.3)

        self.adaptive_pool = nn.AdaptiveAvgPool3d((1,1,1))
        

        self.flatten = nn.Flatten()

        self.linear1 = nn.Linear(in_features=64, out_features=512)
        self.dropout5 = nn.Dropout(0.3)
        self.linear2 = nn.Linear(in_features=512, out_features=128)
        self.dropout6 = nn.Dropout(0.3)
        self.linear3 = nn.Linear(in_features=128, out_features=self.num_classes)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.dropout1(self.batch_norm1(self.relu(self.conv1(x))))
        x = self.average_pool1(x)
        x = self.dropout2(self.batch_norm2(self.relu(self.conv2(x))))
        x = self.dropout3(self.batch_norm3(self.relu(self.conv3(x))))
        x = self.average_pool3(x)
        x = self.dropout4(self.batch_norm4(self.relu(self.conv4(x))))
        x = self.adaptive_pool(x)
        x_flatten = self.flatten(x)
        x = self.relu(self.dropout5(self.linear1(x_flatten)))
        x = self.relu(self.dropout6(self.linear2(x)))
        x = self.linear3(x)


        return x
    

def main():
    data = torch.randn(32, 3, 16, 16, 16)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    obj = MNIST_3D(10).to(device)
    data = data.to(device)
    out = obj(data)
    print(out.shape)

if __name__ == "__main__":
    main()