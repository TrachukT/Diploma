import torch.nn as nn
import torch
import torch.nn.functional as F

class ConvNeuralNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvNeuralNet, self).__init__()
        dropout_rate = 0.3  # Reduced dropout for better feature retention
        
        # First block with more initial filters
        self.conv_layer1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv_layer2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.max_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout1 = nn.Dropout2d(dropout_rate)

        # Second block
        self.conv_layer3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv_layer4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.max_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout2 = nn.Dropout2d(dropout_rate)

        # Third block with residual connection
        self.conv_layer5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv_layer6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.max_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout3 = nn.Dropout2d(dropout_rate)

        # Global pooling
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        
        # Two parallel FC paths
        # Path 1
        self.fc1_1 = nn.Linear(256 * 4 * 4, 512)
        self.bn_fc1_1 = nn.BatchNorm1d(512)
        self.dropout_fc1 = nn.Dropout(0.4)
        
        # Path 2
        self.fc1_2 = nn.Linear(256 * 4 * 4, 512)
        self.bn_fc1_2 = nn.BatchNorm1d(512)
        self.dropout_fc2 = nn.Dropout(0.4)
        
        # Combine paths
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        # First block
        out = self.conv_layer1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv_layer2(out)
        out = self.bn2(out)
        out = F.relu(out)
        identity1 = out
        out = self.max_pool1(out)
        out = self.dropout1(out)

        # Second block
        out = self.conv_layer3(out)
        out = self.bn3(out)
        out = F.relu(out)
        out = self.conv_layer4(out)
        out = self.bn4(out)
        out = F.relu(out)
        out = self.max_pool2(out)
        out = self.dropout2(out)

        # Third block
        out = self.conv_layer5(out)
        out = self.bn5(out)
        out = F.relu(out)
        out = self.conv_layer6(out)
        out = self.bn6(out)
        out = F.relu(out)
        out = self.max_pool3(out)
        out = self.dropout3(out)

        # Global pooling
        out = self.adaptive_pool(out)
        out = out.view(out.size(0), -1)
        
        # Parallel FC paths
        out1 = self.fc1_1(out)
        out1 = self.bn_fc1_1(out1)
        out1 = F.relu(out1)
        out1 = self.dropout_fc1(out1)
        
        out2 = self.fc1_2(out)
        out2 = self.bn_fc1_2(out2)
        out2 = F.relu(out2)
        out2 = self.dropout_fc2(out2)
        
        # Combine paths
        out = torch.cat((out1, out2), dim=1)
        out = self.fc2(out)
        return out