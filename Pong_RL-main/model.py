import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):

    def __init__(self, action_dim, observation_shape=(4, 84, 84), hidden_dim=512):
        super(Model, self).__init__()

        in_channels = observation_shape[0]

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1)

        conv_output_size = self.calculate_conv_output(observation_shape)
        print("Conv output size:", conv_output_size)

        self.fc1 = nn.Linear(conv_output_size, hidden_dim)
        self.output = nn.Linear(hidden_dim, action_dim)

        self.apply(self.weights_init)

    def forward(self, x):
        if x.dtype == torch.uint8 or x.max() > 1.0:
            x = x.float() / 255.0
        else:
            x = x.float()

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        output = self.output(x)
        return output

    def calculate_conv_output(self, observation_shape):
        x = torch.zeros(1, *observation_shape)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x.view(-1).shape[0]

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)