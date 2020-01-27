import torch
import torch.nn.functional as F


class DQN(torch.nn.Module):
    """

    """

    def __init__(self, h, w, outputs):
        super(DQN, self).__init__()
        """
        half (same) zero padding
        unit strides
        """
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(32)

        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.bn2 = torch.nn.BatchNorm2d(64)

        """
        no zero padding
        non-unit strides
        """
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=5, stride=2)
        self.bn3 = torch.nn.BatchNorm2d(128)

        self.conv4 = torch.nn.Conv2d(128, 256, kernel_size=5, stride=2)
        self.bn4 = torch.nn.BatchNorm2d(256)

        self.conv5 = torch.nn.Conv2d(256, 512, kernel_size=5, stride=2)
        self.bn5 = torch.nn.BatchNorm2d(512)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        def pool2d_size_out(size, kernel_size=2, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))

        linear_input_size = convw * convh * 512

        self.do = torch.nn.Dropout2d(p=0.2)
        self.head1 = torch.nn.Linear(linear_input_size, 125)
        self.head2 = torch.nn.Linear(125, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.do(x)
        return self.head2(self.head1(x.view(x.size(0), -1)))
