import torch
import torch.nn as nn
import torch.nn.functional as F

class View(nn.Module):
    """
    For convenience so we can add in in nn.Sequential
    instead of doing it manually in forward()
    """
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

class LeNet5(nn.Module):
    """
    For SVHN/CIFAR experiments
    """
    def __init__(self, n_classes):
        super(LeNet5, self).__init__()
        self.n_classes = n_classes
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        # self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
        # self.conv4 = nn.Conv2d(64, 64, kernel_size=3)
        # self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # self.conv4_drop = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(64*6*6, 128)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        out = F.relu(F.max_pool2d(self.conv1(x), 2))
        # print('out;', out.shape)
        out = F.relu(F.max_pool2d(self.conv2(out), 2))
        activation = out
        #print('out;', out.shape)
        out = out.view(-1, 64*6*6)
        out = F.relu(self.fc1(out))
        out = F.dropout(out, training=self.training)
        out = self.fc2(out)
        return activation, out



class LeNet7_T(nn.Module):
    """
    For SVHN/MNIST experiments
    """
    def __init__(self, n_classes):
        super(LeNet7_T, self).__init__()
        self.n_classes = n_classes
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        # self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3)
        # self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # self.conv4_drop = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(64*4*4, 200)
        self.fc2 = nn.Linear(200, n_classes)

    def forward(self, x):
        out = F.relu((self.conv1(x)))
        # print('out;', out.shape)
        out = F.relu(F.max_pool2d(self.conv2(out), 2))
        out = F.relu((self.conv3(out)))
        out = F.relu(F.max_pool2d(self.conv4(out), 2))
        activation = out
        # print('out;', out.shape)
        out = out.view(-1, 64*4*4)
        out = F.relu(self.fc1(out))
        out = F.dropout(out, training=self.training)
        out = self.fc2(out)
        return activation, out


class LeNet7_S(nn.Module):
    """
    For SVHN/MNIST experiments
    """
    def __init__(self, n_classes):
        super(LeNet7_S, self).__init__()

        self.n_classes = n_classes
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3)
        # self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3)
        # self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # self.conv4_drop = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(128*4*4, 256)
        self.fc2 = nn.Linear(256, n_classes)

    def forward(self, x):
        out = F.relu((self.conv1(x)))
        out = F.relu(F.max_pool2d(self.conv2(out), 2))
        out = F.relu((self.conv3(out)))
        out = F.relu(F.max_pool2d(self.conv4(out), 2))
        activation = out
        # print('out;', out.shape)
        out = out.view(-1, 128*4*4)
        out = F.relu(self.fc1(out))
        out = F.dropout(out, training=self.training)
        out = self.fc2(out)
        return activation, out

class trojan_model(nn.Module):
    """
    For train trojan model
    """
    def __init__(self, n_classes):
        super(trojan_model, self).__init__()

        self.n_classes = n_classes
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3)
        # self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3)
        # self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        # self.conv4_drop = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(128*5*5, 256)
        self.fc2 = nn.Linear(256, n_classes)

    def forward(self, x):
        out = F.relu((self.conv1(x)))
        out = F.relu(F.max_pool2d(self.conv2(out), 2))
        activation1 = out
        out = F.relu((self.conv3(out)))
        activation2 = out
        out = F.relu(F.max_pool2d(self.conv4(out), 2))
        activation3 = out
        # print('out;', out.shape)
        out = out.view(-1, 128*5*5)
        out = F.relu(self.fc1(out))
        out = F.dropout(out, training=self.training)
        out = self.fc2(out)
        return activation1, activation2, activation3,  out

#
# if __name__ == '__main__':
#     import random
#     import sys
#     # from torchsummary import summary
#
#     random.seed(1234)  # torch transforms use this seed
#     torch.manual_seed(1234)
#     torch.cuda.manual_seed(1234)
#
#     ### LENET5
#     x = torch.FloatTensor(64, 3, 32, 32).uniform_(0, 1)
#     true_labels = torch.tensor([[2.], [3], [1], [8], [4]], requires_grad=True)
#     model = LeNet5(n_classes=10)
#     output, act = model(x)
#     print("\nOUTPUT SHAPE: ", output.shape)
#
#     # summary(model, input_size=(3,32,32))

