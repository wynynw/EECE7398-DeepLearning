import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import argparse
import cv2

def load():
    '''
    The fonction to get the CIFAR10 dataset and load them
    :return: train_loader: load the train data
             test_loader: load the test data
             classes: data labels
             transform: transorm matrix
    '''
    # Normalization for tensor
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                             download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=128,
                                               shuffle=True, num_workers=2)

    test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=128,
                                              shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return train_loader, test_loader, classes


class NerualNetwork(torch.nn.Module):
    def __init__(self):
        super(NerualNetwork, self).__init__()
        self.conv11 = nn.Conv2d(3, 32, 5)  # (28x28x32)
        self.conv12 = nn.Conv2d(32, 32, 1)
        self.conv13 = nn.Conv2d(32, 32, 1)
        self.mp1 = nn.MaxPool2d(3, stride=1)  # (26x26x32)
        # self.mp11 = nn.MaxPool2d(4, stride =2)
        self.bn2d1 = nn.BatchNorm2d(32)
        self.conv21 = nn.Conv2d(32, 64, 3)  # (24x24x64)
        self.conv22 = nn.Conv2d(64, 64, 1)
        self.conv23 = nn.Conv2d(64, 64, 1)
        self.mp2 = nn.MaxPool2d(4, stride=2)  # (10x10x16)
        self.bn2d2 = nn.BatchNorm2d(64)
        self.conv31 = nn.Conv2d(64, 128, 3)  # (8x8x128)
        self.conv32 = nn.Conv2d(128, 128, 1)
        self.conv33 = nn.Conv2d(128, 128, 1)
        self.bn2d3 = nn.BatchNorm2d(128)
        self.mp3 = nn.MaxPool2d(3, stride=2)  # (5x5z128)

        # self.conv31 = nn.Conv2d(16,16,3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(4 * 4 * 128, 1024)  # 5x5x16 from image mp2
        self.fc1_bn = nn.BatchNorm1d(1024)
        self.fc2 = nn.Linear(1024, 10)
        self.fc2_bn = nn.BatchNorm1d(10)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv11(x))
        self.conv1_output = x
        x = self.bn2d1(x)
        x = self.mp1(x)
        x = F.relu(self.conv21(x))
        x = self.bn2d2(x)
        x = self.mp2(x)
        x = F.relu(self.conv31(x))
        x = self.bn2d3(x)
        x = self.mp3(x)
        # x = self.conv31(x)
        x = x.view(-1, 4 * 4 * 128)
        x = self.fc1_bn(F.relu(self.fc1(x)))
        x = F.softmax(self.fc2(x), dim=-1)
        return x


def test(net, inputs, criterion=None):
    '''
    The fonction for data testing
    :param: net: network
            inputs: input data for testing
            criterion: default=None. The criterion for the network
    :return: acc: the accurary of inputing data
             loss: the loss of inputing data
    '''
    net.eval()
    correct = 0
    total = 0
    total_loss = 0
    with torch.no_grad():
        for data in inputs:
            images, labels = data[0], data[1]
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total_loss += criterion(outputs, labels)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    acc = format((100 * correct / total), '.4f')
    loss = format((total_loss / len(inputs)), '.4f')
    return acc, loss

def train(total_epoch=10):
    '''
    The fonction for data testing
    :param: total_epoch: the total num of epoches
    '''
    train_loader, test_loader, classes= load()
    net = NerualNetwork()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    # loop over the dataset multiple times
    best_test_acc = 0
    print('Loop\t\tTrain Loss\t\tTrain Acc %\t\tTest Loss\t\tTest Acc  %')
    for epoch in range(total_epoch):
        # get the inputs; data is a list of [inputs, labels]
        for i, data in enumerate(train_loader, 0):
            net.train()
            #inputs, labels = data[0].view(-1, np.array(data[0].size()[1:]).prod()),data[1]
            inputs = data[0]
            labels = data[1]
            optimizer.zero_grad()  # gradient set to zero
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward() #backword
            optimizer.step()
        # print statistics
        train_accuracy, train_loss = test(net, train_loader, criterion)
        test_accuracy, test_loss = test(net, test_loader,criterion)
        if epoch%(total_epoch/10) == 0 or epoch == total_epoch-1:
                if total_epoch<100 and epoch<10:
                    print(epoch + 1, '/', total_epoch, '\t\t', train_loss, '\t\t', train_accuracy,
                        '\t\t', test_loss, '\t\t', test_accuracy)
                else:
                    print(epoch + 1, '/', total_epoch, '\t', train_loss, '\t\t', train_accuracy,
                          '\t\t', test_loss, '\t\t', test_accuracy)
        if best_test_acc < float(test_accuracy):
            best_test_acc = float(test_accuracy)
            if not os.path.isdir('./model'):
                os.mkdir('./model')
            torch.save(net.state_dict(), "./model/model.ckpt")
    print('Model saved in file: ./model/model.ckpt')

def predict(net, inputs):
    '''
    The fonction for data testing
    :param: net: network
            inputs: input data for testing
    :return: predicted: the num of label in class
    '''
    net.eval()
    #net.conv11.register_forward_hook(get_activation('conv11'))
    with torch.no_grad():
        image = inputs.unsqueeze(0)
        output = net(image)
        _, predicted = torch.max(output.data, 1)
        predicted = predicted.int()
        return predicted,net.conv1_output
#Main
if __name__ == '__main__':
    # get the input command args
    parser = argparse.ArgumentParser(description='HW2')
    parser.add_argument('mode',type=str,
                        help='train or test')
    parser.add_argument('imgName',type=str,default='', nargs='?',
                        help='test file')
    arg = parser.parse_args()
    epoch = 20
    # for different args
    if arg.mode == 'train':
        train(epoch)
    elif arg.mode == 'test':
        net = NerualNetwork()
        transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        try:
            net.load_state_dict(torch.load('./model/model.ckpt'))
            print("Load network from ./model/model.ckpt")
        except Exception:
            print("You need to train a model first!")
        try:
            img = cv2.imread(arg.imgName)
            img = cv2.resize(img, (32, 32), )
        except Exception:
            print("File doesn't exist or wrong file")
        else:
            img = transform(img).float()
            label, conv1_output = predict(net, img)
            classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
            print("Prediction result: ", classes[label])
            conv1_output = conv1_output / 2 + 0.5
            for id, output in enumerate(conv1_output):
                fig, axs = plt.subplots(8, 4, figsize=(1, 1))
                fig.subplots_adjust(top=0.95, bottom=0.01, left=0.2, right=0.99, hspace=0.1, wspace=0.1)
                for i, ax in enumerate(axs.flatten()):
                    ax.axis('off')
                    ax.imshow(output[i], interpolation='hamming', cmap='gray')
                fig.savefig('CONV_rslt.png', dpi=500)
            plt.show()