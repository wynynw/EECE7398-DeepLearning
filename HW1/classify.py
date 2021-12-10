import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import numpy as np
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
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=4,
                                               shuffle=True, num_workers=2)

    test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=4,
                                              shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return train_loader, test_loader, classes, transform


class NerualNetwork(nn.Module):  # inherit from torch.nn.Module
    def __init__(self):
        super(NerualNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(3*32*32, 128),
            nn.ReLU(),
            nn.Linear(128, 24),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(24, 10)
        )

    def forward(self, x):  # forward propagation
        x = self.layers(x)
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
            images, labels = data[0].view(-1, np.array(data[0].size()[1:]).prod()), data[1]
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
    train_loader, test_loader, classes, _= load()
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
            inputs, labels = data[0].view(-1, np.array(data[0].size()[1:]).prod()),data[1]
            optimizer.zero_grad()  # gradient set to zero
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward() #backword
            optimizer.step()
        # print statistics
        train_accuracy, train_loss = test(net, train_loader, criterion)
        test_accuracy, test_loss = test(net, test_loader,criterion)
        if epoch in range(0,9):
            print(epoch+1,'/',total_epoch,'\t\t',train_loss,'\t\t',train_accuracy,
            '\t\t',test_loss,'\t\t',test_accuracy)
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
    with torch.no_grad():
        image = inputs.view(-1, np.array(inputs.size()).prod())
        output = net(image)
        _, predicted = torch.max(output.data, 1)
        predicted = predicted.int()
        return predicted

#Main
if __name__ == '__main__':
    # get the input command args
    parser = argparse.ArgumentParser(description='HW1')
    parser.add_argument('mode',type=str,
                        help='train or test')
    parser.add_argument('imgName',type=str,default='', nargs='?',
                        help='test file')
    arg = parser.parse_args()
    epoch = 10

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
        except Exception:
            print("You need to train a model first!")
        try:
            img = cv2.imread(arg.imgName)
            img = cv2.resize(img, (32, 32), )
        except Exception:
            print("File doesn't exist or wrong file")
        else:
            img = transform(img).float()
            # prediction
            label = predict(net, img)
            classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
            print("Prediction result: ",classes[label])

