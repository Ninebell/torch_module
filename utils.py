from torch.utils.tensorboard import SummaryWriter
import tqdm
import numpy as np
import torch
import math


class TrainInfo:
    def __init__(self, path):
        self.save_path = path
        self.train_loss = math.inf
        self.validate_loss = math.inf
        self.train_acc = []
        self.validate_acc = []
        self.min_train_loss = math.inf
        self.min_validate_loss = math.inf
        self.max_train_acc = None
        self.max_validate_acc = None

    def set_train_loss(self, loss):
        self.train_loss = loss

    def set_validate_loss(self, loss):
        self.validate_loss = loss

    def set_train_acc(self, acc):
        self.train_acc = acc

    def set_validate_acc(self, acc):
        self.validate_acc = acc


class TorchBoard:
    def __init__(self, dir_path, comment):
        self.writer = SummaryWriter(log_dir=dir_path, comment=comment)

    def add_train_loss(self, value, n_iter):
        self.writer.add_scalar('Loss/train', value, n_iter)

    def add_train_acc(self, value, n_iter, name):
        self.writer.add_scalar('{}/train'.format(name), value, n_iter)

    def add_validate_loss(self, value, n_iter):
        self.writer.add_scalar('Loss/test', value, n_iter)

    def add_validate_acc(self, value, n_iter, name):
        self.writer.add_scalar('{}/test'.format(name), value, n_iter)


def train_model(epoches, model, loss, optim, train_loader, validate_loader, save_path=None, tag=None, checkpoint=None, accuracy=None):
    print()
    print("{0:^40s}".format('Train Information'))
    print('{0:^40s}'.format("{0:22s}: {1:10,d}".format('model # param', get_param_count(model))))
    print("{0:^40s}".format("{0:22s}: {1:10,d}".format('epoch', epoches)))
    print("{0:^40s}".format("{0:22s}: {1:10,d}".format('batch size', train_loader['conf']['batch'])))

    train_info = TrainInfo(save_path)

    if torch.cuda.is_available():
        print('make cuda')
        model = model.cuda()
    tb = TorchBoard(save_path, tag)
    for epoch in range(1, epoches+1):
        model.train()
        train_loss = 0
        validate_loss = 0
        if accuracy is not None:
            train_acc = [0 for _ in range(len(accuracy))]
        if accuracy is not None:
            validate_acc = [0 for _ in range(len(accuracy))]

        for iter, (x, y) in tqdm.tqdm(enumerate(train_loader['loader'](train_loader['conf']))):
            optim.zero_grad()
            result = model(x)
            iter_loss = loss(y, result)
            train_loss += iter_loss
            iter_loss.backward()
            optim.step()
            if accuracy is not None:
                for idx, acc_di in enumerate(accuracy):
                    metrics = acc_di['metrics']
                    acc = metrics(y, result)
                    train_acc[idx] += acc
            del iter_loss
            del result
        train_loss /= (iter+1)
        train_info.set_train_loss(train_loss)

        if accuracy is not None:
            train_acc = np.array(train_acc)/(iter+1)
            train_info.set_train_acc(train_acc)

        model.eval()
        with torch.no_grad():
            for iter, (x, y) in enumerate(validate_loader['loader'](validate_loader['conf'])):
                result = model(x)
                iter_loss = loss(y, result)
                validate_loss += iter_loss
                if accuracy is not None:
                    for idx, acc_di in enumerate(accuracy):
                        metrics = acc_di['metrics']
                        acc = metrics(y, result)
                        validate_acc[idx] += acc

                del iter_loss
                del result
        validate_loss /= (iter+1)
        train_info.set_validate_loss(validate_loss)
        if accuracy is not None:
            validate_acc = np.array(validate_acc)/(iter+1)
            train_info.set_validate_acc(validate_acc)

        if save_path is not None:
            tb.add_train_loss(train_loss, epoch)
            tb.add_validate_loss(validate_loss, epoch)

            if accuracy is not None:
                for idx, acc_di in enumerate(accuracy):
                    tb.add_validate_acc(validate_acc[idx], epoch, acc_di['name'])
                    tb.add_train_acc(train_acc[idx], epoch, acc_di['name'])

        if checkpoint is not None:
            checkpoint(model, train_info)


def get_param_count(net):
    total_params = sum(p.numel() for p in net.parameters())
    return total_params
