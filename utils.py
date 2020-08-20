from torch.utils.tensorboard import SummaryWriter
import tqdm
import numpy as np
import torch


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
    print('{:,}'.format(get_param_count(model)))
    if torch.cuda.is_available():
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
                    acc = metrics(y, result, 0, x)
                    train_acc[idx] += acc
            del iter_loss
            del result
        train_loss /= (iter+1)
        train_acc = np.array(train_acc)/(iter+1)

        model.eval()
        with torch.no_grad():
            for iter, (x, y) in enumerate(validate_loader['loader'](validate_loader['conf'])):
                result = model(x)
                iter_loss = loss(y, result)
                validate_loss += iter_loss
                if accuracy is not None:
                    for idx, acc_di in enumerate(accuracy):
                        metrics = acc_di['metrics']
                        acc = metrics(y, result, 1, x)
                        validate_acc[idx] += acc

                del iter_loss
                del result
        validate_loss /= (iter+1)
        validate_acc = np.array(validate_acc)/(iter+1)

        if save_path is not None:
            tb.add_train_loss(train_loss, epoch)
            tb.add_validate_loss(validate_loss, epoch)

            if accuracy is not None:
                for idx, acc_di in enumerate(accuracy):
                    tb.add_validate_acc(validate_acc[idx], epoch, acc_di['name'])
                    tb.add_train_acc(train_acc[idx], epoch, acc_di['name'])

        if checkpoint is not None:
            checkpoint(model, train_acc[1], validate_acc[1])


def get_param_count(net):
    total_params = sum(p.numel() for p in net.parameters())
    return total_params


