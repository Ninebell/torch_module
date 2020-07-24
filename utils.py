from torch.utils.tensorboard import SummaryWriter
import torch


class TorchBoard:
    def __init__(self, dir_path, comment):
        self.writer = SummaryWriter(log_dir=dir_path, comment=comment)

    def add_train_loss(self, value, n_iter):
        self.writer.add_scalar('Loss/train', value, n_iter)

    def add_train_acc(self, value, n_iter):
        self.writer.add_scalar('Acc/train', value, n_iter)

    def add_test_loss(self, value, n_iter):
        self.writer.add_scalar('Loss/test', value, n_iter)

    def add_test_acc(self, value, n_iter):
        self.writer.add_scalar('Acc/test', value, n_iter)


def train_model(epoches, model, loss, optim, train_loader, validate_loader, save_path, tag=None, checkpoint=None):
    tb = TorchBoard(save_path, tag)
    for epoch in range(1, epoches+1):
        train_loss = 0
        validate_loss = 0
        for iter, (x, y) in enumerate(train_loader):
            optim.zero_grad()
            result = model(x)
            iter_loss = loss(y, result)
            train_loss += iter_loss
            iter_loss.backward()
            optim.step()
        train_loss /= iter

        with torch.no_grade():
            for iter, (x, y) in enumerate(validate_loader):
                optim.zero_grad()
                result = model(x)
                iter_loss = loss(y, result)
                validate_loss += iter_loss
        validate_loss /= iter

        tb.add_train_loss(train_loss, epoch)
        tb.add_validate_loss(validate_loss, epoch)

        if checkpoint is not None:
            checkpoint(model, train_loss, validate_loss)


def get_param_count(net):
    total_params = sum(p.numel() for p in net.parameters())
    return total_params

