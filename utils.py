from torch.utils.tensorboard import SummaryWriter
import tqdm
import torch



class TorchBoard:
    def __init__(self, dir_path, comment):
        self.writer = SummaryWriter(log_dir=dir_path, comment=comment)

    def add_train_loss(self, value, n_iter):
        self.writer.add_scalar('Loss/train', value, n_iter)

    def add_train_acc(self, value, n_iter):
        self.writer.add_scalar('Acc/train', value, n_iter)

    def add_validate_loss(self, value, n_iter):
        self.writer.add_scalar('Loss/test', value, n_iter)

    def add_validate_acc(self, value, n_iter):
        self.writer.add_scalar('Acc/test', value, n_iter)


def train_model(epoches, model, loss, optim, train_loader, validate_loader, save_path=None, tag=None, checkpoint=None, accuracy=None):
    tb = TorchBoard(save_path, tag)
    for epoch in range(1, epoches+1):
        train_loss = 0
        validate_loss = 0
        validate_acc = 0
        for iter, (x, y) in tqdm.tqdm(enumerate(train_loader['loader'](train_loader['conf']))):
            optim.zero_grad()
            result = model(x)
            iter_loss = loss(y, result)
            train_loss += iter_loss
            iter_loss.backward()
            optim.step()
            del iter_loss
            del result
        train_loss /= iter

        with torch.no_grad():
            for iter, (x, y) in enumerate(validate_loader['loader'](validate_loader['conf'])):
                result = model(x)
                iter_loss = loss(y, result)
                validate_loss += iter_loss
                if accuracy is not None:
                    acc = accuracy(y, result)
                    validate_acc += acc
                del iter_loss
                del result
        validate_loss /= iter
        validate_acc /= iter

        if save_path is not None:
            tb.add_train_loss(train_loss, epoch)
            tb.add_validate_loss(validate_loss, epoch)
            tb.add_validate_acc(validate_acc, epoch)

        if checkpoint is not None:
            checkpoint(model, train_loss, validate_loss)


def get_param_count(net):
    total_params = sum(p.numel() for p in net.parameters())
    return total_params

