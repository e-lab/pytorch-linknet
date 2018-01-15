import torch
import visdom
from tqdm import trange
from torch.autograd import Variable

class Train():
    def __init__(self, model, data_loader, optimizer, criterion, lr, wd, visdom):
        super(Train, self).__init__()
        self.model = model
        self.data_loader = data_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.lr = lr
        self.wd = wd
        self.visdom = visdom

    def forward(self):
        self.model.train()
        # TODO adjust learning rate

        total_loss = 0
        pbar = trange(len(self.data_loader.dataset), desc='Training ')

        if self.visdom:
            vis = visdom.Visdom()

            loss_window = vis.line(X=torch.zeros((1,)).cpu(),
                    Y=torch.zeros((1)).cpu(),
                    opts=dict(xlabel='minibatches',
                    ylabel='Loss',
                    title='Training Loss',
                    legend=['Loss']))

        for batch_idx, (x, yt) in enumerate(self.data_loader):
            x = x.cuda(async=True)
            yt = yt.cuda(async=True)
            input_var = Variable(x, requires_grad=True)
            target_var = Variable(yt)

            # compute output
            y = self.model(input_var)
            loss = self.criterion(y, target_var)

            if self.visdom:
                vis.line(
                        X=torch.ones((1, 1)).cpu() * batch_idx,
                        Y=torch.Tensor([loss.data[0]]).unsqueeze(0).cpu(),
                        win=loss_window,
                        update='append')

            # measure accuracy and record loss
            total_loss += loss.data[0]

            # compute gradient and do SGD step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch_idx % 10 == 0:
                if (batch_idx*len(x) + 10*len(x)) <= len(self.data_loader.dataset):
                    pbar.update(10 * len(x))
                else:
                    pbar.update(len(self.data_loader.dataset) - batch_idx*len(x))

        pbar.close()

        return total_loss/len(self.data_loader.dataset)
