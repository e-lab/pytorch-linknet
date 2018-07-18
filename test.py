import torch
import visdom
from tqdm import trange
from torch.autograd import Variable

class Test(object):
    def __init__(self, model, data_loader, criterion, metrics, batch_size, vis):
        super(Test, self).__init__()
        self.model = model
        self.data_loader = data_loader
        self.criterion = criterion
        self.metrics = metrics
        self.bs = batch_size
        self.vis = None

        if vis:
            self.vis = visdom.Visdom()

            self.loss_window = self.vis.line(X=torch.zeros((1,)).cpu(),
                    Y=torch.zeros((1)).cpu(),
                    opts=dict(xlabel='minibatches',
                    ylabel='Loss',
                    title='Validation Loss',
                    legend=['Loss']))

        self.iterations = 0

    def forward(self):
        self.model.eval()
        # TODO adjust learning rate

        total_loss = 0
        pbar = trange(len(self.data_loader.dataset), desc='Validation ')

        for batch_idx, (x, yt) in enumerate(self.data_loader):
            x = x.cuda(async=True)
            yt = yt.cuda(async=True).type(torch.cuda.LongTensor)
            input_var = Variable(x, requires_grad=False)
            target_var = Variable(yt, requires_grad=False)

            # compute output
            y = self.model(input_var)
            loss = self.criterion(y, target_var)

            # measure accuracy and record loss
            total_loss += loss.item()

            # calculate mIoU
            pred = y.data.cpu().numpy()
            gt = yt.cpu().numpy()
            self.metrics.update_matrix(gt, pred)

            if batch_idx % 10 == 0:
                # Update tqdm bar
                if (batch_idx*self.bs + 10*len(x)) <= len(self.data_loader.dataset):
                    pbar.update(10 * len(x))
                else:
                    pbar.update(len(self.data_loader.dataset) - int(batch_idx*self.bs))

            # Display plot using visdom
            if self.vis:
                self.vis.line(
                        X=torch.ones((1)).cpu() * self.iterations,
                        Y=loss.data.cpu(),
                        win=self.loss_window,
                        update='append')

            self.iterations += 1

        accuracy, avg_accuracy, IoU, mIoU, conf_mat = self.metrics.scores()
        self.metrics.reset()
        pbar.close()

        return (total_loss*self.bs/len(self.data_loader.dataset), accuracy, avg_accuracy, IoU, mIoU, conf_mat)
