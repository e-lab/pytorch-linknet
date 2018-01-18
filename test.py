import torch
import visdom
from tqdm import trange
from torch.autograd import Variable

class Test(object):
    def __init__(self, model, data_loader, criterion, metrics, vis):
        super(Test, self).__init__()
        self.model = model
        self.data_loader = data_loader
        self.criterion = criterion
        self.metrics = metrics
        self.vis = None

    def forward(self):
        self.model.eval()
        # TODO adjust learning rate

        total_loss = 0
        pbar = trange(len(self.data_loader.dataset), desc='Validation ')

        if self.visdom:
            vis = visdom.Visdom()

            loss_window = vis.line(X=torch.zeros((1,)).cpu(),
                    Y=torch.zeros((1)).cpu(),
                    opts=dict(xlabel='minibatches',
                    ylabel='Loss',
                    title='Validation Loss',
                    legend=['Loss']))

    def forward(self):
        self.model.eval()
        # TODO adjust learning rate

        total_loss = 0
        pbar = trange(len(self.data_loader.dataset), desc='Validation ')

        for batch_idx, (x, yt) in enumerate(self.data_loader):
            x = x.cuda(async=True)
            yt = yt.cuda(async=True)
            input_var = Variable(x, requires_grad=False)
            target_var = Variable(yt, requires_grad=False)

            # compute output
            y = self.model(input_var)
            loss = self.criterion(y, target_var)

            # measure accuracy and record loss
            total_loss += loss.data[0]

            # calculate mIoU
            pred = y.data.max(1)[1].cpu().numpy()
            gt = yt.cpu().numpy()
            self.metrics.update(gt, pred)

            if self.visdom:
                vis.line(
                        X=torch.ones((1, 1)).cpu() * batch_idx,
                        Y=torch.Tensor([loss.data[0]]).unsqueeze(0).cpu(),
                        win=loss_window,
                        update='append')

            if batch_idx % 10 == 0:
                if (batch_idx*len(x) + 10*(len(x))) <= len(self.data_loader.dataset):
                    pbar.update(10 * len(x))
                else:
                    pbar.update(len(self.data_loader.dataset) - batch_idx*len(x))

        score, conf_matrix, class_iou = self.metrics.get_scores()
        self.metrics.reset()
        pbar.close()

        return (total_loss/len(self.data_loader.dataset), score, conf_matrix, class_iou)
