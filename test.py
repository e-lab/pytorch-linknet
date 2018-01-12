import torch
from torch.autograd import Variable

class Test(object):
    def __init__(self, model, data_loader, criterion):
        super(Test, self).__init__()
        self.model = model
        self.data_loader = data_loader
        self.criterion = criterion


    def forward(self):
        self.model.eval()
        # TODO adjust learning rate

        total_loss = 0
        for i, (x, yt) in enumerate(self.data_loader):
            x = x.cuda(async=True)
            yt = yt.cuda(async=True)
            input_var = Variable(x, volatile=True)
            target_var = Variable(yt, volatile=True)

            # compute output
            output = self.model(input_var)
            loss = self.criterion(output, target_var)

            # measure accuracy and record loss
            total_loss += loss.data[0]

        return total_loss
