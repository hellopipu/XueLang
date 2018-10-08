########################################################
#### TEAM: VGG19 , MEMBER:Bingyu Xin                ####
########################################################
import numpy as np
## rates  ------------------------------
def get_learning_rate(optimizer):
    lr=[]
    for param_group in optimizer.param_groups:
       lr +=[ param_group['lr'] ]
    return lr


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

## simple stepping rates
class StepLR():
    def __init__(self, pairs):
        super(StepLR, self).__init__()

        N = len(pairs)
        rates = []
        steps = []
        for n in range(N):
            s, r = pairs[n]
            if r < 0: s = s + 1
            steps.append(s)
            rates.append(r)

        self.rates = rates
        self.steps = steps

    def get_rate(self, epoch=None):

        N = len(self.steps)
        lr = -1
        for n in range(N):
            if epoch >= self.steps[n]:
                lr = self.rates[n]
        return lr

    def __str__(self):
        string = 'Step Learning Rates\n' \
                 + 'rates=' + str(['%7.4f' % i for i in self.rates]) + '\n' \
                 + 'steps=' + str(['%7.0f' % i for i in self.steps]) + ''
        return string


## https://github.com/pytorch/tutorials/blob/master/beginner_source/transfer_learning_tutorial.py
class DecayLR():
    def __init__(self, base_lr, decay, step):
        super(DecayLR, self).__init__()
        self.step = step
        self.decay = decay
        self.base_lr = base_lr

    def get_rate(self, epoch=None, num_epoches=None):
        lr = self.base_lr * (self.decay ** (epoch // self.step))
        return lr

    def __str__(self):
        string = '(Exp) Decay Learning Rates\n' \
                 + 'base_lr=%0.3f, decay=%0.3f, step=%0.3f' % (self.base_lr, self.decay, self.step)
        return string


# 'Cyclical Learning Rates for Training Neural Networks'- Leslie N. Smith, arxiv 2017
#       https://arxiv.org/abs/1506.01186
#       https://github.com/bckenstler/CLR

class CyclicLR():

    def __init__(self, base_lr=0.001, max_lr=0.006, step=2000., mode='triangular',
                 gamma=1., scale_fn=None, scale_mode='cycle'):
        super(CyclicLR, self).__init__()

        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step = step
        self.mode = mode
        self.gamma = gamma
        if scale_fn == None:
            if self.mode == 'triangular':
                self.scale_fn = lambda x: 1.
                self.scale_mode = 'cycle'
            elif self.mode == 'triangular2':
                self.scale_fn = lambda x: (0.5) ** (x - 1)
                self.scale_mode = 'cycle'
            elif self.mode == 'exp_range':
                self.scale_fn = lambda x: gamma ** (x)
                self.scale_mode = 'iterations'
        else:
            self.scale_fn = scale_fn
            self.scale_mode = scale_mode
        self.clr_iterations = 0.
        self.trn_iterations = 0.
        self.history = {}

        self._reset()

    def _reset(self, new_base_lr=None, new_max_lr=None,
               new_step=None):
        """Resets cycle iterations.
        Optional boundary/step size adjustment.
        """
        if new_base_lr != None:
            self.base_lr = new_base_lr
        if new_max_lr != None:
            self.max_lr = new_max_lr
        if new_step != None:
            self.step = new_step
        self.clr_iterations = 0.

    def clr(self):
        cycle = np.floor(1 + self.clr_iterations / (2 * self.step))
        x = np.abs(self.clr_iterations / self.step - 2 * cycle + 1)
        if self.scale_mode == 'cycle':
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(cycle)
        else:
            return self.base_lr + (self.max_lr - self.base_lr) * np.maximum(0, (1 - x)) * self.scale_fn(
                self.clr_iterations)

    def get_rate(self, epoch=None, num_epoches=None):

        self.trn_iterations += 1
        self.clr_iterations += 1
        lr = self.clr()

        return lr

    def __str__(self):
        string = 'Cyclical Learning Rates\n' \
                 + 'base_lr=%0.3f, max_lr=%0.3f' % (self.base_lr, self.max_lr)
        return string
