from .LF_DNN import LF_DNN
from .LMF import LMF
from .TFN import TFN


class ATIO():
    def __init__(self):
        self.TRAIN_MAP = {
            'tfn': TFN,
            'lmf': LMF,
            'lf_dnn': LF_DNN,
        }

    def getTrain(self, args):
        return self.TRAIN_MAP[args['model_name']](args)
