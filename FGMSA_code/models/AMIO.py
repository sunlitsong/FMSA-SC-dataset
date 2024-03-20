import torch.nn as nn
from .LF_DNN import LF_DNN
from .LMF import LMF
from .TFN import TFN


class AMIO(nn.Module):
    def __init__(self, args):
        super(AMIO, self).__init__()
        self.MODEL_MAP = {
            'tfn': TFN,
            'lmf': LMF,
            'lf_dnn': LF_DNN,
        }
        lastModel = self.MODEL_MAP[args['model_name']]
        self.Model = lastModel(args)

    def forward(self, text_cutx, audio_cutx, video_cutx, *args, **kwargs):
        return self.Model(text_cutx, audio_cutx, video_cutx, *args, **kwargs)
