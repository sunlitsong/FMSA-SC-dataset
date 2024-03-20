import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_
from torch.nn.parameter import Parameter


class SubNet(nn.Module):

    def __init__(self, in_size, hidden_size, num_layers=1, dropout=0.2, bidirectional=True):
        super(SubNet, self).__init__()
        if num_layers == 1:
            dropout = 0.0
        self.rnn = nn.GRU(in_size, hidden_size, num_layers=num_layers, dropout=dropout, bidirectional=bidirectional,
                          batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(2 * hidden_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        _, final_states = self.rnn(x)
        h = torch.cat((final_states[0], final_states[1]), -1)
        dropped = self.dropout(h)
        y_1 = F.relu(self.linear_1(dropped), inplace=True)
        y_2 = F.relu(self.linear_2(y_1), inplace=True)
        return y_2


class ExpressionSelfAttention(nn.Module):
    def __init__(self, in_size, dropout):
        super(ExpressionSelfAttention, self).__init__()
        self.linear_1 = nn.Linear(in_size, 1)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(2 * in_size, 1)
        self.linear_3 = nn.Linear(2 * in_size, in_size)

    def forward(self, x):
        dropped_1 = self.dropout(x)
        att = nn.Sigmoid()(self.linear_1(dropped_1))
        vm = torch.mul(att, x).mean(0)
        vm = vm.repeat(x.shape[0], 1)
        vs = torch.cat([x, vm], dim=-1)
        dropped_2 = self.dropout(vs)
        att_new = nn.Sigmoid()(self.linear_2(dropped_2))
        y = torch.mul(att * att_new, vs)
        y_1 = F.relu(self.linear_3(y), inplace=True)
        return y_1


class CrossModalAttention(nn.Module):
    def __init__(self, in_size, hidden_size, dropout):
        super(CrossModalAttention, self).__init__()
        self.activation = nn.ReLU()
        self.x_weight_1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.y_weight_1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.t_weight_1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.bias = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.x_weight_1.data.fill_(1)
        self.y_weight_1.data.fill_(1)
        self.t_weight_1.data.fill_(1)
        self.bias.data.fill_(0)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(in_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, y, target):
        x_att = torch.matmul(x, x.transpose(-1, -2))
        x_att = self.activation(x_att)

        y_att = torch.matmul(y, y.transpose(-1, -2))
        y_att = self.activation(y_att)

        t_att = torch.matmul(target, target.transpose(-1, -2))
        t_att = self.activation(t_att)

        fusion_att = self.x_weight_1 * x_att + self.y_weight_1 * y_att + self.t_weight_1 * t_att + self.bias
        fusion_att = nn.Softmax(dim=-1)(fusion_att)
        target_att = torch.matmul(fusion_att, target)

        dropped = self.dropout(target_att)
        y_1 = F.relu(self.linear_1(dropped), inplace=True)
        y_2 = F.relu(self.linear_2(y_1), inplace=True)

        return y_2


class FusionSubNet(nn.Module):

    def __init__(self, in_size, hidden_size, out_size, dropout):
        super(FusionSubNet, self).__init__()
        self.rnn = nn.GRU(in_size, hidden_size, num_layers=1, dropout=0, bidirectional=True, batch_first=True)
        self.linear_1 = nn.Linear(2 * hidden_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, h, p):
        output, _ = self.rnn(h)
        a_1 = F.relu(self.linear_1(output), inplace=True)
        a_2 = nn.Sigmoid()(self.linear_2(a_1))
        y = torch.matmul(a_2.permute(1, 0), p.permute(1, 0)).squeeze()
        return y


class LMF(nn.Module):

    def __init__(self, args):
        super(LMF, self).__init__()

        self.text_in, self.audio_in, self.video_in = args.feature_dims
        self.text_hidden, self.audio_hidden, self.video_hidden, self.fusion_hidden = args.hidden_dims

        self.output_dim = args.num_classes if args.train_mode == "classification" else 1
        self.rank = args.rank

        self.audio_prob, self.video_prob, self.text_prob, self.fusion_prob = args.dropouts

        self.post_text_prob, self.post_audio_prob, self.post_video_prob, self.post_fusion_prob = args.post_dropouts
        self.post_text_dim = args.post_text_dim
        self.post_audio_dim = args.post_audio_dim
        self.post_video_dim = args.post_video_dim
        self.post_fusion_dim = args.post_fusion_dim

        self.audio_subnet = SubNet(self.audio_in, self.audio_hidden, dropout=self.audio_prob)
        self.video_subnet = SubNet(self.video_in, self.video_hidden, dropout=self.video_prob)
        self.text_subnet = SubNet(self.text_in, self.text_hidden, dropout=self.text_prob)

        self.video_attnet = ExpressionSelfAttention(self.video_in, self.video_prob)

        self.audio_linear = nn.Linear(self.audio_in, self.audio_hidden)
        self.video_linear = nn.Linear(self.video_in, self.video_hidden)
        self.text_linear = nn.Linear(self.text_in, self.text_hidden)
        self.audio_cutnet = CrossModalAttention(self.audio_hidden, self.fusion_hidden, self.audio_prob)
        self.video_cutnet = CrossModalAttention(self.video_hidden, self.fusion_hidden, self.video_prob)
        self.text_cutnet = CrossModalAttention(self.text_hidden, self.fusion_hidden, self.text_prob)

        self.audio_factor = Parameter(torch.Tensor(self.rank, self.fusion_hidden + 1, self.output_dim))
        self.video_factor = Parameter(torch.Tensor(self.rank, self.fusion_hidden + 1, self.output_dim))
        self.text_factor = Parameter(torch.Tensor(self.rank, self.fusion_hidden + 1, self.output_dim))

        self.fusion_subnet = FusionSubNet(self.fusion_hidden, self.post_fusion_dim, self.output_dim, self.fusion_prob)

        self.post_text_dropout = nn.Dropout(p=self.post_text_prob)
        self.post_text_layer_1 = nn.Linear(self.text_hidden, self.post_text_dim)
        self.post_text_layer_2 = nn.Linear(self.post_text_dim, self.post_text_dim)
        self.post_text_layer_3 = nn.Linear(self.post_text_dim, self.output_dim)

        self.post_audio_dropout = nn.Dropout(p=self.post_audio_prob)
        self.post_audio_layer_1 = nn.Linear(self.audio_hidden, self.post_audio_dim)
        self.post_audio_layer_2 = nn.Linear(self.post_audio_dim, self.post_audio_dim)
        self.post_audio_layer_3 = nn.Linear(self.post_audio_dim, self.output_dim)

        self.post_video_dropout = nn.Dropout(p=self.post_video_prob)
        self.post_video_layer_1 = nn.Linear(self.video_hidden, self.post_video_dim)
        self.post_video_layer_2 = nn.Linear(self.post_video_dim, self.post_video_dim)
        self.post_video_layer_3 = nn.Linear(self.post_video_dim, self.output_dim)

        self.fusion_weights = Parameter(torch.Tensor(1, self.rank))
        self.fusion_bias = Parameter(torch.Tensor(1, self.output_dim))

        xavier_normal_(self.audio_factor)
        xavier_normal_(self.video_factor)
        xavier_normal_(self.text_factor)
        xavier_normal_(self.fusion_weights)
        self.fusion_bias.data.fill_(0)

    def forward(self, text_cutx, audio_cutx, video_cutx):
        audio_batchs = list()
        video_batchs = list()
        text_batchs = list()
        fusion_batchs = list()
        batch_size = len(text_cutx)
        for i in range(batch_size):
            audio_cutxi = audio_cutx[i]
            video_cutxi = video_cutx[i]
            text_cutxi = text_cutx[i]

            audio_hi = self.audio_subnet(audio_cutxi)
            video_hi = self.video_subnet(video_cutxi)
            text_hi = self.text_subnet(text_cutxi)
            audio_batchs.append(audio_hi.unsqueeze(0))
            video_batchs.append(video_hi.unsqueeze(0))
            text_batchs.append(text_hi.unsqueeze(0))

            video_cutxi = self.video_attnet(video_cutxi)

            audio_cutxi = self.audio_linear(audio_cutxi)
            video_cutxi = self.video_linear(video_cutxi)
            text_cutxi = self.text_linear(text_cutxi)
            audio_cuth = self.audio_cutnet(video_cutxi, text_cutxi, audio_cutxi)
            video_cuth = self.video_cutnet(audio_cutxi, text_cutxi, video_cutxi)
            text_cuth = self.text_cutnet(audio_cutxi, video_cutxi, text_cutxi)

            seq_size = text_cuth.size()[0]
            add_one = torch.ones(size=[seq_size, 1], requires_grad=False).type_as(audio_cuth).to(text_cuth.device)
            _audio_cuth = torch.cat((add_one, audio_cuth), dim=1)
            _video_cuth = torch.cat((add_one, video_cuth), dim=1)
            _text_cuth = torch.cat((add_one, text_cuth), dim=1)

            fusion_audio = torch.matmul(_audio_cuth, self.audio_factor)
            fusion_video = torch.matmul(_video_cuth, self.video_factor)
            fusion_text = torch.matmul(_text_cuth, self.text_factor)

            fusion_zx = audio_cuth * video_cuth * text_cuth
            fusion_zy = fusion_audio * fusion_video * fusion_text

            _fusion = torch.matmul(self.fusion_weights, fusion_zy.permute(1, 0, 2)).squeeze() + self.fusion_bias
            fusion = self.fusion_subnet(fusion_zx, _fusion)
            fusion_batchs.append(fusion.unsqueeze(0))
        audio_h = torch.cat(audio_batchs, dim=0)
        video_h = torch.cat(video_batchs, dim=0)
        text_h = torch.cat(text_batchs, dim=0)
        fusions = torch.cat(fusion_batchs, dim=0)
        output = fusions.view(-1, self.output_dim)

        x_t = self.post_text_dropout(text_h)
        x_t = F.relu(self.post_text_layer_1(x_t), inplace=True)
        x_t = F.relu(self.post_text_layer_2(x_t), inplace=True)
        output_text = self.post_text_layer_3(x_t)

        x_a = self.post_audio_dropout(audio_h)
        x_a = F.relu(self.post_audio_layer_1(x_a), inplace=True)
        x_a = F.relu(self.post_audio_layer_2(x_a), inplace=True)
        output_audio = self.post_audio_layer_3(x_a)

        x_v = self.post_video_dropout(video_h)
        x_v = F.relu(self.post_video_layer_1(x_v), inplace=True)
        x_v = F.relu(self.post_video_layer_2(x_v), inplace=True)
        output_video = self.post_video_layer_3(x_v)

        res = {
            'Feature_t': text_h,
            'Feature_a': audio_h,
            'Feature_v': video_h,

            'M': output,
            'T': output_text,
            'A': output_audio,
            'V': output_video
        }
        return res
