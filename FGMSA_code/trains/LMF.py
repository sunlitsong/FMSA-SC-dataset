import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from ..utils import MetricsTop


class LMF():
    def __init__(self, args):
        assert args.dataset_name == 'fgmsa'

        self.args = args
        self.args.tasks = "MTAV"
        self.criterion = nn.L1Loss() if args.train_mode == 'regression' else nn.CrossEntropyLoss()
        self.metrics = MetricsTop(args.train_mode).getMetics(args.dataset_name)

    def do_train(self, model, dataloader, return_epoch_results=False):
        model_params_other = [p for n, p in list(model.Model.named_parameters())[5:] if 'text_subnet' not in n and \
                              'audio_subnet' not in n and 'video_subnet' not in n and 'text_cutnet' not in n and \
                              'audio_cutnet' not in n and 'video_cutnet' not in n and 'fusion_subnet' not in n]
        optimizer = optim.Adam(
            [{"params": list(model.Model.text_subnet.parameters()), "weight_decay": self.args.text_weight_decay},
             {"params": list(model.Model.audio_subnet.parameters()), "weight_decay": self.args.audio_weight_decay},
             {"params": list(model.Model.video_subnet.parameters()), "weight_decay": self.args.video_weight_decay},
             {"params": list(model.Model.text_cutnet.parameters()), "weight_decay": self.args.text_weight_decay},
             {"params": list(model.Model.audio_cutnet.parameters()), "weight_decay": self.args.audio_weight_decay},
             {"params": list(model.Model.video_cutnet.parameters()), "weight_decay": self.args.video_weight_decay},
             {"params": list(model.Model.fusion_subnet.parameters()), "weight_decay": self.args.fusion_weight_decay},
             {"params": list(model.parameters())[:3], "lr": self.args.factor_lr},
             {"params": list(model.parameters())[3:5], "lr": 0.0},
             {'params': model_params_other}],
            lr=self.args.learning_rate, weight_decay=self.args.weight_decay)

        epochs, best_epoch = 0, 0
        if return_epoch_results:
            epoch_results = {
                'train': [],
                'valid': [],
                'test': []
            }
        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0

        while True:
            epochs += 1

            y_pred = {'M': [], 'T': [], 'A': [], 'V': []}
            y_true = {'M': [], 'T': [], 'A': [], 'V': []}
            losses = []
            model.train()
            train_loss = 0.0
            with tqdm(dataloader['train']) as td:
                for batch_data in td:
                    vision_cut = list(map(lambda x: x.to(self.args.device), batch_data['vision_cut']))
                    audio_cut = list(map(lambda x: x.to(self.args.device), batch_data['audio_cut']))
                    text_cut = list(map(lambda x: x.to(self.args.device), batch_data['text_cut']))
                    labels = batch_data['labels']

                    for k in labels.keys():
                        if self.args.train_mode == 'classification':
                            labels[k] = labels[k].to(self.args.device).view(-1).long()
                        else:
                            labels[k] = labels[k].to(self.args.device).view(-1, 1)

                    optimizer.zero_grad()

                    outputs = model(text_cut, audio_cut, vision_cut)

                    loss = 0.0
                    for m in self.args.tasks:
                        loss += eval('self.args.' + m) * self.criterion(outputs[m], labels[m])

                    loss.backward()

                    optimizer.step()

                    train_loss += loss.item()
                    for m in self.args.tasks:
                        y_pred[m].append(outputs[m].cpu())
                        y_true[m].append(labels['M'].cpu())
            train_loss = train_loss / len(dataloader['train'])

            for m in self.args.tasks:
                pred, true = torch.cat(y_pred[m]), torch.cat(y_true[m])
                train_results = self.metrics(pred, true)

            val_results = self.do_test(model, dataloader['valid'], mode="VAL")
            cur_valid = val_results[self.args.KeyEval]

            isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)

            if isBetter:
                best_valid, best_epoch = cur_valid, epochs

                torch.save(model.cpu().state_dict(), self.args.model_save_path)
                model.to(self.args.device)

            if return_epoch_results:
                train_results["Loss"] = train_loss
                epoch_results['train'].append(train_results)
                epoch_results['valid'].append(val_results)
                test_results = self.do_test(model, dataloader['test'], mode="TEST")
                epoch_results['test'].append(test_results)

            if epochs - best_epoch >= self.args.early_stop:
                return epoch_results if return_epoch_results else None

    def do_test(self, model, dataloader, mode="VAL", return_sample_results=False):
        model.eval()
        y_pred = {'M': [], 'T': [], 'A': [], 'V': []}
        y_true = {'M': [], 'T': [], 'A': [], 'V': []}
        eval_loss = 0.0
        if return_sample_results:
            ids, sample_results = [], []
            all_labels = []
            features = {
                "Feature_t": [],
                "Feature_a": [],
                "Feature_v": [],
                "Feature_f": [],
            }
        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    vision_cut = list(map(lambda x: x.to(self.args.device), batch_data['vision_cut']))
                    audio_cut = list(map(lambda x: x.to(self.args.device), batch_data['audio_cut']))
                    text_cut = list(map(lambda x: x.to(self.args.device), batch_data['text_cut']))
                    labels = batch_data['labels']
                    for k in labels.keys():
                        if self.args.train_mode == 'classification':
                            labels[k] = labels[k].to(self.args.device).view(-1).long()
                        else:
                            labels[k] = labels[k].to(self.args.device).view(-1, 1)
                    outputs = model(text_cut, audio_cut, vision_cut)

                    if return_sample_results:
                        ids.extend(batch_data['id'])
                        for item in features.keys():
                            features[item].append(outputs[item].cpu().detach().numpy())
                        all_labels.extend(labels.cpu().detach().tolist())
                        preds = outputs["M"].cpu().detach().numpy()

                        sample_results.extend(preds.squeeze())

                    loss = 0.0
                    for m in self.args.tasks:
                        loss += eval('self.args.' + m) * self.criterion(outputs[m], labels[m])
                    eval_loss += loss.item()
                    for m in self.args.tasks:
                        y_pred[m].append(outputs[m].cpu())
                        y_true[m].append(labels['M'].cpu())
        eval_loss = round(eval_loss / len(dataloader), 4)

        eval_results = {}
        for m in self.args.tasks:
            pred, true = torch.cat(y_pred[m]), torch.cat(y_true[m])
            results = self.metrics(pred, true)

            eval_results[m] = results
        eval_results = eval_results[self.args.tasks[0]]
        eval_results['Loss'] = round(eval_loss, 4)

        if return_sample_results:
            eval_results["Ids"] = ids
            eval_results["SResults"] = sample_results
            for k in features.keys():
                features[k] = np.concatenate(features[k], axis=0)
            eval_results['Features'] = features
            eval_results['Labels'] = all_labels

        return eval_results
