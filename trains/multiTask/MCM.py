import os
import time
import logging
import math
import copy
import argparse
import numpy as np
import pickle as plk
from glob import glob
from tqdm import tqdm

import torch
import torch.nn as nn
from torch import optim

from utils.functions import dict_to_str
from utils.metricsTop import MetricsTop

logger = logging.getLogger('MSA')

class MCM():
    def __init__(self, args):
        assert args.train_mode == 'regression'

        self.args = args
        self.args.tasks = ['M', 'T', 'A', 'V', 'TA', 'TV', 'VA']
        self.metrics = MetricsTop(args.train_mode).getMetics(args.datasetName)

    def do_train(self, model, dataloader):
        bert_no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        bert_params = list(model.Model.text_model.named_parameters())
        audio_params = list(model.Model.audio_model.named_parameters())
        video_params = list(model.Model.video_model.named_parameters())

        bert_params_decay = [p for n, p in bert_params if not any(nd in n for nd in bert_no_decay)]
        bert_params_no_decay = [p for n, p in bert_params if any(nd in n for nd in bert_no_decay)]
        audio_params = [p for n, p in audio_params]
        video_params = [p for n, p in video_params]
        model_params_other = [p for n, p in list(model.Model.named_parameters()) if 'text_model' not in n and \
                                'audio_model' not in n and 'video_model' not in n]
        
        log_var_M = torch.zeros((1,), requires_grad=True)
        log_var_T = torch.zeros((1,), requires_grad=True)
        log_var_A = torch.zeros((1,), requires_grad=True)
        log_var_V = torch.zeros((1,), requires_grad=True)
        log_var_TA = torch.zeros((1,), requires_grad=True)
        log_var_TV = torch.zeros((1,), requires_grad=True)
        log_var_VA = torch.zeros((1,), requires_grad=True)
        log_var_params = [log_var_M, log_var_T, log_var_A, log_var_V, log_var_TA, log_var_TV, log_var_VA]

        optimizer_grouped_parameters = [
            {'params': bert_params_decay, 'weight_decay': self.args.weight_decay_bert, 'lr': self.args.learning_rate_bert},
            {'params': bert_params_no_decay, 'weight_decay': 0.0, 'lr': self.args.learning_rate_bert},
            {'params': audio_params, 'weight_decay': self.args.weight_decay_audio, 'lr': self.args.learning_rate_audio},
            {'params': video_params, 'weight_decay': self.args.weight_decay_video, 'lr': self.args.learning_rate_video},
            {'params': model_params_other, 'weight_decay': self.args.weight_decay_other, 'lr': self.args.learning_rate_other},
            {'params': log_var_params, 'weight_decay': 0.0, 'lr': 1e-3}
        ]
        optimizer = optim.Adam(optimizer_grouped_parameters)

        #saved_labels = {}
        ## init labels
        #logger.info("Init labels...")
        #with tqdm(dataloader['train']) as td:
        #    for batch_data in td:
        #        labels_m = batch_data['labels']['M'].view(-1).to(self.args.device)
        #        indexes = batch_data['index'].view(-1)
        #        #self.init_labels(indexes, labels_m)

        # initilize results
        logger.info("Start training...")
        epochs, best_epoch = 0, 0
        min_or_max = 'min' if self.args.KeyEval in ['Loss'] else 'max'
        best_valid = 1e8 if min_or_max == 'min' else 0
        # loop util earlystop
        while True: 
            epochs += 1
            # train
            y_pred = {'M': [], 'M2': [], 'T': [], 'A': [], 'V': [], 'TA': [], 'TV': [], 'VA': []}
            y_true = {'M': [], 'M2': [], 'T': [], 'A': [], 'V': [], 'TA': [], 'TV': [], 'VA': []}
            y_params = {'M': log_var_M, 'T': log_var_T, 'A': log_var_A, 'V': log_var_V, 'TA': log_var_TA, 'TV': log_var_TV, 'VA': log_var_VA}
            for m in self.args.tasks:
              y_params[m] = y_params[m].to(self.args.device)
            losses = []
            model.train()
            train_loss = 0.0
            left_epochs = self.args.update_epochs
            #ids = []
            with tqdm(dataloader['train']) as td:
                for batch_data in td:
                    if left_epochs == self.args.update_epochs:
                        optimizer.zero_grad()
                    left_epochs -= 1

                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    #indexes = batch_data['index'].view(-1)
                    #cur_id = batch_data['id']
                    labels = batch_data['labels']['M'].view(-1).to(self.args.device)
                    #ids.extend(cur_id)

                    if not self.args.need_data_aligned:
                        audio_lengths = batch_data['audio_lengths'].to(self.args.device)
                        vision_lengths = batch_data['vision_lengths'].to(self.args.device)
                    else:
                        audio_lengths, vision_lengths = 0, 0

                    # forward
                    outputs = model(text, (audio, audio_lengths), (vision, vision_lengths))
                    # store results
                    for m in self.args.tasks:
                        y_pred[m].append(outputs[m].cpu())
                        y_true[m].append(labels.cpu())
                    # compute loss
                    loss = self.weighted_loss(outputs, labels, y_params, mode='TRAIN')
                    #loss = self.weighted_loss(outputs['M'], labels, mode='VAL')
                    #for m in self.args.tasks:
                    #    loss += self.weighted_loss(outputs[m], self.label_map[self.name_map[m]][indexes], \
                    #                                indexes=indexes, mode=self.name_map[m])
                    # backward
                    loss.backward()
                    train_loss += loss.item()
                    # update features
                    #f_fusion = outputs['Feature_f'].detach()
                    #f_text = outputs['Feature_t'].detach()
                    #f_audio = outputs['Feature_a'].detach()
                    #f_vision = outputs['Feature_v'].detach()
                    #if epochs > 1:
                    #    self.update_labels(f_fusion, f_text, f_audio, f_vision, epochs, indexes, outputs)

                    #self.update_features(f_fusion, f_text, f_audio, f_vision, indexes)
                    #self.update_centers()

                    # update parameters
                    if not left_epochs:
                        # update
                        optimizer.step()
                        left_epochs = self.args.update_epochs
                if not left_epochs:
                    # update
                    optimizer.step()
            train_loss = train_loss / len(dataloader['train'])
            logger.info("TRAIN-(%s) (%d/%d/%d)>> loss: %.4f " % (self.args.modelName, \
                        epochs-best_epoch, epochs, self.args.cur_time, train_loss))
            for m in self.args.tasks:
                pred, true = torch.cat(y_pred[m]), torch.cat(y_true[m])
                train_results = self.metrics(pred, true)
                logger.info('%s: >> ' %(m) + dict_to_str(train_results))
            print("M:{} T:{} A:{} V:{} TA:{} TV:{} VA:{}".format(y_params['M'].item(), y_params['T'].item(),
                                      y_params['A'].item(), y_params['V'].item(),
                                      y_params['TA'].item(), y_params['TV'].item(),
                                      y_params['VA'].item()))
            # validation
            val_results = self.do_test(model, dataloader['valid'], mode="VAL")
            cur_valid = val_results[self.args.KeyEval]
            # save best model
            isBetter = cur_valid <= (best_valid - 1e-6) if min_or_max == 'min' else cur_valid >= (best_valid + 1e-6)
            if isBetter:
                best_valid, best_epoch = cur_valid, epochs
                # save model
                torch.save(model.cpu().state_dict(), self.args.model_save_path)
                model.to(self.args.device)
            # save labels
            #if self.args.save_labels:
            #    tmp_save = {k: v.cpu().numpy() for k, v in self.label_map.items()}
            #    tmp_save['ids'] = ids
            #    saved_labels[epochs] = tmp_save
            ## early stop
            if epochs - best_epoch >= self.args.early_stop:
                #if self.args.save_labels:
                #    with open(os.path.join(self.args.res_save_dir, f'{self.args.modelName}-{self.args.datasetName}-labels.pkl'), 'wb') as df:
                #        plk.dump(saved_labels, df, protocol=4)
                return

    def do_test(self, model, dataloader, mode="VAL"):
        model.eval()
        y_pred = {'M': [], 'T': [], 'A': [], 'V': []}
        y_true = {'M': [], 'T': [], 'A': [], 'V': []}
        eval_loss = 0.0
        # criterion = nn.L1Loss()
        with torch.no_grad():
            with tqdm(dataloader) as td:
                for batch_data in td:
                    vision = batch_data['vision'].to(self.args.device)
                    audio = batch_data['audio'].to(self.args.device)
                    text = batch_data['text'].to(self.args.device)
                    if not self.args.need_data_aligned:
                        audio_lengths = batch_data['audio_lengths'].to(self.args.device)
                        vision_lengths = batch_data['vision_lengths'].to(self.args.device)
                    else:
                        audio_lengths, vision_lengths = 0, 0

                    labels_m = batch_data['labels']['M'].to(self.args.device).view(-1)
                    outputs = model(text, (audio, audio_lengths), (vision, vision_lengths))
                    loss = self.weighted_loss(outputs['M'], labels_m, None, mode='VAL')
                    eval_loss += loss.item()
                    y_pred['M'].append(outputs['M'].cpu())
                    y_true['M'].append(labels_m.cpu())
        eval_loss = eval_loss / len(dataloader)
        logger.info(mode+"-(%s)" % self.args.modelName + " >> loss: %.4f " % eval_loss)
        pred, true = torch.cat(y_pred['M']), torch.cat(y_true['M'])
        eval_results = self.metrics(pred, true)
        logger.info('M: >> ' + dict_to_str(eval_results))
        eval_results['Loss'] = eval_loss
        return eval_results


    def weighted_loss(self, output, labels, log_var, mode):
        loss = 0.0
        loss_list = ['M', 'T', 'A', 'V', 'TA', 'TV', 'VA']
          
        if mode == 'TRAIN':
          #for m in self.args.tasks:
          for m in loss_list:
            y_pred = output[m].view(-1)
            y_true = labels.view(-1)
            #loss += torch.mean(torch.add(y_pred,  -y_true).pow(2))

            precision = torch.exp(-log_var[m])
            diff = (y_pred - y_true) ** 2
            loss += torch.mean(precision * diff + log_var[m], -1)
          cse_loss = 1.0 * torch.mean(torch.add(output['M'].view(-1), -output['M2'].view(-1)).pow(2))
          loss += cse_loss
          return loss
              
        elif mode == 'VAL':
          y_pred = output.view(-1)
          y_true = labels.view(-1)
          loss += torch.mean(torch.add(y_pred, -y_true).pow(2))
          return loss