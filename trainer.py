# -*- coding: utf-8 -*-

# Import statements

import os
import copy
import time
import warnings

import multiprocessing
multiprocessing.set_start_method('spawn', True)

import hashlib
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tabulate import tabulate
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ReduceLROnPlateau
from torch import autograd
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence

from utils.project_utils import cp_save_obj, cp_load_obj

tabulate.PRESERVE_WHITESPACE = True

from utils.project_utils import Timing, repackage_hidden
from utils.evaluation_utils import MultiLabelEval, export_timestep_metrics
from utils.tensor_utils import \
    DatasetWithLength_multi, \
    padded_collate_multi, \
    sort_minibatch_multi, \
    to_multihot_sorted_vectors, \
    fit_input_to_output

import sys
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(message)s')

warnings.filterwarnings('ignore')

eps = 0

class Trainer(nn.Module):
    def __init__(self,
                 model,
                 event_size=None,
                 loss_fn=None,
                 epoch=101, learning_rate=0.001,
                 print_every=10, valid_every=20, save_every=20,
                 model_prefix='', use_cuda=False, batch_size=64,
                 batch_first=True,
                 curriculum_learning=False, curriculum_rate=1.35,
                 max_seq_len_init=2,
                 weight_decay=0,
                 lr_scheduler=False, lr_scheduler_numiter=15,
                 lr_scheduler_multistep=False, lr_scheduler_epochs=[],
                 lr_scheduler_mult=0.5,
                 lr_scheduler_ror=False,
                 num_workers=4,
                 web_logger=None,
                 bptt=0,
                 device=None,
                 event_dic=None,
                 args=None
                 ):
        """

        :rtype: None
        """
        super(Trainer, self).__init__()

        self.model = model

        self.web_logger = args.web_logger

        self.event_size = args.event_size
        self.event_dic = args.event_dic
        self.loss_fn = args.loss_fn

        self.device = args.device

        self.epoch = args.epoch
        self.learning_rate = args.learning_rate
        self.use_cuda = args.use_cuda
        self.batch_size = args.batch_size
        self.print_every = args.print_every
        self.valid_every = args.valid_every
        self.save_every = args.save_every
        self.model_prefix = args.model_prefix
        self.batch_first = args.batch_first
        self.num_workers = args.num_workers

        self.curriculum_learning = args.curriculum_learning
        self.curriculum_rate = args.curriculum_rate
        self.max_seq_len_init = args.curriculum_init

        self.lr_scheduler = args.lr_scheduler
        self.lr_scheduler_multistep = args.lr_scheduler_multistep
        self.lr_scheduler_epochs = args.lr_scheduler_epochs
        self.lr_scheduler_numiter = args.lr_scheduler_numiter
        self.lr_scheduler_mult = args.lr_scheduler_mult
        self.weight_decay = args.weight_decay
        self.optim = args.optimizer

        self.bptt = args.bptt
        self.patient = args.patient_stop

        self.lr_scheduler_ror = args.lr_scheduler_ror

        # time-pred related args (needs to be part of Trainer args)
        self.force_epoch = args.force_epoch

        self.target_event = args.target_event
        self.force_checkpointing = args.force_checkpointing
        self.force_auroc = args.force_auroc
        self.force_plot_auroc = args.force_plot_auroc
        self.track_weight_change = args.weight_change
        self.loss_tol = args.loss_tol
        self.args = args
        self.elapsed_time = args.elapsed_time

        self.adapt_lstm = args.adapt_lstm
        self.adapt_residual = args.adapt_residual
        self.adapt_lstm_only = args.adapt_lstm_only
        self.adapt_fc_only = args.adapt_fc_only
            
        self.event_types = None
        self.pos_weight = None
        self.best_epoch = 0

        if self.event_dic and 'category' in list(self.event_dic.values())[0]:
            self.event_types = list(set([list(self.event_dic.values())[i]['category'] \
                                         for i in
                                         range(len(list(self.event_dic.values())))]))
            self.event_types += ['lab_normal', 'lab_abnormal', 'chart_normal', 
                                 'chart_abnormal']
             
        # counters for adaptive learning
        self.cnt_update = 0
        self.cnt_time_step = 0

    def train_model(self, train_data, valid_data=None,
                    split10_final=False):

        best_metric = 0
        best_loss = 100
        prev_loss = 0
       
        if self.model != None and hasattr(self.model, 'parameters'):
            optimizer = self.get_optimizer()

            if self.lr_scheduler:
                scheduler = self.get_scheduler(optimizer)

        dataloader = self.get_dataloader(train_data, shuffle=False)

        start_time = time.time()
        max_seq_len = 0
        patient_cnt = 0
        item_avg_cnt = []

        print('Start training')

        if hasattr(self.model, "rb_init") and hasattr(self.model, "pred_period") \
                and "prior" in self.model.rb_init and self.model.pred_period:
            
            prior_next_bin = self.model.pp.prior[:, 0].to(self.device)
            
            if "asbias" in self.model.rb_init:
                self.model.r_bias_linear.bias.data.fill_(0)
                self.model.r_bias_linear.bias.data += prior_next_bin
            
            if "asweight" in self.model.rb_init:
                self.model.r_bias_linear.weight.data[list(range(self.model.event_size)), list(
                    range(self.model.event_size))] += prior_next_bin
                self.model.r_bias_linear.bias.data.fill_(0)

        epoch = 0

        while True:
            t_loss = 0
            epoch_start_time = time.time()

            for i, data in enumerate(dataloader):
                update_cnt = 0

                inp, trg, len_inp, len_trg, inp_time, trg_time \
                    = self.process_batch(data, epoch)
                
                if type(inp) == torch.Tensor:
                    batch_size, seq_len, _ = inp.size()
                        
                else:
                    batch_size = len(inp)
                    seq_len = len(inp[0])


                hidden = self.model.init_hidden(batch_size=batch_size)

                bptt_size = self.bptt if self.bptt else seq_len
                for j in range(0, seq_len, bptt_size):

                    if self.bptt:
                        seqlen = min(self.bptt, seq_len - j)
                    else:  # no bptt
                        seqlen = seq_len

                    if type(inp) == torch.Tensor:
                        inp_seq = inp[:, j:j + seqlen]
                    else:
                        inp_seq = [ibatch[j:j + seqlen] for ibatch in inp]

                    if type(trg) == torch.Tensor:
                        trg_seq = trg[:, j:j + seqlen]
                    else:
                        trg_seq = [ibatch[j:j + seqlen] for ibatch in trg]

                    if trg_time is not None:
                        if type(trg_time) == torch.Tensor:
                            trg_time_seq = trg_time[:, j:j + seqlen]
                        else:
                            trg_time_seq = [ibatch[j:j + seqlen] for ibatch in trg_time]
                    else:
                        trg_time_seq = None

                    if inp_time is not None:
                        if type(inp_time) == torch.Tensor:
                            inp_time_seq = inp_time[:, j:j + seqlen]
                        else:
                            inp_time_seq = [ibatch[j:j + seqlen] for ibatch in inp_time]
                    else:
                        if type(inp) == torch.Tensor:
                            inp_time_seq = torch.zeros(
                                inp_seq.size()).to(self.device)
                        else:
                            inp_time_seq = [[[]]]

                    seqlen_v = torch.LongTensor([seqlen] * batch_size)
                    
                    if type(len_inp) == list:
                        len_inp = torch.tensor(len_inp)
                    if type(len_trg) == list:
                        len_trg = torch.tensor(len_trg)

                    len_inp_step = torch.min(len_inp, seqlen_v)
                    len_inp -= len_inp_step
                    len_trg_step = torch.min(len_trg, seqlen_v)
                    len_trg -= len_trg_step

                    # inp_seq: n_batch * max_seq_len * n_events
                    # x : max_seq_len * n_events
                    # len_seq : n_batch
                    
                    if sum(len_inp_step) < 1:
                        continue

                    hidden = repackage_hidden(hidden)

                    # removing zero-lengths batch elements
                    if type(hidden) != list:
                        hidden = hidden.squeeze(0)

                    hidden, inp_seq, trg_seq, len_inp_step, len_trg_step, inp_time_seq, trg_time_seq = \
                        self.remove_zeroed_batch_elems(
                            hidden, inp_seq, trg_seq, len_inp_step,
                            len_trg_step,
                            trg_time_seq=trg_time_seq,
                            inp_time_seq=inp_time_seq,
                        )

                    plain_out, hidden = \
                        self.model(inp_seq, len_inp_step, hidden)
                    pred_seq = F.sigmoid(plain_out)
                    loss = self.loss_fn(pred_seq, trg_seq.float(), len_inp_step)

                    t_loss += loss.item()

                    loss = loss / self.args.grad_accum_steps
                    loss.backward()

                    if (i + 1) % self.args.grad_accum_steps == 0:
                        optimizer.step()
                        self.model.zero_grad()

            # At the end of each epoch

            if (epoch % self.print_every) == 0:
                if self.curriculum_learning:
                    max_seq_len_str = 'max_seq_len={}'.format(max_seq_len)
                else:
                    max_seq_len_str = ''

                print(' epoch {} train_loss = {:.6f} '
                             'epoch time={:.2f}s lr: {:.6f} {}'.format(
                    epoch, (t_loss + eps) / (len(dataloader) + eps),                                   
                    time.time() - epoch_start_time,
                    optimizer.param_groups[0]['lr'],
                    max_seq_len_str)
                )

                self.web_logger.log_metric(
                    "train_loss",
                    ((t_loss + eps) / (len(dataloader) + eps)),
                    step=epoch)

            if epoch % self.valid_every == 0:

                print('\nResults on training data')
                tr_loss, tr_time_loss, tr_events_loss, tr_auroc = \
                    self.infer_model(train_data, cur_epoch=epoch,
                                        test_name='train', return_metrics=True)

                if valid_data is not None:
                    # Do not test when it is the final one,
                    # since it will be called outside

                    print('Results on valid data')
                    valid_loss, val_time_loss, val_events_loss, val_auroc = \
                        self.infer_model(valid_data, cur_epoch=epoch,
                                        test_name='valid', return_metrics=True)
                    
                    self.web_logger.log_metric(
                        "valid_loss", valid_loss, step=epoch
                    )

                    target_metric = val_auroc
                    target_loss = valid_loss
                else:
                    target_metric = tr_auroc
                    target_loss = tr_loss

                print('total time={:.2f}m '.format(
                    (time.time() - start_time)/60))
    
                msg = ''   

                if ((target_metric > best_metric or self.force_checkpointing) 
                        and self.model is not None):
                    best_metric = target_metric
                    self.best_epoch = epoch
                    os.system('rm -rf {}epoch*.model'.format(self.model_prefix))
                    checkpoint_name = '{}epoch_{}.model'.format(
                        self.model_prefix, self.best_epoch)
                    self.save(checkpoint_name)

                    patient_cnt = 0
                    stegnant_auroc = False

                else:
                    if self.patient and (patient_cnt > self.patient):
                        print(
                            'stop training after {} '
                            'patient epochs (at {})'.format(
                                patient_cnt, epoch)
                        )
                        break

                    stegnant_auroc = True


                if (best_loss > target_loss) and (valid_data is None):
                    # NOTE: for hyper-parameter tuning run, do not check steganat_loss
                    # It is only activate for the final test run
                    best_loss = target_loss
                    patient_cnt = 0
                    stegnant_loss = False
                else:
                    stegnant_loss = True
                
                min_epoch_bar = True
                if valid_data is None and self.epoch > epoch:
                    min_epoch_bar = False

                if valid_data is None:
                    # NOTE: when no valid data, stegnant check based on 
                    # train set will be voided
                    loss_diff = abs(prev_loss - target_loss)

                    if (self.loss_tol != -1) and (loss_diff < self.loss_tol) \
                            and min_epoch_bar:
                        print(
                            'loss diff ({:.6f}) is less than tol ({:.6f}). '
                            'Finish train.'.format(loss_diff, self.loss_tol))

                        break

                    prev_loss = target_loss

                    stegnant_loss = True  # default value for valid_data is None

                if self.args.n_warmup_steps <= epoch:
                    check_warmup_over = True
                else:
                    check_warmup_over = False

                if stegnant_loss and stegnant_auroc and min_epoch_bar and check_warmup_over:
                    patient_cnt += 1
                    msg += '| ptn: {}/{} '.format(patient_cnt, self.patient)

                if self.lr_scheduler_ror:
                    scheduler.step(target_loss)

                print(
                    'target loss={:.6f} | best loss={:.6f} '
                    'auroc={:.2f} (epoch {}) {}'.format(
                        target_loss, best_loss, best_metric, self.best_epoch, msg))

            if (self.lr_scheduler and self.model != None
                    and not self.lr_scheduler_ror):
                scheduler.step()
                if self.lr_scheduler_multistep:
                    if epoch in self.lr_scheduler_epochs:
                        print('Current learning rate: {:g}'.format(
                            self.learning_rate * self.lr_scheduler_mult ** (
                                    self.lr_scheduler_epochs.index(
                                        epoch) + 1)))
                else:
                    if epoch % self.lr_scheduler_numiter == 0:
                        print('Current learning rate: {:g}'.format(
                            self.learning_rate * self.lr_scheduler_mult ** (
                                    epoch // self.lr_scheduler_numiter)))

            if self.save_every > 0 and epoch % self.save_every == 0 \
                    and self.model is not None:
                checkpoint_name = '{}epoch_{}.model'.format(
                    self.model_prefix, epoch)
                with Timing('Saving checkpoint to {}...'.format(
                        checkpoint_name)):
                    self.save(checkpoint_name)

            if self.track_weight_change and (self.model is not None):
                raise NotImplementedError

            if epoch >= self.epoch and self.force_epoch:
                break

            epoch += 1

        print(
            'train done. best AUROC:{:.2f} epoch:{}'.format(best_metric, self.best_epoch)
        )

        if (self.model is not None and hasattr(self.model, 'parameters')
                and not split10_final):
            checkpoint_name = '{}epoch_{}.model'.format(
                self.model_prefix, self.best_epoch)
            print(
                'Load best validation-f1 model (at epoch {}): {}...'.format(
                    self.best_epoch, checkpoint_name))
            self.load(checkpoint_name)
            if self.use_cuda:
                self.to(self.device)
        return best_metric

    def infer_model(self, test_data, cur_epoch=None, test_name=None,
                    final=False,
                    export_csv=False, csv_file=None,
                    eval_multithread=False, return_metrics=False,
                    no_other_metrics_but_flat=False
                    ):

        start_test = time.time()
        if self.model != None and hasattr(self.model, 'parameters'):
            self.model.eval()

        dataloader = self.get_dataloader(test_data, shuffle=False)

        t_loss, loss_times, loss_events = 0, 0, 0

        eval = MultiLabelEval(self.event_size, 
                                  use_cuda=self.use_cuda,
                                  macro_aucs=True,
                                  micro_aucs=False,
                                  device=self.device, 
                                  event_dic=self.event_dic,
                                  event_types=self.event_types,
                                  pred_labs=self.args.pred_labs,
                                  pred_normal_labchart=self.args.pred_normal_labchart,)

        for i, data in enumerate(dataloader):

            inp, trg, len_inp, len_trg, inp_time, trg_time = self.process_batch(data, 0)

            # with torch.no_grad():
            self.model.eval()
            loss_cnt = 0

            if type(inp) == torch.Tensor:
                batch_size, seq_len, _ = inp.size()
            else:
                batch_size = len(inp)
                seq_len = len(inp[0])

            if ((self.adapt_lstm or self.adapt_lstm_only or self.adapt_fc_only or self.adapt_residual) and final):
                read_batch_size = 1
            else:
                read_batch_size = batch_size
            
            bptt_time_loss = 0

            bptt_size = self.bptt if self.bptt else seq_len
                    
            if type(len_inp) == list:
                len_inp = torch.tensor(len_inp)
            if type(len_trg) == list:
                len_trg = torch.tensor(len_trg)

            for b in range(0, batch_size, read_batch_size):
                loss_events, loss_times, inp_first_step, loss_cnt, t_loss, loss_times, bptt_time_loss = self.process_inference(
                    b, inp, trg, inp_time, trg_time, 
                    len_inp, len_trg, read_batch_size,
                    seq_len, bptt_size, final, test_name, eval, 
                    loss_events, loss_cnt, t_loss, loss_times,
                    bptt_time_loss, batch_size, i)


            if self.model is not None and hasattr(self.model, 'parameters') \
                    and loss_cnt > 0:
                t_loss /= loss_cnt

        mse_loss_avg = (loss_times) / (len(dataloader))
        event_loss_avg = (loss_events) / (len(dataloader))

        
        # event prediction metrics

        if final and self.target_event > -1:
            eval.eval['flat']['y_probs'][self.target_event] = \
                eval.eval['flat']['y_probs'][0]
            eval.eval['flat']['y_trues'][self.target_event] = \
                eval.eval['flat']['y_trues'][0]

        output = eval.compute(eval.eval['flat'], epoch=cur_epoch,
                                test_name=test_name,
                                logger=self.web_logger,
                                final=final, export_csv=(export_csv and final),
                                event_dic=self.event_dic,
                                csv_file=csv_file,
                                use_multithread=eval_multithread,
                                return_metrics=return_metrics,
                                force_auroc=(self.force_auroc and (
                                    test_name != 'train')),
                                force_plot_auroc=self.force_plot_auroc,
                                # self.args.target_auprc and
                                target_auprc=((test_name != 'train'))
                                )
        if csv_file is not None: 
            np.save(csv_file.replace('.csv', '_event_dic.npy'), self.event_dic)

        if return_metrics:
            (f1, acc, metrics_container) = output

            if self.args.target_auprc:
                output_metric = metrics_container["mac_auprc"]
            else:
                output_metric = metrics_container["mac_auroc"]

        else:
            (f1, acc) = output
            output_metric = f1

        if final and (not no_other_metrics_but_flat):
            
            container_steps = []
            for step in range(len(eval.eval['tstep'])):
                print('step: {}'.format(step))
                _, _, container = eval.compute(
                    eval.eval['tstep'][step], epoch=step,
                    test_name=test_name,
                    logger=False,  # NOTE: web log off for time-step
                    tstep=True, verbose=False,
                    final=final,
                    return_metrics=True,
                    use_multithread=eval_multithread)
                container_steps.append(container)

            if export_csv:
                export_timestep_metrics(
                    csv_file.replace('.csv', '_timestep.csv'),
                    self.model_prefix,
                    container_steps,
                    event_dic=self.event_dic,
                )

            if self.event_dic and 'category' in list(self.event_dic.values())[0]:
                for etype in self.event_types:
                    print('event-type: {}'.format(etype))
                    eval.compute(eval.eval['etypes'][etype],
                                    epoch=cur_epoch,
                                    test_name=test_name,
                                    logger=self.web_logger,
                                    option_str=' ' + etype,
                                    event_dic=self.event_dic,
                                    export_csv=(export_csv and final),
                                    csv_file=csv_file.replace(
                                        '.csv', '_category_{}.csv'.format(etype)),
                                    final=final)

                    # Category AND Time-step
                    container_steps = []
                    for step in range(len(eval.eval['tstep-etype'])):
                        print('step: {}'.format(step))
                        _, _, container = eval.compute(
                            eval.eval['tstep-etype'][step][etype], epoch=step,
                            test_name=test_name,
                            logger=False,  # NOTE: web log off for time-step
                            tstep=True, 
                            verbose=False,
                            final=final,
                            return_metrics=True,
                            use_multithread=eval_multithread)
                        container_steps.append(container)

                    if export_csv:
                        export_timestep_metrics(
                            csv_file.replace('.csv', '_timestep_{}.csv'.format(etype)),
                            self.model_prefix,
                            container_steps
                        )

        end_test = time.time()
        print('Evaluation done in {:.3f}s\n'.format(end_test - start_test))

        if self.model is not None and hasattr(self.model, 'parameters'):
            self.model.train()  # turn off eval mode

        # Adaptive learning update counter 
        if self.cnt_update > 0:
            print('adaptive learning: count update: {}'.format(self.cnt_update))
            print('adaptive learning: total time-steps: {}'.format(self.cnt_time_step))
            print('average # updates/time-step: {:.4f}'.format(self.cnt_update/self.cnt_time_step))
            self.web_logger.log_parameter('cnt_update', self.cnt_update)
            self.web_logger.log_parameter('cnt_time_step', self.cnt_time_step)
            self.web_logger.log_parameter('cnt_avg_update_per_timestep', self.cnt_update/self.cnt_time_step)

        return (t_loss + eps) / (len(dataloader) + eps), \
               (loss_times + eps) / (len(dataloader) + eps), \
               (loss_events + eps) / (len(dataloader) + eps), \
               output_metric

    def save(self, path):
        if self.model is not None:
            if hasattr(self.model, 'parameters'):
                torch.save(self.model.state_dict(), path)
            else:
                pickle.dump(self.model, open(path, 'wb'),
                            protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path):
        if self.model is not None:
            if hasattr(self.model, 'parameters'):
                if self.use_cuda:
                    self.model.cpu().load_state_dict(torch.load(path))
                else:
                    self.model.cpu().load_state_dict(
                        torch.load(path, map_location='cpu'))
            else:
                self.model = pickle.load(open(path, 'rb'))

    def load_best_epoch_model(self):
        self.web_logger.log_parameter('best_epoch', self.best_epoch)

        checkpoint_name = '{}epoch_{}.model'.format(self.model_prefix,
                                                    self.best_epoch)
        print(
            'Load best validation-auroc model (at epoch {}): {}...'.format(
                self.best_epoch, checkpoint_name))
        self.load(checkpoint_name)
        if self.use_cuda:
            self.model = self.model.to(self.device)

    def save_final_model(self):
        final_model_name = '{}_final.model'.format(self.model_prefix)
        with Timing('Saving final model to {} ...'
                    ''.format(final_model_name)):
            if self.model is not None:
                if hasattr(self.model, 'parameters'):
                    torch.save(self.model.state_dict(), final_model_name)
                else:
                    pickle.dump(self.model, open(final_model_name, 'wb'),
                                protocol=pickle.HIGHEST_PROTOCOL)
        os.system('rm -rf {}epoch*.model'.format(self.model_prefix))

    def process_inference(self, b, inp, trg, inp_time, trg_time, len_inp, len_trg, 
                          read_batch_size, seq_len, bptt_size, final, test_name, 
                          eval, loss_events, loss_cnt, t_loss, loss_times, 
                          bptt_time_loss, batch_size, batch_idx):
        # extract small read batch from mini batch
        inp_b = inp[b: b + read_batch_size]
        trg_b = trg[b: b + read_batch_size]
        if inp_time is not None:
            inp_time_b = inp_time[b: b + read_batch_size]
        if trg_time is not None:
            trg_time_b = trg_time[b: b + read_batch_size]

        len_inp_b = len_inp[b: b + read_batch_size]
        len_trg_b = len_trg[b: b + read_batch_size]

        for j in range(0, seq_len, bptt_size):

            if self.bptt:
                seqlen = min(self.bptt, seq_len - j)
            else:  # no bptt
                seqlen = seq_len

            if type(inp) == torch.Tensor:
                inp_seq = inp_b[:, j:j + seqlen]
            else:
                inp_seq = [ibatch[j:j + seqlen]
                            for ibatch in inp_b]

            if type(trg) == torch.Tensor:
                trg_seq = trg_b[:, j:j + seqlen]
            else:
                trg_seq = [ibatch[j:j + seqlen]
                            for ibatch in trg_b]

            if trg_time is not None:
                if type(trg_time) == torch.Tensor:
                    trg_time_seq = trg_time_b[:, j:j + seqlen]
                else:
                    trg_time_seq = [ibatch[j:j + seqlen]
                                    for ibatch in trg_time_b]
            else:
                trg_time_seq = None

            if inp_time is not None:
                if type(inp_time) == torch.Tensor:
                    inp_time_seq = inp_time_b[:, j:j + seqlen]
                else:
                    inp_time_seq = [ibatch[j:j + seqlen]
                                    for ibatch in inp_time_b]
            else:
                if type(inp) == torch.Tensor:
                    inp_time_seq = torch.zeros(
                        inp_seq.size()).to(self.device)
                else:
                    inp_time_seq = [[[]]]

            seqlen_v = torch.LongTensor([seqlen] * read_batch_size)

            len_inp_step = torch.min(len_inp_b, seqlen_v)

            len_inp_b -= len_inp_step

            len_trg_step = torch.min(len_trg_b, seqlen_v)
            len_trg_b -= len_trg_step

            if sum(len_inp_step) < 1:
                continue

            self.model.zero_grad()

            hidden = self.model.init_hidden(
                batch_size=read_batch_size)

            # removing zero-lengths batch elements
            hidden, inp_seq, trg_seq, len_inp_step, len_trg_step, inp_time_seq, trg_time_seq = \
                self.remove_zeroed_batch_elems(hidden, inp_seq,
                                                trg_seq,
                                                len_inp_step,
                                                len_trg_step,
                                                trg_time_seq=trg_time_seq,
                                                inp_time_seq=inp_time_seq)

            if type(hidden) != list:
                hidden = hidden.squeeze(0)

            if ((self.adapt_lstm or self.adapt_lstm_only or self.adapt_fc_only or self.adapt_residual) and final):

                plain_out, time_pred = self.train_adaptive_model(
                    inp_seq, trg_seq, inp_time_seq, trg_time_seq,
                    len_inp_step, self.model, b, batch_size, batch_idx=batch_idx)
                
            else:
                plain_out, _ = self.model(inp_seq, len_inp_step, hidden)
                time_pred = None
            if plain_out.size(0) == inp_seq.size(1) and plain_out.size(1) == inp_seq.size(0):
                plain_out = plain_out.transpose(0, 1)

            sig_out = torch.sigmoid(plain_out)

            loss = self.loss_fn(sig_out, trg_seq.float(), len_inp_step)


            loss_events += loss.item()
            loss_cnt += 1

            t_loss += loss.item()
            loss_times += bptt_time_loss

            # ==========
            # evaluation
            # ----------

            pred = (sig_out > 0.5).float()

            if trg_seq.dim() == 3 and pred.dim() == 2:
                pred = pred.unsqueeze(1)
                sig_out = sig_out.unsqueeze(1)
            elif trg_seq.dim() == 3 and pred.dim() == 1:
                pred = pred.unsqueeze(0).unsqueeze(0)
                sig_out = sig_out.unsqueeze(0).unsqueeze(0)

            if type(inp) == torch.Tensor:
                inp_first_step = inp_seq[:, 0, :].unsqueeze(1)
            else:
                inp_first_step = torch.zeros(
                    read_batch_size, self.event_size).to(self.device).float()
                for b_idx, iseq in enumerate(inp_seq):
                    for event in iseq[0]:
                        inp_first_step[b_idx, event - 1] = 1
                inp_first_step = inp_first_step.unsqueeze(1)

            if self.args.pred_labs or self.args.pred_normal_labchart:
                inp_first_step = fit_input_to_output(inp_first_step,
                    self.args.inv_id_mapping.keys())

            eval.update(pred=pred, trg=trg_seq.float(),
                        len_trg=len_trg_step,
                        base_step=j, final=final, prob=sig_out,
                        force_auroc=(self.force_auroc and (
                            test_name != 'train')),
                        inp_first_step=inp_first_step)

        return loss_events, loss_times, inp_first_step, loss_cnt, t_loss, loss_times, bptt_time_loss


    def get_residual_model(self, popl_model):
        res_model = ResidualSeqNet(popl_model.embedding_dim, popl_model.hidden_dim, popl_model.event_size,
                                 popl_model.num_layers, popl_model.dropout, popl_model.device, rnn_type=self.model.rnn_type)

        res_model.embed_input = copy.deepcopy(popl_model.embed_input)
        res_model.rnn = copy.deepcopy(popl_model.rnn)
        res_model.fc_out = copy.deepcopy(popl_model.fc_out)

        res_model.embed_input.weight.requires_grad = False 
        for param in res_model.rnn.parameters():
            param.requires_grad = False
        for param in res_model.fc_out.parameters():
            param.requires_grad = False

        return res_model


    def train_adaptive_model(self, inp_seq, trg_seq, inp_time_seq, trg_time_seq, 
        len_inp_step, popl_model, b, full_batch_size,
        tol=1e-03, d_loss_tol=1e-04, batch_idx=0):
        
        kernel_bandwidth = self.args.adapt_bandwidth
        loss_type = self.args.adapt_loss
        adapt_LR = self.args.adapt_lr

        # Switch to GPU mode
        gpu = torch.device('cuda')
        self = self.to(gpu)
        self.device = gpu
        self.use_cuda = True

        inp_seq, trg_seq = inp_seq.to(gpu), trg_seq.to(gpu)

        batch_size = len(inp_seq)
        
        plain_out_seq, time_pred_seq = [], []
        
        if loss_type == 'bce':
            criterion = torch.nn.BCELoss(reduction='none')
        elif loss_type == 'mse':
            criterion = torch.nn.MSELoss(reduction='none')

        if batch_idx > 0:
            print('adaptation: batch_idx : {}'.format(batch_idx))

        for sub_step in range(inp_seq.size(1)):
            
            self.cnt_time_step += 1

            if self.args.adapt_pop_based or (sub_step == 0) or self.args.adapt_sw_pop:
                # everystep, we fork from population model, or init new at step 0

                if self.adapt_lstm or self.adapt_lstm_only or self.adapt_fc_only:
                    step_model = copy.deepcopy(popl_model).to(gpu)
                    
                    # freeze weights
                    step_model.embed_input.weight.requires_grad = False 

                    if self.adapt_lstm_only:
                        for param in step_model.fc_out.parameters():
                            param.requires_grad = False

                    if self.adapt_fc_only:
                        for param in step_model.rnn.parameters():
                            param.requires_grad = False

                    optimizer = optim.Adam(step_model.parameters(), lr=adapt_LR)

                elif self.adapt_residual:
                    step_model = self.get_residual_model(popl_model).to(gpu)

                    optimizer = optim.Adam(
                        [
                            {
                                'params': step_model.fc_out_residual.parameters(), 
                                'weight_decay': self.args.adapt_residual_wdecay
                            },
                            {
                                'params': step_model.embed_input.parameters(),
                                'weight_decay': self.weight_decay
                            },
                            {
                                'params': step_model.rnn.parameters(), 
                                'weight_decay': self.weight_decay
                            },
                            {
                                'params': step_model.fc_out.parameters(), 
                                'weight_decay': self.weight_decay
                            }
                        ]
                        , lr=adapt_LR)

            step_model.train()

            hidden = popl_model.init_hidden(batch_size=batch_size, device=gpu)
            
            # print("hidden:{}".format(hidden))

            # autoregressive sequence extract
            sub_inp = inp_seq[:, :(sub_step + 1)]
            sub_trg = trg_seq[:, :(sub_step + 1)]

            # start sub training routine
            prev_loss = 0
            mini_epoch = 1
            patient_cnt = 0

            if self.args.verbose:
                print('-'*16)

            while sub_step > 0:
                step_model.zero_grad()
                # make sure that we use the last step's hidden for the prediction 
                _sub_inp_step = sub_inp[:, :-1]
                _len_inp_step = torch.LongTensor(
                    [_sub_inp_step.size(1)]).to(gpu)

                output, time_pred, hidden = step_model(
                    _sub_inp_step,
                    _len_inp_step,
                    hidden,
                    trg_times=trg_time_seq,
                    inp_times=inp_time_seq,
                )
                output = torch.sigmoid(output)

                loss = self.compute_decayed_loss(criterion, output, sub_trg[:, :-1], kernel_bandwidth, self.decay_kernel)

                loss.backward(retain_graph=True)
                optimizer.step()

                d_loss = prev_loss - loss

                if loss > prev_loss:
                    patient_cnt += 1
                else:
                    patient_cnt = 0

                if self.args.verbose:
                    print('batch:{}/{} seq:{}/{} mini epoch: {}, loss: {:.8f}, d_loss:{:.8f} patient_cnt:{}'.format(
                        b, full_batch_size, sub_step, inp_seq.size(1), mini_epoch, loss, d_loss, patient_cnt))

                if (patient_cnt > self.patient) or (loss < tol) or (abs(d_loss) < d_loss_tol):
                    break

                prev_loss = loss
                mini_epoch += 1
                self.cnt_update += 1

            # output after sub-training 
            step_model.eval()
            _len_inp_step = torch.LongTensor([sub_inp.size(1)]).to(gpu)
            instance_out, time_pred, hidden = step_model(
                sub_inp,
                _len_inp_step,
                hidden,
                trg_times=trg_time_seq,
                inp_times=inp_time_seq,
            )

            # default output
            pred_step = instance_out[:, -1]

            if self.args.adapt_switch:
                # Online Switching model: choose best among population model and instance-specific
                # based on recent error rate.

                hidden = popl_model.init_hidden(batch_size=batch_size, device=gpu)
                popl_model.eval()

                try:
                    _len_inp_step = torch.LongTensor([sub_inp.size(1)]).to(gpu)
                    pop_output, time_pred, hidden = popl_model(
                        sub_inp,
                        _len_inp_step,
                        hidden,
                        trg_times=trg_time_seq,
                        inp_times=inp_time_seq,
                    )
                except RuntimeError as e:
                    print('sub_inp:{}'.format(sub_inp.size()))
                    print('hidden:{}'.format(hidden.size()))
                    print('len_inp_step:{}'.format(len_inp_step))
                    raise e 

                pop_loss = self.compute_decayed_loss(criterion, 
                    torch.sigmoid(pop_output), sub_trg, kernel_bandwidth, 
                    self.decay_kernel)

                # instance models' loss
                instance_loss = self.compute_decayed_loss(criterion, 
                    torch.sigmoid(instance_out), sub_trg, kernel_bandwidth, 
                    self.decay_kernel)

                # compare losses of the two
                if instance_loss > pop_loss:
                    pred_step = pop_output[:, -1]
            

            plain_out_seq.append(pred_step)  # use the last time-step only

            if time_pred is not None:
                time_pred_seq.append(time_pred[:, -1])
        
        plain_out_seq = torch.stack(plain_out_seq).transpose(0, 1)

        if time_pred is not None:
            time_pred_seq = torch.stack(time_pred_seq)
        else:
            time_pred_seq = None
        
        # Switch back to CPU mode
        cpu = torch.device('cpu')
        self = self.to(cpu)
        self.device = cpu
        self.use_cuda = False

        return plain_out_seq.to(cpu), time_pred_seq

    @staticmethod
    def compute_decayed_loss(criterion, pred, trg, kernel_bandwidth, kernel_func):
        loss = criterion(pred, trg)
            
        if loss.dim() == 4 and loss.size(2) == 1:
            loss = loss.squeeze(2)

        # apply decay kernel
        kernel = kernel_func(loss, kernel_bandwidth)
        loss = loss * kernel
        loss = loss.mean()
        
        return loss


    @staticmethod
    def decay_kernel(seq_tensor, bandwidth=3):
        # bandwith: if it is large, it returns uniform dist
        #           if it is small (=1), put high weight on very recent.
        # seq_tensor: n_batch x n_step x n_events
        n_step = seq_tensor.size(1)
        steps = torch.range(1, n_step).to(seq_tensor.device) - 1
        kernel = torch.exp(-(steps / bandwidth))
        kernel = kernel.unsqueeze(0).unsqueeze(2).expand_as(seq_tensor).flip(1)
        return kernel

    def remove_zeroed_batch_elems(self, hidden, inp_seq, trg_seq, len_inp_step,
                                  len_trg_step, inp_time_seq=None, 
                                  trg_time_seq=None):
        # if type(hidden) not in [list, tuple]:
        #     hidden = hidden.unsqueeze(0)

        _len_inp_step = len_inp_step
        for b_idx, b_len in reversed(list(enumerate(_len_inp_step))):
            if b_len == 0:
                
                if self.model.rnn_type in ['GRU']:
                    hidden = torch.cat([hidden[0:b_idx],
                                        hidden[b_idx + 1:]])

                if self.model.rnn_type in ['LSTM', 'MyLSTM', 'GRU',
                                                'NoRNN']:
                    inp_seq = torch.cat([inp_seq[0:b_idx],
                                         inp_seq[b_idx + 1:]])
                    trg_seq = torch.cat([trg_seq[0:b_idx],
                                         trg_seq[b_idx + 1:]])

                    len_inp_step = torch.cat([len_inp_step[0:b_idx],
                                              len_inp_step[b_idx + 1:]])
                    len_trg_step = torch.cat([len_trg_step[0:b_idx],
                                              len_trg_step[b_idx + 1:]])

                if trg_time_seq is not None:
                    trg_time_seq = torch.cat([trg_time_seq[0:b_idx],
                                              trg_time_seq[b_idx + 1:]])
                if inp_time_seq is not None:
                    inp_time_seq = torch.cat([inp_time_seq[0:b_idx],
                                              inp_time_seq[b_idx + 1:]])

        if type(hidden) not in [list, tuple]:
            hidden = hidden.unsqueeze(0)

        return hidden, inp_seq, trg_seq, len_inp_step, len_trg_step, inp_time_seq, trg_time_seq

    def get_optimizer(self):
        if self.optim == 'sparse_adam':
            optimizer = optim.SparseAdam(self.model.parameters(),
                                         lr=self.learning_rate)
        elif self.optim == 'sgd':
            optimizer = optim.SGD(self.model.parameters(),
                                  lr=self.learning_rate,
                                  weight_decay=self.weight_decay)
        elif self.optim == 'rmsprop':
            optimizer = optim.RMSprop(self.model.parameters(),
                                      lr=self.learning_rate,
                                      weight_decay=self.weight_decay)
        elif self.optim == 'clippedadam':
            optimizer = None
            # optimizer = ClippedAdam(self.model.parameters(),
            #                         lr=self.learning_rate)
        else:
            optimizer = optim.Adam(self.model.parameters(),
                                   lr=self.learning_rate,
                                   weight_decay=self.weight_decay)
        return optimizer

    def get_scheduler(self, optimizer):
        if self.lr_scheduler_ror:
            scheduler = ReduceLROnPlateau(
                optimizer, factor=self.lr_scheduler_mult, verbose=True, patience=5)
        elif self.lr_scheduler_multistep:
            scheduler = MultiStepLR(optimizer,
                                    milestones=self.lr_scheduler_epochs,
                                    gamma=self.lr_scheduler_mult)
        else:
            scheduler = StepLR(optimizer,
                               step_size=self.lr_scheduler_numiter,
                               gamma=self.lr_scheduler_mult)
        return scheduler

    def get_dataloader(self, input_data, shuffle=True):
        data = DatasetWithLength_multi(input_data)
        dataloader = torch.utils.data.DataLoader(data,
                                                 batch_size=self.batch_size,
                                                 shuffle=shuffle,
                                                 num_workers=self.num_workers,
                                                 drop_last=False,
                                                 pin_memory=True,
                                                 collate_fn=padded_collate_multi)
        return dataloader


    def process_batch(self, data, epoch):
                
        if self.elapsed_time:
            inp, trg, inp_time, trg_time, len_inp, len_trg = data

            if type(inp_time) == torch.Tensor:
                inp_time = inp_time / 3600
            else:   
                inp_time = [[[tk/3600 for tk in tj] for tj in ti] for ti in inp_time]

            if type(trg_time) == torch.Tensor:
                trg_time = trg_time / 3600
            else:
                trg_time = [[[tk/3600 for tk in tj] for tj in ti] for ti in trg_time]
        else:
            inp, trg, len_inp, len_trg = data
            trg_time = inp_time = None

        """
        inp: batch_size x max_seq_len x n_events
        len_inp: batch_size
        """
        
        if type(inp) == torch.Tensor and type(trg) == torch.Tensor:
            inp, trg, len_inp, len_trg, trg_time, inp_time = \
                sort_minibatch_multi(inp, trg, len_inp, len_trg,
                                    trg_time, inp_time)
            inp, trg = inp.float(), trg.float()
            len_inp = torch.LongTensor(len_inp)
            len_trg = torch.LongTensor(len_trg)

        if self.curriculum_learning:
            max_seq_len = int(self.max_seq_len_init * (
                    self.curriculum_rate ** epoch))
            max_seq_len = min(max_seq_len, inp.size(1))
            index = torch.LongTensor(list(range(max_seq_len)))
            if self.use_cuda:
                index = index.to(self.device)

            inp = torch.index_select(inp, 1, index)
            len_inp = [min(max_seq_len, x) for x in len_inp]

        
        if self.use_cuda:
            if type(inp) == torch.Tensor:
                inp = inp.to(self.device)
            if type(trg) == torch.Tensor:
                trg = trg.to(self.device)
            if inp_time is not None and type(inp_time) == torch.Tensor:
                inp_time = inp_time.to(self.device)
            if trg_time is not None and type(trg_time) == torch.Tensor:
                trg_time = trg_time.to(self.device)

        return inp, trg, len_inp, len_trg, inp_time, trg_time


class ResidualSeqNet(nn.Module):
    def __init__(self, embed_dim, hidden_dim, input_dim, num_layers, dropout, device, rnn_type='MyLayerLSTM'):
        super(ResidualSeqNet, self).__init__()
        self.rnn_type = rnn_type
        self.embed_input = nn.Linear(input_dim, embed_dim, bias=False)

        self.rnn = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            num_layers=num_layers,
            dropout=dropout)
        self.rnn = self.rnn.to(device)
        
        self.fc_out_residual = nn.Linear(hidden_dim, input_dim)
        self.fc_out = nn.Linear(hidden_dim, input_dim)

    def forward(self, seq_events, lengths, hidden, trg_times=None, inp_times=None):
        input_seq = self.embed_input(seq_events)
        _output, hidden = self.rnn(input_seq, hidden)
        plain_output = self.fc_out(_output) + self.fc_out_residual(_output)
        return plain_output, None, hidden



def load_multitarget_data(data_name, set_type, event_size, data_filter=None,
                          base_path=None, x_hr=None, y_hr=None, test=False,
                          midnight=False, labrange=False,
                          excl_ablab=False,
                          excl_abchart=False,
                          split_id=None, icv_fold_ids=None, icv_numfolds=None,
                          remapped_data=False,
                          use_mimicid=False, option_str="", pred_labs=False,
                          pred_normal_labchart=False,
                          inv_id_mapping=None, target_size=None,
                          elapsed_time=False,
                          x_as_list=False,
                          ):

    d_path = get_data_file_path(base_path, data_name, x_hr, y_hr, data_filter,
                                set_type, midnight, labrange, excl_ablab, excl_abchart, test,
                                use_mimicid, option_str, elapsed_time, split_id)

    if remapped_data:
        remap_str = '_remapped'
    else:
        remap_str = ''

    if icv_fold_ids is not None and icv_numfolds is not None \
            and data_name in ['mimic3', 'tipas']:
        # internal cross validation
        bin_xs, bin_ys = {}, {}
        for icv_fold_id in icv_fold_ids:

            d_path_fold = d_path + '/cv_{}_fold_{}'.format(
                icv_numfolds, icv_fold_id)
            logging.info("data path: {}".format(d_path_fold))
            x_path = '{}/hadm_bin_x{}.npy'.format(d_path_fold, remap_str)
            y_path = '{}/hadm_bin_y{}.npy'.format(d_path_fold, remap_str)

            with Timing('load {} ... '.format(x_path)):
                bin_xs.update(cp_load_obj(x_path))
            with Timing('load {} ... '.format(y_path)):
                bin_ys.update(cp_load_obj(y_path))
        bin_x, bin_y = bin_xs, bin_ys

        print('elapsed_time:{}'.format(elapsed_time))
        dataset = to_multihot_sorted_vectors(bin_x, bin_y, 
                                            input_size=event_size,
                                            elapsed_time=elapsed_time, 
                                            pred_labs=pred_labs,
                                            pred_normal_labchart=pred_normal_labchart,
                                            inv_id_mapping=inv_id_mapping,
                                            target_size=target_size,
                                            x_as_list=x_as_list,
                                            )

    else:
        # non internal cv data
        logging.info("data path: {}".format(d_path))

        print("data path: {}".format(d_path))
        x_path = '{}/hadm_bin_x{}.npy'.format(d_path, remap_str)
        y_path = '{}/hadm_bin_y{}.npy'.format(d_path, remap_str)

        dataset_path = x_path.replace('.npy', '.dataset')
        
        print("not found computed multihot vectors. let's processing.")
        with Timing('load {} ... '.format(x_path)):
            bin_x = cp_load_obj(x_path)

        with Timing('load {} ... '.format(y_path)):
            bin_y = cp_load_obj(y_path)

        dataset = to_multihot_sorted_vectors(bin_x, bin_y, input_size=event_size,
                                            elapsed_time=elapsed_time, 
                                            pred_labs=pred_labs,
                                            pred_normal_labchart=pred_normal_labchart,
                                            inv_id_mapping=inv_id_mapping,
                                            target_size=target_size,
                                            x_as_list=x_as_list,
                                            )
    return dataset, d_path


def pack_n_sort(instances):
    # print('instances:{}'.format(instances))
    instances = sorted(instances, key=lambda x: x[0].size(0))
    x_bin_lens = [instance[0].size(0) for instance in instances]
    y_bin_lens = [instance[1].size(0) for instance in instances]
    return (instances, x_bin_lens, y_bin_lens)


def load_multitarget_dic(base_path, data_name, x_hr, y_hr, data_filter=None,
                         set_type=None, midnight=False, labrange=False,
                         excl_ablab=False, excl_abchart=False,
                         test=False, split_id=None, remapped_data=False,
                         use_mimicid=False, option_str="", elapsed_time=False,
                         get_vec2mimic=False
                         ):

    fname = 'vec_idx_2_label_info'
    fname += '_labrange' if labrange else ''
    fname += '_exclablab' if excl_ablab else ''
    fname += '_exclabchart' if excl_abchart else ''
    if test:
        fname += '_TEST'
    if split_id is not None:
        fname += '_split_{}'.format(split_id)

    if remapped_data:
        fname += '_remapped_dic'
        set_type = 'train'
        d_path = get_data_file_path(
            base_path, data_name, x_hr, y_hr, data_filter, set_type, 
            midnight, labrange, excl_ablab, excl_abchart, test, use_mimicid, option_str, 
            elapsed_time
        )
        d_path += '/split_{}'.format(split_id)

    else:
        d_path = base_path


    dicfile = '{}/{}.npy'.format(d_path, fname)


    with Timing('read {} ... '.format(dicfile)):
        vecidx2label = np.load(dicfile).item()

    event_size = len(vecidx2label)

    if get_vec2mimic:
        vec2mimic_f = dicfile.replace('remapped_dic', 'vecidx2mimic')

        with Timing('read {} ... '.format(vec2mimic_f)):
            vecidx2mimic = np.load(vec2mimic_f).item()

        vecidx2label = (vecidx2label, vecidx2mimic)

    return vecidx2label, event_size


def get_data_file_path(base_path, data_name, x_hr, y_hr, data_filter=None,
                       set_type=None, midnight=False, labrange=False,
                       excl_ablab=False, excl_abchart=False,
                       test=False, use_mimicid=False, option_str="", 
                       elapsed_time=False, split_id=None):
    test_str = '_TEST' if test else ''
    opt_str = '_midnight' if midnight else ''

    if use_mimicid:
        opt_str += '_mimicid'

    if elapsed_time:
        opt_str += '_elapsedt'

    opt_str += option_str

    lr_str = '_labrange' if labrange else ''
    lr_str += '_exclablab' if excl_ablab else ''
    lr_str += '_exclabchart' if excl_abchart else ''
    print("lr_str: {}".format(lr_str))
    d_path = '{}/mimic_{}_xhr_{}_yhr_{}_ytype_multi_event{}' \
                '{}_singleseq{}'.format(base_path,
                                        set_type,
                                        x_hr, y_hr,
                                        lr_str, test_str,
                                        opt_str)

    if data_name in ['mimic3']:
        if split_id is not None:
            d_path += '/split_{}'.format(split_id)

    return d_path
