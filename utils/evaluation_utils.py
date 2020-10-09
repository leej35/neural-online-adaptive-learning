
import copy
import torch
import os
import csv
from collections import Counter

import numpy as np
from joblib import Parallel, delayed
import multiprocessing
from tabulate import tabulate

import matplotlib
from functools import reduce
matplotlib.use('Agg')
import matplotlib.pyplot as plt

tabulate.PRESERVE_WHITESPACE = True
from sklearn.metrics import precision_recall_curve, roc_auc_score, auc, \
    average_precision_score, precision_score, recall_score, confusion_matrix,\
    roc_curve

from numba import jit


class MultiLabelEval(object):
    def __init__(self, event_size, use_cuda=False, device=None, event_dic=None,
                 event_types=None, micro_aucs=True, macro_aucs=True,
                 pred_labs=False, pred_normal_labchart=False):
        self.device = device
        self.eval = {}
        et = torch.FloatTensor([])
        self._metric = {'total_cnt': 0,
                        'total_exact_match': 0,
                        'total_hamming_loss': torch.zeros(1).to(device),
                        'total_correct': torch.zeros(1).to(device),
                        'total_match': torch.zeros(1).to(device),
                        'total_pred': torch.zeros(1).to(device),
                        'total_gold': torch.zeros(1).to(device),
                        'total_num': torch.zeros(1).to(device),
                        'pred_count': torch.zeros(event_size).to(device),
                        'total_cnt': 0,
                        'total_exact_match': 0,
                        }

        if macro_aucs:
            self._metric['y_probs'] = {i: [] for i in range(event_size)}
            self._metric['y_trues'] = {i: [] for i in range(event_size)}
        if micro_aucs:
            self._metric['y_ex_probs'] = {i: [] for i in range(event_size)}
            self._metric['y_ex_trues'] = {i: [] for i in range(event_size)}

        self.eval['flat'] = copy.deepcopy(self._metric)
        self.eval['overall_num'] = 0

        self.eval['tstep'] = {}
        self.eval['tstep-etype'] = {}
        self.eval['occur_steps'] = {}

        self.use_cuda = use_cuda
        self.micro_aucs = micro_aucs
        self.macro_aucs = macro_aucs

        self.pred_labs = pred_labs
        self.pred_normal_labchart = pred_normal_labchart
        self.event_dic = event_dic
        self.event_idxs = {}

        if event_dic is not None and event_types is not None:
            self.eval['etypes'] = {}
            self.event_types = event_types
            for e_type in self.event_types:
                self.eval['etypes'][e_type] = copy.deepcopy(
                    self._metric)

            for e_type in self.event_types:
                self.event_idxs[e_type] = []

            for eid, info in event_dic.items():

                # NOTE: as itemid in event_dic starts from 1 
                # but the itemid in the predicted vector starts from 0,
                # let's remap the id -1 (Sep 11 2019)

                eid = eid - 1

                self.event_idxs[info['category']].append(eid)

                if not pred_normal_labchart: 
                    #NOTE: pred_normal_labchart does not separate normal/abnormal on target side

                    if '-NORMAL' in info['label']:
                        if info['category'] == 'chart':
                            self.event_idxs['chart_normal'].append(eid)
                        elif info['category'] == 'lab':
                            self.event_idxs['lab_normal'].append(eid)

                    elif '-ABNORMAL' in info['label']:
                        if info['category'] == 'chart':
                            self.event_idxs['chart_abnormal'].append(eid)
                        elif info['category'] == 'lab':
                            self.event_idxs['lab_abnormal'].append(eid)

    def _to_cuda(self, dic):
        if type(dic) is not dict:
            raise NotImplementedError('input should be a type of dic')
        rtn = {}
        for k, v in dic.items():
            if isinstance(v, torch.Tensor):
                rtn[k] = v.to(self.device)
            else:
                rtn[k] = v

        return rtn

    def update_container(self, dic, prob, trg, mask=None):

        """
        Only save current prediction & true target in container and
        compute metrics later at the end of epoch
        """
        if len(prob.size()) == 2:
            prob = prob.unsqueeze(0)
            trg = trg.unsqueeze(0)
            if mask is not None:
                mask = mask.unsqueeze(0)

        for event_idx in range(prob.size(-1)):
            """
            select patient sequences only with y_true=1 exist
            (skip those don't have y_true=1)
            """
            p = prob[:, :, event_idx].contiguous().view(-1)
            t = trg[:, :, event_idx].contiguous().view(-1)
            if mask is not None:
                m = mask[:, :, event_idx].byte().contiguous().view(-1)
                p = torch.masked_select(p, m)
                t = torch.masked_select(t, m)
                    
            # move to cpu (they will be consumed by cpu eventually)
            p = p.to('cpu')
            t = t.to('cpu')

            # append event-specific sequences to container

            if self.macro_aucs:
                dic['y_probs'][event_idx] += p.tolist()
                dic['y_trues'][event_idx] += t.tolist()

            if self.micro_aucs:
                dic['y_ex_probs'][event_idx].append(p)
                dic['y_ex_trues'][event_idx].append(t)

        return dic

    def trad_update(self, dic, pred, trg, mask=None):
        """

        :param pred: seqlen x n_events
        :param trg: seqlen x n_events
        :param mask: seqlen x n_events
        :return:
        """
        dic['pred_count'] += pred.data.sum()
        correct = torch.sum(torch.mul(pred, trg))
        dic['total_exact_match'] += torch.equal(pred, trg)
        dic['total_match'] += (pred == trg).float().sum().data
        dic['total_gold'] += trg.sum().data  # recall
        dic['total_num'] += reduce(lambda x, y: x * y, trg.size())
        dic['total_correct'] += correct.data
        dic['total_pred'] += pred.sum().data  # prec
        dic['total_hamming_loss'] += (pred != trg).float().sum().data
        dic['total_cnt'] += 1

        if mask is not None:
            # when mask is provided, remove number of non-active entries in
            # the mask on the statsitics (^1 : invert op)
            num_zero_items = (mask.byte() ^ 1).sum().data.float()

            if pred.is_cuda:
                device = pred.get_device()
                num_zero_items = num_zero_items.to(device)

            dic['total_match'] -= num_zero_items
            dic['total_num'] -= num_zero_items

        return dic

    def _get_event_type_mask(self, _trg, event_types, event_idxs):
        # event_idx: element indices of the event type
        masks = {}
        zten = torch.zeros(_trg.size())
        if self.use_cuda:
            zten = zten.to(self.device)

        for etype in event_types:
            masks[etype] = copy.deepcopy(zten)
            for idx in event_idxs[etype]:

                if len(_trg.size()) == 2:
                    masks[etype][:, idx] = 1
                elif len(_trg.size()) == 3:
                    masks[etype][:, :, idx] = 1
        return masks

    def _update_step_mask(self, dic, pred, trg, mask=None, prob=None,
                          final=False, force_auroc=False):

        if (prob is not None) and (final or force_auroc):

            # NOTE: prediction & target update to container
            dic = self.update_container(dic, prob, trg, mask)

        if mask is not None:
            pred = mask * pred
            trg = mask * trg

        dic = self.trad_update(dic, pred, trg, mask)

        return dic

    def update(self, pred, trg, len_trg, base_step=None, final=False,
               prob=None, load_to_cpu=False, 
               skip_detail_eval=False, force_auroc=False,
               inp_first_step=None):

        if trg.size() > pred.size():
            trg = trg.squeeze(0)
        elif trg.size() < pred.size():
            pred = pred.squeeze(0)
            if prob is not None:
                prob = prob.squeeze(0)

        if pred.dim() == 2:
            pred = pred.unsqueeze(1)
        try:
            n_b, n_s, n_e = pred.size()
        except ValueError as e:
            raise e

        trg_len_mask = torch.FloatTensor(
            [[1] * l + [0] * (n_s - l) for l in len_trg.tolist()])
        if self.use_cuda:
            trg_len_mask = trg_len_mask.to(self.device)

        if trg_len_mask.size() == (n_s, n_b):
            trg_len_mask = trg_len_mask.permute(1, 0)

        trg_len_mask = trg_len_mask.expand(n_e, n_b, n_s)
        trg_len_mask = trg_len_mask.permute(1, 2, 0)

        # flat stats
        self.eval['flat'] = self._update_step_mask(
            self.eval['flat'], pred, trg, mask=trg_len_mask,
            prob=prob, final=final, force_auroc=force_auroc)
        self.eval['overall_num'] += trg_len_mask.sum()

        if final and not skip_detail_eval:

            # event category stats
            if self.event_dic \
                    and 'category' in list(self.event_dic.values())[0]:
                e_masks = self._get_event_type_mask(trg, self.event_types,
                                                    self.event_idxs)
                for category in self.event_types:

                    # category overall
                    self.eval['etypes'][category] = self._update_step_mask(
                        self.eval['etypes'][category], pred, trg,
                        mask=e_masks[category] * trg_len_mask, prob=prob,
                        final=final)

            # step specific
            for step in range(n_s):
                if base_step is not None:
                    step += base_step
                if step not in self.eval['tstep']:
                    self.eval['tstep'][step] = copy.deepcopy(self._metric)
                if step >= prob.size(1):  # prevent step += base_step overflow
                    continue
                try:
                    _prob = prob[:, step, :] if prob is not None else None
                except IndexError as e:
                    raise e
                _trg = trg[:, step, :]
                _pred = pred[:, step, :]
                self.eval['tstep'][step] = self._update_step_mask(
                    self.eval['tstep'][step], _pred,
                    _trg, prob=_prob, final=final)

                # event category stats AND timestep
                if self.event_dic \
                        and 'category' in list(self.event_dic.values())[0]:
                    e_masks = self._get_event_type_mask(_trg, self.event_types,
                                                        self.event_idxs)
                                                        
                    if step not in self.eval['tstep-etype']:
                        self.eval['tstep-etype'][step] = {category: copy.deepcopy(
                            self._metric) for category in self.event_types}

                    for category in self.event_types:
                        self.eval['tstep-etype'][step][category] = self._update_step_mask(
                            self.eval['tstep-etype'][step][category], 
                            _pred, _trg,
                            mask=e_masks[category], #* trg_len_mask, 
                            prob=_prob,
                            final=final)



    @staticmethod
    def compute(dic, epoch, logger=None, test_name='', tstep=False,
                avg='micro', cv_str='', option_str='', verbose=True,
                final=False, export_csv=False, export_probs=True, event_dic=None, 
                csv_file=None, return_metrics=False, use_multithread=True, 
                force_auroc=False, force_plot_auroc=False, do_plot=False, 
                target_auprc=False):
        total_exact_match = (dic['total_exact_match'] * 100.0 + 1e-09) / \
                            (dic['total_cnt'] + 1e-09)
        total_hamming_loss = (
                dic['total_hamming_loss'] / dic['total_num']).item()
        total_correct = dic['total_correct'][0]
        total_match = dic['total_match'][0]
        total_pred = dic['total_pred'][0]
        total_gold = dic['total_gold'][0]
        total_num = dic['total_num'][0]
        if total_pred > 0:
            prec = 100.0 * total_correct / total_pred
        else:
            prec = 0
        if total_gold > 0:
            recall = 100.0 * total_correct / total_gold
        else:
            recall = 0
        if prec + recall > 0:
            f1 = 2 * prec * recall / (prec + recall)
        else:
            f1 = 0
        if total_num > 0:
            acc = 100.0 * total_match / total_num
        else:
            acc = 0

        micro_prior = total_gold / total_num

        container = {}

        mac_auprc, mac_auroc, mac_ap, mi_auprc, mi_auroc, mi_ap, \
        mac_auprc_std, mac_auroc_std, mac_ap_std = [0.0] * 9
        less_point_five_auroc = 0  # count events with AUROC less than 0.5

        if 'y_probs' in dic:

            # macro averaged auroc, auprc, and ap

            """
            Multithread
            """

            def _auprc_auroc_ap_wrapper(i, probs, trues, auroc_only):
                spr, sro, sap, sprec, srecall, sspec, stn, sfp, sfn, stp, sacc \
                    = get_auprc_auroc_ap(trues, probs, auroc_only=auroc_only)
                return {i: spr}, {i: sro}, {i: sap}, {i: sprec}, {i: srecall}, \
                    {i: sspec}, {i: stn}, {i: sfp}, {i: sfn}, {i: stp}, {i: sacc}

            num_cores = 40
            output = list(zip(*Parallel(n_jobs=num_cores, prefer="threads")(
                delayed(_auprc_auroc_ap_wrapper)(
                    i,
                    dic['y_probs'][i],
                    dic['y_trues'][i],
                    auroc_only=force_auroc and (not final) and (not target_auprc),
                ) \
                for i in dic['y_probs'].keys() \
                if (len(dic['y_trues'][i]) > 0
                    and dic['y_probs'][i] != []
                    and dic['y_trues'][i] != []
                    )
            )))

            if output == []:
                mac_auprcs, mac_aurocs, mac_aps, mac_precs, mac_recs, \
                    mac_specs, mac_tn, mac_fp, mac_fn, mac_tp, mac_acc = [{0: 0}] * 11
            else:
                (prcs, rocs, aps, precisions, recalls, specificities, tns, fps, fns, tps, accs) = output
                # mac_auprcs = {list(x.keys())[0]: list(x.values())[0] for x in prcs \
                #                 if not np.isnan(list(x.values())[0])}
                mac_auprcs = {list(x.keys())[0]: list(x.values())[0] for x in prcs
                                if list(x.values())[0] is not None}
                mac_aurocs = {list(x.keys())[0]: list(x.values())[0] for x in rocs \
                                if list(x.values())[0] is not None}
                mac_aps = {list(x.keys())[0]: list(x.values())[0] for x in aps \
                            if not np.isnan(list(x.values())[0])}
                mac_precs = {list(x.keys())[0]: list(x.values())[0] for x in precisions \
                            if not np.isnan(list(x.values())[0])}
                mac_recs = {list(x.keys())[0]: list(x.values())[0] for x in recalls \
                            if not np.isnan(list(x.values())[0])}
                mac_specs = {list(x.keys())[0]: list(x.values())[0] for x in specificities \
                            if not np.isnan(list(x.values())[0])}
                mac_tn = {list(x.keys())[0]: list(x.values())[0] for x in tns
                            if not np.isnan(list(x.values())[0])}
                mac_fp = {list(x.keys())[0]: list(x.values())[0] for x in fps
                            if not np.isnan(list(x.values())[0])}
                mac_fn = {list(x.keys())[0]: list(x.values())[0] for x in fns
                            if not np.isnan(list(x.values())[0])}
                mac_tp = {list(x.keys())[0]: list(x.values())[0] for x in tps
                            if not np.isnan(list(x.values())[0])}
                mac_acc = {list(x.keys())[0]: list(x.values())[0] for x in accs
                            if not np.isnan(list(x.values())[0])}

            if export_csv:
                # merge two dicts by sum values of same key
                pos_points = dict(Counter(mac_tp) + Counter(mac_fn))
                neg_points = dict(Counter(mac_fp) + Counter(mac_tn))
                export_event_metrics(
                    csv_file, event_dic,
                    {
                        'mac_auprc': mac_auprcs,
                        'mac_auroc': mac_aurocs,
                        'mac_aps': mac_aps,
                        'mac_prec' : mac_precs,
                        'mac_recs' : mac_recs,
                        'mac_specs' : mac_specs,
                        'tn': mac_tn,
                        'fp': mac_fp,
                        'fn': mac_fn,
                        'tp': mac_tp,
                        'pos_n_points': pos_points,
                        'neg_n_points': neg_points,
                        'acc': mac_acc,
                    }
                )

            if final or force_auroc:
                for event_id, event_auroc in mac_aurocs.items():
                    if not np.isnan(event_auroc) and event_auroc < 0.5:
                        less_point_five_auroc += 1

                    # Plot AUROC / AUPRC figures
                    if (do_plot and final and event_dic is not None 
                            and event_id in event_dic 
                            and csv_file is not None
                        ):

                        y_true = dic['y_trues'][event_id]
                        y_prob = dic['y_probs'][event_id]

                        # NOTE: event_id in event_dic starts from 1. 
                        event_name = event_dic[event_id + 1]["label"]
                        event_category = event_dic[event_id + 1]["category"]

                        _precision, _recall, thresholds = \
                            precision_recall_curve(y_true, y_prob, pos_label=1)
                        auprc = auc(_recall, _precision)

                        fpr, tpr, _ = roc_curve(y_true, y_prob,
                            sample_weight=None,
                            pos_label=1)
                        tr_test = 'train' if '_train.csv' in csv_file else 'test'
                        f_path = '/'.join(csv_file.split('/')[:-1]) + '/plots'
                        os.system("mkdir -p {}".format(f_path))

                        event_name = event_name.replace('/','_')
                        event_name = event_name.replace('[', '_')
                        event_name = event_name.replace(']', '_')
                        event_name = event_name.replace(' ', '_')

                        f_path += '/{}plot_auroc_{:.2f}_auprc_{:.2f}_e_{}_{}_{}.png'.format(
                            tr_test, event_auroc, auprc, event_id, event_name, event_category
                        )
                        draw_roc_curve(
                            event_category+' '+event_name,
                            fpr, tpr, event_auroc, 
                            _precision, _recall, auprc,
                            f_path
                        )   

            # Macro averagings

            print('cnt nan (auprc) : {}'.format(sum(np.isnan(list(mac_auprcs.values())))))
            print('cnt nan (auroc) : {}'.format(sum(np.isnan(list(mac_aurocs.values())))))

            mac_auprc = np.nanmean(list(mac_auprcs.values())) * 100
            mac_auroc = np.nanmean(list(mac_aurocs.values())) * 100
            mac_ap = np.nanmean(list(mac_aps.values())) * 100

            # precision, recall, specificity
            mac_prec = np.nanmean(list(mac_precs.values())) * 100
            mac_rec = np.nanmean(list(mac_recs.values())) * 100
            mac_spec = np.nanmean(list(mac_specs.values())) * 100

            mac_auprc_std = np.nanstd(list(mac_auprcs.values()))
            mac_auroc_std = np.nanstd(list(mac_aurocs.values()))
            mac_ap_std = np.nanstd(list(mac_aps.values()))

            mac_prec_std = np.nanstd(list(mac_precs.values())) * 100
            mac_rec_std = np.nanstd(list(mac_recs.values())) * 100
            mac_spec_std = np.nanstd(list(mac_specs.values())) * 100

            mac_acc = np.nanmean(list(mac_acc.values())) * 100

            # Weighted Averagings

            occur_events_num = {k : sum(list(v)) for k, v in dic['y_trues'].items()}

            all_occurs = sum(list(occur_events_num.values())) + 1e-10
            weight_events = {k: v/all_occurs for k, v in occur_events_num.items()}

            wgh_auprc = sum([val * weight_events[idx] for idx,
                             val in mac_auprcs.items() if ~np.isnan(val)]) * 100
            wgh_auroc = sum([val * weight_events[idx] for idx, 
                             val in mac_aurocs.items() if ~np.isnan(val)]) * 100

            if return_metrics:
                container = {
                    'mac_auprc': mac_auprc,
                    'mac_auroc': mac_auroc,
                    'mac_ap': mac_ap,
                    'mac_auprc_std': mac_auprc_std,
                    'mac_auroc_std': mac_auroc_std,
                    'mac_ap_std': mac_ap_std,
                    'mac_prec': mac_prec,
                    'mac_rec': mac_rec,
                    'mac_spec': mac_spec,
                    'mac_prec_std': mac_prec_std,
                    'mac_rec_std': mac_rec_std,
                    'mac_spec_std': mac_spec_std,
                    'wgh_auprc': wgh_auprc,
                    'wgh_auroc': wgh_auroc,
                    'events_mac_auprcs': mac_auprcs,
                    'events_mac_aurocs': mac_aurocs,
                }


        if logger:
            tstep_str = ' time' if tstep else ''


            for met_name, score in zip(
                    ['precision', 'recall', 'f1', 'acc', 'micro_prior'],
                    [prec, recall, f1, acc, micro_prior]
            ):
                if type(score) == torch.Tensor:
                    score = score.cpu()
                logger.log_metric("{}: {}({}) {}{}{}".format(
                    test_name, met_name, avg, cv_str, tstep_str, option_str),
                    score, step=epoch)

            for met_name, score in zip(
                    [
                        'auprc', 'auroc', 'wgh_auprc', 'wgh_auroc',
                        'less_point_five_auroc'
                    ],
                    [
                        mac_auprc, mac_auroc, wgh_auprc, wgh_auroc,
                        less_point_five_auroc
                    ]
                ):
                if type(score) == torch.Tensor:
                    score = score.cpu()

                logger.log_metric("{}: {}({}) {}{}{}".format(
                    test_name, met_name, 'macro', cv_str, tstep_str,
                    option_str),
                    score, step=epoch)

        if return_metrics:
            return f1, acc, container

        return f1, acc


def get_auprc_auroc_ap(y_true, y_prob, skip_ap=True, auroc_only=False):
    precision, recall, thresholds \
        = precision_recall_curve(y_true, y_prob, pos_label=1)

    auprc = auc(recall, precision)

    # calculate average precision score
    if skip_ap:
        ap = 0
    else:
        ap = average_precision_score(y_true, y_prob)
        
    try:
        # auroc = roc_auc_score(y_true, y_prob)
        fpr, tpr, _ = roc_curve(y_true, y_prob,
                                sample_weight=None,
                                pos_label=1)
        auroc = auc(fpr, tpr)          

    except ValueError:
        auroc = None
    
    # print('y_prob: {} {}'.format(len(y_prob), y_prob))
    # print('y_true: {} {}'.format(len(y_true), y_true))

    y_pred = (np.array([y_prob]) > 0.5).astype('float').flatten()

    recall = recall_score(y_true, y_pred, pos_label=1, average='binary')
    prec = precision_score(y_true, y_pred, pos_label=1, average='binary')

    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    except ValueError:
        tn, fp, fn, tp = -1, -1, -1, -1

    specificity = tn / (tn+fp)
    acc = (tp + tn) / sum([tn, fp, fn, tp])

    return auprc, auroc, ap, prec, recall, specificity, tn, fp, fn, tp, acc


def export_event_metrics(filename, event_dic, metrics_dic):
    np.save(filename.replace('.csv', '_metric_dic.npy'), metrics_dic)
    with open(filename, mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',',
                                quotechar='"', quoting=csv.QUOTE_MINIMAL)

        # write header row
        csv_writer.writerow(
            ["category", "label"] + list(metrics_dic.keys())
        )

        # NOTE: in event_dic, event id starts from 1.  (so k is)
        #       in the multi-hot vector, event id starts from 0. (so keys in metrics_dic)
        #       So transpose (-1) needed.
        for k, v in event_dic.items():
            csv_writer.writerow(
                [v["category"], v["label"], ]
                + ['{:.8f}'.format(metric[k-1]) for metric in
                   list(metrics_dic.values()) if k-1 in metric]
            )



def draw_roc_curve(event_name, fpr, tpr, auroc, precision, recall, auprc, fname):
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
                lw=lw, label='ROC curve (area = %0.2f)' % auroc)
    plt.plot(recall, precision, color='navy',
             lw=lw, label='PR curve (area = %0.2f)' % auprc)

    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate // Precision')
    plt.ylabel('True Positive Rate // Recall')
    plt.title('ROC: {}'.format(event_name))
    plt.legend(loc="lower right")
    plt.savefig(fname, format='png')


def export_timestep_metrics(filename, model_name, metrics, event_dic=None):
    """

    :param filename: string csv file name
    :param model_name: string model name (1nd column name)
    :param metrics: list of dictionary of metrics over time series.
        order of elements in list represents order in time series
        e.g. [ {'metrics_one': 0.1, 'metrics_two': 0.1}, ... ]
    """
    with open(filename, mode='w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=',',
                                quotechar='"', quoting=csv.QUOTE_MINIMAL)

        # NOTE: manually inserting metric names
        metric_names = ["mac_auprc", "mac_auprc_std", "mac_auroc",
                        "mac_auroc_std"]
        # metric_names = metrics[0].keys()

        # write header row
        csv_writer.writerow([model_name] + metric_names)

        for step, metric_step in enumerate(metrics):

            csv_writer.writerow(
                [str(step), ]
                + ['{:.4f}'.format(metric_step[name]) for name in metric_names]
            )
    # Event-Type-Specific Results
    if event_dic is not None:
        for metric_name in ['events_mac_auprcs', 'events_mac_aurocs']:
            _filename = filename.replace('.csv', '_spec_{}.csv'.format(metric_name))

            with open(_filename, mode='w') as csv_file:
                csv_writer = csv.writer(csv_file, delimiter=',',
                                        quotechar='"', quoting=csv.QUOTE_MINIMAL)

                csv_writer.writerow(
                    ['steps', ] + [event_dic[itemidx+1]["category"] + "--" + event_dic[itemidx+1]["label"] 
                    for itemidx in metric_step[metric_name].keys()]
                )

                for step, metric_step in enumerate(metrics):

                    csv_writer.writerow(
                        [str(step), ] + ['{:.8f}'.format(metric) for itemidx, metric in
                            metric_step[metric_name].items()]
                    )
