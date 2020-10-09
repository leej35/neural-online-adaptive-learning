
import itertools
import multiprocessing
multiprocessing.set_start_method('spawn', True)

from comet_ml import Experiment
import os
import sys

import torch
from multiprocessing.reduction import ForkingPickler
from torch.multiprocessing import reductions
from torch.utils.data import dataloader
import traceback

import copy
import collections
import logging
import time
import csv

from pathos.multiprocessing import ProcessingPool as Pool

import numpy as np

from utils.project_utils import Timing
from trainer import Trainer, load_multitarget_data, load_multitarget_dic
from models.base_seq_model import masked_bce_loss
from main_utils import (get_parser, create_model, add_main_setup, get_weblog,
                        print_args, load_model)

SEED = 5
API_KEY = ''

np.random.seed(SEED)
torch.manual_seed(SEED)
logger = logging.getLogger(__name__)
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')

logger.setLevel(logging.DEBUG)


def main():
    
    # args
    args = get_parser()
    args = add_main_setup(args)

    # single machine from here
    print('split: {}'.format(args.split_id))

    args = get_dict_and_info(args)

    if not args.skip_hypertuning:
        hyperparam_settings = get_hyperparam_setttings(args)
        best_hyper_param, best_epoch = hyperparam_tuning(copy.deepcopy(args),
                                                         hyperparam_settings)
        args = combine_namedtuple(args, best_hyper_param)

    # training for final run
    print('\n{}'.format('=' * 64))
    print('start final run')
    print('\n{}'.format('-' * 64))
    args.web_logger = get_weblog(API_KEY, args)
    args.web_logger.log_other('model_name', args.model_name)
    args.web_logger.log_other('run_mode', 'final_run')
    args.web_logger.log_other('split_id', args.split_id)
    args.web_logger.log_other('code_name', args.code_name)
    print_args(args)

    if not args.skip_hypertuning:
        for name, value in best_hyper_param._asdict().items():
            args.web_logger.log_parameter(name, value)

    with Timing('Loading data files...\n'):
        train_dataset, test_dataset, seqlen_idx, train_data_path, target_size, \
            inv_id_mapping \
            = get_dataset(args, args.split_id, args.event_size, args.base_path,
                          use_valid=args.use_valid,
                          simulated_data=args.simulated_data)
        
        args.target_size = target_size
        args.inv_id_mapping = inv_id_mapping

        if args.use_valid:
            (test_dataset, valid_dataset) = test_dataset
        else:
            valid_dataset = None  # means no early stopping

    with Timing('Creating model and trainer...'):
        model = create_model(args, args.event_size, args.device,
                             args.target_type, args.vecidx2label,
                             train_dataset,
                             train_data_path,
                             hidden_dim=args.hidden_dim)

        trainer = Trainer(model, args=args)
        if not args.skip_hypertuning:
            trainer.epoch = best_epoch
            trainer.force_epoch = True  # on final train, stop by best epoch
            args.web_logger.log_other('best_epoch', best_epoch)
            print('best_epoch from hypertuning: {}'.format(best_epoch))

    if args.load_model_from is not None:
        trainer = load_model(args, trainer, args.device)
    if not args.eval_only:
        trainer.train_model(train_dataset, split10_final=True,
                            valid_data=valid_dataset)

    with Timing('Save final trained model\n'):
        trainer.save('{}_final.model'.format(args.model_prefix))

    if args.eval_on_cpu:
        print('='*24)
        print('Run evaluation on CPU')
        cpu = torch.device('cpu')
        trainer = trainer.to(cpu)
        trainer.device = cpu
        trainer.model.device = cpu
        trainer.use_cuda = False
        trainer.model.use_cuda = False
        trainer.model.to(cpu)

    print('='*24)
    with Timing('Doing final evaluation...', one_line=False):
        if args.load_model_from is None and valid_dataset is not None:
            trainer.load_best_epoch_model()
            trainer.save_final_model()

        print('\n{}'.format('-' * 64))
        print('Eval stats')
        print('{}'.format('-' * 64))
        
        last_slash = args.model_prefix.rfind('/')
        csvfile = args.model_prefix[:last_slash + 1] + 'metric'

        eval_stats = trainer.infer_model(test_dataset,
                                         test_name='final_test', final=True,
                                         export_csv=True,
                                         csv_file=csvfile + '_test.csv',
                                         eval_multithread=args.eval_multithread)

        print('\n{}'.format('-' * 64))

    print('eval stats: {}'.format(eval_stats))

def get_hyperparam_setttings(args):

    hyperparam_settings = []

    HyperArgs = collections.namedtuple('HyperArgs',
                                    'batch_size learning_rate embedding_dim '
                                    'hidden_dim bptt dropout dropout_emb '
                                    'wdrop weight_decay')


    # dropout on hidden states as output of LSTM/RNN
    if args.rnn_type is not 'NoRNN':
        dropouts = [0] # , 0.5
        embedding_dims = [args.embedding_dim]
    else:
        dropouts = [0]
        embedding_dims = [0]

    # dropout on word embedding output
    dropout_embs = [0]
    # dropout on hidden-to-hidden parameter of LSTM/RNN
    wdrops = [0]

    if args.hyper_batch_size is not None:
        batch_sizes = [int(x) for x in args.hyper_batch_size]
        print("batch_sizes: {}".format(batch_sizes))
    else:
        batch_sizes = [args.batch_size]

    if args.hyper_learning_rate is not None:
        learning_rates = [float(x) for x in args.hyper_learning_rate]
        print("learning_rates: {}".format(learning_rates))
    else:
        learning_rates = [args.learning_rate]

    if args.hyper_weight_decay is not None:
        weight_decays = [float(x) for x in args.hyper_weight_decay]
        print("weight_decays: {}".format(weight_decays))
    else:
        weight_decays = [1e-06, 1e-07, 1e-08]

    if args.hyper_bptt is not None:
        bptts = [int(x) for x in args.hyper_bptt]
        print("bptts: {}".format(bptts))
    else:
        bptts = [args.bptt]

    if args.hyper_hidden_dim is not None:
        hidden_dims = [int(x) for x in args.hyper_hidden_dim]
        print("hidden_dims: {}".format(hidden_dims))
    else:
        hidden_dims = [args.hidden_dim]  # [64, 128, 256]

    for batch_size in batch_sizes:
        for lr in learning_rates:
            for e_dim in embedding_dims:
                for h_dim in hidden_dims:
                    for bptt in bptts:
                        for dropout in dropouts:
                            for dropout_emb in dropout_embs:
                                for wdrop in wdrops:
                                    for weight_decay in weight_decays:
                                        hyperparam_settings.append(
                                            HyperArgs(
                                                batch_size, lr, e_dim, h_dim,
                                                bptt, dropout, dropout_emb,
                                                wdrop,
                                                weight_decay
                                            )
                                        )

    return hyperparam_settings


def _runner(args):
    try:
        hyper_idx, package, fold_id, args, n_settings = args
        args = copy.deepcopy(args)
        hyper_args = package['hyper_args']
        args.model_prefix += '/h{}_f{}/'.format(hyper_idx, fold_id)
        os.system("mkdir -p {}".format(args.model_prefix))
        args = combine_namedtuple(args, hyper_args)
        if hasattr(args, 'web_logger'):
            args.web_logger.end()
            del args.web_logger
        args.web_logger = get_weblog(API_KEY, args)
        args.web_logger.log_other('model_name', args.model_name)
        args.web_logger.log_other('run_mode', 'hypertune')
        args.web_logger.log_other('split_id', args.split_id)
        args.web_logger.log_other('hyper_idx', hyper_idx)
        args.web_logger.log_other('fold_id', fold_id)
        args.web_logger.log_parameters(hyper_args._asdict())
        print('\n{}\nhyper parameter tuning {}/{}  '
                    '\nsplit:{} '
                    '\nhyper parameter trying: {} '
                    '\ncv-fold:{}'
                    '\n{}'
                    ''.format('-' * 64, hyper_idx,
                            n_settings,
                            args.split_id, hyper_args, fold_id,
                            '-' * 64))

        print_args(args)

        with Timing('Loading data files...\n'):
            train_dataset_cv, valid_dataset_cv, seqlen_idx, train_data_path,\
                target_size, inv_id_mapping \
                = get_dataset(args, args.split_id, args.event_size,
                            args.base_path, fold_id, args.num_folds, 
                            simulated_data=args.simulated_data)

        args.target_size = target_size
        args.inv_id_mapping = inv_id_mapping
        with Timing('Creating model and trainer...'):
            model = create_model(args, args.event_size, args.device,
                                args.target_type, args.vecidx2label,
                                train_dataset_cv,
                                train_data_path)

            trainer = Trainer(model, args=args)

        _ = trainer.train_model(train_dataset_cv,
                                valid_dataset_cv)
        eval_stats = trainer.infer_model(valid_dataset_cv,
                                        test_name='hyper_valid',
                                        final=True,
                                        export_csv=False,
                                        eval_multithread=args.eval_multithread,
                                        return_metrics=True,
                                        no_other_metrics_but_flat=True
                                        )
        best_mac_auprc = eval_stats[-1]
        print("best metric: {:.4f}".format(best_mac_auprc))
        package['stats'].append(best_mac_auprc)
        package['best_epochs'].append(trainer.best_epoch)
        del trainer
        del model
        del train_dataset_cv
        del valid_dataset_cv
        os.system('rm -rf {}epoch*.model'.format(args.model_prefix))
        args.web_logger.end()

    except Exception:
        print("Exception in worker:")
        traceback.print_exc()
        raise

    return package


def hyperparam_tuning(args, hyperparam_settings):
    Box = collections.namedtuple('Box', 'stats param best_epochs')
    boxs = []

    homeworks = []
    for hyper_idx, hyper_args in enumerate(hyperparam_settings):

        package = {'hyper_idx': hyper_idx, 'stats': [],
                   'hyper_args': hyper_args, 'best_epochs': []}

        num_folds = args.num_folds if args.fast_folds is None else args.fast_folds

        for fold_id in range(num_folds):
            homeworks.append(
                (
                    hyper_idx, package, fold_id, args,
                    len(hyperparam_settings)
                )
            )


    # packages = [_runner(hw) for hw in homeworks] ## when debug
    # run workers asynchly

    # method : pathos 
    pool = Pool(nodes=args.multiproc)
    packages = pool.amap(_runner, homeworks)
    while not packages.ready():
        time.sleep(5)

    packages = packages.get()

    print('='*20)
    print('hyper param tuning done!')
    print('='*20)
    # packages = packages.get()

    # merge by same hyper_idxs (from different internal-cv sets)
    boxs = {}
    for pack in packages:
        hyper_idx = pack['hyper_idx']

        if hyper_idx not in boxs:
            boxs[hyper_idx] = {'stats': [], 'best_epochs': [],
                               'hyper_args': pack['hyper_args']}

        boxs[hyper_idx]['stats'] += pack['stats']
        boxs[hyper_idx]['best_epochs'] += pack['best_epochs']

    # get best hyperparam
    best_mean, best_std, best_param, best_epoch = 0, 0, None, 0

    with open(('{}hypertune_result.csv'.format(args.model_prefix)), 'w') as f_csv:
        writer = csv.writer(f_csv)
        fields = hyperparam_settings[0]._fields
        writer.writerow(list(fields) + ['avg', 'std', 'best_epoch'])
        for box in list(boxs.values()):
            writer.writerow(
                list(box['hyper_args']._asdict().values())
                + [
                    np.mean(box['stats']),
                    np.std(box['stats']),
                    int(np.mean(box['best_epochs']))
                ]
            )
            if np.mean(box['stats']) > best_mean:
                best_param = box['hyper_args']
                best_mean = np.mean(box['stats'])
                best_std = np.std(box['stats'])
                best_epoch = int(np.mean(box['best_epochs']))

    print('\n{}'.format('=' * 64))
    print('best hyperparam info \n param:{} \n mean:{} std: {}'
                 '\n best epoch: {}'.format(
                     best_param, best_mean, best_std, best_epoch
                 ))
    print('\n{}'.format('-' * 64))

    return best_param, best_epoch


def combine_namedtuple(args, hyper_args):
    """
    Overwrite hyper_args into args
    """
    for name in hyper_args._fields:
        setattr(args, name, getattr(hyper_args, name))
    return args


def get_dict_and_info(args):
    target_type = 'multi'

    if args.data_name == 'mimic3':
        base_path = '{}'.format(
            args.base_path)
    else:
        base_path = args.base_path

    loss_fn = masked_bce_loss

    vecidx2label, event_size = load_multitarget_dic(base_path,
                                                    data_name=args.data_name,
                                                    data_filter=args.data_filter,
                                                    x_hr=args.window_hr_x,
                                                    y_hr=args.window_hr_y,
                                                    set_type=None,
                                                    midnight=args.midnight,
                                                    labrange=args.labrange,
                                                    excl_ablab=args.excl_ablab,
                                                    excl_abchart=args.excl_abchart,
                                                    test=args.testmode,
                                                    split_id=args.split_id,
                                                    remapped_data=args.remapped_data,
                                                    use_mimicid=args.use_mimicid,
                                                    option_str=args.opt_str,
                                                    elapsed_time=args.elapsed_time,
                                                    get_vec2mimic=args.prior_from_mimic,
                                                    )

    if args.prior_from_mimic:
        vecidx2label, vecidx2mimic = vecidx2label
        args.vecidx2mimic = vecidx2mimic
    else:
        args.vecidx2mimic = None
        
    args.event_size = event_size
    args.vecidx2label = vecidx2label
    args.event_dic = vecidx2label
    args.loss_fn = loss_fn
    args.target_type = target_type
    args.base_path = base_path
    return args


def get_merged_id_for_abnormal_normal(args):
    assert args.event_dic is not None
    assert not args.excl_ablab
    assert args.labrange
    # support for lab range, non-excl_ablab and non-excl_abchart
    # merge item-id index for abnormal and normal ids into normal ones.
    ab2n_mapping = {} # args.event_dic
    non_labchart_items = []
    for orig_idx, item in args.event_dic.items():
        if item['category'] in ['lab', 'chart']:

            label = item['label']
            norm_label = label.replace('-NORMAL', '').replace('-ABNORMAL_LOW', '').replace('-ABNORMAL_HIGH', '').replace('-ABNORMAL', '')

            if norm_label not in ab2n_mapping:
                ab2n_mapping[norm_label] = {'normal':None,'abnormals':[]}

            if label.endswith('-ABNORMAL_LOW') or label.endswith('-ABNORMAL_HIGH') or label.endswith('-ABNORMAL'):
                ab2n_mapping[norm_label]['abnormals'].append(orig_idx)

            else: # label.endswith('-NORMAL'):
                ab2n_mapping[norm_label]['normal'] = orig_idx

        else:
            non_labchart_items.append(orig_idx)
    
    # check where no-normal item exist (if happens, assign itself for normal)
    for norm_label, entry in ab2n_mapping.items():
        if entry['normal'] == None:
            # print('None: {}'.format(norm_label))
            # print('entry: {}'.format(entry))
            entry['normal'] = entry['abnormals'][0]

    ab2n_mapping = {v['normal']: v['abnormals']
                    for k, v in ab2n_mapping.items()}  # discard name
    
    inv_ab2n_mapping = {}
    for normal_item, abnormal_items in ab2n_mapping.items():
        for ab_item in abnormal_items:
            inv_ab2n_mapping[ab_item] = normal_item

    normal_items = list(ab2n_mapping.keys())
    target_items = normal_items + non_labchart_items
    non_target_items = itertools.chain.from_iterable(ab2n_mapping.values())

    print('!! target_items: {}'.format(len(target_items)))
    print('!! non_target_items: {}'.format(len(list(non_target_items))))

    # create id mapping first (orig_id -> new_id) for target items

    id_mapping = {}
    inv_id_mapping = {}
    for orig_idx in target_items:

        # supports excl lab only!! 
        item = args.event_dic[orig_idx]
        new_idx = len(id_mapping) + 1
        id_mapping[new_idx] = {'orig_idx': orig_idx, 
                            'category': item['category'],
                            'label': item['label']}    
        inv_id_mapping[orig_idx] = new_idx

    # then, add lab and chart abnormal items to inv_id_mapping
    for orig_idx in non_target_items:
        normal_item_orig_idx = inv_ab2n_mapping[orig_idx]
        new_idx = inv_id_mapping[normal_item_orig_idx]
        inv_id_mapping[orig_idx] = new_idx

    return id_mapping, inv_id_mapping, target_items


def test_get_merged_id_for_abnormal_normal():
    event_dic = {
        1:{'category':'lab','label':'A'}, 
        2:{'category':'lab','label':'A-ABNORMAL_LOW'},
        3:{'category':'lab','label':'A-ABNORMAL'},
        4:{'category':'lab','label':'A-ABNORMAL_HIGH'},
        5:{'category':'lab','label':'B'},
    }
    id_mapping, inv_id_mapping = get_merged_id_for_abnormal_normal(event_dic)
    print('id_mapping:\n{}'.format(id_mapping))
    print('inv_id_mapping:\n{}'.format(inv_id_mapping))


def get_lab_target_id_mapping(args):
    
    assert args.event_dic is not None
    assert args.excl_ablab 
    assert not args.labrange
    id_mapping = {}
    inv_id_mapping = {}
    for orig_idx, item in args.event_dic.items():
        if item['category'] == 'lab':
            
            # supports excl lab only!! 
            
            new_idx = len(id_mapping) + 1
            id_mapping[new_idx] = {'orig_idx': orig_idx, 
                                'category': item['category'],
                                'label': item['label']}    
            inv_id_mapping[orig_idx] = new_idx

    return id_mapping, inv_id_mapping



def get_dataset(args, split_id, event_size, base_path, fold_id=None,
                num_folds=None, count_mode=False,
                use_valid=False, simulated_data=False):
    print('event_size: {}'.format(event_size))
    

    if args.pred_labs or args.pred_normal_labchart:
        if args.pred_labs:
            args.event_dic, inv_id_mapping = get_lab_target_id_mapping(args)
            target_items = None
        elif args.pred_normal_labchart:
            args.event_dic, inv_id_mapping, target_items = get_merged_id_for_abnormal_normal(
                args)
        target_size = len(args.event_dic)
        np.save('{}_event_dic_target_id.npy'.format(args.model_prefix), args.event_dic)
        np.save('{}_inv_id_mapping_target_id.npy'.format(
            args.model_prefix), inv_id_mapping)
        print('target_size: {}'.format(target_size))
    else:
        target_size, inv_id_mapping = event_size, None

    if fold_id is not None:
        valid_fold_id = [fold_id]
        train_fold_ids = list(range(num_folds))
        train_fold_ids.remove(fold_id)

        if args.testmode_by_onefold:
            # NOTE: 2020/01 for fast testmode run, only use first icv set.
            train_fold_ids = [train_fold_ids[0]]

    else:
        valid_fold_id = train_fold_ids = None

    seqlen_idx = 2

    train_dataset, train_data_path = \
        load_multitarget_data(args.data_name,
                            'train',
                            event_size,
                            data_filter=args.data_filter,
                            base_path=base_path,
                            x_hr=args.window_hr_x,
                            y_hr=args.window_hr_y,
                            test=args.testmode,
                            midnight=args.midnight,
                            labrange=args.labrange,
                            excl_ablab=args.excl_ablab,
                            excl_abchart=args.excl_abchart,
                            split_id=split_id,
                            icv_fold_ids=train_fold_ids,
                            icv_numfolds=num_folds,
                            remapped_data=args.remapped_data,
                            use_mimicid=args.use_mimicid,
                            option_str=args.opt_str,
                            pred_labs=args.pred_labs,
                                pred_normal_labchart=args.pred_normal_labchart,
                            inv_id_mapping=inv_id_mapping,
                            target_size=target_size,
                            elapsed_time=args.elapsed_time,
                            x_as_list=args.x_as_list,
                            )

    tr_len = len(train_dataset[seqlen_idx])
    print("# train seq: {}".format(tr_len))

    if use_valid:
        valid_type_str = 'valid'
    elif ((count_mode or fold_id is None)):
        valid_type_str = 'test'
    else:
        valid_type_str = 'train'

    valid_dataset, _ = \
        load_multitarget_data(args.data_name,
                            valid_type_str,
                            event_size,
                            data_filter=args.data_filter,
                            base_path=base_path,
                            x_hr=args.window_hr_x,
                            y_hr=args.window_hr_y,
                            test=args.testmode,
                            midnight=args.midnight,
                            labrange=args.labrange,
                            excl_ablab=args.excl_ablab,
                            excl_abchart=args.excl_abchart,
                            split_id=split_id,
                            icv_fold_ids=valid_fold_id,
                            icv_numfolds=num_folds,
                            remapped_data=args.remapped_data,
                            use_mimicid=args.use_mimicid,
                            option_str=args.opt_str,
                            pred_labs=args.pred_labs,
                            pred_normal_labchart=args.pred_normal_labchart,
                            inv_id_mapping=inv_id_mapping,
                            target_size=target_size,
                            elapsed_time=args.elapsed_time,
                            x_as_list=args.x_as_list,
                            )

    va_len = len(valid_dataset[seqlen_idx])
    print("# valid seq: {}".format(va_len))

    if use_valid:
        test_dataset, _ = \
            load_multitarget_data(args.data_name,
                                'test',
                                event_size,
                                data_filter=args.data_filter,
                                base_path=base_path,
                                x_hr=args.window_hr_x,
                                y_hr=args.window_hr_y,
                                test=args.testmode,
                                midnight=args.midnight,
                                labrange=args.labrange,
                                excl_ablab=args.excl_ablab,
                                excl_abchart=args.excl_abchart,
                                split_id=split_id,
                                icv_fold_ids=valid_fold_id,
                                icv_numfolds=num_folds,
                                remapped_data=args.remapped_data,
                                use_mimicid=args.use_mimicid,
                                option_str=args.opt_str,
                                pred_labs=args.pred_labs,
                                    pred_normal_labchart=args.pred_normal_labchart,
                                inv_id_mapping=inv_id_mapping,
                                target_size=target_size,
                                elapsed_time=args.elapsed_time,
                                x_as_list=args.x_as_list,
                                )
        valid_dataset = (test_dataset, valid_dataset)

    return train_dataset, valid_dataset, seqlen_idx, train_data_path, \
        target_size, inv_id_mapping


if __name__ == '__main__':
    main()
