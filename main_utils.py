import multiprocessing
multiprocessing.set_start_method('spawn', True)


import os
import sys
import socket
import logging
import pickle
from argparse import ArgumentParser

from comet_ml import Experiment
import torch

from utils.project_utils import Timing, count_parameters

from models.base_seq_model import BaseMultiLabelLSTM, masked_bce_loss
from trainer import load_multitarget_data, load_multitarget_dic

optimizer = None
do_multithreading = False

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p')
logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def create_model(args, event_size, device, target_type, vecidx2label,
                 train_dataset, train_data_path, hidden_dim=None, embedding_dim=None, batch_size=None,
                 num_layers=None, dropout=None, ):
    model = BaseMultiLabelLSTM(event_size=event_size,
                                window_size_y=args.window_hr_y,
                                target_size=args.target_size,
                                hidden_dim=hidden_dim \
                                    if hidden_dim else args.hidden_dim,
                                embed_dim=embedding_dim if embedding_dim \
                                    else args.embedding_dim,
                                batch_size=batch_size \
                                    if batch_size else args.batch_size,
                                use_cuda=args.use_cuda,
                                num_layers=num_layers \
                                    if num_layers else args.num_layers,
                                dropout=dropout if dropout \
                                    else args.dropout,
                                rnn_type=args.rnn_type,
                                batch_first=args.batch_first,
                                device=device,
                                dropouth=args.dropouth,
                                dropouti=args.dropouti,
                                remap=args.remapped_data,
                                inv_id_mapping=args.inv_id_mapping,
                                elapsed_time=args.elapsed_time,
                                )
      
    if args.use_cuda and model is not None and hasattr(model, 'parameters'):
        model = model.to(device)

    num_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('num_total_params: {}'.format(num_total_params))
    args.web_logger.log_other('num_total_params', num_total_params)

    return model



def load_model(args, trainer, device):
    with Timing('Loading model from {}...'.format(args.load_model_from)):
        try:
            trainer.load(args.load_model_from)

        except AttributeError:
            print(
                'Model is not just state_dict, loading whole model '
                '(source code change might affect the performance).')
            if not args.use_cuda:
                trainer.model = torch.load(args.load_model_from,
                                           map_location='cpu')
            else:
                trainer.model = torch.load(args.load_model_from)
        if args.use_cuda:
            trainer.model = trainer.model.to(device)
    return trainer



def get_parser():

    parser = ArgumentParser(description='To run RNN Timeseries Prediction Model')

    # data
    parser.add_argument('--data-name', dest='data_name',
                        default='None',
                        help='data name')
    parser.add_argument('--data-filter', dest='data_filter',
                        default='None')

    parser.add_argument('--window-hr-x', dest='window_hr_x', type=int, default=6,
                        help='window x size in hour')
    parser.add_argument('--window-hr-y', dest='window_hr_y', type=int, default=48,
                        help='window y size in hour')
    parser.add_argument('--load-model-from', dest='load_model_from',
                        help='Loads the model parameters from the given path')
    parser.add_argument('--num-workers', type=int, default=2, dest='num_workers',
                        help='Number of workers for data loader class')
    parser.add_argument('--batch-size', dest='batch_size', type=int, default=2,
                        help='The batch size. Default to 128')
    parser.add_argument('--base-path', dest='base_path', type=str,
                        default='../../',
                        help='base path for the dataset')
    parser.add_argument('--testmode', action='store_true',
                        dest='testmode', default=False)
    parser.add_argument('--midnight', action='store_true',
                        dest='midnight', default=False)
    parser.add_argument('--labrange', action='store_true',
                        dest='labrange', default=False)

    # model parameters


    parser.add_argument('--hidden-dim', dest='hidden_dim',
                        type=int, default=128,
                        help=('The size of hidden activation and '
                              'memory cell of LSTM. Default is 128'))

    parser.add_argument('--embedding-dim', dest='embedding_dim',
                        type=int, default=128,
                        help=('The size of word embedding '
                              'Default is 128'))

    parser.add_argument('--n-layers', dest='num_layers', type=int, default=1,
                        help=('Number of layers in RNN model.'))

    parser.add_argument('--rnn-type', dest='rnn_type', type=str,
                        default='NoRNN',
                        help='Type of RNN: (RNN, LSTM, GRU) ')

    parser.add_argument('--cuda', action='store_true', default=False,
                        dest='use_cuda', help='Use GPU for computation.')

    parser.add_argument('--not-batch-first', dest='batch_first',
                        action='store_false', default=True,
                        help='Batch first')
    parser.add_argument('--eval-on-cpu', action='store_true',
                        default=False)

    # optimization and training

    parser.add_argument('--learning-rate', dest='learning_rate', type=float,
                        default=0.005,
                        help='The learning rate for the model.')
    parser.add_argument('--epoch', dest='epoch', type=int, default=20,
                        help='The number of epoch to train on.')
    parser.add_argument('--max-doc-len', dest='max_doc_len', type=int, default=0,
                        help='Limit the maximum document length')
    parser.add_argument('--bptt', type=int, default=False)
    parser.add_argument('--patient-stop', type=int, default=10,
                        help='Patient stopping criteria')
    parser.add_argument('--use-bce-logit', action='store_true',
                        dest='use_bce_logit', default=False,
                        help='Use BCEWithLogit loss function.')
    parser.add_argument('--use-bce-stable', action='store_true',
                        dest='use_bce_stable', default=False,
                        help='Add eps (1e-12) to pred on bce loss.')
    parser.add_argument('--optimizer', type=str, default="adam")


    # print, load and save options
    parser.add_argument('--validate-every', dest='valid_every', type=int,
                        metavar='n', default=10,
                        help='Validate on validation data every n epochs')
    parser.add_argument('--print-every', type=int, default=5,)
    parser.add_argument('--save-every', type=int, default=-1,
                        help=('Save the model every this number of epochs. '
                              'Default to be the same as --validate-every. '
                              'If set to 0, will save only the final model.'))
    parser.add_argument('--model-prefix', type=str, default="logs/_tmp_",
                        help='Binarized model file will have '
                             'this prefix on its name.')

    # curriculum learning
    parser.add_argument('--curriculum', action='store_true',
                        dest='curriculum_learning', default=False,
                        help='Do curriculum learning.')
    parser.add_argument('--curriculum-rate', type=float,
                        dest='curriculum_rate', default=1.0005,
                        help='The rate for curriculum learning')
    parser.add_argument('--curriculum-init', type=int,
                        dest='curriculum_init', default=1,
                        help='The initial curriculum max seq length')

    # learning rate scheduler
    parser.add_argument('--lr-scheduler', dest='lr_scheduler',
                        action='store_true', default=False,
                        help='Learning Rate Opitmization Scheduler')
    parser.add_argument('--lr-scheduler-multistep',
                        dest='lr_scheduler_multistep', action='store_true',
                        default=False,
                        help='Use multi-step learning rate scheduler. '
                             'Specify the milestones using '
                             '--lr-scheduler-epochs.')
    parser.add_argument('--lr-scheduler-ror',
                        dest='lr_scheduler_ror', action='store_true',
                        default=False,
                        help='Use ReduceLROnPlateau learning rate scheduler. ')

    parser.add_argument('--lr-scheduler-epochs', dest='lr_scheduler_epochs',
                        type=int, nargs='+',
                        default=[45, 90, 135, 180, 225],
                        help='The epochs in which to reduce the learning rate.')
    parser.add_argument('--lr-scheduler-numiter', dest='lr_scheduler_numiter',
                        type=int, default=15,
                        help='Number of epochs before reducing learning rate.')
    parser.add_argument('--lr-scheduler-mult', dest='lr_scheduler_mult',
                        type=float, default=0.5,
                        help='The multiplier used to reduce the learning rate.')
    parser.add_argument('--gpu-id', type=int, dest='gpu_id', default=0)

    parser.add_argument('--aime-eval', action='store_true', default=False,
                        help='for evaluation code, run like AIME-2019 paper')
    parser.add_argument('--aime-eval-macro', dest='aime_eval_macro',
                        action='store_true', default=False,
                        help='macro auprc, auroc')
    parser.add_argument('--jit', action='store_true', default=False,
                        help='for pyro HMM implementation, use jit mode for '
                             'faster computation')

    parser.add_argument('--skip-hypertuning', action='store_true', default=False,
                        dest='skip_hypertuning',
                        help='skip hypertuning for Split10 setting')
    parser.add_argument('--not-eval-multithread', action='store_false', default=True,
                        dest='eval_multithread',
                        help='Multithread eval')

    # split10 & cross validation
    parser.add_argument('--num-folds', type=int, dest='num_folds', default=5)
    parser.add_argument('--split-id', type=int, dest='split_id', default=1)
    parser.add_argument('--model-name', type=str, dest='model_name')
    parser.add_argument('--code-name', type=str, dest='code_name', default='')
    parser.add_argument('--remapped-data', action='store_true', default=False,
                        dest='remapped_data')

    # regularizations (https://github.com/salesforce/awd-lstm-lm)
    parser.add_argument('--weight-decay', type=float, dest='weight_decay',
                        default=0,
                        help=('L2 Regularization coefficient'))
    parser.add_argument('--dropout', type=float, default=0,
                        help=('Dropout rate (0:none)'))
    parser.add_argument('--dropouth', type=float, default=0,
                        help=('Dropout rate (0:none) for RNN hidden states'))
    parser.add_argument('--dropouti', type=float, default=0,
                        help=('Dropout rate (0:none) for input embedding'))
    parser.add_argument('--wdrop', type=float, default=0,
                        help=('weight dropout to apply to the RNN hidden to hidden matrix'))
    parser.add_argument('--tie-weights', action='store_true', default=False,
                        help='share input embedding & output embedding')
    
    parser.add_argument('--single-run-cv', action='store_true', default=False,
                        help='Run cross validation with only one run')
    parser.add_argument('--force-epoch', action='store_true', default=False,
                        help='Turn off the ')
    parser.add_argument('--skip-hidden-state', action='store_true', default=False,
                        dest='skip_hidden_state')
    
    parser.add_argument('--use-mimicid', dest='use_mimicid', action="store_true",
                        default=False,
                        help='instead of mapped vec_index, use mimic itemid with'
                        'value string (abnormal/normal/etc.)')
    parser.add_argument('--opt-str', dest='opt_str', type=str, default="none",
                        help='optional string attached to output path name')
    parser.add_argument('--use-valid', dest='use_valid', action="store_true",
                        default=False)

    parser.add_argument('--clamp-prob', dest='clamp_prob', type=float, default=None,
                        help=('clamp output probability with this minimum bound'))

    # debug LR-last (oct-02-2019)
    parser.add_argument('--target-event', type=int, default=-1,
                        help=('only use this target id for predict & evaluation'))
    parser.add_argument('--force-checkpoint', dest='force_checkpointing', action="store_true",
                        default=False)
    parser.add_argument('--force-auroc', dest='force_auroc', action="store_true",
                        default=False)
    parser.add_argument('--force-comet', dest='force_comet', action="store_true",
                        default=False)
    parser.add_argument('--force-plot-auroc', dest='force_plot_auroc', action="store_true",
                        default=False)

    parser.add_argument('--multiproc', type=int, default=1, help='multiprocessing (number of cores)')
    
    parser.add_argument('--weight-change', dest='weight_change', action="store_true",
                        default=False)

    parser.add_argument('--loss-tol', type=float, default=1, help='loss tolerance')

    parser.add_argument('--weight-decay-range', nargs='+', type=float,
                            help='weight decay range for hyperparam tuning')
    
    parser.add_argument('--excl-ablab', dest='excl_ablab', action="store_true",
                        default=False)

    parser.add_argument('--excl-abchart', dest='excl_abchart', action="store_true",
                        default=False)

    parser.add_argument('--hyper-weight-decay', dest='hyper_weight_decay', 
                        action='append', default=None)

    parser.add_argument('--hyper-bptt', dest='hyper_bptt',
                        action='append', default=None)
                        
    parser.add_argument('--hyper-hidden-dim', dest='hyper_hidden_dim',
                        action='append', default=None)
    parser.add_argument('--hyper-batch-size', dest='hyper_batch_size',
                        action='append', default=None)
    parser.add_argument('--hyper-learning-rate', dest='hyper_learning_rate',
                        action='append', default=None)
    parser.add_argument('--warmup', dest='n_warmup_steps', type=int, 
                        default=1, help='number of warmup steps for Transformer')
    parser.add_argument('--testmode-by-onefold', action="store_true", default=False)

    parser.add_argument('--rb-init', dest='rb_init', type=str, default="None", 
                        help="xavier-prior-asweight or asbias")
    parser.add_argument('--manual-alpha', type=float, default=-1, 
                        help='manual alpha (importance weight) for periodicity module')
    
    parser.add_argument('--eval-only', dest='eval_only', action="store_true",
                        default=False, help="run evaluation only (skip training)")
    
    parser.add_argument('--freeze-loaded-model', dest='freeze_loaded_model', action="store_true",
                        default=False, help="freeze loaded model for finetuning")

    parser.add_argument('--fast-folds', type=int, default=None,)

    parser.add_argument('--skip-train-eval', dest='skip_train_eval',
                        action="store_true", default=False,)

    parser.add_argument('--target-auprc', action="store_true", default=False)
    parser.add_argument('--f-window', action="store_true", default=False)
    parser.add_argument('--pred-labs', action="store_true", default=False)
    parser.add_argument('--elapsed-time', action="store_true", default=False)
    parser.add_argument('--simulated-data', action="store_true", default=False)
    parser.add_argument('--simulated-data-name', type=str, 
                        default="None")
    parser.add_argument('--prior-from-mimic', action="store_true", default=False)

    parser.add_argument('--x-as-list', action="store_true", default=False)
    parser.add_argument('--force-comet-off', action="store_true", default=False)
    parser.add_argument('--grad-accum-steps', default=1, type=int)

    # NIPS20 - Adaptive Prediction Module
    parser.add_argument('--adapt-lstm', dest='adapt_lstm', action="store_true", default=False)
    parser.add_argument('--adapt-bandwidth', dest='adapt_bandwidth', type=int, default=3,
                        help="kernel bandwidth size.")
    parser.add_argument('--adapt-loss', dest='adapt_loss', type=str, default="bce", help="[bce, mse]")
    parser.add_argument('--adapt-lr', dest='adapt_lr', type=float, default=0.005)
    parser.add_argument('--adapt-pop-based', dest='adapt_pop_based', action="store_true", default=False)
    parser.add_argument('--adapt-residual', dest='adapt_residual', action="store_true", default=False)
    parser.add_argument('--adapt-residual-wdecay', dest='adapt_residual_wdecay', type=float, default=1e-06)
    parser.add_argument('--adapt-switch', dest='adapt_switch', action="store_true", default=False)
    parser.add_argument('--adapt-lstm-only', dest='adapt_lstm_only', action="store_true", default=False)
    parser.add_argument('--adapt-fc-only', dest='adapt_fc_only', action="store_true", default=False)
    parser.add_argument('--adapt-sw-pop', dest='adapt_sw_pop', action="store_true", default=False, 
        help="when start new sequence, compare previous step's model and population "
            "based on performance and choose one that gives lower error.")

    # NIPS20 - change target: shut off abnormal for labs and charts
    parser.add_argument('--pred-normal-labchart', dest='pred_normal_labchart', action="store_true", default=False)
    parser.add_argument('--verbose', dest='verbose', action="store_true", default=False)

    args = parser.parse_args()

    return args


def print_args(args):
    for arg in sorted(vars(args)):  # print all args
        itm = str(getattr(args, arg))
        if (itm is not 'None' and itm is not 'False'
                and arg is not 'vecidx2label' and arg is not 'event_dic'):
            print('{0: <20}: {1}'.format(arg, itm))  #


def get_weblog(api_key, args):
    if args.force_comet_off:
        class Empty():
            def log_parameter(self, *args, **kwargs):
                pass
            def log_metric(self, *args, **kwargs):
                pass
            def log_other(self, *args, **kwargs):
                pass
            def log_parameters(self, *args, **kwargs):
                pass
            def end(self, *args, **kwargs):
                pass
        web_logger = Empty()

    elif not args.testmode or args.force_comet:
        web_logger = Experiment(api_key=api_key, auto_param_logging=True,
                                auto_output_logging="",
                                auto_metric_logging=True,
                                workspace="",
                                project_name=args.data_name)
    else:
        class Empty():
            def log_parameter(self, *args, **kwargs):
                pass
            def log_metric(self, *args, **kwargs):
                pass
            def log_other(self, *args, **kwargs):
                pass
            def log_parameters(self, *args, **kwargs):
                pass
            def end(self, *args, **kwargs):
                pass
        web_logger = Empty()
    return web_logger


def add_main_setup(args):
    device = torch.device("cuda:{}".format(args.gpu_id) \
                          if args.use_cuda else 'cpu')

    if args.data_filter is None:
        args.data_filter = ''

    os.system("mkdir -p trained")

    args.device = device

    return args