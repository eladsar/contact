import argparse
import time
import numpy as np
import socket
import os
import pwd
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
from torch.utils.tensorboard import SummaryWriter
from distutils.dir_util import copy_tree
import sys
import torch
import pandas as pd
import shutil
from loguru import logger
import random
from collections import defaultdict

project_name = 'contact'
username = pwd.getpwuid(os.geteuid()).pw_name


def boolean_feature(feature, default, help):

    global parser
    featurename = feature.replace("-", "_")
    feature_parser = parser.add_mutually_exclusive_group(required=False)
    feature_parser.add_argument('--%s' % feature, dest=featurename, action='store_true', help=help)
    feature_parser.add_argument('--no-%s' % feature, dest=featurename, action='store_false', help=help)
    parser.set_defaults(**{featurename: default})

parser = argparse.ArgumentParser(description=project_name)
# Arguments

# global parameters

parser.add_argument('--algorithm', type=str, default='pg', help='[pg|ddpg|td3|ppo|trpo|sac|rbi|egl]')
parser.add_argument('--num', type=int, default=-1, help='Resume experiment number, set -1 for new experiment')
parser.add_argument('--cpu-workers', type=int, default=4, help='How many CPUs will be used for the data loading')
parser.add_argument('--cuda', type=int, default=0, help='GPU Number')
parser.add_argument('--identifier', type=str, default='debug', help='The name of the model to use')

# booleans

boolean_feature("tensorboard", True, "Log results to tensorboard")
boolean_feature("reload", False, "Load saved model")
boolean_feature("half", True, 'Use half precision calculation')
boolean_feature("supervised", False, 'Supervised Learning')
boolean_feature("reinforcement", False, 'Reinforcement Learning')
boolean_feature("lognet", False, 'Log  networks parameters')
boolean_feature("render", False, 'Render environment image')

# experiment parameters
parser.add_argument('--environment', type=str, default='HopperBulletEnv-v0', help='gym name of the environmet')
parser.add_argument('--init', type=str, default='ortho', help='Initialization method [ortho|N02|xavier|]')
parser.add_argument('--seed', type=int, default=0, help='Seed for reproducability')
parser.add_argument('--board', type=int, default=1, help='Board Size [9|10|11]')

# Netwoks parameters

parser.add_argument('--lr-p', type=float, default=1e-4, metavar='α', help='learning rate for the policy network')
parser.add_argument('--lr-q', type=float, default=1e-3, metavar='α', help='learning rate for the Q-function network')
parser.add_argument('--weight-decay', type=float, default=0., help='L2 regularization coefficient')
parser.add_argument('--dropout', type=float, default=0., help='Dropout regularization coefficient')
parser.add_argument('--clip-p', type=float, default=0., help='Clip Pi Gradient L2 norm')
parser.add_argument('--clip-q', type=float, default=0., help='Clip Pi Gradient L2 norm')

# rbi parameters
parser.add_argument('--rbi-samples', type=int, default=100, help='policy samples for rbi training')
parser.add_argument('--cmin', type=float, default=0.5, metavar='c_min', help='Lower reroute threshold')
parser.add_argument('--cmax', type=float, default=1.5, metavar='c_max', help='Upper reroute threshold')
parser.add_argument('--rbi-epsilon', type=float, default=0.01, metavar='ε', help='Uniform sampling in RBI update')
parser.add_argument('--rbi-greed', type=float, default=0.1, help='Greedy part in RBI update')

parser.add_argument('--total-steps', type=int, default=int(1e6), metavar='STEPS', help='Total number of environment steps')
parser.add_argument('--train-epoch', type=int, default=500, metavar='BATCHES', help='Length of each epoch (in batches)')
parser.add_argument('--target-update', type=int, default=1000, metavar='BATCHES', help='update targets every number of steps')
parser.add_argument('--test-epoch', type=int, default=10, metavar='BATCHES', help='Length of test epoch (in batches)')
parser.add_argument('--batch', type=int, default=64, help='Batch Size')

# parser.add_argument('--warmup-steps', type=int, default=2000, help='warm-up random steps')
# parser.add_argument('--min-replay-buffer', type=int, default=1000, help='minimal replay buffer size')
# parser.add_argument('--start-policy-update', type=int, default=200, help='minimal Q-learning steps before policy update')

parser.add_argument('--warmup-steps', type=int, default=0, help='warm-up random steps')
parser.add_argument('--min-replay-buffer', type=int, default=200, help='minimal replay buffer size')
parser.add_argument('--start-policy-update', type=int, default=0, help='minimal Q-learning steps before policy update')


# RL parameters

parser.add_argument('--gamma', type=float, default=0.99, help='Discount Factor')
parser.add_argument('--epsilon', type=float, default=0.2, metavar='ε', help='exploration parameter')
parser.add_argument('--epsilon-warmup', type=float, default=0.2, metavar='ε', help='warm-up exploration parameter')
parser.add_argument("--tau", default=0.001, help="Update factor for the soft update of the target networks")
parser.add_argument('--steps-per-train', type=int, default=1, metavar='STEPS', help='number of steps between training epochs')
parser.add_argument('--consecutive-train', type=int, default=1, metavar='STEPS', help='number of consecutive training iterations')
parser.add_argument('--replay-memory-size', type=int, default=100000, help='Total replay memory size')
parser.add_argument('--n-steps', type=int, default=1, metavar='STEPS', help='Number of steps for multi-step learning')
parser.add_argument('--delayed-policy-update', type=int, default=1, metavar='STEPS', help='steps between policy updates')

args = parser.parse_args()
seed = args.seed


def set_seed(seed=seed):

    if 'cnt' not in set_seed.__dict__:
        set_seed.cnt = 0
    set_seed.cnt += 1

    if seed is None:
        seed = args.seed * set_seed.cnt

    if seed > 0:
        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class Experiment(object):

    def __init__(self):

        set_seed()

        torch.set_num_threads(100)
        logger.info("Welcome to: Deep Hex Agent")
        logger.info(' ' * 26 + 'Simulation Hyperparameters')
        for k, v in vars(args).items():
            logger.info(' ' * 26 + k + ': ' + str(v))

        # consts

        self.uncertainty_samples = 1
        # parameters

        self.start_time = time.time()
        self.exptime = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        self.device = torch.device("cuda:%d" % args.cuda)
        self.opt_level = "O1" if args.half else "O0"

        if "gpu" in socket.gethostname():
            self.root_dir = os.path.join('/home/dsi/', username, 'data', project_name)
        elif "root" == username:
            self.root_dir = os.path.join('/data/data', project_name)
        else:
            self.root_dir = os.path.join('/data/', username, project_name)

        self.base_dir = os.path.join(self.root_dir, 'results')

        for folder in [self.base_dir, self.root_dir]:
            if not os.path.exists(folder):
                os.makedirs(folder)

        dirs = os.listdir(self.base_dir)

        self.resume = args.num
        temp_name = "%s_%s_%s_exp" % (args.algorithm, args.identifier, args.environment.split('-')[0])
        self.exp_name = ""
        self.load_model = True
        if self.resume >= 0:
            for d in dirs:
                if "%s_%04d_" % (temp_name, self.resume) in d:
                    self.exp_name = d
                    self.exp_num = self.resume
                    break
        elif self.resume == -1:

            ds = [d for d in dirs if temp_name in d]
            ns = np.array([int(d.split("_")[-3]) for d in ds])
            if len(ns):
                self.exp_name = ds[np.argmax(ns)]
        else:
            raise Exception("Non-existing experiment")

        if not self.exp_name:
            # count similar experiments
            n = max([-1] + [int(d.split("_")[-3]) for d in dirs if temp_name in d]) + 1
            self.exp_name = "%s_%04d_%s" % (temp_name, n, self.exptime)
            self.exp_num = n
            self.load_model = False

        # init experiment parameters
        self.root = os.path.join(self.base_dir, self.exp_name)

        # set dirs
        self.tensorboard_dir = os.path.join(self.root, 'tensorboard')
        self.checkpoints_dir = os.path.join(self.root, 'checkpoints')
        self.results_dir = os.path.join(self.root, 'results')
        self.code_dir = os.path.join(self.root, 'code')
        self.checkpoint = os.path.join(self.checkpoints_dir, 'checkpoint')

        if self.load_model and args.reload:
            print("Resuming existing experiment")

        else:

            if not self.load_model:
                print("Creating new experiment")

            else:
                print("Deleting old experiment")
                shutil.rmtree(self.root)

            os.makedirs(self.root)
            os.makedirs(self.tensorboard_dir)
            os.makedirs(self.checkpoints_dir)
            os.makedirs(self.results_dir)
            os.makedirs(self.code_dir)

            # make log dirs
            os.makedirs(os.path.join(self.results_dir, 'train'))
            os.makedirs(os.path.join(self.results_dir, 'eval'))

            # copy code to dir
            copy_tree(os.path.dirname(os.path.realpath(__file__)), self.code_dir)

            # write args to file
            filename = os.path.join(self.root, "args.txt")
            with open(filename, 'w') as fp:
                fp.write('\n'.join(sys.argv[1:]))

            pd.to_pickle(vars(args), os.path.join(self.root, "args.pkl"))

        # initialize tensorboard writer
        if args.tensorboard:
            self.writer = SummaryWriter(log_dir=self.tensorboard_dir, comment=args.identifier)

    def log_data(self, train_results, test_results, n, alg=None):

        defaults_argv = defaultdict(dict)

        for param, val in train_results['scalar'].items():
            if type(val) is dict:
                for p, v in val.items():
                    val[p] = np.mean(v)
            else:
                train_results['scalar'][param] = np.mean(val)

        if test_results is not None:
            for param, val in test_results['scalar'].items():
                if type(val) is dict:
                    for p, v in val.items():
                        val[p] = np.mean(v)
                else:
                    test_results['scalar'][param] = np.mean(val)

        if args.tensorboard:

            if alg is not None:
                networks = alg.get_networks()
                for net in networks:
                    for name, param in networks[net]():
                        try:
                            self.writer.add_histogram("weight_%s/%s" % (net, name), param.data.cpu().numpy(), n,
                                                      bins='tensorflow')
                            self.writer.add_histogram("grad_%s/%s" % (net, name), param.grad.cpu().numpy(), n,
                                                      bins='tensorflow')
                            if hasattr(param, 'intermediate'):
                                self.writer.add_histogram("iterm_%s/%s" % (net, name), param.intermediate.cpu().numpy(),
                                                          n,
                                                          bins='tensorflow')
                        except:
                            pass

            for log_type in train_results:
                log_func = getattr(self.writer, f"add_{log_type}")
                for param in train_results[log_type]:

                    if type(train_results[log_type][param]) is dict:
                        for p, v in train_results[log_type][param].items():
                            log_func(f"train_{param}/{p}", v, n, **defaults_argv[log_type])
                    elif type(train_results[log_type][param]) is list:
                        log_func(f"eval/{param}", *train_results[log_type][param], n, **defaults_argv[log_type])
                    else:
                        log_func(f"train/{param}", train_results[log_type][param], n, **defaults_argv[log_type])

            if test_results is not None:
                for log_type in test_results:
                    log_func = getattr(self.writer, f"add_{log_type}")
                    for param in test_results[log_type]:

                        if type(test_results[log_type][param]) is dict:
                            for p, v in test_results[log_type][param].items():
                                log_func(f"eval_{param}/{p}", v, n, **defaults_argv[log_type])
                        elif type(test_results[log_type][param]) is list:
                            log_func(f"eval/{param}", *test_results[log_type][param], n, **defaults_argv[log_type])
                        else:
                            log_func(f"eval/{param}", test_results[log_type][param], n, **defaults_argv[log_type])

        stat_line = 'Train: '
        for param in train_results['scalar']:
            if type(train_results['scalar'][param]) is not dict:
                stat_line += '  %s %g \t|' % (param, train_results['scalar'][param])
        logger.info(stat_line)

        if test_results is not None:
            stat_line = 'Eval: '
            for param in test_results['scalar']:
                if type(test_results['scalar'][param]) is not dict:
                    stat_line += '  %s %g \t|' % (param, test_results['scalar'][param])
            logger.info(stat_line)

    def log_alg(self, alg):
        pass
        # self.writer.add_hparams(hparam_dict=vars(args), metric_dict={'x': 0})
        # for name, net in alg.networks_dict:
        #     self.writer.add_graph(net)
        #     self.writer.flush()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if args.tensorboard:
            self.writer.export_scalars_to_json(os.path.join(self.tensorboard_dir, "all_scalars.json"))
            self.writer.close()


exp = Experiment()
