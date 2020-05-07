from config import args, exp
import numpy as np
from loguru import logger
import pandas as pd
import os
from pg import PG
from ddpg import DDPG
from td3 import TD3
from rbi import RBI
from tqdm import tqdm
from collections import defaultdict
from environment import BulletEnv


def get_algorithm(*argv, **kwargs):

    if args.algorithm == 'pg':
        return PG(*argv, **kwargs)
    if args.algorithm == 'ddpg':
        return DDPG(*argv, **kwargs)
    if args.algorithm == 'td3':
        return TD3(*argv, **kwargs)
    if args.algorithm == 'rbi':
        return RBI(*argv, **kwargs)
    raise NotImplementedError


def reload(alg):

    aux = defaultdict(lambda: 0)
    if exp.load_model and args.reload:
        try:
            aux = alg.load_checkpoint(exp.checkpoint)
        except Exception as e:
            logger.error(str(e))

    return aux


def supervised(alg):
    aux = reload(alg)
    n_offset = aux['n']

    # test_results = next(evaluation)
    for epoch, (train_results, test_results) in enumerate(zip(alg.train_supervised(), alg.eval_supervised())):
        n = n_offset + (epoch + 1) * args.train_epoch

        exp.log_data(train_results, test_results, n, alg=alg if args.lognet else None)

        aux = {'n': n}
        alg.save_checkpoint(exp.checkpoint, aux)


def reinforcement(alg):
    aux = reload(alg)
    n_offset = aux['n']

    # test_results = next(evaluation)
    for epoch, train_results in enumerate(alg.train()):
        n = n_offset + alg.env_steps + 1

        exp.log_data(train_results, None, n, alg=alg if args.lognet else None)

        aux = {'n': n}
        alg.save_checkpoint(exp.checkpoint, aux)


def main():

    env = BulletEnv(args.environment, n_steps=args.n_steps, gamma=args.gamma)
    alg = get_algorithm(env)

    exp.log_alg(alg)

    if args.supervised:
        logger.info("Supervised Learning session")
        supervised(alg)
    elif args.reinforcement:
        logger.info("Reinforcement Learning session")
        reinforcement(alg)
    else:
        raise NotImplementedError

    logger.info("End of simulation")


if __name__ == '__main__':
    main()

