"""
 * argparser
 * Created on 03.04.19
 * Author: doering
"""

import argparse


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('-ss', '--sample_size', type=int, default=384)
    parser.add_argument('-tbs', '--train_batch_size', type=int, default=64)
    parser.add_argument('-nw', '--num_worker', type=int, default=8)
    parser.add_argument('-ne', '--num_epochs', type=int, default=160)
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3)

    parser.add_argument('-cp', '--checkpoint_path', type=str, default='')
    parser.add_argument('-log', '--log', type=str, default='data/logs/tensorboardX')
    parser.add_argument('-msp', '--model_save_path', type=str, default='data/models/')

    return parser.parse_args()
