#!python3

import os
import sys
import time
import datetime
import gc
import json
import copy
import gzip
import math
import random
import tqdm
import glob
import psutil
import argparse

import socket
import subprocess
from multiprocessing import Pool as ProcessPool

import torch


def test_gpu(args):
    print('[test]', vars(args))
    D = 1024
    vram_bytes_i = D * D * 4
    vram_bytes = float(args.hold) * (1024 ** 3)
    N = max(int(vram_bytes / vram_bytes_i * 0.77), 3)
    print(D, 'x', D, N)
    m_list = [torch.randn(D, D).type(torch.float32).cuda() for _ in range(0, N)]
    with torch.no_grad():
        while True:
            for i in range(0, N):
                m_list[i] = torch.matmul(m_list[i], m_list[i])
                m_list[i] -= m_list[i].mean()
                m_list[i] /= m_list[i].std()
            time.sleep(0.25)


def cmd_executor(cmd_list):
    t0 = time.time()
    for i in range(0, len(cmd_list)):
        c, e, o = cmd_list[i]
        # time.sleep(2)
        with open(o, 'w') as fp:
            p = subprocess.Popen(c, env=e, stdout=fp, stderr=fp)
            p.wait()
        print('[%d/%d finished]' % (i + 1, len(cmd_list)), '[%.1f hours]' % ((time.time() - t0) / 3600.0), '[%s]' % ' '.join(c), '>>>', '[%s]' % o, flush=True)


def run_mrt(args):
    basedir = os.path.dirname(__file__)
    assert os.access(os.path.join(basedir, 'main.py'), os.R_OK)
    assert os.access(os.path.join(basedir, 'r50_model_best.pth'), os.R_OK)
    assert len(args.gpus) > 0
    vids = list(args.ids)
    random.shuffle(vids)
    print(vids)
    python_path = str(subprocess.run(['which', 'python'], capture_output=True, text=True, env=os.environ).stdout).strip()
    curr_env = os.environ.copy()
    commands = []
    for i in range(0, len(args.gpus)):
        commands_i = []
        if i < len(args.gpus) - 1:
            vids_batch = vids[len(vids) // len(args.gpus) * i : len(vids) // len(args.gpus) * (i + 1)]
        else:
            vids_batch = vids[len(vids) // len(args.gpus) * i :]
        vids_batch = sorted(vids_batch)
        print(args.gpus[i], vids_batch)
        env_i = curr_env.copy()
        env_i['CUDA_VISIBLE_DEVICES'] = str(args.gpus[i])

        for v in vids_batch:
            log_i = 'log_mrt_cross_%s_%s_GPU%s.log' % (v, socket.gethostname(), env_i['CUDA_VISIBLE_DEVICES'])
            log_i = os.path.join(basedir, log_i)
            cmd_i = [python_path, os.path.join(basedir, 'main.py'), '--backbone', 'resnet50', '--num_encoder_layers', '6', '--num_decoder_layers', '6', '--num_classes', '3', '--dropout', '0.0', '--data_root', '../MSCOCO2017', '--source_dataset', 'mscoco', '--target_dataset', v, '--batch_size', '1', '--eval_batch_size', '6', '--lr', '2e-5', '--lr_backbone', '2e-6', '--lr_linear_proj', '2e-6', '--epoch', '3', '--epoch_lr_drop', '2', '--mode', 'cross_domain_mae', '--output_dir', 'outputs/cross/%s' % v, '--resume', os.path.join(basedir, 'r50_model_best.pth')]
            commands_i.append([cmd_i, env_i, log_i])

            log_i = 'log_mrt_teach_%s_%s_GPU%s.log' % (v, socket.gethostname(), env_i['CUDA_VISIBLE_DEVICES'])
            log_i = os.path.join(basedir, log_i)
            cmd_i = [python_path, os.path.join(basedir, 'main.py'), '--backbone', 'resnet50', '--num_encoder_layers', '6', '--num_decoder_layers', '6', '--num_classes', '3', '--dropout', '0.0', '--data_root', '../MSCOCO2017', '--source_dataset', 'mscoco', '--target_dataset', v, '--batch_size', '1', '--eval_batch_size', '6', '--lr', '2e-5', '--lr_backbone', '2e-6', '--lr_linear_proj', '2e-6', '--epoch', '3', '--epoch_lr_drop', '2', '--mode', 'teaching', '--output_dir', 'outputs/teach/%s' % v, '--resume', os.path.join(basedir, 'outputs/cross/%s/model_last.pth' % v)]
            commands_i.append([cmd_i, env_i, log_i])

        commands_i.append([[python_path, os.path.join(basedir, 'run_experiments.py'), '--opt', 'test', '--hold', args.hold], env_i, os.path.join(basedir, 'log_mrt_cross_999_%s_GPU%s.log' % (socket.gethostname(), env_i['CUDA_VISIBLE_DEVICES']))])
        commands.append(commands_i)
    pool = ProcessPool(processes=len(commands))
    _ = pool.map_async(cmd_executor, commands).get()
    pool.close()
    pool.join()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Finetune Script')
    parser.add_argument('--opt', type=str)
    # parser.add_argument('--model', type=str)
    parser.add_argument('--ids', nargs='+', default=[])
    parser.add_argument('--gpus', nargs='+', default=[])
    parser.add_argument('--hold', default='0.005', type=str)
    # parser.add_argument('--ddp_port', type=str, default='50137')
    args = parser.parse_args()
    print(args)

    if args.opt == 'run':
        run_mrt(args)
    elif args.opt == 'test':
        test_gpu(args)
    else: pass

# python run_experiments.py --opt run --ids 001 003 006 --gpus 0
