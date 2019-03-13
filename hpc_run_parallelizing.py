""""
Powerful code to significantly decrease training time when using HPC-Cambridge.
Author: Jin Zhu (jin.zhu@cl.cam.ac.uk)
"""

import argparse
import socket
import GPUtil
from os.path import join
import glob
import os
from time import sleep

import numpy as np

from multiprocessing import Pool


def main():
    parser = argparse.ArgumentParser(description='Multi processing train on HPC')
    parser.add_argument('--data-dir', type=str, required=True, metavar='Dir',
                        help='Dir of input data (within files of .npy or .npz)')
    parser.add_argument('--project-name', type=str, required=True, metavar='Pro_id',
                        help='Name of the project')
    parser.add_argument('--memory', type=int, default=1000, metavar='Mem',
                        help='GPU memory cost for each sub-process (default: 1000 MB)')
    parser.add_argument('--sleep-seconds', type=float, default=1., metavar='Secs',
                        help='Sleep after starting each sub-process, waiting for data pre-processing'
                             '(default: 1.s)')
    parser.add_argument('--pool-limit', type=int, choices=range(41), default=5, metavar='P',
                        help='Maximum number of training, must be in [0, 40] (default: 5)')
    parser.add_argument('--code-file', type=str, required=True, metavar='*.py',
                        help='The python code, must be in the same dir.')
    args = parser.parse_args()

    data_dir = args.data_dir
    project_name = args.project_name
    memory_need = args.memory
    sleep_seconds = args.sleep_seconds
    pool_limit = args.pool_limit
    code_file = args.code_file

    # ## the min memory require for one task
    memory_need = int(memory_need)

    # ## running on which computer
    host = socket.gethostname()

    on_hpc = False
    if 'cozy' in host:
        path_prefix = '/local/sdd/jz426/superResolution'
    elif 'crunchy' in host:
        path_prefix = '/local/sdd/jz426/superResolution'
    elif 'kiiara' in host:
        path_prefix = '/local/scratch/jz426/superResolution'
    elif 'bovat' in host:
        path_prefix = '/local/scratch/jz426/superResolution'
    elif 'iiat-DGX' in host:
        path_prefix = '/home/gyang1/SuperResolution'
    else:
        path_prefix = '/rds-d2/user/jz426/rds-t2-cs056/jz426/superResolution'
        on_hpc = True

    print('Running on ', host)

    # ## find the paths of all slices
    data_path = join(path_prefix, 'Data', data_dir)
    paths_of_all_patients = glob.glob(join(data_path, '*.np*'))

    output_dir = join(path_prefix, 'output', project_name)

    training_pool = Pool(pool_limit)
    # ## for loop for all slices
    for p in paths_of_all_patients:
        case_name = p.split('/')[-1].replace('.npz', '').replace('.npy', '')

        flag_not_trained = True
        while flag_not_trained:

            gpus = GPUtil.getGPUs()
            gpu_mems = np.array([g.memoryFree for g in gpus])
            gpu_id = gpu_mems.argmax()
            if gpus[gpu_id].memoryFree > memory_need:
                # print('Available GPU', gpu_id, 'with free space:', g.memoryFree, 'MB')
                print('Patient', case_name, 'starts training on GPU', gpu_id,
                      'with free space:', gpus[gpu_id].memoryFree, 'MB')
                # ##todo: multiprocessing
                if on_hpc:
                    c = '/home/jz426/workspace/srVE36/bin/python3 -W ignore' + ' ' \
                        '/home/jz426/workspace/code/pytorch_zssr/' + code_file + ' ' \
                        + p + ' ' + output_dir + ' ' + case_name + ' ' + str(gpu_id) + ' ' + '0'
                else:
                    c = 'python -W ignore ' + code_file + ' ' \
                        + p + ' ' + output_dir + ' ' + case_name + ' ' + str(gpu_id) + ' ' + '0'
                # c = 'python ' + \
                #     'pytorch_test.py --gpu-id ' + str(gpu_id)
                training_pool.apply_async(func=os.system, args=(c,))
                # os.system(c)
                flag_not_trained = False
                sleep(sleep_seconds)

    training_pool.close()
    training_pool.join()


if __name__ == '__main__':
    main()