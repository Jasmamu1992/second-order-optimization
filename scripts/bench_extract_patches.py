# -*- coding: utf-8 -*-
from __future__ import print_function
import torch
import time
from kfac.utils import _extract_patches
import reader


def _str_args(args):
    return 'Conv2d(%d, %d, kernel_size=%s, stride=%s, padding=%s, bias=False)' % (args['input_c'], args['output_c'], args['kernel_size'], args['stride'], args['padding'])


def bench_extract_patches(bs, feature_map_shape, input_c, output_c, kernel_size, stride, padding):
    shape = (bs, input_c, feature_map_shape[0], feature_map_shape[1])
    x = torch.rand(shape).float().cuda()
    warmup = 5
    num_iters = 100
    for i in range(warmup):
        _extract_patches(x, kernel_size, stride, padding)
    torch.cuda.synchronize()
    stime = time.time()
    for i in range(num_iters):
        _extract_patches(x, kernel_size, stride, padding)
    torch.cuda.synchronize()
    etime = time.time()
    time_used = (etime-stime)/num_iters
    return time_used



def bench():
    workloads = reader.read_conv_shapes('./logs/resnet50-matrixsize.log')
    bs = 32
    feature_map_shape = [224, 224]
    total_time = 0
    for w in workloads:
        time_used = bench_extract_patches(bs, feature_map_shape, w['input_c'], w['output_c'], w['kernel_size'], w['stride'], w['padding'])
        fms = feature_map_shape[0]
        fms = (fms + 2 * w['padding'][0] - 1)//w['stride'][0] + 1
        feature_map_shape = [fms, fms]
        total_time += time_used

        print(_str_args(w), ', time: ', time_used)
    print('All conv: ', total_time)


if __name__ == '__main__':
    bench()
