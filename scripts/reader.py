# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np


def read_conv_shapes(logfile):
    def _extract_args(string):
        args = {}
        items = string.split(',')
        input_c = int(items[0].split('Conv2d(')[-1])
        output_c = int(items[1])
        args['input_c'] = input_c
        args['output_c'] = output_c
        keys = ['kernel_size', 'stride', 'padding']
        for k in keys:
            if string.find(k) >= 0:
                args[k] = tuple([int(i) for i in string.split(k)[1].split('),')[0][2:].split(',')])
            else:
                args[k] = (0, 0)
        return args
    workloads = []
    with open(logfile) as f:
        for line in f.readlines():
            if line.find('Conv2d') >= 0:
                args = _extract_args(line)
                workloads.append(args)
    #print(workloads)
    return workloads

def read_tensor_sizes(logfile):
    def _extract_args(string):
        args = {}
        items = string.split('torch.Size([')
        str_tz = items[-1][:-3]
        tensor_size = tuple([int(i) for i in str_tz.split(',')])
        return tensor_size
    workloads = []
    with open(logfile) as f:
        for line in f.readlines():
            if line.find('torch.Size') >= 0:
                args = _extract_args(line)
                workloads.append(args)
    print(workloads)
    return workloads

def read_tensorsize_vs_time(logfile):
    with open(logfile) as f:
        sizes = []
        times = []
        for line in f.readlines():
            items = line.split(',')
            if len(items) == 2:
                size = int(items[0])
                t = float(items[1][:-1])
                sizes.append(size)
                times.append(t)
        return sizes, times

def read_times_from_nccl_log(logfile, mode='allreduce', start=0, end=512*1024*1024, original=False, bw=False):
    print('fn: ', logfile)
    f = open(logfile)
    sizes = []
    times = []
    size_comms = {}
    for line in f.readlines():
        if original and line[0:2] != '--':
            items = ' '.join(line.split()).split(' ')
            if (len(items) == 11 or len(items) == 12) and items[0] != '#':
                try:
                    size = int(items[0])
                except:
                    continue
                if size == 8:
                    continue
                if (size >= start and size <= end):
                    if size not in size_comms:
                        size_comms[size] = [] 
                    try:
                        if mode == 'allreduce':
                            t = float(items[4])/(1000*1000)
                            if bw:
                                t = float(items[6])*8
                        else:
                            t = float(items[3])/(1000*1000)
                            if bw:
                                t = float(items[6])*8
                        size_comms[size].append(t)
                        sizes.append(size)
                        times.append(t)
                    except:
                        continue
        elif line[0:2] == '--':
            items = ' '.join(line.split()).split(' ')
            size = int(items[0][2:])
            t = float(items[1])/(1000*1000)
            times.append(t)
            if size not in size_comms:
                size_comms[size] = [] 
            size_comms[size].append(t)
            
    f.close()
    sizes = list(size_comms.keys())
    sizes.sort()
    #comms = [np.mean(size_comms[s]) for s in sizes]
    comms = [np.max(size_comms[s]) for s in sizes]
    comms = []
    for s in sizes:
        a = np.array(size_comms[s])
        a.sort()
        comms.append(np.mean(a))
    errors = [np.std(size_comms[s]) for s in sizes]
    #print('sizes: ', sizes)
    #print('comms: ', comms)
    #print('errors: ', errors)
    return np.array(sizes), np.array(comms), np.array(errors)

