from __future__ import print_function
import numpy as np

def read_speed(logfile):
    print('filename: ', logfile)
    f = open(logfile, 'r')
    speeds = []
    num_of_layers = 0
    for line in f.readlines():
        if line.find(', time: ') > 0:
            speedstr = line.split(', time: ')[-1].split(',')[0]
            speed = float(speedstr)
            speeds.append(speed)
        #if line.find('FW+BW') > 0:
        #    comptime = line.split('FW+BW: ')[-1].split(',')[0]
        #    comptime = float(comptime)
        #    computations.append(comptime)
        #if line.find(', average forward (') > 0:
        #    forwardtime = float(line.split(', average forward (')[1].split(')')[0])
        #    backwardtime = float(line.split('and backward (')[1].split(')')[0])
        #    iotime = float(line.split('iotime: ')[-1][:-1])
        #    forwards.append(forwardtime)
        #    backwards.append(backwardtime)
        #    iotimes.append(iotime)
        #if line.find('Number of groups:') > 0:
        #    num_of_layers = int(line.split('Number of groups:')[-1][:-1])
        #if line.find('Total compress:') > 0:
        #    linestr = line.split('Total compress:')[-1]
        #    compression_time = float(linestr.split(',')[0])
        #    compressions.append(compression_time)
        #elif line.find('total[') > 0:
        #    compression_time = float(line.split(':')[-1].split(',')[2])+float(line.split(':')[-1].split(',')[4])
        #    compressions.append(compression_time)
    si = 1
    avg_speed = np.mean(speeds[si:])
    std_speed = np.std(speeds)
    f.close()
    print('avg speed: ', avg_speed, ' std: ', std_speed)
    return avg_speed, std_speed


def read_multiple_speeds():
    LOGHOME='./logs'
    exclude_parts_full='CommunicateInverse,ComputeInverse,CommunicateFactor,ComputeFactor'
    kfac_name='inverse_opt'

    exclude_parts = ['']+exclude_parts_full.split(',')
    #dnn='resnet50';density=1;bs=32;lr=1.2;nw=16
    #dnn='resnet152';density=1;bs=8;lr=1.2;nw=16
    dnn='resnet34';density=1;bs=64;lr=1.2;nw=64
    speeds = []

    if kfac_name.find('opt') >= 0:
        speed_avg0 = 0
    else:
        exclude_part = '\'\''
        fn = '%s/timing_imagenet_%s_kfac0_gpu%d_bs%d_%s_ep_%s.log' % (LOGHOME, dnn, nw, bs, kfac_name, exclude_part)
        speed_avg0, _ = read_speed(fn)
    for idx, ep in enumerate(exclude_parts):
        if idx == 0:
            exclude_part = '\'\''
        elif idx == 1:
            exclude_part = ep 
        else:
            exclude_part = ','.join(exclude_parts[1:idx+1])
        fn = '%s/timing_imagenet_thres1024_%s_kfac1_gpu%d_bs%d_%s_ep_%s.log' % (LOGHOME, dnn, nw, bs, kfac_name, exclude_part)
        try:
            speed_avg, _ = read_speed(fn)
        except:
            speed_avg = 0
        speeds.append(speed_avg)
    speeds.append(speed_avg0)
    print(speeds)

if __name__ == '__main__':
    #logfile = 'logs/timing_imagenet_resnet50_kfac1_gpu64_bs64_inverse_ep_\'\'.log'
    #avg_speed, std_speed = read_speed(logfile)
    #print(avg_speed)
    read_multiple_speeds()
