import itertools
import subprocess
from random import shuffle
from timeit import default_timer as timer
import datetime
import numpy as np
import random
# Start timer
start=timer()
start_date = str(datetime.datetime.now()).split('.')[0]

# Parameters
path = '/home/stefbraun/anaconda2/envs/ptl/bin/python'
experiment = 'SRU'
max_procs = 1

epochs = 150
draws = 1
CUDA_DEVICE = 1

print('==========================================================================')
print('Total number of experiments: {}'.format(draws))
print('==========================================================================')

procs = []
curr_procs = 0
for draw in range(draws):
    seed = random.choice([2])
    cla_layers = random.choice([9])
    cla_size = random.choice([350])
    cla_dropout = np.round(0.0,2)

    filename = 'seed{}_cla_layers{}_cla_size{}_cla_dropout{}_relu'.format(seed, cla_layers, cla_size, cla_dropout)
    command = 'CUDA_VISIBLE_DEVICES={} {} train_stan.py --seed {} --filename {} --experiment {} --num_epochs {} \
              --cla_layers {} --cla_size {} --cla_dropout {}'.format(
        CUDA_DEVICE, path, seed, filename, experiment, epochs, cla_layers, cla_size, cla_dropout)

    # procs.append(subprocess.Popen(command, shell=True, stdout=subprocess.PIPE))
    procs.append(subprocess.Popen(command, shell=True, stdout=None, stderr=subprocess.STDOUT))
    curr_procs += 1
    if curr_procs >= max_procs:
        for proc in procs:
            proc.wait()
        curr_procs = 0

# Catch last batch of processes
for proc in procs:
    proc.wait()

end = timer()
end_date = str(datetime.datetime.now()).split('.')[0]

seconds = end-start
run_time = str(datetime.timedelta(seconds=seconds)).split('.')[0]

print('==========================================================================')
print('::: Experiment: {}'.format(experiment))
print('>>> {} runs on CUDA device {} have been completed.'.format(len(ALL), CUDA_DEVICE))
print('>>> Parallel processes:  {}'.format(max_procs))
print('>>> Start data:          {}'.format(start_date))
print('>>> End date:            {}'.format(end_date))
print('>>> Total run time:      {} hh:mm:ss'.format(run_time))
print('>>> Total run time:      {:.2f} seconds'.format(seconds))
print('==========================================================================')

