import numpy as np
from timeit import default_timer as timer

MEAN = [0.485, 0.456, 0.406]
STDDEV = [0.229, 0.224, 0.225]

def unnormalize(x):
	x = x.transpose(0, 2,3, 1)
	x = x * STDDEV + MEAN
	return x

def normalize(x):
	x = (x-MEAN)/STDDEV
	x = x.transpose(0,3,1,2)
	return x

def speedtest(pipe, batch):
    pipe.build()

    # warmup
    for i in range(5):
        pipe.run()
    # test
    n_test = 20
    t_start = timer()
    for i in range(n_test):
        pipe.run()
    t = timer() - t_start
    print("Speed: {} imgs/s".format((n_test * batch)/t))