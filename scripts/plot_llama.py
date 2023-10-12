#!/usr/bin/python3

import sys
import matplotlib.pyplot as plt
import numpy as np

data={}

def main():
    global batch_initial, batch_max, output_initial, output_max, tmp_dir
    args = sys.argv
    if len(args) != 6:
        print("Usage: python3 plot_llama.py <batch_intial> <batch_max> <output_initial> <output_max> <tmp_dir>")
        exit(1)
    
    batch_initial = int(args[1])
    batch_max = int(args[2])
    output_initial = int(args[3])
    output_max = int(args[4])
    tmp_dir = args[5]


    batch = batch_initial
    while batch <= batch_max:
        output = output_initial
        while output <= output_max:
            data_collect("{}/manifold_llama-{}-{}.log".format(tmp_dir, batch, output), batch, output)
            output *= 2
        batch *= 2

    output_plot()
    batch_plot()
                
def data_collect(filename, batch, output):
    time = 0.0
    with open(filename) as f:
        for line in f:
            if "FT-CPP-decoding-beamsearch-time" in line:
                time += float(line.split()[-2])
    
    data[(batch, output)] = time/2.0
                

def output_plot():
    batch = batch_initial
    while batch <= batch_max:
        x = []
        y = []

        for key in data.keys(): #key[0]: batch, key[1]: output
            if batch == key[0]:  
                x.append(key[1])
                y.append(data[key])
        
        plt.figure()
        plt.xlabel("output size")
        plt.ylabel("time (ms)")
        plt.xscale("log", base=2)
        plt.xticks(x, x)
        plt.plot(x, y, label="Execution time when batch size is {}".format(batch), marker="o")
        plt.savefig("{}/fig/llama-batch-{}.svg".format(tmp_dir, batch))
        batch *= 2

def batch_plot():
    output = output_initial
    while output <= output_max:
        x = []
        y = []

        for key in data.keys(): #key[0]: batch, key[1]: output
            if output == key[1]:
                x.append(key[0])
                y.append(data[key])

        plt.figure()
        plt.xlabel("batch size")
        plt.ylabel("time (ms)")
        plt.xscale("log", base=2)
        plt.xticks(x, x)
        plt.plot(x, y, label="Execution time when output size is {}".format(output), marker="o")
        plt.savefig("{}/fig/llama-output-{}.svg".format(tmp_dir, output))
        output *= 2

if __name__ == '__main__':
    main()
