#!/usr/bin/python3

import sys
import matplotlib.pyplot as plt

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
    overall_plot()
                
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
        label_x = ["{}".format(i) for i in x]
        plt.xticks([i for i in range(len(label_x))], label_x)
        plt.bar(label_x, y, label="Execution time when batch size is {}".format(batch))
        plt.title("Execution time when batch size is {}".format(batch))
        plt.tight_layout()
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
        label_x = ["{}".format(i) for i in x]
        plt.xticks([i for i in range(len(label_x))], label_x)
        plt.bar(label_x, y, label="Execution time when output size is {}".format(output))
        plt.title("Execution time when output size is {}".format(output))
        plt.tight_layout()
        plt.savefig("{}/fig/llama-output-{}.svg".format(tmp_dir, output))
        output *= 2

def overall_plot():
    batch = batch_initial
    x = []
    y = []
    while batch <= batch_max:
        output = output_initial
        if not batch in [1, 4, 16, 64]:
            batch *= 2
            continue
        while output <= output_max:
            if not output in [2, 8, 32, 128, 512]:
                output *= 2
                continue
            
            x.append((batch, output))
            y.append(data[(batch, output)])
            output *= 2
        
        batch *= 2

    plt.figure()
    plt.xlabel("(batch size, output size)")
    plt.ylabel("time (ms)")
    label_x = ["({}, {})".format(i[0], i[1]) for i in x]
    plt.xticks([i for i in range(len(x))], label_x)
    plt.xticks(rotation=90)
    plt.plot(label_x, y, label="Execution time", marker="o")
    plt.title("Execution time")
    plt.tight_layout()
    plt.savefig("{}/fig/llama-overall.svg".format(tmp_dir))
            

if __name__ == '__main__':
    main()
