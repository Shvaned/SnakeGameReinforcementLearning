import matplotlib.pyplot as plt
import pandas as pd
import os

plt.ion()

def plot(scores, mean_scores, save_path=None):
    plt.figure(figsize=(8,4))
    plt.clf()
    plt.title('Training progress')
    plt.xlabel('Episodes')
    plt.ylabel('Score')
    plt.plot(scores, label='score')
    plt.plot(mean_scores, label='mean')
    plt.legend()
    plt.pause(0.001)
    if save_path:
        dirname = os.path.dirname(save_path)
        if dirname and not os.path.exists(dirname):
            os.makedirs(dirname)
        plt.savefig(save_path)

def append_log(logfile, row_dict):
    df = pd.DataFrame([row_dict])
    header = not os.path.exists(logfile)
    df.to_csv(logfile, mode='a', header=header, index=False)
