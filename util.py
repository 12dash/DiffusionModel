import os
import torch
import matplotlib.pyplot as plt

def build_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def plot_loss(history, result_dir):
    fig = plt.figure(figsize=(6,4))
    fig = plt.plot(history['loss'])
    _ = plt.gca().set(xlabel='Epochs', ylabel='Loss', title='Training Loss')
    plt.savefig(f'{result_dir}/loss.png')
    plt.close()

def plot_samples(samples, epoch, result_dir, rows, cols, labels = None):
    num_rows, num_cols = rows, cols
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(12,18))
    for row in range(num_rows):
        for col in range(num_cols):
            axs[row][col].imshow((samples[row*num_cols + col]+1)/2)
            axs[row][col].set(xticks=[], yticks=[])
            if labels is not None:  axs[row][col].set(title=labels[row*num_cols + col] )
    fig.suptitle(f"Epoch : {epoch}")
    plt.savefig(f"{result_dir}/samples.png")
    plt.close()

def save_models(model, result_dir, model_name):
    torch.save(model.state_dict(), f"{result_dir}/{model_name}.pt")
    return