import os
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets.MnistDataset import Mnist
from datasets.CifarDataset import Cifar

from model.SimplifiedModel import PredictNoise

from util import build_dir, plot_loss, plot_samples, save_models

def load_mnist_data():
    global RESULT_DIR
    global BASE_DIR
    global IN_CHANNEL
    global CLASS_SIZE
    global IMG_SIZE

    RESULT_DIR = 'result/mnist'
    IN_CHANNEL = 1
    CLASS_SIZE = 10
    IMG_SIZE = 28

    img_gzip = "train-images.idx3-ubyte"
    label_gzip = "train-labels.idx1-ubyte"
    data_dir = BASE_DIR + '/data/mnist/'

    dataset = Mnist(img_gzip = img_gzip, label_gzip = label_gzip, base_dir = data_dir)
    return dataset

def load_cifar_data():
    global RESULT_DIR
    global BASE_DIR
    global IN_CHANNEL
    global CLASS_SIZE
    global IMG_SIZE

    RESULT_DIR = 'result/cifar'
    IN_CHANNEL = 3
    CLASS_SIZE = 10
    IMG_SIZE = 32

    img_gzip = ["/data_batch_1","/data_batch_2","/data_batch_3", "/data_batch_4"]
    val_gzip = ['/data_batch_5']
    test_img_gzip = ["/test_batch"]
    gzip_dir = img_gzip + val_gzip + test_img_gzip

    label_name_zip = '/batches.meta'
    data_dir = BASE_DIR + '/data/cifar-10'

    dataset = Cifar(img_gzip = gzip_dir, label_name_zip = label_name_zip, base_dir = data_dir)

    return dataset

def get_beta_alpha():
    global TIMESTEPS
    global DEVICE

    beta_start, beta_end = (1e-4, 0.02)
    beta = torch.linspace(beta_start, beta_end, TIMESTEPS, device = DEVICE)
    alpha = 1 - beta
    alpha_hat = torch.cumprod(alpha, dim = 0)
    return beta, alpha, alpha_hat

def get_samples(model, beta, alpha_hat, alpha, labels=None, samples = 4):
    global DEVICE
    global TIMESTEPS
    global IMG_SIZE
    global IN_CHANNEL
    global CLASS_SIZE

    model.eval()

    initial_sample = samples
    samples = CLASS_SIZE * initial_sample

    with torch.no_grad():
        x_sample = torch.tensor(np.random.normal(size=(samples, IN_CHANNEL, IMG_SIZE, IMG_SIZE)), 
                                        dtype=torch.float32, 
                                        device = DEVICE)
        
        y = torch.tensor([i for i in range(CLASS_SIZE)], device=DEVICE)
        y = y.unsqueeze(1).expand(-1, initial_sample).contiguous().view(-1)

        if labels is not None:
            labels =[labels[i] for i in y.cpu().numpy().reshape(-1)]

        for t in range(TIMESTEPS-1, 0, -1):
            z = torch.tensor(np.random.normal(size=(samples, IN_CHANNEL, IMG_SIZE, IMG_SIZE)), 
                                        dtype=torch.float32, 
                                        device = DEVICE)
            
            if t == 0 : z = 0 
            time_step = t

            t = torch.tensor(t, device = DEVICE).unsqueeze(0).expand(samples)

            alpha_batch = alpha_hat[t].reshape(samples, 1)
            alpha_batch = alpha_batch.unsqueeze(2).unsqueeze(3).expand(samples, IN_CHANNEL, IMG_SIZE, IMG_SIZE)
            noise_pred = model(x_sample, t, y)
            x_sample = 1/torch.sqrt(alpha[time_step]) * (x_sample - ((1-alpha[time_step])/torch.sqrt(1-alpha_batch))*noise_pred)
            x_sample = x_sample + torch.sqrt(beta[time_step]) * z
        
        x_sample = x_sample.clamp(-1,1)
        x_sample = x_sample.cpu().numpy()
        
        x_sample = np.transpose(x_sample, (0, 2, 3, 1))
        return x_sample, labels

def train_dataset(dataloader, model, loss_fn, optimizer):
    global DEVICE
    global TIMESTEPS
    global IMG_SIZE
    global IN_CHANNEL
    
    loss_dataset = []
    model.train()
    for _, (x, y) in enumerate(dataloader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        batch_size = x.size(0)

        time = torch.randint(TIMESTEPS, size=(batch_size,), device = DEVICE)
        eps = torch.tensor(np.random.normal(size=x.shape), dtype=torch.float32, device = DEVICE)

        alpha_batch = alpha_hat[time].reshape(batch_size, 1)
        alpha_batch = alpha_batch.unsqueeze(2).unsqueeze(3).expand(batch_size, IN_CHANNEL, IMG_SIZE, IMG_SIZE)

        x = torch.sqrt(alpha_batch)*x + torch.sqrt(1-alpha_batch)*eps

        pred_eps = model(x, time, y)
        loss = loss_fn(pred_eps, eps)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_dataset.append(loss.item())
    return np.mean(loss_dataset)

def save_checkpoints(model, beta, alpha_hat, alpha, epoch, result_dir, model_name, labels, num_samples = 4):
    samples, labels = get_samples(model, beta, alpha_hat, alpha, samples=num_samples, labels = labels)
    plot_samples(samples, epoch, result_dir, CLASS_SIZE, num_samples, labels)
    plot_loss(history, result_dir)  
    save_models(model, result_dir, model_name)
    return

if __name__ == "__main__":
    global DEVICE
    global BASE_DIR
    global RESULT_DIR
    global IN_CHANNEL
    global CLASS_SIZE
    global TIMESTEPS
    global IMG_SIZE
    global IS_CONDITIONAL

    BASE_DIR = os.getcwd() 
    IS_CONDITIONAL = True
    TIMESTEPS = 1000
    hidden_dim = 256

    DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else DEVICE

    dataset = load_mnist_data()
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    build_dir(RESULT_DIR)

    model = PredictNoise(input_channel = IN_CHANNEL,
                         device = DEVICE, 
                         hidden_dim=hidden_dim,
                         embedding_dim = hidden_dim, 
                         time_dimension = 128,
                         isConditional = IS_CONDITIONAL,
                         class_size = CLASS_SIZE).to(DEVICE)
    
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr = 1e-4)

    beta, alpha, alpha_hat = get_beta_alpha()

    epochs = 100
    history = {'loss':[]}

    for epoch in tqdm(range(epochs)):
        loss = train_dataset(dataloader, model, loss_fn, optimizer)
        history['loss'].append(loss)
        print(f"[{epoch}] Train Loss : {loss:.4f}")
        if epoch % 5 == 0:
            save_checkpoints(model, beta, alpha_hat, alpha, epoch, result_dir=RESULT_DIR, 
                             model_name= f"model_{hidden_dim}", labels=dataset.label_name, num_samples = 5)  
    
    save_checkpoints(model, beta, alpha_hat, alpha, epochs, RESULT_DIR, f"model_{hidden_dim}", labels=dataset.label_name, num_samples = 5)  