import numpy as np
import torch
import torchvision
import time
from torchvision import datasets, transforms
from torch import nn, optim
from torch.nn import functional as F

transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (0.5,)),
                                ])

trainset = datasets.MNIST('dataset/train',
                          download=True, train=True, transform=transform)
valset = datasets.MNIST('dataset/test',
                        download=True, train=False, transform=transform)

train_dl = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True)
val_dl = torch.utils.data.DataLoader(valset, batch_size=128, shuffle=True)

model = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(2, 2)),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=(2, 2)),
    nn.Flatten(),
    nn.Linear(in_features=5408, out_features=128),
    nn.Dropout(0.2),
    nn.Linear(in_features=128, out_features=10),
    nn.Softmax(dim=1)
)

print(model)


def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)


def fit(epochs, model, loss_func, opt, train_dl, val_dl):
    for epoch in range(epochs):
        step_counter = []
        time_pre_loop = time.perf_counter()
        model.train()
        for xb, yb in train_dl:
            time_pre_step = time.perf_counter()
            loss_batch(model, loss_func, xb, yb, opt)
            time_post_step = time.perf_counter()
            step_counter.append(time_post_step - time_pre_step)
        time_post_loop = time.perf_counter()
        step_time_avg = sum(step_counter) / len(step_counter)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in val_dl]
            )
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)

        print("Epoch: {epoch}. Epoch Time {time:0.2f}s,"
              " Step Time: {step:0.2f}ms Validation Loss: {val_loss}"
              .format(epoch=epoch, time=time_post_loop - time_pre_loop,
                      step=step_time_avg * 1000, val_loss=val_loss))


fit(epochs=6, model=model, loss_func=F.cross_entropy,
    opt=optim.Adam(model.parameters()), train_dl=train_dl, val_dl=val_dl)
