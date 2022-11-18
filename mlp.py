import matplotlib.pyplot as plt
from tqdm import tqdm
import torch


### Set hyperparameters
LR = 0.1 # learning rate
EPOCHS = 20000 # amount of forward/backward propagations
BATCH_SIZE = 2 # amount of backproped data per epoch


### Load Data
torch.manual_seed(871)
X = torch.Tensor([[0.1, 0.3, 0.1, 0.6, 0.4, 0.6, 0.5, 0.9, 0.4, 0.7],
                  [0.1, 0.4, 0.5, 0.9, 0.2, 0.3, 0.6, 0.2, 0.4, 0.6]]).T
y = torch.nn.functional.one_hot(torch.LongTensor(
                        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]), num_classes=2)


### Make model
w1 = torch.rand((2,5))
b1 = torch.rand((1,5))
w2 = torch.rand((5,5))
b2 = torch.rand((1,5))
w3 = torch.rand((5,2))
b3 = torch.rand((1,2))


### Training
loss = torch.empty(size=(EPOCHS,))

for epoch in tqdm(range(EPOCHS), "Train"):
    for ids in torch.split(torch.randperm(X.shape[0]), BATCH_SIZE):
        # Forward propagation
        a0 = X[ids, :]
        a1 = torch.sigmoid(a0@w1 + b1)
        a2 = torch.sigmoid(a1@w2 + b2)
        a3 = torch.sigmoid(a2@w3 + b3)
        loss[epoch] = torch.mean((a3 - y[ids])**2).detach()/BATCH_SIZE

        # Backward propagation
        da3 = a3*(1-a3)*(a3-y[ids,:])/BATCH_SIZE
        da2 = a2*(1-a2)*(da3@w3.T)
        da1 = a1*(1-a1)*(da2@w2.T)
        da0 = a0*(1-a0)*(da1@w1.T)

        # Update parameters
        w3 -= LR*a2.T@da3
        w2 -= LR*a1.T@da2
        w1 -= LR*a0.T@da1
        b3 -= LR*da3.sum(dim=0)
        b2 -= LR*da2.sum(dim=0)
        b1 -= LR*da1.sum(dim=0)


### Results
fig, (ax1,ax2,ax3) = plt.subplots(1, 3, figsize=(12,4))
ax1.set_title(f"loss over {EPOCHS} epochs")
ax1.plot(loss)
ax2.set_title("predicted")
fig.colorbar(ax2.imshow(torch.sigmoid(torch.sigmoid(torch.sigmoid(
    X@w1 + b1)@w2 + b2)@w3 + b3), cmap="gray", vmin=0, vmax=1), ax=ax2)
ax3.set_title("truth")
fig.colorbar(ax3.imshow(y, cmap="gray", vmin=0, vmax=1), ax=ax3)
plt.show()