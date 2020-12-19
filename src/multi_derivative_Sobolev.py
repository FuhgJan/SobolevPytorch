import torch
from torch.autograd import Variable
import torch.utils.data as Data
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.optim as optim

from matplotlib import cm
import numpy as np
import copy


torch.manual_seed(1)  # reproducible

def franke(X, Y):
    term1 = .75*torch.exp(-((9*X - 2).pow(2) + (9*Y - 2).pow(2))/4)
    term2 = .75*torch.exp(-((9*X + 1).pow(2))/49 - (9*Y + 1)/10)
    term3 = .5*torch.exp(-((9*X - 7).pow(2) + (9*Y - 3).pow(2))/4)
    term4 = .2*torch.exp(-(9*X - 4).pow(2) - (9*Y - 7).pow(2))

    f = term1 + term2 + term3 - term4
    dfx = -2*(9*X - 2)*9/4 * term1 - 2*(9*X + 1)*9/49 * term2 + \
          -2*(9*X - 7)*9/4 * term3 + 2*(9*X - 4)*9 * term4
    dfy = -2*(9*Y - 2)*9/4 * term1 - 9/10 * term2 + \
          -2*(9*Y - 3)*9/4 * term3 + 2*(9*Y - 7)*9 * term4

    return f, dfx, dfy




class Net(nn.Module):
    def __init__(self,inp,out, activation, num_hidden_units=100, num_layers=1):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(inp, num_hidden_units, bias=True)
        self.fc2 = nn.ModuleList()
        for i in range(num_layers):
            self.fc2.append(nn.Linear(num_hidden_units, num_hidden_units, bias=True))
        self.fc3 = nn.Linear(num_hidden_units, out, bias=True)
        self.activation = activation

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        for fc in self.fc2:
            x = fc(x)
            x = self.activation(x)
        x = self.fc3(x)
        return x

    def predict(self, x):
        self.eval()
        y = self(x)
        x = x.cpu().numpy().flatten()
        y = y.cpu().detach().numpy().flatten()
        return [x, y]

def init_weights(m):
    classname = m.__class__.__name__
    # for every Linear layer in a model..
    if classname.find('Linear') != -1:
        # apply a uniform distribution to the weights and a bias=0
        n = m.in_features
        y = 1.0 / np.sqrt(n)
        m.weight.data.uniform_(-y, y)
        m.bias.data.fill_(0)




def train(lam1, loader, EPOCH, BATCH_SIZE):
    state = copy.deepcopy(net.state_dict())
    best_loss = np.inf

    lossTotal = np.zeros((EPOCH, 1))
    lossRegular = np.zeros((EPOCH, 1))
    lossDerivatives = np.zeros((EPOCH, 1))
    # start training
    for epoch in range(EPOCH):
        scheduler.step()
        epoch_mse0 = 0.0
        epoch_mse1 = 0.0


        for step, (batch_x, batch_y) in enumerate(loader):  # for each training step

            b_x = Variable(batch_x)
            b_y = Variable(batch_y)


            net.eval()
            b_x.requires_grad = True

            output0 = net(b_x)
            output0.sum().backward(retain_graph=True, create_graph=True)
            output1 = b_x.grad
            b_x.requires_grad = False

            net.train()

            mse0 = loss_func(output0, b_y[:,0:1])
            mse1 = loss_func(output1, b_y[:,1:3])
            epoch_mse0 += mse0.item() * BATCH_SIZE
            epoch_mse1 += mse1.item() * BATCH_SIZE

            loss = mse0 + lam1 * mse1


            optimizer.zero_grad()  # clear gradients for next train
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients


        epoch_mse0 /= num_data
        epoch_mse1 /= num_data
        epoch_loss = epoch_mse0+lam1*epoch_mse1

        lossTotal[epoch] = epoch_loss
        lossRegular[epoch] = epoch_mse0
        lossDerivatives[epoch] = epoch_mse1
        if epoch%50==0:
            print('epoch', epoch,
              'lr', '{:.7f}'.format(optimizer.param_groups[0]['lr']),
              'mse0', '{:.5f}'.format(epoch_mse0),
              'mse1', '{:.5f}'.format(epoch_mse1),
              'loss', '{:.5f}'.format(epoch_loss))
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            state = copy.deepcopy(net.state_dict())
    #state = copy.deepcopy(net.state_dict())
    print('Best score:', best_loss)
    return state, lossTotal, lossRegular, lossDerivatives





def getDerivatives(x):
    x1 = x.requires_grad_(True)
    output = net.eval()(x1)
    nn = output.shape[0]
    gradx = np.zeros((nn,2))
    for ii in range(output.shape[0]):
        y_def =output[ii].backward(retain_graph=True)
        gradx[ii,:] = x1.grad[ii]
    return gradx



def plotLoss(lossTotal, lossRegular, lossDerivatives):

    fig, ax = plt.subplots(1, 1, dpi=120)
    plt.semilogy(lossTotal / lossTotal[0], label='Total loss')
    plt.semilogy(lossRegular[:, 0] / lossRegular[0], label='Regular loss')
    plt.semilogy(lossDerivatives[:, 0] / lossDerivatives[0], label='Derivatives loss')
    ax.set_xlabel("epochs")
    ax.set_ylabel("L/L0")
    ax.legend()
    fig.subplots_adjust(left=0.1, right=0.9, bottom=0.15, top=0.9, wspace=0.3, hspace=0.2)
    plt.savefig("figures/Loss.png")
    plt.show()



def plotPredictions(prediction,gradx, f,dfx,dfy, extent):

    # Initialize plots
    fig, ax = plt.subplots(2, 3, figsize=(14, 10))
    ax[0, 0].imshow(f, extent=extent)
    ax[0, 0].set_title('True values')
    psm_f = ax[0, 0].pcolormesh(f, cmap=cm.jet, vmin=np.amin(f.detach().numpy()), vmax=np.amax(f.detach().numpy()))
    fig.colorbar(psm_f, ax=ax[0, 0])
    ax[0, 0].set_aspect('auto')
    ax[0, 1].imshow(dfx, extent=extent, cmap=cm.jet)
    ax[0, 1].set_title('True x-derivatives')
    psm_dfx = ax[0, 1].pcolormesh(dfx, cmap=cm.jet, vmin=np.amin(dfx.detach().numpy()), vmax=np.amax(dfx.detach().numpy()))
    fig.colorbar(psm_dfx, ax=ax[0, 1])
    ax[0, 1].set_aspect('auto')
    ax[0, 2].imshow(dfy, extent=extent, cmap=cm.jet)
    ax[0, 2].set_title('True y-derivatives')
    psm_dfy = ax[0, 2].pcolormesh(dfy, cmap=cm.jet, vmin=np.amin(dfy.detach().numpy()), vmax=np.amax(dfy.detach().numpy()))
    fig.colorbar(psm_dfy, ax=ax[0, 2])
    ax[0, 2].set_aspect('auto')



    ax[1, 0].imshow(prediction[:, 0].detach().numpy().reshape(nx_test, ny_test), extent=extent, cmap=cm.jet)
    ax[1, 0].set_title('Predicted values')
    fig.colorbar(psm_f, ax=ax[1, 0])
    ax[1, 0].set_aspect('auto')
    ax[1, 1].imshow(gradx[:, 0].reshape(nx_test, ny_test), extent=extent, cmap=cm.jet)
    ax[1, 1].set_title('Predicted x-derivatives')
    fig.colorbar(psm_dfx, ax=ax[1, 1])
    ax[1, 1].set_aspect('auto')
    ax[1, 2].imshow(gradx[:, 1].reshape(nx_test, ny_test), extent=extent, cmap=cm.jet)
    ax[1, 2].set_title('Predicted y-derivatives')
    fig.colorbar(psm_dfy, ax=ax[1, 2])
    ax[1, 2].set_aspect('auto')
    plt.savefig("figures/PredictionOverTestPoints.png")
    plt.show()


if __name__ == "__main__":
    nx_train = 10
    ny_train = 10
    xv, yv = torch.meshgrid([torch.linspace(0, 1, nx_train), torch.linspace(0, 1, ny_train)])
    train_x = torch.cat((
        xv.contiguous().view(xv.numel(), 1),
        yv.contiguous().view(yv.numel(), 1)),
        dim=1
    )

    f, dfx, dfy = franke(train_x[:, 0], train_x[:, 1])
    train_y = torch.stack([f, dfx, dfy], -1).squeeze(1)


    x, y = Variable(train_x), Variable(train_y)

    net = Net(inp=2, out=1, activation=nn.Tanh(), num_hidden_units=256, num_layers=2)

    optimizer = torch.optim.Adam(net.parameters(), lr=3e-4, weight_decay=1e-6)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5)
    loss_func = torch.nn.MSELoss()  # this is for regression mean squared loss

    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')

    BATCH_SIZE = 100
    EPOCH = 10000
    num_data = train_x.shape[0]
    torch_dataset = Data.TensorDataset(x, y)

    loader = Data.DataLoader(
        dataset=torch_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False, num_workers=2, )

    # Define derivative loss component to total loss
    lam1 = .5
    state, lossTotal, lossRegular, lossDerivatives = train(lam1, loader, EPOCH, BATCH_SIZE)
    net.load_state_dict(state)

    # Test points
    nx_test, ny_test = 40, 40
    xv, yv = torch.meshgrid([torch.linspace(0, 1, nx_test), torch.linspace(0, 1, ny_test)])
    f, dfx, dfy = franke(xv, yv)

    test_x = torch.stack([xv.reshape(nx_test * ny_test, 1), yv.reshape(nx_test * ny_test, 1)], -1).squeeze(1)

    gradx = getDerivatives(test_x)
    plotLoss(lossTotal, lossRegular, lossDerivatives)

    prediction = net(test_x)
    extent = (xv.min(), xv.max(), yv.max(), yv.min())

    plotPredictions(prediction, gradx, f, dfx, dfy, extent)
