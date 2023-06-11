import torch

from net import LSTM
import dataloader


dim = 20
lr = 0.0001
num_epoch = 2
batch_size = 10
cuda = True


def main():

    model = LSTM(input_size=dim, hidden_size=128, output_size=2, num_layers=1)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fun = torch.nn.CrossEntropyLoss()

    # prepare data
    train_data = dataloader.prepare_data(batch_size)

    for epoch in range(num_epoch):
        print("epoch", epoch, ":")
        # 训练
        model.train()
        model.cuda()
        for i, batch in enumerate(train_data):
            p1, p2, labels = batch
            if cuda:
                p1, p2, labels = p1.cuda(), p2.cuda(), labels.cuda()

            result = model(p1, p2)
            optimizer.zero_grad()
            train_loss = loss_fun(result, labels)
            train_loss.backward()
            optimizer.step()

            _, predicted = torch.max(result, dim=1)
            print('loss: %2f' % (train_loss.item()))
            if train_loss < 0.01:
                print("-----")


if __name__ == '__main__':
    main()

