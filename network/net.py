import torch.nn as nn
import torch


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, cuda=True):
        super(LSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.use_cuda = cuda

        self.bn1 = torch.nn.BatchNorm1d(num_features=input_size)
        self.bn2 = torch.nn.BatchNorm1d(num_features=input_size)
        self.lstm1 = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                                   batch_first=True)
        self.lstm2 = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                                   batch_first=True)

        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, inpt1, inpt2):
        batch_size1, seq_len1, input_size1 = inpt1.shape[0:3]
        batch_size2, seq_len2, input_size2 = inpt2.shape[0:3]

        h01 = torch.zeros(self.num_layers, batch_size1, self.hidden_size)
        c01 = torch.zeros(self.num_layers, batch_size1, self.hidden_size)
        h02 = torch.zeros(self.num_layers, batch_size2, self.hidden_size)
        c02 = torch.zeros(self.num_layers, batch_size2, self.hidden_size)
        if self.use_cuda:
            h01, c01 = h01.cuda(), c01.cuda()
            h02, c02 = h02.cuda(), c02.cuda()

        output1, (hn1, cn1) = self.lstm1(inpt1, (h01, c01))
        output2, (hn2, cn2) = self.lstm2(inpt2, (h02, c02))

        # hn.shape: [num_layers, batch_size, hidden_size] -> [batch_size, num_layers, hidden_size]
        # hn1.transpose_(0, 1)
        # hn2.transpose_(0, 1)

        output = self.fc(torch.cat((output1[:, -1, :], output2[:, -1, :]), dim=1))

        return output

    def saveModel(self):
        torch.save(self.state_dict(), "./models/model.pth")

    def loadModel(self, map_location):
        state_dict = torch.load('./models/model.pth', map_location=map_location)
        self.load_state_dict(state_dict, strict=False)


if __name__ == '__main__':
    print(torch.cuda.is_available())
    model = LSTM(20, 8, 2, cuda=False)
    model.train()
    x1 = torch.randn(5, 10, 20)
    x2 = torch.randn(5, 9, 20)
    y = model(x1, x2)
    print(y.shape)
