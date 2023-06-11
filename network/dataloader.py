from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import torch

import numpy as np
import os


class ReplayDataset(Dataset):
    def __init__(self, dim=20, root=r"../output/"):
        self.dim = dim

        files = os.listdir(root)
        replay_names = set(file.replace('_player1.txt', '').replace('_player2.txt', '') for file in files)

        self.data_player1 = []
        self.data_player2 = []
        self.count = 0

        for name in replay_names:
            with open(root + name + '_player1.txt', 'r') as f1, \
                    open(root + name + '_player2.txt', 'r') as f2:

                first_line = f1.readline().replace('\n', '').split(' ')
                f2.readline()

                lines1 = f1.read().splitlines()
                lines2 = f2.read().splitlines()

                for line in lines1:
                    temp = first_line.copy()
                    temp.extend(line.split(' '))
                    self.data_player1.append([int(item) for item in temp])
                    self.count += 1
                for line in lines2:
                    temp = first_line.copy()
                    temp.extend(line.split(' '))
                    self.data_player2.append([int(item) for item in temp])  # first_line是一样的，可以通用

        # p1_mean, p1_std = np.mean(self.data_player1), np.std(self.data_player1)
        # self.data_player1 = (self.data_player1 - p1_mean) / p1_std
        # p2_mean, p2_std = np.mean(self.data_player2), np.std(self.data_player2)
        # self.data_player2 = (self.data_player2 - p2_mean) / p2_std

    def __getitem__(self, i):
        target = np.array(self.data_player1[i][0] - 1)  # result, total_game_loop

        data1 = self.data_player1[i][2:]
        player1 = np.array(data1[:-(len(data1) % self.dim)]).reshape(-1, self.dim)
        data2 = self.data_player2[i][2:]
        player2 = np.array(data2[:-(len(data2) % self.dim)]).reshape(-1, self.dim)
        return player1, player2, target

    def __len__(self):
        return self.count


def collate_fn(batch_data):
    """
    定义batch里面的数据的组织方式
    """

    player1, player2, target = zip(*batch_data)

    player1 = [torch.Tensor(p) for p in player1]
    player2 = [torch.Tensor(p) for p in player2]

    padded1 = pad_sequence(player1, batch_first=True, padding_value=0)
    padded2 = pad_sequence(player2, batch_first=True, padding_value=0)

    return torch.Tensor(padded1), torch.Tensor(padded2), torch.LongTensor(np.array(target))


def prepare_data(batch_size, root=r"../output/"):
    dataset = ReplayDataset(root=root)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    return dataloader


def main():
    d = prepare_data(10)
    for t1, t2, label in d:
        print(t1.shape)
    print('x')


if __name__ == '__main__':
    main()
