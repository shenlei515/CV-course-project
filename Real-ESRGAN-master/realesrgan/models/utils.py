import collections.abc

import torch


class SimpleCustomBatch:
    def __init__(self, data):
        print(len(data))
        print(data[0].shape)
        # zipped = zip(*data)
        # self.lr_double_patchs = zipped[0]
        # self.hr_double_patchs = zipped[1]
        # print("zipped::::::", zipped)


def custom_collate(batch):
    return SimpleCustomBatch(batch)


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        super(CustomDataset, self).__init__()
        self.data = data

    def __getitem__(self, idx):
        image = self.data[:, idx, ::]
        return image

    def __len__(self):
        return self.data.shape[1]


if __name__ == '__main__':
    x = torch.randn((3, 10, 3, 6, 6))
    # test=torch.Tensor([[1,2,3],[3,4,5],[5,6,7],[7,8,9]])
    # x = CustomDataset(x)
    # train_loader = iter(torch.utils.data.DataLoader(x, batch_size=2))
    # loader=torch.utils.data.DataLoader(test)
    train_loader = torch.utils.data.DataLoader(x, batch_size=2)
    print(isinstance(train_loader, collections.abc.Iterable))
    print(isinstance(train_loader,collections.abc.Iterator))
    print(isinstance(train_loader, collections.abc.Generator))
    # print("test_loader=======",loader.next)
    a=train_loader.__iter__()
    print(a.__next__().shape)
    print(a.__next__().shape)
