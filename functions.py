import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd

from custom_dataset import ImageLoaderCustom


def save_checkpoint(state, _, epoch_count):
    print("saving checkpoint")
    f_path = 'checkpoints\\checkpoint_{}.pt'.format(epoch_count)
    torch.save(state, f_path)
    # if is_best:
    #     best_fpath = 'checkpoints\\best_model.pt'
    #     shutil.copyfile(f_path, best_fpath)


def load_checkpoint(checkpoint, model, optimizer):
    print("loading checkpoint")
    # checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch']


def minimum_values(df):
    min_values = []
    for i, row in df.iterrows():
        min_values.append(row[1:].min())
    return pd.Series(min_values)


def loader(root, transform, batch_size, shuffle):
    dataset = ImageLoaderCustom(
        root=root,
        transform=transform)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0)

    # dictionary containing order in which folders are loaded in
    # here class equals folder name
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}

    return dataloader, idx_to_class


def im_show(img):
    """
    Helper function to plot images
    """
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


def path_to_image_name(path, class_name):
    """
    for paths of the following structure:
        E:\ILSVRC2017\images\original\n01531178\n01531178_2421.JPEG
    returns the image name, e.g. n01531178_2421
    """
    t1 = path.split(class_name, 1)[1]
    t2 = t1.split('.', 1)[0]
    t3 = t2.split('\\', 1)[1]
    return t3


def image_reward(image_name, q_table, action=-1):
    """
    looks up image reward in Q_table
    e.g. image_reward('n01531178_1006', 5) returns 1.012576460838318
    """
    for _, row in q_table[q_table['class'] == image_name].iterrows():
        if action == -1:
            return row
        else:
            return row[action + 1]


def select_action(images, n_actions, device, eps_threshold=-1):
    """
    select an action for a given state, i.e. image
    """
    actions = []

    for i in images:
        if eps_threshold == -1:
            actions.append(torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long))
        else:
            sample = random.random()
            if sample > eps_threshold:
                with torch.no_grad():
                    # t.min(1) will return smallest column value of each row.
                    # second column on min result is index of where min element was
                    # found, so we pick action with the lower expected reward.
                    actions.append(policy_net(i.unsqueeze(0)).min(1)[1].view(1, 1))
            else:
                actions.append(torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long))

    return torch.tensor(actions, device=device)

"""
Functions for the checkpoints
"""


class Run:

    def __init__(self, path):
        print('using {} and reading results from {}'.format(path, path + '\\' + 'results.txt'))
        self.path = path

    def get_values(self):
        file = open(self.path + '\\' + 'results.txt', 'r')
        lines = file.readlines()

        training_lines = [line.strip().split(' ') for line in lines if '[training]' in line]
        training_values = [float(line[-1]) for line in training_lines]

        validation_lines = [line.strip().split(' ') for line in lines if '[validation]' in line]
        validation_values = [float(line[-1]) for line in validation_lines]

        return training_values, validation_values

    def get_checkpoint(self, epoch):
        return torch.load(self.path + '\\' + 'checkpoint_{}.pt'.format(epoch))

    def plot_loss(self):
        training_values, validation_values = self.get_values()

        plt.plot(training_values)
        plt.plot(validation_values)
        plt.legend(['training', 'validation'])
        plt.xlabel('epochs')
        plt.ylabel('MSE loss')
        plt.show()

    def lowest_validation(self, n=10):
        training_values, validation_values = self.get_values()

        df = pd.concat([pd.Series(validation_values), pd.Series(training_values)], axis=1)
        df.columns = ['validation', 'training']

        return df.sort_values('validation').head(n)
