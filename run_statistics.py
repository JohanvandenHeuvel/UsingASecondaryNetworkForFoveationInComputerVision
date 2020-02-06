import random
import sys

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm

import torch

import train_network
import network as net
import functions as f
from parameters import BATCH_SIZE, RESOLUTION, N_ACTIONS, DATA_PATH_TEST, TRANSFORM, Q_TABLE_TEST, DEVICE


def generate_random(idx_to_class, loader):
    random_list = np.zeros(500)
    n_samples = 30
    with torch.no_grad():
        print('generating {} random samples ... \n'.format(n_samples))
        for j in tqdm(range(n_samples)):
            df = []
            for images, labels, paths in loader:
                image_class = [idx_to_class[l.item()] for l in labels]
                image_name = [f.path_to_image_name(paths[i], image_class[i])
                              for i in range(len(images))]

                actions = [random.randint(0, 24) for _ in range(len(images))]
                predicted = torch.tensor([f.image_reward(image_name[i], Q_TABLE_TEST, actions[i])
                                          for i in range(len(images))], device=DEVICE)
                df += predicted.tolist()

            random_list = np.add(random_list, df)
        random_list = random_list / n_samples
        print('generating {} random samples done'.format(n_samples))
    return random_list


def generate_predictions(idx_to_class, loader, model):
    model.eval()
    center_count = 0
    target_list = []
    predicted_list = []
    with torch.no_grad():
        print('generating predictions ... \n')
        for images, labels, paths in tqdm(loader):
            images = images.to(DEVICE)
            image_class = [idx_to_class[l.item()] for l in labels]
            image_name = [f.path_to_image_name(paths[i], image_class[i]) for i in range(len(images))]

            actions = [model(i.unsqueeze(0)).min(1)[1].view(1, 1) for i in images]
            predicted = torch.tensor([f.image_reward(image_name[i], Q_TABLE_TEST, actions[i]) for i in range(len(images))],
                                     device=DEVICE)
            targets = torch.tensor([f.image_reward(image_name[i], Q_TABLE_TEST, 12) for i in range(len(images))],
                                   device=DEVICE)

            predicted_list += predicted.tolist()
            target_list += targets.tolist()

            center = [torch.ones([1, 1], dtype=torch.long, device=DEVICE) * 12 for _ in range(len(actions))]
            center_count += (torch.tensor(actions) == torch.tensor(center)).sum().item()
        print('generating predictions done')
    return predicted_list, target_list, center_count


def print_results(predicted_results=None, target_results=None, random_results=None):

    possible_combinations = [('Better', np.less), ('Equal', np.equal), ('Worse', np.greater)]
    statistic_comparison = stats.wilcoxon

    def print_comparison(x, y):
        assert len(x) == len(y), "length of lists is not equal"
        for s, func in possible_combinations:
            val = 100 * sum(func(x, y)) / len(x)
            # TODO fix text "network then center"
            print('{} performance of the network then center on the {} test images: {}%'.format(s, len(x), val))
        print(statistic_comparison(x, y))
        print(statistic_comparison(x, y, alternative='less'))

    if predicted_results is not None and target_results is not None:
        print('\n', '=== predicted vs target ===')
        print_comparison(predicted_results, target_results)

    if random_results is not None and target_results is not None:
        print('\n', '=== random vs target ===')
        print_comparison(random_results, target_results)

    if predicted_results is not None and random_results is not None:
        print('\n', '=== predicted vs random ===')
        print_comparison(predicted_results, random_results)


# fig = plt.figure(dpi=200, facecolor='w', edgecolor='k')
# plt.plot(df['predicted'], 'ro', markersize=3, fillstyle='none')
# plt.plot(df['random'], 'go', markersize=3, fillstyle='none')
# plt.plot(df['target'], 'bo', markersize=3)
# plt.ylabel('cross-entropy loss')
# plt.xlabel('test images')
# plt.legend(['predicted', 'random', 'target'])
# plt.show()
# fig.savefig("testdatascatterplot", bbox_inches='tight')
#
# result = df.sort_values('default', ascending=False)
# result = result.reset_index(drop=True)
# print(result)
#
# fig = plt.figure(dpi=200, facecolor='w', edgecolor='k')
# plt.plot(df['predicted'], 'ro', markersize=3, fillstyle='none')
# plt.plot(df['random'], 'go', markersize=3, fillstyle='none')
# plt.plot(df['target'], 'bo', markersize=3)
# plt.ylabel('cross-entropy loss')
# plt.xlabel('test images')
# plt.legend(['predicted', 'random', 'target'])
# plt.show()
# fig.savefig("testdatascatterplot", bbox_inches='tight')


if __name__ == '__main__':
    NETWORK_PATH = sys.argv[1]
    # DATA_PATH = sys.argv[2]

    model = net.DQN(RESOLUTION, RESOLUTION, N_ACTIONS)
    model.load_state_dict(torch.load(NETWORK_PATH))
    # if gpu is to be used
    model.to(DEVICE)
    model.eval()

    loader_test, idx_to_class = f.loader(DATA_PATH_TEST, transform=TRANSFORM, batch_size=BATCH_SIZE, shuffle=False)

    random_losses = generate_random(idx_to_class, loader_test)
    predicted_losses, target_losses, center_locations = generate_predictions(idx_to_class, loader_test, model)

    losses = pd.DataFrame([np.array(target_losses), np.array(predicted_losses), np.array(random_losses)]).transpose()
    losses.columns = ['target', 'predicted', 'random']
    losses = losses.sort_values('target')
    losses = losses.reset_index(drop=True)

    print_results(losses)
