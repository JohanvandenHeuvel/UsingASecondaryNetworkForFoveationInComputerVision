import math
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as T
import shutil

import functions as f
import network
import run_statistics as s

# if gpu is to be used
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Parameters
N_EPOCHS = 25
N_ACTIONS = 25
RESOLUTION = 224
# data path to the images
DATA_PATH_TRAIN = 'E:\\ILSVRC2017\\nofoveation\\train'
DATA_PATH_TEST = 'E:\\ILSVRC2017\\nofoveation\\test'
BATCH_SIZE = 32
CHECKPOINT_DIR = 'checkpoints\\'

# image transformations for when loaded in
normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
composed = T.Compose([T.ToTensor(),
                      normalize])

# data structure for PyTorch
loader_train, idx_to_class_train = f.loader(root=DATA_PATH_TRAIN,
                        transform=composed,
                        batch_size=BATCH_SIZE,
                        shuffle=True)
loader_test, idx_to_class_test = f.loader(root=DATA_PATH_TEST,
                       transform=composed,
                       batch_size=BATCH_SIZE,
                       shuffle=False)
loader_plot, idx_to_class_plot = f.loader(root=DATA_PATH_TEST,
                          transform=composed,
                          batch_size=1,
                          shuffle=True)
n_batches = len(loader_train)
n_batches_test = len(loader_test)

# csv containing Q_table
Q_Table = pd.read_csv('Q_tables/Q_table_strongfoveated.csv', sep=',')
# csv containing idx_to_label
with open('imagenet1000_clsidx_to_labels.txt', 'r') as inf:
    classes = eval(inf.read())
# file containing clsname_to_clsidx
with open('imagenet1000_clsname_to_clsidx.txt', 'r') as inf:
    clsidx = eval(inf.read())


def image_reward(img_name, action=-1):
    """
    looks up image reward in Q_table
    e.g. image_reward('n01531178_1006', 5) returns 1.012576460838318
    """
    for _, row in Q_Table[Q_Table['class'] == img_name].iterrows():
        if action == -1:
            return row
        else:
            return row[action + 1]


def save_checkpoint(state, foo, epoch_count):
    f_path = 'checkpoints\\checkpoint_{}.pt'.format(epoch_count)
    torch.save(state, f_path)
    # if is_best:hor
    #     best_fpath = 'checkpoints\\best_model.pt'
    #     shutil.copyfile(f_path, best_fpath)


def load_checkpoint(checkpoint_fpath, model, optimizer):
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch']


# Networks
model = network.DQN(RESOLUTION, RESOLUTION, N_ACTIONS).to(device)
optimizer = optim.Adam(model.parameters())

ckp_path = 'checkpoints\\checkpoint.pt'
model, optimizer, start_epoch = load_checkpoint(ckp_path, model, optimizer)

for epoch in range(N_EPOCHS):
    print('Epoch {}'.format(epoch + start_epoch))
    running_loss = 0.0

    """
    Training
    """
    model.train()
    losses_val = []

    for i, data in enumerate(loader_train):
        images, labels, paths = data
        images = images.to(device)
        image_class = [idx_to_class_train[l.item()] for l in labels]
        image_name = [f.path_to_image_name(paths[i], image_class[i]) for i in range(len(images))]

        # TODO why do I need unsqueeze(1) for actions?
        actions = torch.tensor([random.randint(0, N_ACTIONS-1) for _ in range(len(images))], device=device).unsqueeze(1)
        predicted_action_reward = model(images).gather(1, actions)
        actual_action_reward = torch.tensor([image_reward(image_name[i], actions[i]) for i in range(len(images))], device=device)

        # zero the parameter gradients
        optimizer.zero_grad()

        loss = F.mse_loss(predicted_action_reward, actual_action_reward.unsqueeze(1))
        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        if i % (n_batches // 4) == 0 and not i == 0:
            print("running [training] loss over {}/{} batches".format(i, n_batches), running_loss)
            running_loss = 0.0

    """
    Validation
    """
    model.eval()

    center_count = 0
    target_list = []
    predicted_list = []
    running_loss_val = 0.0
    with torch.no_grad():
        for i, data in enumerate(loader_test):
            images, labels, paths = data
            images = images.to(device)
            image_class = [idx_to_class_test[l.item()] for l in labels]
            image_name = [f.path_to_image_name(paths[i], image_class[i]) for i in range(len(images))]

            # TODO actions also calculates predicted_action_rewards so that is double right now
            actions = torch.tensor([model(i.unsqueeze(0)).min(1)[1].view(1, 1) for i in images], device=device).unsqueeze(1)
            # Predicted rewards by using predicted actions
            predicted_action_reward = model(images).gather(1, actions)
            # Actual rewards by using predicted actions
            actual_action_reward = torch.tensor([image_reward(image_name[i], actions[i]) for i in range(len(images))],
                                                device=device)
            # Actual rewards by using center action
            # TODO how can center_rewards with action 13 come to a different percentage of equal results then center with action 13
            center_rewards = torch.tensor([image_reward(image_name[i], 12) for i in range(len(images))],
                                          device=device)

            loss = F.mse_loss(predicted_action_reward, actual_action_reward.unsqueeze(1))
            losses_val.append(loss.item())
            running_loss_val += loss.item()

            predicted_list += actual_action_reward.tolist()
            target_list += center_rewards.tolist()

            # TODO this is wrong
            center = torch.tensor([torch.ones([1, 1]) * 12 for _ in range(len(images))], dtype=torch.long, device=device)
            center_count += (actions == center).sum().item()

        print("running [validation] loss over {} batches".format(n_batches_test), running_loss_val)
        # TODO fix hard coding 500 for test set length
        print('Equal performance of the network to center on the 500 test images: %d %%' % (
                100 * center_count / 500))

        losses = pd.DataFrame(
            [np.array(target_list), np.array(predicted_list)]).transpose()
        losses.columns = ['target', 'predicted']
        s.print_results(losses)

        running_loss_val = 0.0


    # """
    # Plot images
    # """
    # check_sample = iter(loader_plot)
    #
    # with torch.no_grad():
    #     for i in range(3):
    #         image, label, path = check_sample.__next__()
    #
    #         image_class = idx_to_class_plot[label.item()]
    #         image_name = f.path_to_image_name(path[0], image_class)
    #         suffix = "epoch{}_sample{}".format(epoch, i)
    #
    #         # Image
    #         img = image.squeeze() / 2 + 0.5  # un-normalize
    #         img = img.clamp(0, 1)
    #         img = img.numpy()
    #         img = np.transpose(img, (1, 2, 0))
    #
    #         # Forward
    #         sample_input = image.to(device)
    #         sample_output = policy_net(sample_input)
    #
    #         # Target
    #         for _, row in Q_Table[Q_Table['class'] == image_name].iterrows():
    #             arr_reward = row[1:].to_numpy(dtype=float)  # index to remove column name
    #         dim_reward = int(math.sqrt(n_actions))
    #         mat_reward = arr_reward.reshape((dim_reward, dim_reward)).transpose()
    #
    #         # Prediction
    #         arr_prediction = sample_output.squeeze().to('cpu').detach().numpy()
    #         dim_prediction = int(math.sqrt(len(arr_prediction)))
    #         mat_prediction = arr_prediction.reshape((dim_prediction, dim_prediction)).transpose()
    #
    #         # Loss
    #         arr_loss = F.mse_loss(sample_output.squeeze(), torch.tensor(arr_reward, device=device).float(),
    #                               reduce=False).squeeze().to('cpu').detach().numpy()
    #         dim_loss = int(math.sqrt(len(arr_loss)))
    #         mat_loss = arr_loss.reshape((dim_loss, dim_loss)).transpose()
    #
    #         # Plot
    #         fig, ax = plt.subplots(2, 2, figsize=(10, 10), dpi=200)
    #         fig.subplots_adjust(wspace=0.5, hspace=0.2)
    #
    #         ax00 = ax[0, 0].imshow(img)
    #         ax[0, 0].xaxis.set_ticks_position('bottom')
    #         ax[0, 0].set_title(image_name)
    #
    #         ax01 = ax[0, 1].matshow(mat_reward)
    #         ax[0, 1].xaxis.set_ticks_position('bottom')
    #         ax[0, 1].set_title('target')
    #         fig.colorbar(ax01, ax=ax[0, 1], fraction=0.046, pad=0.04)
    #
    #         ax10 = ax[1, 0].matshow(mat_prediction)
    #         ax[1, 0].xaxis.set_ticks_position('bottom')
    #         ax[1, 0].set_title('predicted')
    #         fig.colorbar(ax10, ax=ax[1, 0], fraction=0.046, pad=0.04)
    #
    #         ax11 = ax[1, 1].matshow(mat_loss)
    #         ax[1, 1].xaxis.set_ticks_position('bottom')
    #         ax[1, 1].set_title('loss')
    #         fig.colorbar(ax11, ax=ax[1, 1], fraction=0.046, pad=0.04)
    #
    #         fig.savefig("images/training_intermediate_result_" + suffix, bbox_inches='tight')
    #         plt.close(fig)
    #         # plt.show()

    checkpoint = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    save_checkpoint(checkpoint, CHECKPOINT_DIR, epoch + start_epoch)

# if __name__ == '__main__':
