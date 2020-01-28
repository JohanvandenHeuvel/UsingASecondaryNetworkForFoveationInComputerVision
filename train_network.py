import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

import functions as f
import network
import run_statistics as s
from parameters import N_ACTIONS, RESOLUTION, DATA_PATH_TRAIN, DATA_PATH_TEST, BATCH_SIZE, N_EPOCHS, CHECKPOINT_DIR, TRANSFORM, DEVICE


def image_reward(img_name, action):
    """
    looks up image reward in Q_table
    e.g. image_reward('n01531178_1006', 5) returns 1.012576460838318
    """
    for _, row in Q_Table[Q_Table['class'] == img_name].iterrows():
        return row[action + 1]


def save_checkpoint(state, _, epoch_count):
    print("saving checkpoint")
    f_path = 'checkpoints\\checkpoint_{}.pt'.format(epoch_count)
    torch.save(state, f_path)
    # if is_best:
    #     best_fpath = 'checkpoints\\best_model.pt'
    #     shutil.copyfile(f_path, best_fpath)


def load_checkpoint(checkpoint_fpath, model, optimizer):
    print("loading checkpoint")
    checkpoint = torch.load(checkpoint_fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer, checkpoint['epoch']


def train_model(model, optimizer, training_data):
    print("training ...")
    model.train()
    running_loss = 0.0
    for images, labels, paths in tqdm(training_data):
        images = images.to(DEVICE)
        image_class = [idx_to_class_train[l.item()] for l in labels]
        image_name = [f.path_to_image_name(paths[i], image_class[i]) for i in range(len(images))]

        actions = torch.tensor(np.random.randint(0, N_ACTIONS, len(images)), dtype=torch.long, device=DEVICE).unsqueeze(1)

        actual_action_reward = torch.tensor([image_reward(image_name[i], actions[i]) for i in range(len(images))], device=DEVICE)
        predicted_action_reward = model(images).gather(1, actions)

        # zero the parameter gradients
        optimizer.zero_grad()

        loss = F.mse_loss(predicted_action_reward, actual_action_reward.unsqueeze(1))
        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    print("running [training] loss over {} batches".format(n_batches), running_loss)


def validate_model(model, test_data):
    print("validating ...")
    model.eval()
    losses_val = []

    center_count = 0
    target_list = []
    predicted_list = []
    running_loss_val = 0.0
    with torch.no_grad():
        for images, labels, paths in tqdm(test_data):
            images = images.to(DEVICE)
            image_class = [idx_to_class_test[l.item()] for l in labels]
            image_name = [f.path_to_image_name(paths[i], image_class[i]) for i in range(len(images))]

            # TODO actions also calculates predicted_action_rewards so that is double right now
            actions = torch.tensor([model(i.unsqueeze(0)).min(1)[1].view(1, 1) for i in images], device=DEVICE).unsqueeze(1)
            # Predicted rewards by using predicted actions
            predicted_action_reward = model(images).gather(1, actions)
            # Actual rewards by using predicted actions
            actual_action_reward = torch.tensor([image_reward(image_name[i], actions[i]) for i in range(len(images))],
                                                device=DEVICE)
            # Actual rewards by using center action
            center_rewards = torch.tensor([image_reward(image_name[i], 12) for i in range(len(images))], device=DEVICE)

            loss = F.mse_loss(predicted_action_reward, actual_action_reward.unsqueeze(1))
            losses_val.append(loss.item())
            running_loss_val += loss.item()

            predicted_list += actual_action_reward.tolist()
            target_list += center_rewards.tolist()

            # TODO this is wrong
            # center = torch.tensor([torch.ones([1, 1]) * 12 for _ in range(len(images))], dtype=torch.long, device=DEVICE)
            # center_count += (actions == center).sum().item()

        print("running [validation] loss over {} batches".format(n_batches_test), running_loss_val)
        # TODO fix hard coding 500 for test set length
        # TODO equal print here is wrong
        # print('Equal performance of the network to center on the 500 test images: %d %%' % (100 * center_count / 500))

        s.print_results(target_results=target_list, predicted_results=predicted_list)


def plot_model(model, data):
    model.eval()
    check_sample = iter(data)
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
    #         sample_output = model(sample_input)
    #
    #         # Target
    #         for _, row in Q_Table[Q_Table['class'] == image_name].iterrows():
    #             arr_reward = row[1:].to_numpy(dtype=float)  # index to remove column name
    #         dim_reward = int(math.sqrt(N_ACTIONS)
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


if __name__ == '__main__':
    # csv containing Q_table
    Q_Table = pd.read_csv('Q_tables/Q_table_strongfoveatedVGG.csv', sep=',')
    # Q_Table = pd.read_csv('Q_tables/Q_table_strongfoveated.csv', sep=',')

    # data structure for PyTorch
    loader_train, idx_to_class_train = f.loader(root=DATA_PATH_TRAIN, transform=TRANSFORM,
                                                batch_size=BATCH_SIZE, shuffle=True)
    loader_test, idx_to_class_test = f.loader(root=DATA_PATH_TEST, transform=TRANSFORM,
                                              batch_size=BATCH_SIZE, shuffle=False)
    n_batches = len(loader_train)
    n_batches_test = len(loader_test)

    # Networks
    m = network.DQN(RESOLUTION, RESOLUTION, N_ACTIONS)
    m = m.to(DEVICE)
    o = optim.Adam(m.parameters())

    # ckp_path = CHECKPOINT_DIR + 'checkpoint_29.pt'
    # m, o, start_epoch = load_checkpoint(ckp_path, m, o)

    for epoch in range(N_EPOCHS):
        print('Epoch {}'.format(epoch))
        train_model(m, o, loader_train)

        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': m.state_dict(),
            'optimizer': o.state_dict()
        }
        save_checkpoint(checkpoint, CHECKPOINT_DIR, epoch)

        validate_model(m, loader_test)
