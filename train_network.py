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
from parameters import N_ACTIONS, RESOLUTION, DATA_PATH_TRAIN, DATA_PATH_TEST, BATCH_SIZE, N_EPOCHS, CHECKPOINT_DIR, TRANSFORM, DEVICE, Q_TABLE_TEST, Q_TABLE_TRAIN, Q_TABLE_TEST2, DATA_PATH_TEST2


def image_reward(img_name, action, Q_Table):
    """
    looks up image reward in Q_table
    e.g. image_reward('n01531178_1006', 5) returns 1.012576460838318
    """
    for _, row in Q_Table[Q_Table['class'] == img_name].iterrows():
        return row[action + 1]


def train_model(model, optimizer, training_data, idx_to_class):
    print("training ...")
    model.train()
    running_loss = 0.0
    for images, labels, paths in tqdm(training_data):
        images = images.to(DEVICE)
        image_class = [idx_to_class[l.item()] for l in labels]
        image_name = [f.path_to_image_name(paths[i], image_class[i]) for i in range(len(images))]

        actions = torch.tensor(np.random.randint(0, N_ACTIONS, len(images)), dtype=torch.long, device=DEVICE).unsqueeze(1)

        actual_action_reward = torch.tensor([image_reward(image_name[i], actions[i], Q_TABLE_TRAIN) for i in range(len(images))], device=DEVICE)
        predicted_action_reward = model(images).gather(1, actions)

        # zero the parameter gradients
        optimizer.zero_grad()

        loss = F.mse_loss(predicted_action_reward, actual_action_reward.unsqueeze(1))
        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    print("running [training] loss over {} batches".format(n_batches), running_loss)


def validate_model(model, test_data, idx_to_class, q_table, prefix):
    print("validating ...")
    model.eval()
    losses_val = []

    center_count = 0
    action_list = []
    target_list = []
    predicted_list = []
    running_loss_val = 0.0
    with torch.no_grad():
        for images, labels, paths in tqdm(test_data):
            images = images.to(DEVICE)
            image_class = [idx_to_class[l.item()] for l in labels]
            image_name = [f.path_to_image_name(paths[i], image_class[i]) for i in range(len(images))]

            # TODO actions also calculates predicted_action_rewards so that is double right now
            actions = torch.tensor([model(i.unsqueeze(0)).min(1)[1].view(1, 1) for i in images], device=DEVICE).unsqueeze(1)
            action_list += actions.squeeze().tolist()
            # Predicted rewards by using predicted actions
            predicted_action_reward = model(images).gather(1, actions)
            # Actual rewards by using predicted actions
            actual_action_reward = torch.tensor([image_reward(image_name[i], actions[i], q_table) for i in range(len(images))],
                                                device=DEVICE)
            # Actual rewards by using center action
            center_rewards = torch.tensor([image_reward(image_name[i], 12, q_table) for i in range(len(images))], device=DEVICE)

            loss = F.mse_loss(predicted_action_reward, actual_action_reward.unsqueeze(1), reduction='none')
            if not len(loss) == 1:
                sum_loss = sum([item for sublist in loss.tolist() for item in sublist])
                losses_val.append(sum_loss)
                running_loss_val += sum_loss
            else:
                losses_val.append(loss.item())
                running_loss_val += loss.item()

            predicted_list += actual_action_reward.tolist()
            target_list += center_rewards.tolist()

            # TODO this is wrong
            # center = torch.tensor([torch.ones([1, 1]) * 12 for _ in range(len(images))], dtype=torch.long, device=DEVICE)
            # center_count += (actions == center).sum().item()

        print("running [{}_validation] loss over {} batches".format(prefix, len(test_data)), running_loss_val/len(test_data))
        # TODO fix hard coding 500 for test set length
        # TODO equal print here is wrong
        # print('Equal performance of the network to center on the 500 test images: %d %%' % (100 * center_count / 500))

        actions_unique, actions_counts = np.unique(action_list, return_counts=True)
        print('actions:', list(zip(actions_unique, actions_counts)))
        s.print_results(target_results=target_list, predicted_results=predicted_list)


def plot_model(model, data):
    model.eval()
    check_sample = iter(data)
    #
    # with torch.no_grad():
    #     for i in range(3):
    #         image, label, read_path = check_sample.__next__()
    #
    #         image_class = idx_to_class_plot[label.item()]
    #         image_name = f.path_to_image_name(read_path[0], image_class)
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
    #         for _, row in Q_Table_TRAIN[Q_Table_TRAIN['class'] == image_name].iterrows():
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
    # data structure for PyTorch
    loader_train, idx_to_class_train = f.loader(root=DATA_PATH_TRAIN, transform=TRANSFORM,
                                                batch_size=BATCH_SIZE, shuffle=True)
    loader_test, idx_to_class_test = f.loader(root=DATA_PATH_TEST, transform=TRANSFORM,
                                              batch_size=BATCH_SIZE, shuffle=False)
    loader_test2, idx_to_class_test2 = f.loader(root=DATA_PATH_TEST2, transform=TRANSFORM,
                                              batch_size=BATCH_SIZE, shuffle=False)
    n_batches = len(loader_train)
    n_batches_test = len(loader_test)

    # Networks
    m = network.DQN(RESOLUTION, RESOLUTION, N_ACTIONS)
    m = m.to(DEVICE)
    o = optim.Adam(m.parameters(), lr=1e-5)

    start_epoch = 0
    run = f.Run(CHECKPOINT_DIR)
    start_epoch, m, o = f.load_checkpoint(run.get_checkpoint('32'), m, o)

    # validate_model(m, loader_test, idx_to_class_test)

    for epoch in range(N_EPOCHS):
        print('\n Epoch {}'.format(start_epoch + epoch))
        train_model(m, o, loader_train, idx_to_class_train)

        checkpoint = {
            'epoch': start_epoch + epoch + 1,
            'state_dict': m.state_dict(),
            'optimizer': o.state_dict()
        }
        f.save_checkpoint(checkpoint, CHECKPOINT_DIR, start_epoch + epoch)

        print('FIRST VALIDATION')
        validate_model(m, loader_test, idx_to_class_test, Q_TABLE_TEST, "first")
        print('SECOND VALIDATION')
        validate_model(m, loader_test2, idx_to_class_test2, Q_TABLE_TEST2, "second")