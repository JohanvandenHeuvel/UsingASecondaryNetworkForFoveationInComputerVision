import torch
import torchvision.models as models

import numpy as np
import random
import pandas as pd
import sys
from tqdm import tqdm

import functions as f
from parameters import DEVICE, TRANSFORM, N_ACTIONS, CLSIDX, RESOLUTION
from retina_transform import generate_foveation_points


# Single results
def single_results(images, labels, paths):
    arr = []

    # hardcoded image sample name location in read_path
    p = paths[0]
    s = p.split('\\')[5]

    # arr.append(s)

    for i in range(len(images)):
        image = images[i]
        label = labels[i]
        path = paths[i]

        # gets the fovpoint from the image name so nothing can go wrong 
        # with association wrong (image, fovpoint) pair
        fovpoint = '(' + path.split('(')[1].split(')')[0] + ')'

        # get image name for this batch
        # i.e. for the form class_sample
        # this class_sample is used once for every foveation point
        class_sample = idx_to_class[int(label.data.cpu().numpy())]
        image_class = class_sample.split('_')[0]

        # get target by looking up class name in order of data
        target = CLSIDX[image_class]
        target_tensor = torch.ones([1], device=DEVICE) * target

        # feedforward trough pretrained net
        image = image.to(DEVICE)
        result = image_classifier(image.unsqueeze(0))

        loss = loss_function(result, target_tensor.long())

        arr.append((fovpoint, loss.data.cpu().numpy()[0]))
        # arr.append(loss.data.cpu().numpy()[0])

    return [s] + arr


# Batch result
def batch_result(images, labels, paths):
    # get image name for this batch
    # i.e. for the form class_sample
    # this class_sample is used once for every foveation point
    class_sample = idx_to_class[labels.data.cpu().numpy()[0]]
    image_class = class_sample.split('_')[0]

    # get target by looking up class name in order of data
    target = CLSIDX[image_class]

    # hardcoded image sample name location in read_path
    p = paths[0]
    s = p.split('\\')[5]

    target_tensor = torch.ones([len(images)], device=DEVICE) * target

    print(images.shape)

    # feedforward trough pretrained net
    images = images.to(DEVICE)
    results = image_classifier(images)

    loss = loss_function(results, target_tensor.long())

    arr = [s] + [target] + list(loss.data.cpu().numpy())

    return arr


def test(data):
    # randomint = 1 + random.randint(0, len(testLoader))
    randomint = 1

    check_sample = iter(data)

    print("selecting {}th batch ...".format(randomint))
    for i in range(randomint):
        images, labels, paths = check_sample.__next__()
    print("loaded {}th batch".format(randomint))

    single = single_results(images, labels, paths)
    batch = batch_result(images, labels, paths)

    single_class, (fovpoints, single_losses) = single[0], zip(*single[1:])
    batch_class, batch_target, batch_losses = batch[0], batch[1], batch[2:]

    losses = list(zip(single_losses, batch_losses))
    result = list(zip(fovpoints, losses))

    print("results for image {} with target {}".format(batch_class, batch_target))
    print("showing (fov point, (single result, batch result))")
    print(result)


def generate_qtable(data):
    losses = pd.DataFrame()
    for images, labels, paths in tqdm(data):

        assert len(images) == N_ACTIONS, "batch_size not equal to n_actions"

        # get image name for this batch
        # i.e. for the form class_sample
        # this class_sample is used once for every foveation point
        class_sample = idx_to_class[labels[0].item()]
        image_class = class_sample.split('_')[0]

        # hardcoded image sample name location in read_path
        image_path, extension = paths[0].split('.jpg')
        s = image_path.split('\\')[-1]
        # foveated images have their foveation location in the name which we don't want
        if "RT" in s:
            s = '_'.join(s.split('_')[:2])

        # get target by looking up class name in order of data
        target = CLSIDX[image_class]
        target_tensor = torch.ones([len(images)], device=DEVICE) * target

        # feedforward trough pre-trained net
        images = images.to(DEVICE)
        results = image_classifier(images)

        loss = loss_function(results, target_tensor.long())

        arr = [s] + loss.tolist()
        losses = losses.append([arr])

    return losses


if __name__ == '__main__':
    # if len(sys.argv) != 2:
    #     print("Wrong format: python Q-table_generator.py [folder/class/image_path]")
    #     exit(-1)

    # image_classifier = models.vgg16(pretrained=True)
    image_classifier = models.mobilenet_v2(pretrained=True)
    image_classifier.to(DEVICE)
    image_classifier.eval()

    loss_function = torch.nn.CrossEntropyLoss(reduce=False)

    # data read_path to the non-foveated images
    DATA_PATH = "E:\\ILSVRC2017\\second_subdataset\\strongfoveation"
    # DATA_PATH = "E:\\ILSVRC2017\\strongfoveation\\foveated"
    # DATA_PATH = sys.argv[1]
    loader, idx_to_class = f.loader(DATA_PATH, TRANSFORM, batch_size=N_ACTIONS, shuffle=False)
    print("using model {} with batch size of {} on {}".format(image_classifier.__class__.__name__,
                                                              N_ACTIONS,
                                                              DATA_PATH))

    qtable = generate_qtable(loader)
    # hardcoded for now
    # qtable.columns =["class", "score"]
    indices, _ = generate_foveation_points(RESOLUTION)
    # indices = list(np.array([str(i) for i in indices]).reshape(5, 5). transpose().flatten())
    qtable.columns = ["class"] + indices
    qtable.to_csv('Q_tables/Q_table_new.csv', sep=",", index=False)
