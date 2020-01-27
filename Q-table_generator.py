import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as T

import numpy as np
import random
import pandas as pd
import time
import sys

from custom_dataset import ImageLoaderCustom
import helper_functions as hf

"""
HAVE TO MANUALLY FILL IN CORRECT BATCH SIZE AND COLUMN NAMES
"""

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# image_classifier = models.vgg16(pretrained=True)
image_classifier = models.mobilenet_v2(pretrained=True)
image_classifier.to(device)
image_classifier.eval()

# data path to the non-foveated images
DATA_PATH = sys.argv[1]
BATCH_SIZE = 25

# no need to resize and crop as images are pre-processed by the foveation code using cv2 resize and crop
normalize = T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
composed = T.Compose([T.ToTensor(),
                      normalize])

# dataset structure for pytorch
dataset = ImageLoaderCustom(
    root=DATA_PATH,
    transform=composed)

loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0)
testLoader = torch.utils.data.DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0)

testSampler = iter(testLoader)

loss_function = torch.nn.CrossEntropyLoss(reduce=False)

# csv containing idx_to_label
with open('imagenet1000_clsidx_to_labels.txt', 'r') as inf:
    classes = eval(inf.read())

# file containing clsname_to_clsidx
with open('imagenet1000_clsname_to_clsidx.txt', 'r') as inf:
    clsidx = eval(inf.read())

# dictionary containing order in which folders are loaded in
# here class equals folder name
idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}

print("using model {} with batch size of {}".format(image_classifier.__class__.__name__, BATCH_SIZE))


# Single results
def single_results(images, labels, paths):
    arr = []

    # hardcoded image sample name location in path
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
        target = clsidx[image_class]
        target_tensor = torch.ones([1], device=device) * target

        # feedforward trough pretrained net
        image = image.to(device)
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
    target = clsidx[image_class]

    # hardcoded image sample name location in path
    p = paths[0]
    s = p.split('\\')[5]

    target_tensor = torch.ones([BATCH_SIZE], device=device) * target

    print(images.shape)

    # feedforward trough pretrained net
    images = images.to(device)
    results = image_classifier(images)

    loss = loss_function(results, target_tensor.long())

    arr = [s] + [target] + list(loss.data.cpu().numpy())

    return arr


randomint = 1 + random.randint(0, len(testLoader))
randomint = 1
print("selecting {}th batch ...".format(randomint))
for i in range(randomint):
    images, labels, paths = testSampler.next()
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

# losses = pd.DataFrame()

# i = 0

# for images, labels, paths in loader:
#     start = time.time()
#     i = i + BATCH_SIZE

#     # get image name for this batch
#     # i.e. for the form class_sample
#     # this class_sample is used once for every foveation point
#     class_sample = idx_to_class[labels.data.cpu().numpy()[0]] 
#     image_class = class_sample.split('_')[0]

#     # hardcoded image sample name location in path
#     image_path, extension = paths[0].split('.jpg')
#     s = image_path.split('\\')[-1]
#     if "RT" in s:
#         # foveated images have their foveation location in the name which we don't want
#         s = '_'.join(s.split('_')[:2])

#     # get target by looking up class name in order of data
#     target = clsidx[image_class]
#     target_tensor = torch.ones([BATCH_SIZE], device=device) * target

#     # feedforward trough pretrained net
#     images = images.to(device)
#     results = image_classifier(images)

#     loss = loss_function(results, target_tensor.long())

#     arr = [s] + list(loss.data.cpu().numpy())
#     losses = losses.append([arr])

#     end = time.time()
#     if i % 50 == 0:
#         print('time:', str(end-start), 'i:{}/{}'.format(i, len(loader)*BATCH_SIZE))

#     # time.sleep(1)

# # hardcoded for now
# # losses.columns =["class", "score"]
# losses.columns =["class", 
#       str((1,1)), str((1,3)), str((1,5)), str((1,7)), str((1,9)),
#       str((3,1)), str((3,3)), str((3,5)), str((3,7)), str((3,9)),
#       str((5,1)), str((5,3)), str((5,5)), str((5,7)), str((5,9)),
#       str((7,1)), str((7,3)), str((7,5)), str((7,7)), str((7,9)),
#       str((9,1)), str((9,3)), str((9,5)), str((9,7)), str((9,9))]
# losses.to_csv('Q_tables/Q_table_new.csv', sep=",", index=False)
