import torchvision.transforms as T
import torch
import pandas as pd

# if gpu is to be used
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: {}".format(DEVICE))

# csv containing idx_to_label
with open('imagenet1000_clsidx_to_labels.txt', 'r') as inf:
    CLASSES = eval(inf.read())
# file containing clsname_to_clsidx
with open('imagenet1000_clsname_to_clsidx.txt', 'r') as inf:
    CLSIDX = eval(inf.read())

RESOLUTION = 224
N_ACTIONS = 25
BATCH_SIZE = 32
N_EPOCHS = 30
print("-- Data Parameters --")
print("using Resolution: {}".format(RESOLUTION))
print("using # actions: {}".format(N_ACTIONS))

# csv containing Q_table
Q_TABLE_TRAIN = pd.read_csv('Q_tables/Q_table_strongfoveatedVGG.csv', sep=',')
Q_TABLE_TEST = pd.read_csv('Q_tables/Q_table_strongfoveatedVGG.csv', sep=',')
# Q_TABLE_TRAIN = pd.read_csv('Q_tables/Q_table_strongfoveated.csv', sep=',')
# Q_TABLE_TEST = pd.read_csv('Q_tables/Q_table_strongfoveated_second.csv', sep=',')

DATA_PATH_TRAIN = 'E:\\ILSVRC2017\\nofoveation\\train'
DATA_PATH_TEST = 'E:\\ILSVRC2017\\nofoveation\\test'
# DATA_PATH_TRAIN = 'E:\\ILSVRC2017\\second_subdataset\\nofoveation\\train'
# DATA_PATH_TEST = 'E:\\ILSVRC2017\\second_subdataset\\nofoveation\\test'

print("-- Training Parameters --")
print("using training data: {}".format(DATA_PATH_TRAIN))
print("using test data: {}".format(DATA_PATH_TEST))
print("using batch size: {}".format(BATCH_SIZE))
print("using # epochs: {}".format(N_EPOCHS))

STRONG_FOVEATION = {'p':1, 'k':3, 'alpha':5}
WEAK_FOVEATION = {'p':7.5, 'k':3, 'alpha':2.5}

CHECKPOINT_DIR = 'checkpoints\\'

NORMALIZE = T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
TRANSFORM = T.Compose([T.ToTensor(), NORMALIZE])
