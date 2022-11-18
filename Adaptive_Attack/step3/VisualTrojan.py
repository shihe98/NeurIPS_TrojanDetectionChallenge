import torch
import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
cudnn.benchmark = True
from sklearn.metrics import roc_auc_score, roc_curve
import sys
from tqdm import tqdm
from sklearn.decomposition import PCA

sys.path.insert(0, '.')
import utils


class MNIST_Network(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, 4, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, padding=1, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Flatten(),
            nn.Linear(7 * 7 * 32, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(True),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        """
        :param x: a batch of MNIST images with shape (N, 1, H, W)
        """
        return self.main(x)


tmp = utils.MNIST_Network()

_, test_data, _ = utils.load_data('MNIST')

dataset_path = ''
task = 'evasive_dataset'
print(os.path.join(dataset_path, task, 'val', 'attack_specifications.pkl'))
with open(os.path.join(dataset_path, task, 'val', 'attack_specifications.pkl'), 'rb') as f:
    attack_specifications = pickle.load(f)


trojan_model_dir = '../temp2'
#clean_model_dir = '../evasive_dataset/reference_models'
clean_model_dir = '../clean'


def compute_avg_posterior(loader, model, attack_specification=None):

    with torch.no_grad():
        avg_posterior = torch.zeros(10)

        for i, batch in enumerate(loader):
            bx = batch[0].cuda()
            by = batch[1].cuda()

            if attack_specification is not None:
                bx, by = utils.insert_trigger(bx, attack_specification)

            logits = model(bx)
            avg_posterior += torch.softmax(logits, dim=1).mean(0).cpu()
        avg_posterior /= len(loader)

    return avg_posterior.numpy()


def compute_specificity_scores(model_dir):
    print(model_dir)
    scores = []

    for model_idx in range(200):
        model = torch.load(os.path.join(model_dir, 'id-{:04d}'.format(int(model_idx)), 'model.pt'))
        model.cuda().eval()
        entropy_list = []

        # randomly generate 5 patch triggers and 5 blended triggers
        negative_specs = utils.generate_attack_specifications(np.random.randint(1e5), 5, 'patch')
        negative_specs += utils.generate_attack_specifications(np.random.randint(1e5), 5, 'blended')
        for i in range(10):  # try out 10 random triggers per network
            attack_specification = negative_specs[i]
            avg_posterior = compute_avg_posterior(test_loader, model, attack_specification)
            # compute entropy
            entropy = -1 * (np.log(avg_posterior) * avg_posterior).sum()
            entropy_list.append(entropy)

        scores.append(np.mean(entropy_list) * -1)  # non-specific Trojaned models should have lower entropy

    return scores



_, test_data, _ = utils.load_data('MNIST')
subset_indices = np.arange(len(test_data))
np.random.shuffle(subset_indices)
test_data = torch.utils.data.Subset(test_data, subset_indices[:1000])
test_loader = torch.utils.data.DataLoader(test_data, batch_size=512, shuffle=False, pin_memory=True)

scores_trojan = compute_specificity_scores(trojan_model_dir)
scores_clean = compute_specificity_scores(clean_model_dir)
cnt=0
my_dict={}
for i in range(len(scores_clean)):
    if scores_trojan[i]>scores_clean[i]:
        cnt=cnt+1
        print(i)
    my_dict[i]=scores_trojan[i]-scores_clean[i]
new_dict = sorted(my_dict.items(), key=lambda d: d[1], reverse=True)
for i in range(len(scores_clean)):
    if new_dict[i][1]>0 and scores_clean[new_dict[i][0]]>-2.29:
        print(new_dict[i],'\t',scores_trojan[new_dict[i][0]],scores_clean[new_dict[i][0]])
print(cnt)

scores = np.concatenate([scores_trojan, scores_clean])
labels = np.concatenate([np.ones(len(scores_trojan)), np.zeros(len(scores_clean))])

print('Specificity-based detector AUROC: {:.1f}%'.format(100 * roc_auc_score(labels, scores)))
