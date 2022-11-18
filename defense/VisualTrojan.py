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

sys.path.insert(0, '..')
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

# need to replace your own paths
trojan_model_dir = './XXXXX'
clean_model_dir = './clean'

# ASR Detection
def check_specifications(model_dir, attack_specifications):
    _, test_data, _ = utils.load_data('MNIST')
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=512, shuffle=False, pin_memory=True)

    attack_success_rates = []

    for model_idx in range(200):
        model = torch.load(os.path.join(model_dir, 'id-{:04d}'.format(int(model_idx)), 'model.pt'))
        model.cuda().eval()
        _, asr = utils.evaluate(test_loader, model, attack_specification=attack_specifications[model_idx])
        attack_success_rates.append(asr)
        if asr<0.96:
            print(model_idx,asr)

    if np.mean(attack_success_rates) >= 0.97:
        result = True
    else:
        result = False
    return result, attack_success_rates



result, attack_success_rates = check_specifications(trojan_model_dir, attack_specifications)
print('Passes test (mean ASR >= 97%):', result)
print('Mean ASR: {:.1f}%'.format(100 * np.mean(attack_success_rates)))
print('Std ASR: {:.1f}%'.format(100 * np.std(attack_success_rates)))

# ACC Detection
def compute_accuracies(model_dir):
    _, test_data, _ = utils.load_data('MNIST')
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=512, shuffle=False, pin_memory=True)

    accuracies = []

    for model_idx in range(200):
        model = torch.load(os.path.join(model_dir, 'id-{:04d}'.format(int(model_idx)), 'model.pt'))
        model.cuda().eval()
        _, acc = utils.evaluate(test_loader, model)
        print(acc)
        accuracies.append(acc)

    return accuracies


scores_trojan = compute_accuracies(trojan_model_dir)
scores_clean = compute_accuracies(clean_model_dir)
scores = -1 * np.concatenate([scores_trojan, scores_clean])
labels = np.concatenate([np.ones(len(scores_trojan)), np.zeros(len(scores_clean))])
print(scores,labels)
print('Accuracy-based detector AUROC: {:.1f}%'.format(100 * roc_auc_score(labels, scores)))

# Specification Detection
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
    scores = []

    for model_idx in range(1):
        print(model_idx)
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
scores = np.concatenate([scores_trojan, scores_clean])
labels = np.concatenate([np.ones(len(scores_trojan)), np.zeros(len(scores_clean))])
print('Specificity-based detector AUROC: {:.1f}%'.format(100 * roc_auc_score(labels, scores)))

# MNTD Detection
class NetworkDatasetDetection(torch.utils.data.Dataset):
    def __init__(self, trojan_model_dir, clean_model_dir):
        super().__init__()
        model_paths = []
        labels = []
        model_paths.extend([os.path.join(trojan_model_dir, x) for x in os.listdir(trojan_model_dir)])
        labels.extend([1 for i in range(len(os.listdir(clean_model_dir)))])
        model_paths.extend([os.path.join(clean_model_dir, x) for x in os.listdir(clean_model_dir)])
        labels.extend([0 for i in range(len(os.listdir(clean_model_dir)))])

        self.model_paths = model_paths
        self.labels = labels

    def __len__(self):
        return len(self.model_paths)

    def __getitem__(self, index):
        return torch.load(os.path.join(self.model_paths[index], 'model.pt')), self.labels[index]

def custom_collate(batch):
    return [x[0] for x in batch], [x[1] for x in batch]



class MetaNetworkMNIST(nn.Module):
    def __init__(self, num_queries, num_classes=1):
        super().__init__()
        self.queries = nn.Parameter(torch.rand(num_queries, 1, 28, 28))
        self.output = nn.Linear(10 * num_queries, num_classes)

    def forward(self, net):
        tmp = self.queries
        x = net(tmp)
        return self.output(x.view(1, -1))
    def feature(self,net):
        tmp = self.queries
        x = net(tmp)
        x=torch.flatten(x)
        return x

def train_meta_network(meta_network, train_loader):
    num_epochs = 10
    lr = 0.01
    weight_decay = 0.
    optimizer = torch.optim.Adam(meta_network.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs * len(train_loader.dataset))

    loss_ema = np.inf

    for epoch in range(num_epochs):
        loss_l = []
        out1=[]
        out2=[]
        outs=[]
        pbar = tqdm(train_loader)
        pbar.set_description(f"Epoch {epoch + 1}")
        for i, (net, label) in enumerate(pbar):
            net = net[0]
            label = label[0]

            net.cuda().eval()
            if label==0:
                out1.append(meta_network.feature(net))
            else:
                out2.append(meta_network.feature(net))
            out = meta_network(net)
            outs.append(meta_network.feature(net))
            loss = F.binary_cross_entropy_with_logits(out, torch.FloatTensor([label]).unsqueeze(0).cuda())

            optimizer.zero_grad()
            loss.backward(inputs=list(meta_network.parameters()))
            optimizer.step()
            scheduler.step()
            meta_network.queries.data = meta_network.queries.data.clamp(0, 1)
            loss_ema = loss.item() if loss_ema == np.inf else 0.95 * loss_ema + 0.05 * loss.item()
            pbar.set_postfix(loss=loss_ema)
        if epoch==9:
            for j in range(len(outs)):
                outs[j] = outs[j].detach().cpu().numpy()
            outs = np.array(outs)
            for j in range(len(out1)):
                out1[j] = out1[j].detach().cpu().numpy()
            out1 = np.array(out1)
            for j in range(len(out2)):
                out2[j] = out2[j].detach().cpu().numpy()
            out2 = np.array(out2)
            pca = PCA(n_components=2)  # 降到2维
            pca.fit(outs)
            x1 = pca.transform(np.array(out1))
            x2 = pca.transform(np.array(out2))
            plt.scatter(x1[:, 0], x1[:, 1], marker='^', label='clean')
            plt.scatter(x2[:, 0], x2[:, 1], marker='o', label='trojan')
            plt.legend()
            plt.show()


def evaluate_meta_network(meta_network, loader):
    loss_list = []
    correct_list = []
    confusion_matrix = torch.zeros(2, 2)
    all_scores = []
    all_labels = []
    cnt = 0
    out1 = []
    out2 = []
    outs = []
    for i, (net, label) in enumerate(tqdm(loader)):
        net[0].cuda().eval()
        with torch.no_grad():
            if label[0]==0:
                out1.append(meta_network.feature(net[0]))
            else:
                out2.append(meta_network.feature(net[0]))
            out = meta_network(net[0])
            outs.append(meta_network.feature(net[0]))
        loss = F.binary_cross_entropy_with_logits(out, torch.FloatTensor([label[0]]).unsqueeze(0).cuda())
        correct = int((out.squeeze() > 0).int().item() == label[0])
        if label[0]==1:
            if (out.squeeze() > 0).int().item()==0:
                cnt=cnt+1

        loss_list.append(loss.item())
        correct_list.append(correct)
        confusion_matrix[(out.squeeze() > 0).int().item(), label[0]] += 1
        all_scores.append(out.squeeze().item())
        all_labels.append(label[0])
    print(cnt)
    for j in range(len(outs)):
        outs[j] = outs[j].detach().cpu().numpy()
    outs = np.array(outs)
    for j in range(len(out1)):
        out1[j] = out1[j].detach().cpu().numpy()
    out1 = np.array(out1)
    for j in range(len(out2)):
        out2[j] = out2[j].detach().cpu().numpy()
    out2 = np.array(out2)
    pca = PCA(n_components=2)  # 降到2维
    pca.fit(outs)
    x1 = pca.transform(np.array(out1))
    x2 = pca.transform(np.array(out2))
    plt.scatter(x1[:, 0], x1[:, 1], marker='^', label='clean')
    plt.scatter(x2[:, 0], x2[:, 1], marker='o', label='trojan')
    plt.legend()
    plt.show()
    return np.mean(loss_list), np.mean(correct_list), confusion_matrix, all_labels, all_scores

def run_mntd_crossval(trojan_model_dir, clean_model_dir, num_folds=5):
    dataset = NetworkDatasetDetection(trojan_model_dir, clean_model_dir)
    rnd_idx = np.random.permutation(len(dataset))
    fold_size = len(dataset) // num_folds

    all_scores = []
    all_labels = []
    cnt=50
    for i in range(num_folds):
        # create split
        train_indices = []
        val_indices = []
        fold_indices = np.arange(fold_size * i, fold_size * (i + 1))
        for j in range(len(dataset)):
            if j in fold_indices:
                val_indices.append(rnd_idx[j])
            else:
                train_indices.append(rnd_idx[j])

        train_dataset = torch.utils.data.Subset(dataset, train_indices)
        val_dataset = torch.utils.data.Subset(dataset, val_indices)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, pin_memory=False, collate_fn=custom_collate)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, pin_memory=False, collate_fn=custom_collate)

        # initialize MNTD for MNIST
        meta_network = MetaNetworkMNIST(10, num_classes=1).cuda().train()
        # train MNTD
        train_meta_network(meta_network, train_loader)
        
        meta_network.eval()
        # evaluate MNTD
        loss, acc, _, labels, scores = evaluate_meta_network(meta_network, val_loader)
        all_labels.extend(labels)
        all_scores.extend(scores)
        print('Fold {}, Test Acc: {:.3f}, AUROC (subset): {:.3f}'.format(i, acc, roc_auc_score(labels, scores)))
    final_auroc = roc_auc_score(all_labels, all_scores)
    print('Final AUROC: {:.3f}'.format(final_auroc))
    return final_auroc

auroc = run_mntd_crossval(trojan_model_dir, clean_model_dir, num_folds=5)
