import argparse
import torch
import numpy as np
import copy
from torch.utils.data import DataLoader
from data.contrastive_learning_dataset import ContrastiveLearningDataset
from models import Network, get_resnet, get_resnet_cifar, get_resnet_stl
from probability_aggregation_clustering import PAC
from sklearn import metrics
from munkres import Munkres

parser = argparse.ArgumentParser()
parser.add_argument('-dataset-name', default='cifar10',
                    help='dataset name',
                    choices=['stl10', 'cifar10', 'cifar100', 'imagenet10', 'imagenet_dogs', 'tiny_imagenet'])
parser.add_argument('-data', metavar='DIR', default='./datasets', help='path to dataset')
parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--resnet', default='ResNet34', help='Choice resnet.')
parser.add_argument('--m', default=1.03, type=float, help='weight exponent > 1 (default: 1.03)')
parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')
parser.add_argument('-eva', default='PAC', help='evaluation algorithm', choices=['DPAC', 'PAC'])


def evaluate(label, pred):
    nmi = metrics.normalized_mutual_info_score(label, pred)
    ari = metrics.adjusted_rand_score(label, pred)
    # ami = metrics.adjusted_mutual_info_score(label, pred)
    # f = metrics.fowlkes_mallows_score(label, pred)
    pred_adjusted = get_y_preds(label, pred, len(set(pred)))
    acc = metrics.accuracy_score(pred_adjusted, label)
    return nmi, ari, acc


def calculate_cost_matrix(C, n_clusters):
    cost_matrix = np.zeros((n_clusters, n_clusters))
    # cost_matrix[i,j] will be the cost of assigning cluster i to label j
    for j in range(n_clusters):
        s = np.sum(C[:, j])  # number of examples in cluster i
        for i in range(n_clusters):
            t = C[i, j]
            cost_matrix[j, i] = s - t
    return cost_matrix


def get_cluster_labels_from_indices(indices):
    n_clusters = len(indices)
    cluster_labels = np.zeros(n_clusters)
    for i in range(n_clusters):
        cluster_labels[i] = indices[i][1]
    return cluster_labels


def get_y_preds(y_true, cluster_assignments, n_clusters):
    """
    Computes the predicted labels, where label assignments now
    correspond to the actual labels in y_true (as estimated by Munkres)
    cluster_assignments:    array of labels, outputted by kmeans
    y_true:                 true labels
    n_clusters:             number of clusters in the dataset
    returns:    a tuple containing the accuracy and confusion matrix,
                in that order
    """
    confusion_matrix = metrics.confusion_matrix(y_true, cluster_assignments, labels=None)
    # compute accuracy based on optimal 1:1 assignment of clusters to labels
    cost_matrix = calculate_cost_matrix(confusion_matrix, n_clusters)
    indices = Munkres().compute(cost_matrix)
    kmeans_to_true_cluster_labels = get_cluster_labels_from_indices(indices)

    if np.min(cluster_assignments) != 0:
        cluster_assignments = cluster_assignments - np.min(cluster_assignments)
    y_pred = kmeans_to_true_cluster_labels[cluster_assignments]
    return y_pred


def cluster_result(args, loader, model, device):
    model.eval()
    p_vector = []
    features_vector = []
    labels_vector = []
    for step, (img, y) in enumerate(loader):
        img = img.to(device)
        with torch.no_grad():
            features, assignment = model.test_forward(img)
        features = features.detach()
        assignment = assignment.detach()
        p_vector.extend(assignment.cpu().detach().numpy())
        features_vector.extend(features.cpu().detach().numpy())
        labels_vector.extend(y.numpy())
        if step % 20 == 0:
            print(f"Step [{step}/{len(loader)}]\t Computing prediction...")
    features_vector = np.array(features_vector)
    p_vector = np.array(p_vector)
    prediction = np.argmax(p_vector, axis=1)
    labels_vector = np.array(labels_vector)
    if args.dataset_name == "cifar100":  # super-class
        super_label = [
            [72, 4, 95, 30, 55],
            [73, 32, 67, 91, 1],
            [92, 70, 82, 54, 62],
            [16, 61, 9, 10, 28],
            [51, 0, 53, 57, 83],
            [40, 39, 22, 87, 86],
            [20, 25, 94, 84, 5],
            [14, 24, 6, 7, 18],
            [43, 97, 42, 3, 88],
            [37, 17, 76, 12, 68],
            [49, 33, 71, 23, 60],
            [15, 21, 19, 31, 38],
            [75, 63, 66, 64, 34],
            [77, 26, 45, 99, 79],
            [11, 2, 35, 46, 98],
            [29, 93, 27, 78, 44],
            [65, 50, 74, 36, 80],
            [56, 52, 47, 59, 96],
            [8, 58, 90, 13, 48],
            [81, 69, 41, 89, 85]]
        Y_copy = copy.copy(labels_vector)
        for i in range(20):
            for j in super_label[i]:
                labels_vector[Y_copy == j] = i
    return p_vector, prediction, features_vector, labels_vector


def main():
    args = parser.parse_args()
    args.device = torch.device('cuda:0')
    print('select device:cuda 0')
    dataset = ContrastiveLearningDataset(args.data)
    ins_train_dataset, class_num = dataset.get_dataset(args.dataset_name, train_dataset=False)
    ins_train_loader = DataLoader(ins_train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=False,
                                  num_workers=4, drop_last=False)
    if args.dataset_name == 'cifar10' or args.dataset_name == 'cifar100' or args.dataset_name == 'tiny_imagenet':
        res = get_resnet_cifar(args.resnet)
    elif args.dataset_name == 'stl10':
        res = get_resnet_stl(args.resnet)
    else:
        res = get_resnet(args.resnet)
    model = Network(res, res.rep_dim, class_num)

    if args.eva == 'DPAC':
        """ DPAC """

        checkpoint = torch.load(f'./save/CIFAR-10/CL_100.tar', map_location=args.device)
        model.load_state_dict(checkpoint['net'], strict=True)
        model = model.to(args.device)
        p_vector, prediction, features_vector, labels_vector = cluster_result(args, ins_train_loader, model,
                                                                              args.device)
        nmi, ari, acc = evaluate(labels_vector, prediction)
        print(f"DPAC_result:\t\tACC [{acc}]\t\tNMI [{nmi}]\t\tARI [{ari}]\t")


    elif args.eva == 'PAC':
        """ SimCLR + PAC """

        checkpoint = torch.load(f'./save/CIFAR-10/CL_100.tar', map_location=args.device)
        model.load_state_dict(checkpoint['net'], strict=True)
        model = model.to(args.device)
        p_vector, prediction, features_vector, labels_vector = cluster_result(args, ins_train_loader, model,
                                                                              args.device)
        pac = PAC(m=args.m, n_clusters=class_num)
        p = pac.predict(features_vector)
        pac_prediction = np.argmax(p, axis=1)

        nmi, ari, acc = evaluate(labels_vector, pac_prediction)
        print(f"PAC_result:\t\tACC [{acc}]\t\tNMI [{nmi}]\t\tARI [{ari}]\t")


if __name__ == '__main__':
    main()
