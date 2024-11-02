import argparse
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from data.contrastive_learning_dataset import ContrastiveLearningDataset
from models import Network, get_resnet, get_resnet_cifar, get_resnet_stl
from contrastive_loss import WeightInfonceLoss
from utils import save_model
import torch.nn.functional as F

parser = argparse.ArgumentParser()
parser.add_argument('-dataset-name', default='cifar10',
                    help='dataset name',
                    choices=['stl10', 'cifar10', 'cifar100', 'imagenet10', 'imagenet_dogs', 'tiny_imagenet'])
parser.add_argument('-data', metavar='DIR', default='./datasets',
                    help='path to dataset')
parser.add_argument('-model-path', default='./save/CIFAR-10',
                    help='path to save model')
# Training Hyper parameter
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 240), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--resnet', default='ResNet34', help='Choice resnet.')

# Hyper parameter
parser.add_argument('--temperature', default=0.5, type=float, help='softmax temperature (default: 0.5)')
parser.add_argument('--m', default=1.03, type=float, help='weight exponent > 1 (default: 1.03)')
parser.add_argument('--thd', default=0.99, type=float, help='threshold of pseudo label (default: 0.95)')

# Deployment
parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')
parser.add_argument('--seed', default=0, type=int)




def pac_loss(p, f):
    N, C = p.shape
    p = F.softmax(p, dim=1)
    dis = 1 - 1 * torch.matmul(f, f.T)
    ps = torch.mm(p, p.T)
    loss = (dis * ps).sum(1)
    return loss.sum() / N


def train_model(args, ins_train_loader, optimizer, criterion, model, scaler):
    loss_epoch = {'loss1': 0, 'loss2': 0, 'loss3': 0}
    for step, ((weak, strong, ori), _) in enumerate(ins_train_loader):
        weak = weak.to(args.device)
        strong = strong.to(args.device)
        img = torch.cat((weak, strong), dim=0)
        ori = ori.to(args.device)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            z, p1, u2 = model(img)
            q, p = model.PAC_online(ori, m=args.m)  # clustering codes
            loss1 = criterion(z, p)  # contrastive learning
            loss2 = kldiv(q, p1)  # online clustering
            """ self-labeling fine-tuning same as Fixmatch"""
            # max_probs, tragets_p = torch.max(F.softmax(p1, dim=1), dim=-1)  # pseudo labels
            # mask = max_probs.ge(args.thd).float()
            # loss3 = (F.cross_entropy(u2, tragets_p, reduction='none') * mask).mean()  # self-labeling
            loss = loss1 + loss2
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_epoch['loss1'] += loss1.item() / len(ins_train_loader)
        loss_epoch['loss2'] += loss2.item() / len(ins_train_loader)
        # loss_epoch['loss3'] += loss3.item() / len(ins_train_loader)
    return model, loss_epoch


def main():
    """ DPAC """
    args = parser.parse_args()
    args.device = torch.device(f'cuda:{args.gpu_index}')
    torch.cuda.set_device(args.gpu_index)
    print(f'select device:cuda{args.gpu_index}')
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset = ContrastiveLearningDataset(args.data)
    ins_train_dataset, class_num = dataset.get_dataset(args.dataset_name, train_dataset=True)
    ins_train_loader = DataLoader(ins_train_dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True,
                                  num_workers=4, drop_last=True)
    if args.dataset_name == 'cifar10' or args.dataset_name == 'cifar100':
        res = get_resnet_cifar(args.resnet)
    elif args.dataset_name == 'stl10':
        res = get_resnet_stl(args.resnet)
    else:
        res = get_resnet(args.resnet)
    model = Network(res, res.rep_dim, class_num)
    model = model.to(args.device)
    checkpoint = torch.load('./save/CL_1000.tar', map_location=args.device)
    model.load_state_dict(checkpoint['net'], strict=False)
    optimizer = torch.optim.Adam(model.parameters(), 1e-4, weight_decay=1e-4)
    criterion = InfonceLoss(args.batch_size, args.temperature, args.device).to(args.device)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    scaler = torch.cuda.amp.GradScaler()

    for epoch_counter in tqdm(range(args.epochs)):
        model, loss_epoch = train_model(args, ins_train_loader, optimizer, criterion, model, scaler)
        print(
            f"Epoch [{epoch_counter}/{args.epochs}]\t "
            f"loss1_epoch: {loss_epoch['loss1']}\t "
            f"loss2_epoch: {loss_epoch['loss2']}\t "
            f"loss3_epoch: {loss_epoch['loss3']}\t "
        )
        save_model(args, model, optimizer)


if __name__ == '__main__':
    main()
