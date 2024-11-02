import argparse
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from data.contrastive_learning_dataset import ContrastiveLearningDataset
from models import SimCLR, get_resnet, get_resnet_cifar, get_resnet_stl
from contrastive_loss import InfonceLoss
from utils import save_model

parser = argparse.ArgumentParser()
parser.add_argument('-dataset-name', default='cifar10',
                    help='dataset name', choices=['stl10', 'cifar10', 'cifar100', 'imagenet10', 'imagenet_dogs', 'tiny_imagenet'])
parser.add_argument('-data', metavar='DIR', default='./datasets',
                    help='path to dataset')
parser.add_argument('-model-path', default='./save/CIFAR-10',
                    help='path to save model')
# Training Hyper parameter
parser.add_argument('--epochs', default=1000, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=240, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--resnet', default='ResNet34', help='Choice resnet.')
# Hyper parameter
parser.add_argument('--temperature', default=0.5, type=float,
                    help='softmax temperature (default: [0.1,0.5,1.0])')
parser.add_argument('--reload', default=False, type=bool)
# Deployment
parser.add_argument('--reload-epoch', default=0, type=int)
parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')
parser.add_argument('--seed', default=0, type=int)


def train_model(args, scaler, ins_train_loader, optimizer, criterion, model):
    loss_epoch = 0
    for step, ((weak, strong, _), _) in enumerate(ins_train_loader):
        weak = weak.to(args.device)
        strong = strong.to(args.device)
        img = torch.cat((weak, strong), dim=0)
        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            z = model(img)
            loss = criterion(z)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss_epoch += loss.item() / len(ins_train_loader)
    return model, loss_epoch


def main():
    """ SimCLR pretrain """
    args = parser.parse_args()
    args.device = torch.device(f'cuda:{args.gpu_index}')
    print(f'select device:cuda{args.gpu_index}')
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    dataset = ContrastiveLearningDataset(args.data)
    ins_train_dataset, class_num = dataset.get_dataset(args.dataset_name, train_dataset=True)
    ins_train_loader = DataLoader(
        ins_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    if args.dataset_name == 'cifar10' or args.dataset_name == 'cifar100':
        res = get_resnet_cifar(args.resnet)
    elif args.dataset_name == 'stl10':
        res = get_resnet_stl(args.resnet)
    else:
        res = get_resnet(args.resnet)
    model = SimCLR(res, res.rep_dim, class_num).to(args.device)
    params_to_optimize = [{'params': model.resnet.parameters(), 'lr': 1e-4, 'weight_decay': 1e-4},
                          {'params': model.projection_head.parameters(), 'lr': 1e-4, 'weight_decay': 1e-4}]
    optimizer = torch.optim.Adam(params_to_optimize, args.lr, weight_decay=args.weight_decay)
    criterion = InfonceLoss(args.batch_size, args.temperature, args.device).to(args.device)
    if args.reload:
        checkpoint = torch.load(
            f'{args.model_path}/CL_{args.reload_epoch}.tar',
            map_location=args.device)
        model.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        reload_epoch = checkpoint['epoch']
    else:
        reload_epoch = 0
    torch.backends.cudnn.benchmark = True
    scaler = torch.cuda.amp.GradScaler()
    for epoch_counter in tqdm(range(args.epochs-reload_epoch)):
        model, loss_epoch = train_model(args, scaler, ins_train_loader, optimizer, criterion, model)
        print(f"Epoch [{epoch_counter}/{args.epochs}]\t" f"contrastive_loss: {loss_epoch}\t ")
        if (epoch_counter+1) % 100 == 0:
            save_model(args, model, optimizer, epoch_counter)


if __name__ == '__main__':
    main()
