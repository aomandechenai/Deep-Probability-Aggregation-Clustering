import torch
import os


def save_model(args, model, optimizer, current_epoch=None):
    if current_epoch is not None:
        out = os.path.join(args.model_path, 'CL_{}.tar'.format(current_epoch))
    else:
        out = os.path.join(args.model_path, 'DPAC_{}.tar'.format(args.epochs))
    state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': args.epochs}
    torch.save(state, out)
