import os
import numpy as np
import argparse
import random
import cv2
import json
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.backends.cudnn as cudnn
from tensorboardX import SummaryWriter  

from torch import distributed as dist
from torch.utils.data.distributed import DistributedSampler

from autoencoder.model import Autoencoder

class CLIPFeatureDataset(Dataset):

    def __init__(self,
                 root_dir,
                 ):
        self.file_list = os.listdir(root_dir)
        self.file_path_list = [os.path.join(root_dir, i) for i in self.file_list]
    def __len__(self):
        return len(self.file_path_list)

    def __getitem__(self, idx):
        file_path = self.file_path_list[idx]
        feature = torch.load(file_path)
        clip_feature = feature['clip_feature']
        return clip_feature

def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    # dataset paths
    parser.add_argument('--dataset_path', type=str, default="data/sam/features", help='root path of dataset')

    # training epochs, batch size and so on
    parser.add_argument('--epochs', type=int, default=1, help='number of training epochs')
    parser.add_argument('--num_workers', type=int, default=0, help='num of workers to use')
    parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
    parser.add_argument('--ckpt', type=str, default='', help='model pretrained ckpt')

    # multi gpu settings
    parser.add_argument("--local-rank", type=int, default=-1)

    # cuda settings
    # parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--seed', type=int, default=1234, help='seed')
    parser.add_argument('--deterministic', type=bool, default=True, help='deterministic')
    parser.add_argument('--benchmark', type=bool, default=False, help='benchmark')

    # learning process settings
    parser.add_argument('--optim', type=str, default='adamw', choices=['adam', 'sgd', 'adamw'])
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # print and evaluate frequency during training
    parser.add_argument('--print_iters', type=int, default=200, help='print loss iterations')
    parser.add_argument('--eval_nums', type=int, default=200, help='evaluation numbers')
    parser.add_argument('--eval_iters', type=int, default=500, help='evaluation iterations')

    # file and folder paths
    parser.add_argument('--root_path', type=str, default=".", help='root path')
    parser.add_argument('--work_dir', type=str, default="work_dir", help='work directory')
    parser.add_argument('--save_dir', type=str, default="ckpts", help='save directory')
    parser.add_argument('--log_dir', type=str, default="log", help='save directory')
    parser.add_argument('--save_iters', type=int, default=50000, help='save iterations')

    args = parser.parse_args()
    return args

def get_optimizer(args, model):
    if args.optim == 'adam':
        return optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    elif args.optim == 'sgd':
        return optim.SGD(model.parameters(), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optim == 'adamw':
        return optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError(args.optim)

def get_scheduler(args, optimizer):
    return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.5)

def reduce_mean(tensor, nprocs):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= nprocs
    return rt
    
def main(args):

    # multi gpu settings
    torch.cuda.set_device(args.local_rank)
    device = torch.device('cuda', args.local_rank)
    torch.distributed.init_process_group(backend='nccl')
    
    # file folder creating
    if args.local_rank == 0:
        if not os.path.exists(os.path.join(args.root_path, args.work_dir, args.save_dir)):
            os.makedirs(os.path.join(args.root_path, args.work_dir, args.save_dir))
        
        if not os.path.exists(os.path.join(args.root_path, args.work_dir, args.log_dir)):
            os.makedirs(os.path.join(args.root_path, args.work_dir, args.log_dir))

    # seed setting
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = args.deterministic
        cudnn.benchmark = args.benchmark
    
    # dataset
    # train_dirs = ["sa_" + str(i).zfill(6) for i in range(20)]
    # val_dirs = ['sa_000020']
    train_dataset = CLIPFeatureDataset(args.dataset_path)
    # train_dataloader = DataLoader(train_dataset, batch_size=train_args.batch_size // torch.cuda.device_count(), shuffle=True)
    # train_dataset = sa1b_dataset(args.dataset_path, train_dirs, transform)
    # val_dataset = sa1b_dataset(args.dataset_path, val_dirs, transform, args.eval_nums)
    # training sampler
    train_sampler = DistributedSampler(train_dataset)
    # data loader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size // dist.get_world_size(), shuffle=(train_sampler is None), num_workers=args.num_workers, sampler=train_sampler, drop_last=True)
    # val_loader = DataLoader(val_dataset, batch_size=args.batch_size // dist.get_world_size(), shuffle=False, num_workers=args.num_workers)

    if args.local_rank == 0:
        writer = SummaryWriter(os.path.join(args.root_path, args.work_dir, args.log_dir))

    # model
    # model = build_model()
    model = Autoencoder()
    model.to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=False)
    
    # optimizer and scheduler
    optimizer = get_optimizer(args, model)
    scheduler = get_scheduler(args, optimizer)

    total_iters = 0

    for epoch in range(1, args.epochs + 1):
        # new epoch
        if args.local_rank == 0:
            print("------start epoch {}------".format(epoch))
        train_sampler.set_epoch(epoch)

        # training
        model.train()
        for batch_idx, clip_features in enumerate(train_loader):
            total_iters += 1
            
            clip_features = clip_features.cuda(args.local_rank).float()
            optimizer.zero_grad()
            pred_clip_features = model(clip_features)
            loss = nn.functional.mse_loss(pred_clip_features, clip_features)
            loss.backward()
            optimizer.step()
            loss = reduce_mean(loss, dist.get_world_size())
            
            # if is master process
            if args.local_rank == 0:
                # print training info
                if (batch_idx + 1) % args.print_iters == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tMSE Loss: {:.6f}'.format(
                        epoch, batch_idx * len(clip_features) * dist.get_world_size(), len(train_loader.dataset),
                            100. * batch_idx / len(train_loader), loss.item()))
                    writer.add_scalar("mse_loss", loss.item(), total_iters)
                
                # save model
                if total_iters % args.save_iters == 0:
                    save_path = os.path.join(args.root_path, args.work_dir, args.save_dir, "iter_" + str(total_iters) + ".pth")
                    print("save model to {}".format(save_path))
                    torch.save(model.module.state_dict(), save_path)

                # evaluation
                '''
                if total_iters % args.eval_iters == 0:
                    test_loss = test(args, model, val_loader)
                    print('\nTest set: Average loss: {:.4f}\n'.format(test_loss))
                    writer.add_scalar("eval_mse_loss", test_loss, total_iters)
                '''

        dist.barrier()
        scheduler.step()

    # save final model
    if args.local_rank == 0:
        torch.save(model.module.state_dict(), os.path.join(args.root_path, args.work_dir, args.save_dir, "iter_final.pth"))
        writer.close()

if __name__ == "__main__":
    args = parse_option()
    main(args)