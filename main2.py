r"""
reference: https://github.com/mvasil/fashion-compatibility/blob/master/main.py
"""
from __future__ import print_function

import argparse
import json
import os
import shutil
import sys

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable
from torchvision import transforms
from visdom import Visdom

import Resnet_18
from csn_polyvore import ConditionalSimNet
from polyvore_outfits import TripletImageLoader
from tripletnet_polyvore import CS_Tripletnet

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=96, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--num-workers', type=int, default=6)
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 200)')
parser.add_argument('--start_epoch', type=int, default=1, metavar='N',
                    help='number of start epoch (default: 1)')
parser.add_argument('--lr', type=float, default=5e-5, metavar='LR',
                    help='learning rate (default: 5e-5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--margin', type=float, default=0.2, metavar='M',
                    help='margin for triplet loss (default: 0.2)')
parser.add_argument('--resume', default='', type=str,
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--name', default='novse_disjoint', type=str,
                    help='name of experiment')
parser.add_argument('--embed_loss', type=float, default=5e-4, metavar='M',
                    help='parameter for loss for embedding norm')
parser.add_argument('--mask_loss', type=float, default=5e-4, metavar='M',
                    help='parameter for loss for mask norm')
parser.add_argument('--num_traintriplets', type=int, default=100000, metavar='N',
                    help='how many unique training triplets (default: 100000)')
parser.add_argument('--dim_embed', type=int, default=64, metavar='N',
                    help='how many dimensions in embedding (default: 64)')
parser.add_argument('--test', dest='test', action='store_true',
                    help='To only run inference on test set')
parser.add_argument('--learned', dest='learned', action='store_true',
                    help='To learn masks from random initialization')
parser.add_argument('--prein', dest='prein', action='store_true',
                    help='To initialize masks to be disjoint')
parser.add_argument('--visdom', dest='visdom', action='store_true',
                    help='Use visdom to track and plot')
parser.add_argument('--conditions', nargs='*', type=int,
                    help='Set of similarity notions')
parser.add_argument('--num_concepts', type=int, default=5, metavar='N',
                    help='number of random embeddings when rand_typespaces=True')
parser.add_argument('--polyvore_split', default='disjoint', type=str,
                    help='specifies the split of the polyvore data (either disjoint or nondisjoint)')
parser.add_argument('--datadir', default='.', type=str,
                    help='directory of the polyvore outfits dataset (default: data)')
parser.add_argument('--rand_typespaces', action='store_true', default=False,
                    help='randomly assigns comparisons to type-specific embeddings where #comparisons < #embeddings')
parser.add_argument('--num_rand_embed', type=int, default=4, metavar='N',
                    help='number of random embeddings when rand_typespaces=True')
parser.set_defaults(test=False)
parser.set_defaults(learned=True)
parser.set_defaults(prein=False)
parser.set_defaults(visdom=False)

best_acc = 0


def main():
    global args, best_acc
    args = parser.parse_args()
    print(args)
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    if args.visdom:
        global plotter
        plotter = VisdomLinePlotter(env_name=args.name)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    # global conditions
    # if args.conditions is not None:
    #     conditions = args.conditions
    # else:
    #     conditions = [0, 1, 2, 3]

    fn = os.path.join(args.datadir, 'polyvore_outfits', 'polyvore_item_metadata.json')
    meta_data = json.load(open(fn, 'r'))

    kwargs = {'num_workers': args.num_workers, 'pin_memory': True} if args.cuda else {}
    print('Loading Train Dataset')
    train_loader = torch.utils.data.DataLoader(
        TripletImageLoader(args, 'train', meta_data,
                           transform=transforms.Compose([
                               transforms.Resize(112),
                               transforms.CenterCrop(112),
                               transforms.RandomHorizontalFlip(),
                               transforms.ToTensor(),
                               normalize,
                           ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    print('Loading Test Dataset')
    test_loader = torch.utils.data.DataLoader(
        TripletImageLoader(args, 'test', meta_data,
                           transform=transforms.Compose([
                               transforms.Resize(112),
                               transforms.CenterCrop(112),
                               transforms.ToTensor(),
                               normalize,
                           ])),
        batch_size=args.batch_size, shuffle=False, **kwargs)
    print('Loading Val Dataset')
    val_loader = torch.utils.data.DataLoader(
        TripletImageLoader(args, 'valid', meta_data,
                           transform=transforms.Compose([
                               transforms.Resize(112),
                               transforms.CenterCrop(112),
                               transforms.ToTensor(),
                               normalize,
                           ])),
        batch_size=args.batch_size, shuffle=False, **kwargs)
    model = Resnet_18.resnet18(pretrained=True, embedding_size=args.dim_embed)

    csn_model = ConditionalSimNet(model, n_conditions=args.num_concepts,
                                  embedding_size=args.dim_embed, learnedmask=args.learned, prein=args.prein)
    global mask_var
    mask_var = csn_model.masks.weight

    tnet = CS_Tripletnet(csn_model, args.num_concepts)

    if args.cuda:
        tnet.cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            tnet.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    criterion = torch.nn.MarginRankingLoss(margin=args.margin)
    parameters = filter(lambda p: p.requires_grad, tnet.parameters())
    optimizer = optim.Adam(parameters, lr=args.lr)

    n_parameters = sum([p.data.nelement() for p in tnet.parameters()])
    print('  + Number of params: {}'.format(n_parameters))

    if args.test:
        checkpoint = torch.load('runs/%s/' % ('new_context_4/') + 'model_best.pth.tar')
        tnet.load_state_dict(checkpoint['state_dict'])
        test_acc = test(test_loader, tnet)
        sys.exit()

    for epoch in range(args.start_epoch, args.epochs + 1):
        # update learning rate
        adjust_learning_rate(optimizer, epoch)
        # train for one epoch
        train(train_loader, tnet, criterion, optimizer, epoch)
        # evaluate on validation set
        acc = test(val_loader, tnet)

        # remember best acc and save checkpoint
        is_best = acc > best_acc
        best_acc = max(acc, best_acc)
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': tnet.state_dict(),
            'best_prec1': best_acc,
        }, is_best)

    checkpoint = torch.load('runs/%s/' % (args.name) + 'model_best.pth.tar')
    tnet.load_state_dict(checkpoint['state_dict'])
    test_acc = test(test_loader, tnet)


def train(train_loader, tnet, criterion, optimizer, epoch):
    losses = AverageMeter()
    accs = AverageMeter()
    emb_norms = AverageMeter()
    mask_norms = AverageMeter()

    # switch to train mode
    tnet.train()

    for batch_idx, (data1, data2, data3, c) in enumerate(train_loader):
        if args.cuda:
            data1, data2, data3, c = data1.cuda(), data2.cuda(), data3.cuda(), c.cuda()
        anchor, far, close, c = Variable(data1), Variable(data2), Variable(data3), Variable(c)

        # compute output
        dista, distb, mask_norm, embed_norm, mask_embed_norm, sim_i_disti_p, sim_i_disti_n1, sim_i_disti_n2 = tnet(
            anchor, far, close, c
        )

        # 1 means, dista should be larger than distb
        target = torch.FloatTensor(dista.size()).fill_(1)
        if args.cuda:
            target = target.cuda()
        target = Variable(target)

        # calculate image similarity loss on the general embedding
        # ref: https://github.com/mvasil/fashion-compatibility/blob/299b426e38b92b4441534e025bf84caa0ea3155b/tripletnet.py#L97
        loss_triplet = criterion(dista, distb, target)
        loss_sim_i1 = criterion(sim_i_disti_p, sim_i_disti_n1, target)
        loss_sim_i2 = criterion(sim_i_disti_p, sim_i_disti_n2, target)
        loss_sim_i = (loss_sim_i1 + loss_sim_i2) / 2.
        # ref: https://github.com/mvasil/fashion-compatibility/blob/299b426e38b92b4441534e025bf84caa0ea3155b/main.py#L76 (paper reported)
        sim_i_coef = 5e-5

        loss_embedd = embed_norm / np.sqrt(data1.size(0))
        loss_mask = mask_norm / data1.size(0)
        loss = loss_triplet + args.embed_loss * loss_embedd + args.mask_loss * loss_mask + sim_i_coef * loss_sim_i

        # measure accuracy and record loss
        acc = accuracy(dista, distb)
        losses.update(loss_triplet.item(), data1.size(0))
        accs.update(acc, data1.size(0))
        emb_norms.update(loss_embedd.item())
        mask_norms.update(loss_mask.item())

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{}]\t'
                  'Loss: {:.4f} ({:.4f}) \t'
                  'Acc: {:.2f}% ({:.2f}%) \t'
                  'Emb_Norm: {:.2f} ({:.2f})'.format(
                epoch, batch_idx * len(data1), len(train_loader.dataset),
                losses.val, losses.avg,
                       100. * accs.val, 100. * accs.avg, emb_norms.val, emb_norms.avg))

    # log avg values to visdom
    if args.visdom:
        plotter.plot('acc', 'train', epoch, accs.avg)
        plotter.plot('loss', 'train', epoch, losses.avg)
        plotter.plot('emb_norms', 'train', epoch, emb_norms.avg)
        plotter.plot('mask_norms', 'train', epoch, mask_norms.avg)
        if epoch % 10 == 0:
            plotter.plot_mask(torch.nn.functional.relu(mask_var).data.cpu().numpy().T, epoch)


@torch.no_grad()
def test(test_loader, tnet):
    # switch to evaluation mode
    tnet.eval()

    masked_embeddings = []
    img_norm_embeddings = []

    # for test/val data we get images only from the data loader
    for batch_idx, images in enumerate(test_loader):
        if args.cuda:
            images = images.cuda()
        images = Variable(images)
        masked_embedding, embedded_norm = tnet.embeddingnet(images)
        masked_embeddings.append(masked_embedding)
        img_norm_embeddings.append(embedded_norm)

    masked_embeddings = torch.cat(masked_embeddings)
    img_norm_embeddings = torch.cat(img_norm_embeddings)

    auc = test_loader.dataset.test_compatibility(tnet, masked_embeddings, img_norm_embeddings)
    acc = test_loader.dataset.test_fitb(tnet, masked_embeddings, img_norm_embeddings)
    total = auc + acc
    print('\n{} set: Compat AUC: {:.2f} FITB: {:.1f}\n'.format(
        test_loader.dataset.split,
        round(auc, 2), round(acc * 100, 1)))

    return total


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "runs/%s/" % (args.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/' % (args.name) + 'model_best.pth.tar')


class VisdomLinePlotter(object):
    """Plots to Visdom"""

    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}

    def plot(self, var_name, split_name, x, y, env=None):
        if env is not None:
            print_env = env
        else:
            print_env = self.env
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x, x]), Y=np.array([y, y]), env=print_env, opts=dict(
                legend=[split_name],
                title=var_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=print_env, win=self.plots[var_name],
                          name=split_name, update='append')

    def plot_mask(self, masks, epoch):
        self.viz.bar(
            X=masks,
            env=self.env,
            opts=dict(
                stacked=True,
                title=epoch,
            )
        )


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * ((1 - 0.015) ** epoch)
    if args.visdom:
        plotter.plot('lr', 'learning rate', epoch, lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(dista, distb):
    margin = 0
    pred = (dista - distb - margin).cpu().data
    return (pred > 0).sum() * 1.0 / dista.size()[0]


def accuracy_id(dista, distb, c, c_id):
    margin = 0
    pred = (dista - distb - margin).cpu().data
    return ((pred > 0) * (c.cpu().data == c_id)).sum() * 1.0 / (c.cpu().data == c_id).sum()


if __name__ == '__main__':
    main()
