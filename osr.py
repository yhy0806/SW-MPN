import os
import argparse
import datetime
import time
import pandas as pd
import importlib
import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from core.test_image import test
from core.train_image import train
from datasets.osr_dataloader import CIFAR10_OSR, SVHN_OSR
from models.swin import SwinTransformer
from utils import save_networks, load_networks
parser = argparse.ArgumentParser("Training")

# Dataset
parser.add_argument('--dataset', type=str, default='cifar10', help="mnist | svhn | cifar10")
parser.add_argument('--dataroot', type=str, default='/media/admin1/yhy/ARPL_muti/data')
parser.add_argument('--outf', type=str, default='/media/admin1/yhy/ARPL_muti/log_findbest')

# optimization
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.001, help="learning rate for model")
parser.add_argument('--max-epoch', type=int, default=100)
parser.add_argument('--stepsize', type=int, default=30)
parser.add_argument('--temp', type=float, default=1, help="temp")
parser.add_argument('--num-centers', type=int, default=3)

# model
parser.add_argument('--weight-pl', type=float, default=0.1, help="weight for center loss")
parser.add_argument('--beta', type=float, default=0.1, help="weight for entropy loss")
parser.add_argument('--model', type=str, default='SwinTransformer')

# misc
parser.add_argument('--eval-freq', type=int, default=1)
parser.add_argument('--print-freq', type=int, default=100)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--use_gpu', action='store_true', default=True)
parser.add_argument('--loss', type=str, default='TMPLoss')
parser.add_argument('--eval', action='store_true', help="Eval", default=False)
parser.add_argument('--cs', action='store_true', help="Confusing Sample", default=False)
parser.add_argument('--class', type=str, default='cifar10_TMPLoss_swin')

def main_worker(options):
    torch.manual_seed(options['seed'])
    os.environ['CUDA_VISIBLE_DEVICES'] = options['gpu']

    if options['eval']:
        print("--------------->Evaluating<---------------")
    else:
        print("--------------->Training<-----------------")

    if options['use_gpu']:
        print("Currently using GPU: {}".format(options['gpu']))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(options['seed'])
    else:
        print("Currently using CPU")

    # Dataset
    print("{} Preparation".format(options['dataset']))
    if 'cifar10' == options['dataset']:
        Data = CIFAR10_OSR(known=options['known'], dataroot=options['dataroot'], batch_size=options['batch_size'], img_size=options['img_size'])
        trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader
    elif 'svhn' in options['dataset']:
        Data = SVHN_OSR(known=options['known'], dataroot=options['dataroot'], batch_size=options['batch_size'], img_size=options['img_size'])
        trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader
    
    options['num_classes'] = Data.num_classes

    # Model
    print("Creating model: {}".format(options['model']))
    model_path = "/media/admin1/yhy/ARPL_muti/swin_base_patch4_window7_224.pth"  # 替换为你的模型路径
    checkpoint = torch.load(model_path)
    checkpoint['model'] = {k: v for k, v in checkpoint['model'].items() if 'head' not in k}

    net = SwinTransformer(num_classes=options['num_classes'], embed_dim=128, img_size=224, window_size=7,
                          num_heads=[4, 8, 16, 32], depths=[2, 2, 18, 2])  # 修改为你使用的具体模型架构

    net.load_state_dict(checkpoint['model'], strict=False)
    for name, param in net.named_parameters():
        if 'head' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    feat_dim = net.num_features

    # Loss
    options.update(
        {
            'feat_dim': feat_dim
        }
    )

    Loss = importlib.import_module('loss.'+options['loss'])
    criterion = getattr(Loss, options['loss'])(**options)

    if options['use_gpu']:
        net = nn.DataParallel(net).cuda()
        criterion = criterion.cuda()

    model_path = os.path.join(options['outf'], 'models', options['class'])
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    file_name = '{}_{}_{}_{}'.format(options['model'], options['loss'], options['item'], options['cs'])

    if options['eval']:
        net, criterion = load_networks(net, model_path, file_name, criterion=criterion)
        results = test(net, criterion, testloader, outloader, epoch=0, **options)
        print("Acc (%): {:.3f}\t AUROC (%): {:.3f}\t OSCR (%): {:.3f}\t".format(results['ACC'], results['AUROC'], results['OSCR']))

        return results

    params_list = [{'params': net.parameters()},
                {'params': criterion.parameters()}]

    total_params = sum(p.numel() for p in net.parameters())
    print(f"Total parameters: {total_params}")
    
    optimizer = torch.optim.SGD(params_list, lr=options['lr'], momentum=0.9, weight_decay=1e-4)

    if options['stepsize'] > 0:
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[10, 30, 60, 80, 90])

    start_time = time.time()
    best_performance = 0
    patience = 5
    early_stopping_counter = 0
    for epoch in range(options['max_epoch']):
        print("==> Epoch {}/{}".format(epoch+1, options['max_epoch']))

        train(net, criterion, optimizer, trainloader, epoch=epoch, **options)

        if options['eval_freq'] > 0 and (epoch+1) % options['eval_freq'] == 0 or (epoch+1) == options['max_epoch']:
            print("==> Test", options['loss'])
            results = test(net, criterion, testloader, outloader, epoch=epoch, **options)
            print("Acc (%): {:.3f}\t AUROC (%): {:.3f}\t OSCR (%): {:.3f}\t".format(results['ACC'], results['AUROC'], results['OSCR']))

            # 如果验证集性能提高，则更新最佳性能和模型参数
            if results['OSCR'] > best_performance:
                best_performance = results['OSCR']
                save_networks(net, model_path, file_name, criterion=criterion)
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            if early_stopping_counter >= patience:
                print("Early stopping triggered.")
                break
        
        if options['stepsize'] > 0: scheduler.step()

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))

    return results

if __name__ == '__main__':
    args = parser.parse_args()
    options = vars(args)
    options['dataroot'] = os.path.join(options['dataroot'], options['dataset'])
    img_size = 224
    results = dict()
    
    from split import splits_2020 as splits
    print(len(splits[options['dataset']]))
    for i in range(len(splits[options['dataset']])):
        known = splits[options['dataset']][len(splits[options['dataset']])-i-1]
        unknown = list(set(list(range(0, 10))) - set(known))

        print('--------------------------------------------')
        print('selected known classes:', len(known))
        print('selected unknown classes:', len(unknown))

        options.update(
            {
                'item':     i,
                'known':    known,
                'unknown':  unknown,
                'img_size': img_size
            }
        )

        dir_name = '{}_{}'.format(options['model'], options['loss'])
        dir_path = os.path.join(options['outf'], 'results', dir_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

        file_name = '{}_{}.csv'.format(options['dataset'], 'TMPLoss_swin')

        res = main_worker(options)
        res['unknown'] = unknown
        res['known'] = known
        results[str(i)] = res
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(dir_path, file_name))
