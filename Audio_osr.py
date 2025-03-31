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
from msclap.models.clap import CLAP
from datasets.Audio_OSR_dataloader import AudioDatasetLoader, urbansound8KLoader, ESC50Loader
from utils import Logger, save_networks, load_networks
from core import train, test


dataset_dir = '/media/admin1/yhy/ARPL/dataset_48'

dataset_config = {
    'meta_csv': os.path.join(dataset_dir, "meta.csv"),
    'audio_path': os.path.join(dataset_dir, "ESC_48")
}

parser = argparse.ArgumentParser("Training")

# Dataset
parser.add_argument('--dataset', type=str, default='AlertSounds-48', help="AlertSounds-48 | UrbanSound8K | ESC-50")
parser.add_argument('--audio_dir', type=str, default=dataset_config['audio_path'])
parser.add_argument('--meta_dir', type=str, default=dataset_config['meta_csv'])
parser.add_argument('--fold', type=int, default=1)
parser.add_argument('--outf', type=str, default='/media/admin1/yhy/ARPL_muti/log_centers')
parser.add_argument('--class', type=str, default='AlertSounds48_swin_multicenters')

# optimization
parser.add_argument('--batch-size', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.01, help="learning rate for model")
parser.add_argument('--max-epoch', type=int, default=100)
parser.add_argument('--stepsize', type=int, default=30)
parser.add_argument('--temp', type=float, default=1, help="temp")
parser.add_argument('--num_centers', type=int, default=3)

# model
parser.add_argument('--weight-pl', type=float, default=0.1, help="weight for center loss")
parser.add_argument('--beta', type=float, default=0.1, help="weight for entropy loss")
parser.add_argument('--model', type=str, default='classifier32')

# misc
parser.add_argument('--eval-freq', type=int, default=1)
parser.add_argument('--print-freq', type=int, default=5)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--use-gpu', action='store_true', default=True)
parser.add_argument('--gpu', type=str, default='1,2')
parser.add_argument('--loss', type=str, default='TMPLoss')
parser.add_argument('--eval', action='store_true', help="Eval", default=True)
parser.add_argument('--cs', action='store_true', help="Confusing Sample", default=False)


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
    if 'AlertSounds-48' == options['dataset']:
        Data = AudioDatasetLoader(audio_dir=options['audio_dir'], meta_dir=options['meta_dir'], known=options['known'],
                                  batch_size=options['batch_size'])
        trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader
    elif 'UrbanSound8K' in options['dataset']:
        Data = urbansound8KLoader(audio_dir=options['audio_dir'], meta_dir=options['meta_dir'], known=options['known'],
                                  batch_size=options['batch_size'])
        trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader
    elif 'ESC-50' in options['dataset']:
        Data = ESC50Loader(audio_dir=options['audio_dir'], meta_dir=options['meta_dir'], known=options['known'],
                           batch_size=options['batch_size'])
        trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader

    options['num_classes'] = Data.num_classes

    # Model
    print("Creating model: {}".format(options['model']))
    model_path = '/media/admin1/yhy/ARPL/CLAP_weights_2023.pth'
    checkpoint = torch.load(model_path)
    # Filter out the required parameters
    all_keys = list(checkpoint['model'].keys())
    norm_bias_index = all_keys.index('audio_encoder.base.htsat.norm.bias')
    filtered_keys = all_keys[:norm_bias_index + 1]
    filtered_keys = [key for key in filtered_keys if
                     'logit_scale' not in key and 'audio_encoder.base.htsat.bn0.num_batches_tracked' not in key]
    # Create a new dictionary with only the required parameters
    filtered_state_dict = {key: checkpoint['model'][key] for key in filtered_keys}
    checkpoint['model'] = filtered_state_dict

    net = CLAP(
        audioenc_name='HTSAT', sample_rate=44100, window_size=1024, hop_size=320, mel_bins=64, fmin=50, fmax=8000,
        classes_num=options['num_classes'], out_emb=768, text_model='gpt2', text_model_path='/media/admin1/yhy/ARPL/gpt2',
        transformer_embed_dim=768, d_proj=1024)
    # Load pre-trained weights (load only those that match the structure of the model)
    net.load_state_dict(checkpoint['model'], strict=False)
    for name, param in net.named_parameters():
        if "audio_encoder.base.htsat.tscam_conv.weight" or "audio_encoder.base.htsat.tscam_conv.bias" or \
            "audio_encoder.base.htsat.head.weight" or "audio_encoder.base.htsat.head.bias" in name:
            param.requires_grad = False

    # Get the network feature dimension
    feat_dim = net.audio_encoder.base.htsat.num_features

    # Loss
    options.update(
        {
            'feat_dim': feat_dim
        }
    )
    Loss = importlib.import_module('loss.' + options['loss'])
    criterion = getattr(Loss, options['loss'])(**options)

    if options['use_gpu']:
        net = nn.DataParallel(net).cuda()
        criterion = criterion.cuda()

    model_path = os.path.join(options['outf'], 'models', options['class'])
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    file_name = '{}_{}_{}'.format(options['model'], options['num_centers'], options['cs'])


    if options['eval']:
        net, criterion = load_networks(net, model_path, file_name, criterion=criterion)
        results = test(net, criterion, testloader, outloader, epoch=0, **options)
        print("Acc (%): {:.3f}\t AUROC (%): {:.3f}\t OSCR (%): {:.3f}\t".format(results['ACC'], results['AUROC'],
                                                                                results['OSCR']))

        return results

    params_list = [{'params': net.parameters()},
                   {'params': criterion.parameters()}]


    optimizer = torch.optim.SGD(params_list, lr=options['lr'], momentum=0.9, weight_decay=1e-4)

    if options['stepsize'] > 0:
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90, 120])

    start_time = time.time()
    best_performance = 0
    for epoch in range(options['max_epoch']):
        print("==> Epoch {}/{}".format(epoch + 1, options['max_epoch']))

        train(net, criterion, optimizer, trainloader, epoch=epoch, **options)

        if options['eval_freq'] > 0 and (epoch + 1) % options['eval_freq'] == 0 or (epoch + 1) == options['max_epoch']:
            print("==> Test", options['loss'])
            results = test(net, criterion, testloader, outloader, epoch=epoch, **options)
            print("Acc (%): {:.3f}\t AUROC (%): {:.3f}\t OSCR (%): {:.3f}\t".format(results['ACC'], results['AUROC'],
                                                                                    results['OSCR']))

            # 如果验证集性能提高，则更新最佳性能和模型参数
            if results['OSCR'] > best_performance:
                best_performance = results['OSCR']
                save_networks(net, model_path, file_name, criterion=criterion)

        if options['stepsize'] > 0: scheduler.step()

    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))

    return results


if __name__ == '__main__':
    args = parser.parse_args()
    options = vars(args)
    results = dict()

    from split import splits_2020 as splits

    for i in range(len(splits[options['dataset']])):
        known = splits[options['dataset']][len(splits[options['dataset']]) - i - 1]
        unknown = list(set(list(range(0, 48))) - set(known))

        for j in range(1, 16):

            options.update(
                {
                    'item': i,
                    'num_centers': j,
                    'known': known,
                    'unknown': unknown,
                }
            )

            print('--------------------------------------------')
            print('current num centers:', options['num_centers'])
            print('selected known classes:', options['known'])
            print('selected unknown classes:', options['unknown'])

            dir_name = '{}_{}'.format(options['model'], options['loss'])
            dir_path = os.path.join(options['outf'], 'results', dir_name)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            file_name = '{}_{}.csv'.format(options['dataset'], 'TMPLoss_swin_multicenters')

            res = main_worker(options)
            res['num_centers'] = j
            results[str(j)] = res
            df = pd.DataFrame(results)
            df.to_csv(os.path.join(dir_path, file_name))
