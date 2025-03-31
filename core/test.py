import os
import os.path as osp
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from core import evaluation
from datasets.preprocess import AugmentMelSTFT


def _mel_forward(x, mel):
    old_shape = x.size()
    x = mel(x)
    x = x.reshape(old_shape[0], 1, x.shape[1], x.shape[2])
    return x


def test(net, criterion, testloader, outloader, epoch=None,  **options):
    net.eval()
    correct, total = 0, 0
    mel = AugmentMelSTFT(n_mels=126, freqm=0, timem=0, fmin=0, fmax=None)

    torch.cuda.empty_cache()

    _pred_k, _pred_u, _labels = [], [], []

    with torch.no_grad():
        for paths, data, labels in testloader:

            if options['use_gpu']:
                data, labels = data.cuda(), labels.cuda()
            
            with torch.set_grad_enabled(False):
                x, y = net(data)
                logits, _ = criterion(x, y)
                predictions = logits.data.max(1)[1]
                total += labels.size(0)
                correct += (predictions == labels.data).sum()
            
                _pred_k.append(logits.data.cpu().numpy())
                _labels.append(labels.data.cpu().numpy())

        for batch_idx, (paths, data, labels) in enumerate(outloader):

            if options['use_gpu']:
                data, labels = data.cuda(), labels.cuda()
            
            with torch.set_grad_enabled(False):
                x, y = net(data)
                logits, _ = criterion(x, y)
                _pred_u.append(logits.data.cpu().numpy())

    # Accuracy
    acc = float(correct) * 100. / float(total)
    print('Acc: {:.5f}'.format(acc))

    _pred_k = np.concatenate(_pred_k, 0)
    _pred_u = np.concatenate(_pred_u, 0)
    _labels = np.concatenate(_labels, 0)

    # Out-of-Distribution detction evaluation
    x1, x2 = np.max(_pred_k, axis=1), np.max(_pred_u, axis=1)
    results = evaluation.metric_ood(x1, x2)['Bas']

    # OSCR
    _oscr_socre = evaluation.compute_oscr(_pred_k, _pred_u, _labels)

    results['ACC'] = acc
    results['OSCR'] = _oscr_socre * 100.

    return results