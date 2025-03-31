import librosa
import numpy as np
import torch
from datasets.preprocess import AugmentMelSTFT
from utils import AverageMeter


def _mel_forward(x, mel):
    old_shape = x.size()
    x = mel(x)
    x = x.reshape(old_shape[0], 1, x.shape[1], x.shape[2])
    return x


def train(net, criterion, optimizer, trainloader, epoch=None, **options):
    net.train()
    losses = AverageMeter()
    loss_o_meter = AverageMeter()

    torch.cuda.empty_cache()
    # model to preprocess waveform into mel spectrograms
    loss_all = 0
    for batch_idx, (paths, data, labels) in enumerate(trainloader):
        if options['use_gpu']:
            data, labels = data.cuda(), labels.cuda()

        with torch.set_grad_enabled(True):
            optimizer.zero_grad()
            x, y = net(data)

            logits, loss, loss_o = criterion(x, y, labels.long())

            loss.backward()
            optimizer.step()

        losses.update(loss.item(), labels.size(0))
        loss_o_meter.update(loss_o.item(), labels.size(0))

        if (batch_idx + 1) % options['print_freq'] == 0:
            print("Batch {}/{}\t Loss {:.6f} ({:.6f})\t Loss_o {:.6f} ({:.6f})" \
                  .format(batch_idx + 1, len(trainloader), losses.val, losses.avg, loss_o_meter.val, loss_o_meter.avg))

        loss_all += losses.avg

    return loss_all