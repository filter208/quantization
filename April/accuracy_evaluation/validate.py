import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.distributed as dist
import time
import torchvision.transforms.functional as tf
import torch.nn.functional as F
import sys
from tqdm import tqdm

def validate(val_loader, model, criterion, device):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    # ===== 新增：张量爆炸追踪 Hook =====
    debug_hooks = []
    def get_activation_hook(name):
        def hook(model, input, output):
            # 获取当前 Block 输出的最大绝对值，排查是否撞到了 31.0
            max_val = output.abs().max().item()
            mean_val = output.mean().item()
            if max_val > 25.0:
                print(f"[警告] {name} 输出即将溢出: Max={max_val:.4f}, Mean={mean_val:.4f}")
        return hook

    # 挂载到所有的残差块上
    from models.mobilenet_v2_quantized_approx import QuantizedInvertedResidual
    for name, layer in model.named_modules():
        if type(layer).__name__ == 'QuantizedInvertedResidual':
            h = layer.register_forward_hook(get_activation_hook(name))
            debug_hooks.append(h)

    val_start_time = end = time.time()
    loop = tqdm(enumerate(val_loader), leave=True, total=len(val_loader))
    print(f"DEBUG: val_loader length is {len(val_loader)}")
    for i, (data, target) in loop:
        data = data.to(device)
        target = target.to(device)

        with torch.no_grad():
            output = model(data)
            #植入探针
            if i == 0:
                print("\n" + "="*50)
                print("【深度透视：前 15 张图片的标签 vs 预测】")
                print("数据集给的答案 (Target) :", target[:15].cpu().numpy().tolist())
                _, preds = output.topk(1, 1, True, True)
                print("模型给出的预测 (Predict) :", preds[:15].squeeze().cpu().numpy().tolist())
                print("="*50 + "\n")
                #结束
            # data_cpu = data.cpu()
            # imshow(data_cpu)
            
        loss = criterion(output, target)
        print(f"\n--- Debug: 观测网络输出 ---")
        print(f"当前 Batch 的 Loss 值: {loss.item()}")
        print(f"第一个样本的前 10 个神经元输出(Logits): \n{output[0][:10]}")
        exit(0) # 测完第一个 Batch 直接终止程序，方便看日志

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        # print(f'acc1: {prec1}  acc5: {prec5}  loss: {loss}')
        # break
        losses.update(loss.data.item(), data.size(0))
        top1.update(prec1.data.item(), data.size(0))
        top5.update(prec5.data.item(), data.size(0))
        if i >=50:
            loop.close() # --- 强制关闭进度条 ---
            print(f"\n[Quick Check] Batch {i} reached. Prec@1 Avg: {top1.avg:.3f}")
            sys.stdout.flush() # 强制刷新缓冲区
            break
        # print(f'top1: {top1.val:.3f} ({top1.avg:.3f})  top5: {top5.val:.3f} ({top5.avg:.3f}')

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        loop.set_description('Test ')
        loop.set_postfix_str('Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
                            'Loss {loss.val:.4f} ({loss.avg:.4f})   '
                            'Prec@1 {top1.val:.3f} ({top1.avg:.3f})   '
                            'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                                batch_time=batch_time,
                                loss=losses,
                                top1=top1,
                                top5=top5,
                            ))
        # if i % args.print_freq == 0:
        #     print('Test: [{0}/{1}]\t'
        #           'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #           'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        #           'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
        #           'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
        #               i,
        #               len(val_loader),
        #               batch_time=batch_time,
        #               loss=losses,
        #               top1=top1,
        #               top5=top5,
        #           ))
    val_end_time = time.time()
    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Time {time:.3f}'.
          format(top1=top1, top5=top5, time=val_end_time - val_start_time))

    return all_reduce_mean(losses.avg), all_reduce_mean(top1.avg), all_reduce_mean(top5.avg)


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


def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    # print(pred, target)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size > 1:
        x_reduce = torch.tensor(x).cuda()
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x