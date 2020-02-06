# TODO: dataparallel疑似未正常工作，鉴于多卡训练时不同卡的GPU占用差别巨大
# TODO: 多线程dataloader Broken Pipe错误
# TODO: 动态学习率调整，参考自己的代码
# TODO: 搞懂各种概念
import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import multiprocessing

from model_search import Network
from architect import Architect

'''
data:数据集的目录   batchsize:不能调太大
learning_rate   learning_rate_min   momentum   weight_decay: optimizer4件套
report_freq: 打印报告的频率 TODO 康康报告格式   gpu:TODO 解决多GPU显存不平衡
epoch: 默认50 TODO 为什么这么短？
init_channels, layers: supernet的大小？TODO
cutout, cutout_length: TODO 是否使用cutout及其参数？？？
drop_path_prob: TODO drop path的概率？？？
save: 保存路径名    seed:随机种子
grad_clip:TODO 梯度裁剪是干啥的？ train_portion:训练数据的比例，剩下的会当作“验证数据”（但不在验证集中
unrolled: one-step unrolled validation loss TODO
arch_learning_rate/arch_weight_decay: arch encoding的参数？？？TODO
'''
parser = argparse.ArgumentParser("cifar")
parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.025, help='init learning rate')
parser.add_argument('--learning_rate_min', type=float, default=0.001, help='min learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-4, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--gpu', type=str, default='0', help='gpu device id, split with ","')
parser.add_argument('--epochs', type=int, default=50, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=16, help='num of init channels')
parser.add_argument('--layers', type=int, default=8, help='total number of layers')
parser.add_argument('--model_path', type=str, default='saved_models', help='path to save the model')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='EXP', help='experiment name')
parser.add_argument('--seed', type=int, default=12450, help='random seed')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--train_portion', type=float, default=0.5, help='portion of training data')
parser.add_argument('--unrolled', action='store_true', default=False, help='use one-step unrolled validation loss')
parser.add_argument('--arch_learning_rate', type=float, default=3e-4, help='learning rate for arch encoding')
parser.add_argument('--arch_weight_decay', type=float, default=1e-3, help='weight decay for arch encoding')
args = parser.parse_args()

args.save = 'search-{}-{}'.format(args.save, time.strftime("%Y%m%d-%H%M%S")) # 生成search目录
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py')) # 把cnn内所有py脚本拷到search目录里
# glob.glob()查找符合特定规则的文件路径名
'''
log
'''
log_format = '%(asctime)s %(message)s' # %(asctime)s 当前时间，%(message)s 用户输出的消息 
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


CIFAR_CLASSES = 10



def main():
  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)

  np.random.seed(args.seed)
  # torch.cuda.set_device(args.gpu)
  gpus = [int(i) for i in args.gpu.split(',')] # argparser传入的参数转为int list
  if len(gpus) == 1:
    torch.cuda.set_device(int(args.gpu))
  # cudnn.benchmark = True
  torch.manual_seed(args.seed)
  # cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)
  logging.info('gpu device = %s' % args.gpu)
  logging.info("args = %s", args)

  # loss function 
  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda()
  # 初始化模型
  model = Network(args.init_channels, CIFAR_CLASSES, args.layers, criterion)
  model = model.cuda()
  if len(gpus)>1:
    print("True")
    model = nn.parallel.DataParallel(model, device_ids=gpus, output_device=gpus[0])
    model = model.module #FIXME:去掉这段

  arch_params = list(map(id, model.arch_parameters())) # FIXME:去掉时连锁报错
  weight_params = filter(lambda p: id(p) not in arch_params,
                         model.parameters()) # FIXME:去掉时连锁报错

  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  optimizer = torch.optim.SGD(
      # model.parameters(),
      weight_params,
      args.learning_rate,
      momentum=args.momentum,
      weight_decay=args.weight_decay)
  optimizer = nn.DataParallel(optimizer, device_ids=gpus)

  train_transform, valid_transform = utils._data_transforms_cifar10(args)
  train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
  # dset:torchvision.dataset的缩写
  num_train = len(train_data)
  indices = list(range(num_train))
  split = int(np.floor(args.train_portion * num_train)) # 
  print("使用多线程做dataloader会报错！") # FIXME: 确定没人占用的时候多线程是否报错!
  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
      pin_memory=True, num_workers=0)

  valid_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
      pin_memory=True, num_workers=0)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer.module, float(args.epochs), eta_min=args.learning_rate_min)

  architect = Architect(model, criterion, args) #有一个专门的architect.py 不知道是干嘛的，train要输入

  for epoch in range(args.epochs):
    scheduler.step() # FIXME:这个地方要改，不然学习率无法动态调整
    lr = scheduler.get_lr()[0]
    logging.info('epoch %d lr %e', epoch, lr)

    genotype = model.genotype() # model_search.py里待搜索的Network类型自带的参数
    logging.info('genotype = %s', genotype)# 打印当前epoch 的cell的网络结构

    print(F.softmax(model.alphas_normal, dim=-1))
    print(F.softmax(model.alphas_reduce, dim=-1))

    # training
    train_acc, train_obj = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr)
    logging.info('train_acc %f', train_acc)

    # validation
    with torch.no_grad():
      valid_acc, valid_obj = infer(valid_queue, model, criterion)
    logging.info('valid_acc %f', valid_acc)

    utils.save(model, os.path.join(args.save, 'weights.pt'))


def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr):
  objs = utils.AvgrageMeter() # 这三行的对象里，avg，sum，cnt三个数都置零
  top1 = utils.AvgrageMeter() # objs:loss的值， top1:前1正确率，top5：前5正确率
  top5 = utils.AvgrageMeter()

  for step, (input, target) in enumerate(train_queue):
    model.train()
    n = input.size(0)

    input = input.cuda()
    target = target.cuda()

    # get a random minibatch from the search queue with replacement
    input_search, target_search = next(iter(valid_queue)) # 用于架构参数更新的一个batch
    input_search = input_search.cuda()
    target_search = target_search.cuda()

    architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)

    optimizer.zero_grad()
    logits = model(input)
    loss = criterion(logits, target)

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.module.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    if step % args.report_freq == 0:
      logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  for step, (input, target) in enumerate(valid_queue):
    input = input.cuda()
    target = target.cuda()

    logits = model(input)
    loss = criterion(logits, target)

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)

    if step % args.report_freq == 0:
      logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)

  return top1.avg, objs.avg


if __name__ == '__main__':
  main()

