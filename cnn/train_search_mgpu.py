# FIXME: 解决 Architecture的并行！
import os
import sys
import time
import glob
import numpy as np
import logging
import argparse
import multiprocessing
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.utils
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.distributed as dist

import utils
from model_search import Network
from architect import Architect



def main():
  '''
  data:数据集的目录   batchsize:不能调太大
  learning_rate   learning_rate_min   momentum   weight_decay: optimizer4件套
  report_freq: 打印报告的频率 
  epoch: 默认50
  init_channels: 初始特征通道数，随着网络加深特征通道数会成倍增长
  layers: 进行cell的搜索时，网络框架由几个cell组成
  cutout, cutout_length: TODO 是否使用cutout及其参数？？？
  drop_path_prob: 减少搜索过程中的计算时间以及内存占用的一个参数
  save: 保存路径名    seed:随机种子
  grad_clip:梯度裁剪用以解决梯度爆炸 train_portion:训练数据的比例，剩下的会当作“验证数据”（但不在验证集中
  unrolled: one-step unrolled validation loss TODO
  arch_learning_rate/arch_weight_decay: 架构参数学习率，用以更新网络架构参数
  '''
  parser = argparse.ArgumentParser("cifar")
  parser.add_argument('--data', type=str, default='../data', 
                      help='location of the data corpus')
  parser.add_argument('--batch_size', type=int, default=64, 
                      help='batch size')
  parser.add_argument('--learning_rate', type=float, default=0.025, 
                      help='init learning rate')
  parser.add_argument('--learning_rate_min', type=float, default=0.001, 
                      help='min learning rate')
  parser.add_argument('--momentum', type=float, default=0.9,
                      help='momentum')
  parser.add_argument('--weight_decay', type=float, default=3e-4, 
                      help='weight decay')
  parser.add_argument('--report_freq', type=float, default=50, 
                      help='report frequency')
  parser.add_argument('--gpu', type=str, default='0,1', 
                      help='gpu device id, split with ","')
  parser.add_argument('--epochs', type=int, default=50,
                      help='num of training epochs')
  parser.add_argument('--init_channels', type=int, default=16, 
                      help='num of init channels')
  parser.add_argument('--layers', type=int, default=8, 
                      help='total number of layers')
  parser.add_argument('--model_path', type=str, default='saved_models', 
                      help='path to save the model')
  parser.add_argument('--cutout', action='store_true', default=False, 
                      help='use cutout')
  parser.add_argument('--cutout_length', type=int, default=16, 
                      help='cutout length')
  parser.add_argument('--drop_path_prob', type=float, default=0.3, 
                      help='drop path probability')
  parser.add_argument('--save', type=str, default='EXP', 
                      help='experiment name')
  parser.add_argument('--seed', type=int, default=12450, 
                      help='random seed')
  parser.add_argument('--grad_clip', type=float, default=5, 
                      help='gradient clipping')
  parser.add_argument('--train_portion', type=float, default=0.5, 
                      help='portion of training data')
  parser.add_argument('--unrolled', action='store_true', default=False, 
                      help='use one-step unrolled validation loss')
  parser.add_argument('--arch_learning_rate', type=float, default=3e-4, 
                      help='learning rate for arch encoding')
  parser.add_argument('--arch_weight_decay', type=float, default=1e-3, 
                      help='weight decay for arch encoding')
  # pasers about distributed data parallel
  parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N',
                      help='number of data loading workers(default:1)')
  parser.add_argument('-nr','--nr',default=0, type=int,
                      help='ranking within the nodes')
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


  args.CIFAR_CLASSES = 10
  # Setting GPU device
  os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

  if not torch.cuda.is_available():
    logging.info('no gpu device available')
    sys.exit(1)
  gpus = [int(i) for i in args.gpu.split(',')] # argparser传入的参数转为int list
  num_gpu = len(gpus)
  if num_gpu == 1:
    torch.cuda.set_device(int(args.gpu))

  logging.info('gpu device = %s' % args.gpu)
  logging.info("args = %s", args)

  args.world_size = num_gpu * args.nodes 
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = '23456'
  print('OMG')
  mp.spawn(train_search, nprocs=num_gpu, args=(args,))

def train_search(gpu,args):
  print('START TRAIN')
  # Setting random seed
  print("Setting random seed",args.seed)
  np.random.seed(args.seed)
  cudnn.benchmark = True
  torch.manual_seed(args.seed)
  cudnn.enabled=True
  torch.cuda.manual_seed(args.seed)

  torch.cuda.set_device(gpu)

  num_gpu = len([int(i) for i in args.gpu.split(',')])    
  rank = args.nr * num_gpu + gpu
  dist.init_process_group(backend= 'nccl', init_method='env://',  world_size=args.world_size, rank=rank)

  # loss function 
  criterion = nn.CrossEntropyLoss()
  criterion = criterion.cuda(gpu) 
  
  # 初始化模型，构建一个超网，并将其部署到GPU上
  model = Network(args.init_channels, args.CIFAR_CLASSES, args.layers, criterion)
  model = model.cuda(gpu)
  arch_params = list(map(id, model.arch_parameters())) 
  weight_params = filter(lambda p: id(p) not in arch_params, #暂时没看到怎么用
                         model.parameters()) 

  logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

  optimizer = torch.optim.SGD(
      model.parameters(), # 优化器更新的参数
      # weight_params,
      args.learning_rate, # 学习率
      momentum=args.momentum,
      weight_decay=args.weight_decay)

  scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, float(args.epochs), eta_min=args.learning_rate_min)


  train_transform, valid_transform = utils._data_transforms_cifar10(args)
  train_data = dset.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)
  # dset:torchvision.dataset的缩写
  '''
  #  FIXME: 在Distributed DataParallel中，看起来无法通过直接指定indices分割数据集
  num_train = len(train_data)
  indices = list(range(num_train))
  split = int(np.floor(args.train_portion * num_train)) # 
  print("使用多线程做dataloader会报错！")
  # 数据集划分为训练和验证集，并打包成有序的结构
  '''
  train_sampler = torch.utils.data.distributed.DistributedSampler(train_data,
      num_replicas= args.world_size,
      rank= rank)
  train_queue = torch.utils.data.DataLoader(
      dataset= train_data, 
      batch_size= args.batch_size,
      shuffle= False,
      sampler= train_sampler,
      pin_memory= True,
      num_workers= 0)
  valid_data = dset.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)
  # valid_sampler = torch.utils.data.distributed.DistributedSampler(
  #     valid_data,
  #     num_replicas= args.world_size,
  #     rank= rank
  # )
  valid_queue = torch.utils.data.DataLoader(
      dataset= valid_data, batch_size=args.batch_size,
      pin_memory=True, num_workers=0)
  '''
  # FIXME: 
  train_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
      pin_memory=True, num_workers=0)

  valid_queue = torch.utils.data.DataLoader(
      train_data, batch_size=args.batch_size,
      sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
      pin_memory=True, num_workers=0)
  '''
  # 在Architecture中创建架构参数和架构参数更新函数
  architect = Architect(model, criterion, args) #有一个专门的architect.py 不知道是干嘛的，train要输入
  
  model = nn.parallel.DistributedDataParallel(model,device_ids=[gpu]) 


  for epoch in range(args.epochs):
    lr = scheduler.get_lr()[0]
    logging.info('epoch %d lr %e', epoch, lr)
    genotype = model.module.genotype() # model_search.py里待搜索的Network类型自带的参数
    logging.info('genotype = %s', genotype)# 打印当前epoch 的cell的网络结构
    print(F.softmax(model.module.alphas_normal, dim=-1))
    print(F.softmax(model.module.alphas_reduce, dim=-1))

    # training
    train_acc, train_obj = train(train_queue, valid_queue, model, architect, criterion, optimizer, lr,
             args, gpu)
    logging.info('train_acc %f', train_acc)

    # validation
    with torch.no_grad():
      valid_acc, valid_obj = infer(valid_queue, model.module, criterion,
             args, gpu)
    logging.info('valid_acc %f', valid_acc)
    scheduler.step()

    if gpu == 0:
      utils.save(model.module, os.path.join(args.save, 'weights.pt'))


def train(train_queue, valid_queue, model, architect, criterion, optimizer, lr, args, gpu):
  objs = utils.AvgrageMeter() # 这三行的对象里，avg，sum，cnt三个数都置零
  top1 = utils.AvgrageMeter() # objs:loss的值， top1:前1正确率，top5：前5正确率
  top5 = utils.AvgrageMeter()
  print('正在进行训练！！！！',len(train_queue))
  for step, (input, target) in enumerate(train_queue):
    model.train()
    n = input.size(0)

    input = input.cuda(non_blocking=True)
    target = target.cuda(non_blocking=True)
    # get a random minibatch from the search queue with replacement
    input_search, target_search = next(iter(valid_queue)) # 用于架构参数更新的一个batch
    input_search = input_search.cuda()
    target_search = target_search.cuda()

    # PC darts中使用了在epoch>=15时才更新参数。
    architect.step(input, target, input_search, target_search, lr, optimizer, unrolled=args.unrolled)

    optimizer.zero_grad()
    logits = model(input)
    loss = criterion(logits, target)

    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)
    if gpu == 0:
      if step % args.report_freq == 0:
        logging.info('train %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
        print("Train Step: %03d Loss: %.3e Top1: %.3f Top5: %.3f"%(step, objs.avg, top1.avg, top5.avg))
  return top1.avg, objs.avg


def infer(valid_queue, model, criterion, args, gpu):
  '''
  在最后一个epoch后打印验证集计算结果
  '''
  print('正在进行测试！！！！')
  objs = utils.AvgrageMeter()
  top1 = utils.AvgrageMeter()
  top5 = utils.AvgrageMeter()
  model.eval()

  for step, (input, target) in enumerate(valid_queue):
    input = input.cuda(non_blocking=True)
    target = target.cuda(non_blocking=True)
    logits = model(input) # 计算预测结果
    loss = criterion(logits, target)

    prec1, prec5 = utils.accuracy(logits, target, topk=(1, 5))
    n = input.size(0)
    objs.update(loss.item(), n)
    top1.update(prec1.item(), n)
    top5.update(prec5.item(), n)
    if gpu == 0:
      if step % args.report_freq == 0:
        logging.info('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
        print('valid %03d %e %f %f', step, objs.avg, top1.avg, top5.avg)
  return top1.avg, objs.avg


if __name__ == '__main__':
  main()

