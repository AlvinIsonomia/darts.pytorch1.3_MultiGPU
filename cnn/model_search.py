# TODO: PC-DARTS还引入了channel_shuffle
import torch
import torch.nn as nn
import torch.nn.functional as F
from operations import *

from genotypes import PRIMITIVES
from genotypes import Genotype


class MixedOp(nn.Module):
  '''
  获得混合操作
  '''
  def __init__(self, C, stride):
    super(MixedOp, self).__init__()
    self._ops = nn.ModuleList()
    # TODO: PC-DARTS在这里加了个最大池化
    # 形成每一层的操作
    for primitive in PRIMITIVES: # PRIMITIVES中存储了各种操作的名称
      op = OPS[primitive](C, stride, False) # OPS中存储了各种操作的定义
      if 'pool' in primitive:
        op = nn.Sequential(op, nn.BatchNorm2d(C, affine=False))
      self._ops.append(op)

  def forward(self, x, weights):
    '''
    PC-DARTS中这里实现的很麻烦（只取了部分的特征进行运算），并进行了channel_shuffle
    '''
    return sum(w * op(x) for w, op in zip(weights, self._ops))

class Cell(nn.Module):
  '''
  初始基本单元的构建
  '''
  def __init__(self, steps, multiplier, C_prev_prev, C_prev, C, reduction, reduction_prev):
    super(Cell, self).__init__()
    self.reduction = reduction

    if reduction_prev: # 前两层固定不参与搜索（PC？），之后根据之前接的Cell类型进行设定
      self.preprocess0 = FactorizedReduce(C_prev_prev, C, affine=False)
    else:
      self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0, affine=False)
    self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0, affine=False)
    self._steps = steps # 每个cell中有4个结点的连接状态待确定
    self._multiplier = multiplier

    self._ops = nn.ModuleList() # 构建operation的ModuleList
    self._bns = nn.ModuleList() 
    for i in range(self._steps):# 顺序构建四个节点的混合Operation
      for j in range(2+i):
        stride = 2 if reduction and j < 2 else 1
        op = MixedOp(C, stride) # MixedOp构建一个混合操作
        self._ops.append(op) # 将混合操作添加到ops

  def forward(self, s0, s1, weights):
    '''
    cell中的计算过程，前向传播时自动调用
    '''
    s0 = self.preprocess0(s0)
    s1 = self.preprocess1(s1)

    states = [s0, s1]
    offset = 0
    for i in range(self._steps): # 顺序计算四个结点的输出，并添加到states中，每一个结点的输入为之前节点的输出
      s = sum(self._ops[offset+j](h, weights[offset+j]) for j, h in enumerate(states))
      offset += len(states)
      states.append(s)

    return torch.cat(states[-self._multiplier:], dim=1) # 后四个节点进行concat作为输出 


class Network(nn.Module):

  def __init__(self, C, num_classes, layers, criterion, steps=4, multiplier=4, stem_multiplier=3):
    super(Network, self).__init__()
    self._C = C # 初始通道数
    self._num_classes = num_classes
    self._layers = layers # 网络由layers个cell组成
    self._criterion = criterion # 损失函数
    self._steps = steps # 一个cell内有 4 个节点要进行operation操作的搜索
    self._multiplier = multiplier

    C_curr = stem_multiplier*C # 当前Sequnential模块的输出通道数
    self.stem = nn.Sequential( # 一个Sequential表示一个序列模块
      nn.Conv2d(3, C_curr, 3, padding=1, bias=False), # 二维卷积层, 输入的尺度是(N, C_in,H,W)，输出尺度（N,C_out,H_out,W_out）
      # C_in,C_out,卷积核大小，groups：卷积核个数，默认为1个
      nn.BatchNorm2d(C_curr)   # 输入特征通道数
    )
 
    C_prev_prev, C_prev, C_curr = C_curr, C_curr, C # 通道数更新
    self.cells = nn.ModuleList() # 创建一个空ModuleList类型数据
    reduction_prev = False # 之前链接的是否是Reduce Cell
    for i in range(layers): # 遍历所有基本单元模块，构建相应的Cell
      if i in [layers//3, 2*layers//3]: # 在特定的layer使用Reduce Cell
        C_curr *= 2 # 每Reduce一次，通道数 * 2
        reduction = True # 由于使用了Reduce Cell
      else:
        reduction = False
        # list中增加一个normal 或者reduction cell，cell属于normal或者redunction由参数决定
      cell = Cell(steps, multiplier, C_prev_prev, C_prev, C_curr, reduction, reduction_prev) # Cell的具体功能在后面
      reduction_prev = reduction
      self.cells += [cell] # 在ModuleList中增加一个Cell
      C_prev_prev, C_prev = C_prev, multiplier*C_curr # 更新当前的通道数

    self.global_pooling = nn.AdaptiveAvgPool2d(1) # 创建一个平均池化层
    self.classifier = nn.Linear(C_prev, num_classes) # 构建一个线性分类器

    self._initialize_alphas() # 架构参数初始化

  def new(self):
    model_new = Network(self._C, self._num_classes, self._layers, self._criterion).cuda()
    for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
        x.data.copy_(y.data)
    return model_new

  def forward(self, input):
    '''
    前向计算
    '''
    s0 = s1 = self.stem(input)
    for i, cell in enumerate(self.cells):
      if cell.reduction: # Reduce和Normal模块分开进行
        weights = F.softmax(self.alphas_reduce, dim=-1) # 首先将架构参数中的操作系数用softmax进行归一化
        # TODO: PC darts在这里引入了边缘归一化架构参数
      else:
        weights = F.softmax(self.alphas_normal, dim=-1)
      s0, s1 = s1, cell(s0, s1, weights) # 实际计算和更新输出数据
    out = self.global_pooling(s1) # 全局池化
    logits = self.classifier(out.view(out.size(0),-1)) # 分类器输出结果
    return logits

  # def _loss(self, input, target):
  #   logits = self(input)
  #   return self._criterion(logits, target)

  def _initialize_alphas(self):
    k = sum(1 for i in range(self._steps) for n in range(2+i))
    num_ops = len(PRIMITIVES)

    self.alphas_normal = nn.Parameter(1e-3*torch.randn(k, num_ops))
    self.alphas_reduce = nn.Parameter(1e-3*torch.randn(k, num_ops))
    self._arch_parameters = [
      self.alphas_normal,
      self.alphas_reduce,
    ]

  def arch_parameters(self):
    return self._arch_parameters

  def genotype(self):

    def _parse(weights):
      gene = []
      n = 2
      start = 0
      for i in range(self._steps):
        end = start + n
        W = weights[start:end].copy()
        edges = sorted(range(i + 2), key=lambda x: -max(W[x][k] for k in range(len(W[x])) if k != PRIMITIVES.index('none')))[:2]
        for j in edges:
          k_best = None
          for k in range(len(W[j])):
            if k != PRIMITIVES.index('none'):
              if k_best is None or W[j][k] > W[j][k_best]:
                k_best = k
          gene.append((PRIMITIVES[k_best], j))
        start = end
        n += 1
      return gene

    gene_normal = _parse(F.softmax(self.alphas_normal, dim=-1).data.cpu().numpy())
    gene_reduce = _parse(F.softmax(self.alphas_reduce, dim=-1).data.cpu().numpy())

    concat = range(2+self._steps-self._multiplier, self._steps+2)
    genotype = Genotype(
      normal=gene_normal, normal_concat=concat,
      reduce=gene_reduce, reduce_concat=concat
    )
    return genotype

