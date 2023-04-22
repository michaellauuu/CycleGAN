import torch
import torch.nn as nn

from nn.cyclegan import *
import pytorch_lightning as pl
from nn.utils import set_requires_grad

class CycleGAN_PL(pl.LightningModule):
  '''
  Reference: https://github.com/nocotan/pytorch-lightning-gans/blob/master/models/gan.py
  '''
  def __init__(self, lr, lambd = 0.5, device = 'cpu'):
    '''
    @params:
      lr: float, lr
      lambd: lambda for cyclic constrain
    '''
    super().__init__()
    
    self.model = CycleGAN().to(device)
    self.lr = lr
    self.lambd = lambd
    # both losses hv reduction = mean, so we can compute est. exp value
    self.loss_gan = nn.BCEWithLogitsLoss()
    self.loss_cycle = nn.L1Loss() 

    #self.device = device


  def configure_optimizers(self):
    opt_G = torch.optim.Adam(self.model.G.parameters(), lr=self.lr)
    opt_F = torch.optim.Adam(self.model.F.parameters(), lr=self.lr)
    opt_D_x = torch.optim.Adam(self.model.D_x.parameters(), lr=self.lr)
    opt_D_y = torch.optim.Adam(self.model.D_y.parameters(), lr=self.lr)
    
    return [opt_G, opt_F, opt_D_x, opt_D_y],[]

  def training_step(self, batch, batch_idx,optimizer_idx):
    '''
    There are two ways to do this, lets first use optimizer_idx
    @params:
      batch: (batch_size, image shape), (batch_size, image shape); corrspound to out get_item from dataset but vertically stacked 
    '''
    x, y = batch # x source domain, y target domain
    batch_size = x.shape[0]

    # train generators
    if optimizer_idx == 0 or optimizer_idx == 1:
      # turn off require grad for discrminators and on for generators
      set_requires_grad(self.model.G, True)
      set_requires_grad(self.model.F, True)
      set_requires_grad(self.model.D_x, False)
      set_requires_grad(self.model.D_y, False)

      # loss_gan G
      g_x = self.model.G(x)
      d_g_x = self.model.D_x(g_x)

      g_loss = -1*self.loss_gan(d_g_x, torch.zeros(1).expand_as(d_g_x).to(self.device)) # technical detail: we treat each patch's prediction as one output
      # loss_gan F
      f_x = self.model.F(y)
      d_f_x = self.model.D_y(f_x)
      f_loss = -1*self.loss_gan(d_f_x, torch.zeros(1).expand_as(d_f_x).to(self.device))

      cyclic_loss = self.loss_cycle(self.model.F(g_x), x ) + self.loss_cycle(self.model.G(f_x), y )

      loss = g_loss + f_loss + cyclic_loss
      #print(f' train generator g_loss:{g_loss} f_loss:{f_loss} cyclic_loss:{cyclic_loss} ')


    # train discrmin
    elif optimizer_idx == 2 or optimizer_idx == 3:
      set_requires_grad(self.model.G, False)
      set_requires_grad(self.model.F, False)
      set_requires_grad(self.model.D_x, True)
      set_requires_grad(self.model.D_y, True)

      d_x_y = self.model.D_x(y)
      d_g_x_loss = self.loss_gan(d_x_y, torch.ones(1).expand_as(d_x_y).to(self.device))
      d_y_x = self.model.D_y(x)
      d_g_y_loss = self.loss_gan(d_y_x, torch.ones(1).expand_as(d_y_x).to(self.device))

      g_x = self.model.G(x)
      f_x = self.model.F(y)
      d_g_x = self.model.D_x(g_x)
      g_loss = self.loss_gan(d_g_x, torch.zeros(1).expand_as(d_g_x).to(self.device))
      d_f_x = self.model.D_y(f_x)
      f_loss = self.loss_gan(d_f_x, torch.zeros(1).expand_as(d_f_x).to(self.device))

      loss = d_g_x_loss + d_g_y_loss+g_loss + f_loss
      #print(f'g_loss:{g_loss} f_loss:{f_loss} d_g_x_loss:{d_g_x_loss} d_g_y_loss:{d_g_y_loss} ')

    return {'loss':loss}

