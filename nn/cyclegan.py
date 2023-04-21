from nn.unet import *
from nn.disc import *

class CycleGAN(nn.Module):
  def __init__(self):
    super(CycleGAN, self).__init__()

    self.G = UNet(3)
    self.F = UNet(3)
    self.D_x = NLayerDiscriminator(input_nc = 3,ndf = 64  )
    self.D_y = NLayerDiscriminator(input_nc = 3,ndf = 64  )

  def forward(self, x, y):
    '''
    want to translate from x[i] to y[i]
    @params:
      x: (batch_size, 2, 256, 256) tensor source domain image
      y: (batch_size, 2, 256, 256) tensor target domain image

    '''

    g_x = self.G(x)
    f_x = self.F(y)
    f_g_x = self.F(g_x) # F(G(x))
    g_f_x = self.G(f_x) # G(F(x))

    return g_x, f_g_x, f_x, g_f_x
    