'''Code borrowed from: https://github.com/milesial/Pytorch-UNet and modified'''

""" 

Parts of the U-Net model 


Difference between bilinear True and False in notebook

Modificaitons made:
1) removed bilinear, we use deconv here only like original paper
2) double conv for encoder has leaky relu


"""

import torch as T
import torch.nn as nn
import torch.nn.functional as F


def init_weights(net, init_type='normal', init_gain=0.02):
        """Initialize network weights.
        Parameters:
            net (network)   -- network to be initialized
            init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
            init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
        We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
        work better for some applications. Feel free to try yourself.
        """
        def init_func(m):  # define the initialization function
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1  ): # also init conv2dtran according tho: https://discuss.pytorch.org/t/initalize-the-weights-of-nn-convtranspose2d/946
                if init_type == 'normal':
                    T.nn.init.normal_(m.weight.data, 0.0, init_gain)
                elif init_type == 'xavier':
                    T.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
                elif init_type == 'kaiming':
                    T.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    T.nn.init.orthogonal_(m.weight.data, gain=init_gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    T.nn.init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
                T.nn.init.normal_(m.weight.data, 1.0, init_gain)
                T.nn.init.constant_(m.bias.data, 0.0)

        #print('initialize network with %s' % init_type)
        net.apply(init_func)  # apply the initialization function <init_func>


class DoubleConv_Encoder(nn.Module):
    """
    Used for Encoder, now just a single conv

    We used leakyrelu instead of relu, just like pix2pix

    (convolution => [BN] => ReLU) * 2
    """

    def __init__(self, in_channels, out_channels,in_layer = False , bottle_neck = False):
        '''
        @params:
            in_channels: number of input channels
            out_channels: number of output channels
            in_layer: true if we are at first layer
            bottle_neck: see Down for doc
        '''
        super().__init__()

        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=4,stride = 2, padding=1,  bias=False))
        if not in_layer and not bottle_neck :
            layers.append( nn.BatchNorm2d(out_channels))
        
        if not bottle_neck:
            layers.append(nn.LeakyReLU(negative_slope = 0.2, inplace=True))
        else: 
            layers.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*layers)

        # added init layer weights
        init_weights(self.conv, init_type='kaiming')

    def forward(self, x):
        return self.conv(x)




class Conv_Decoder(nn.Module):
    """
    Used for Up

    Originally DoubleConv

    (convolution => [BN] => ReLU) * 2
    
    """

    def __init__(self, in_channels, out_channels, out_layer = False, use_dropout = False):
        '''
        @params:
            in_channels: number of input channels
            out_channels: number of output channels
        '''
        super().__init__()
        
        layers = []
        layers.append(nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding = 1 ,bias = False))
        if not out_layer :
            layers.append( nn.BatchNorm2d(out_channels))
            if use_dropout:            
                layers.append( nn.Dropout(0.5, inplace= True))
            layers.append(nn.ReLU(inplace=True))
        else:
            layers.append(nn.Tanh())
        

        self.deconv = nn.Sequential(*layers)

        # added init layer weights
        init_weights(self.deconv,init_type='kaiming')


    def forward(self, x):
        return self.deconv(x)



class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, in_layer = False, bottle_neck = False):
        '''
        @params:
            in_channels: number of input channels
            out_channels: number of output channels
            in_layer: true if we are at first layer
            bottle_neck: if bottle_neck no batch norm and use Relu instead, can change this though
        '''
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            #nn.MaxPool2d(2),
            DoubleConv_Encoder(in_channels, out_channels, in_layer=in_layer, bottle_neck = bottle_neck)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, out_layer = False, use_dropout = False):
        '''
        @params:
            in_channels: number of input channels
            out_channels: number of output channels
            out_layer: true if we are at last layer
        '''
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels

        #self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = Conv_Decoder(in_channels, out_channels, out_layer = out_layer, use_dropout = use_dropout)

    def forward(self, x1, x2 = None):
        '''
        Two modes 

        1) we concatentate inputs @ normal layers
 
        2) no concat (for now) @ bottleneck
        '''
        #x1 = self.up(x1)


        # input is CHW
        
        #diffY = x2.size()[2] - x1.size()[2]
        #diffX = x2.size()[3] - x1.size()[3]

        #x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                diffY // 2, diffY - diffY // 2])

        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        #x = T.cat([x2, x1], dim=1)
        

        if x2 != None:
            #print("         ", x1.shape, x2.shape)
            return self.conv(T.cat([x2,x1], 1))
        else:
            return self.conv(x1)
        #return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)



class UNet(nn.Module):
    '''
    Example: Takes (1,3,256,256) as input and output (1,3,256,256)

    '''
    def __init__(self,  n_classes, h_conv_size=[],  bilinear=False, use_dropout = False):
        '''
        @params:
          n_classes: int, number of output channels
        '''
        super(UNet, self).__init__()

        
        self.h_conv_size = h_conv_size # not used right now 
        self.use_dropout = use_dropout


        self.n_channels = 3
        self.bilinear = bilinear
        # how many class are there in total 
        self.n_classes = n_classes

        

        layers = []
        self._build_layers_normal()
            


    
    def _build_layers_normal(self):
        self.inc = Down(self.n_channels, 64, in_layer = True)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.down5 = Down(512, 512)
        self.down6 = Down(512, 512)
        self.down7 = Down(512, 512, bottle_neck= True)

        self.up1 = Up(512, 512 )
        self.up2 = Up(1024, 512 )
        self.up3 = Up(1024, 512 )
        self.up4 = Up(1024, 512 )
        self.up5 = Up(1024, 256 )
        self.up6 = Up(512 , 128 )
        self.up7 = Up(256 , 64  )

        self.outc = Up(128, self.n_classes, out_layer=True)
        return 



    def forward_normal(self,x):


        x1 = self.inc(x)
        x2 = self.down1(x1)


        x3 = self.down2(x2)

        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x6 = self.down5(x5)

        x7 = self.down6(x6)
        x8 = self.down7(x7)


        x = self.up1(x8)
        x = self.up2(x, x7)
        x = self.up3(x, x6)
        x = self.up4(x, x5)
        x = self.up5(x, x4)
        x = self.up6(x, x3)
        x = self.up7(x, x2)


        logits = self.outc(x,x1)



        return logits


    

    def forward(self, x):
        '''      
        @params
            x: input image, for example has shape (batch, 3,25,25). 
                If deconv then not image but vector ie (batch, 10,)

                for mode f_x, pa is a dictionary and we hv to manually concat
        '''

        
        return self.forward_normal(x)
        
    