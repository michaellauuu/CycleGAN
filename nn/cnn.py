import torch as T
import torch.nn as nn



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

        print('initialize network with %s' % init_type)
        net.apply(init_func)  # apply the initialization function <init_func>



class CNN(nn.Module):
    def __init__(self, h_conv_size=[], h_mlp_size=[], use_sigmoid=True, use_layer_norm=True , deconv = False, norm_layer=nn.BatchNorm2d):
        """
        This class build a CNN [cnn_layers, mlp_layers]. This class could allow just cnn_layers or mlp_layers as defined 
        by the inputs 

        Each class also stores the size of input, parents and u/noise

        @params:
            h_conv_size: [] of {} that define the network. number of tuples = number of layers. ie [[in_channels, out_channels... same as doc]]
            h_mlp_size:
            use_layer_norm: if True adds batchnorm in _buildlayers
            deconv:  True if we want to build a deconvution network instead


        """
        super().__init__()
        self.h_conv_size = h_conv_size
        self.h_mlp_size = h_mlp_size

        self.use_layer_norm = use_layer_norm
        
        self.deconv = deconv

        '''
        layers = [nn.Conv2d(self.i_size, self.h_size, 3, padding =  'same')]
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(self.h_size , 2*self.h_size , 3, padding = 'same'))
        layers.append(nn.ReLU())
        layers.append(nn.Conv2d(2*self.h_size , 4*self.h_size , 3, padding = 'same'))
        #layers.append(nn.Flatten())
        self.nn = nn.Sequential(*layers)

        '''
        
        
        #layers.append(nn.Flatten())
        self.nn = nn.Sequential(* self._build_layers())

        init_weights(self.nn)

    def _build_layers(self):
        layers = []
        
        # if we want conv layers, h_conv_size not empty
        if len(self.h_conv_size) != 0:
            for i,lay in enumerate(self.h_conv_size):
                if self.deconv:
                    layers.append(nn.ConvTranspose2d( **self.h_conv_size[i]))
                else:
                    layers.append(nn.Conv2d( **self.h_conv_size[i]))

                if self.use_layer_norm:
                    layers.append(nn.BatchNorm2d(self.h_conv_size[i]['out_channels']))
                    if i == len(self.h_conv_size)-1:
                        layers.append(nn.Tanh())
                    else:
                        layers.append(nn.ReLU(inplace=True))

        if len(self.h_mlp_size) != 0:
            for i, lay in enumerate(self.h_mlp_size):
                layers.append(nn.Linear(**self.h_mlp_size[i] ))
                if self.use_layer_norm:
                    layers.append(nn.BatchNorm1d(self.h_conv_size[i]['out_channels']))

        return layers

    def forward(self,x):
        '''
        modified forward function that deals forward pass with noise as well
        
        

        @params
            pa: input image, for example has shape (batch, 3,25,25). 
                If deconv then not image but vector ie (batch, 10,)
            u: noise, we assume noise same shape as input
        '''

        return self.nn(x)



