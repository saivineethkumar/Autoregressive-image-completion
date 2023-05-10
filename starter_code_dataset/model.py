import torch.nn as nn

class MaskedCNN(nn.Conv2d):
    """
    Masked convolution as explained in the PixelCNN variant of
    Van den Oord et al, “Pixel Recurrent Neural Networks”, NeurIPS 2016
    It inherits from Conv2D (uses the same parameters, plus the option to select a mask including
    the center pixel or not, as described in class and in the Fig. 2 of the above paper)
    """

    def __init__(self, mask_type, *args, **kwargs):
        self.mask_type = mask_type
        super(MaskedCNN, self).__init__(*args, **kwargs)
        self.register_buffer('mask', self.weight.data.clone())
        _, _, height, width = self.weight.size()
        self.mask.fill_(1)
        if mask_type == 'A':
            # center too is zero here
            self.mask[:, :, height//2, width//2:] = 0
            self.mask[:, :, height//2+1:, :] = 0
        else:
            # center is not zero here
            self.mask[:, :, height//2, width//2+1:] = 0
            self.mask[:, :, height//2+1:, :] = 0

    def forward(self, x):
        self.weight.data *= self.mask
        return super(MaskedCNN, self).forward(x)


class PixelCNN(nn.Module):
    """
    A PixelCNN variant you have to implement according to the instructions
    """

    def __init__(self):
        super(PixelCNN, self).__init__()

        # WRITE CODE HERE TO IMPLEMENT THE MODEL STRUCTURE
        self.mask_conv1 = MaskedCNN(in_channels=1, out_channels=16, kernel_size=3, stride=1, 
                                    padding_mode='reflect', padding=3, dilation=3, mask_type='A')
        self.batch_norm1 = nn.BatchNorm2d(16)
        self.leaky_relu1 = nn.LeakyReLU(negative_slope=0.001)

        self.mask_conv2 = MaskedCNN(in_channels=16, out_channels=16, kernel_size=3, stride=1, 
                                    padding_mode='reflect', padding=3, dilation=3, mask_type='B')
        self.batch_norm2 = nn.BatchNorm2d(16)
        self.leaky_relu2 = nn.LeakyReLU(negative_slope=0.001)


        self.mask_conv3 = MaskedCNN(in_channels=16, out_channels=16, kernel_size=3, stride=1, 
                                    padding_mode='reflect', padding=3, dilation=3, mask_type='B')
        self.batch_norm3 = nn.BatchNorm2d(16)
        self.leaky_relu3 = nn.LeakyReLU(negative_slope=0.001)

        self.conv4 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        # WRITE CODE HERE TO IMPLEMENT THE FORWARD PASS
        x = self.mask_conv1(x)
        x = self.batch_norm1(x)
        x = self.leaky_relu1(x)
        
        x = self.mask_conv2(x)
        x = self.batch_norm2(x)
        x = self.leaky_relu2(x)
        
        x = self.mask_conv3(x)
        x = self.batch_norm3(x)
        x = self.leaky_relu3(x)

        x = self.conv4(x)
        x = self.sigmoid(x)
        
        return x


