
import torch.nn as nn


class TIR2BRI(nn.Module):
    """Model converts thermal ir image to grayscale image
    """
    def __init__(self, norm_layer=nn.BatchNorm2d):
        super(TIR2BRI, self).__init__()

        model1 = [nn.Conv2d(1, 64, kernel_size=3, stride=1,
                            padding=1, bias=True),]
        model1 += [nn.ReLU(True),]
        # 256 x d64
        model1 += [nn.Conv2d(64, 64, kernel_size=3,
                             stride=2, padding=1, bias=True),]
        model1 += [nn.ReLU(True),]
        # 128 x d64
        model1 += [norm_layer(64),]

        model2 = [nn.Conv2d(64, 128, kernel_size=3,
                            stride=1, padding=1, bias=True),]
        model2 += [nn.ReLU(True),]
        # 128 x d128
        model2 += [nn.Conv2d(128, 128, kernel_size=3,
                             stride=1, padding=1, bias=True),]
        model2 += [nn.ReLU(True),]
        model2 += [norm_layer(128),]

        model3 = [nn.Conv2d(128, 256, kernel_size=3,
                            stride=1, padding=1, bias=True),]
        model3 += [nn.ReLU(True),]
        model3 += [nn.Conv2d(256, 256, kernel_size=3,
                             stride=1, padding=1, bias=True),]
        model3 += [nn.ReLU(True),]
        model3 += [nn.Conv2d(256, 256, kernel_size=3,
                             stride=1, padding=1, bias=True),]
        model3 += [nn.ReLU(True),]
        #model3 += [nn.Dropout(p=0.2)]
        model3 += [norm_layer(256),]

        model4 = [nn.Conv2d(256, 512, kernel_size=3,
                            stride=1, padding=1, bias=True),]
        model4 += [nn.ReLU(True),]
        model4 += [nn.Conv2d(512, 512, kernel_size=3,
                             stride=1, padding=1, bias=True),]
        model4 += [nn.ReLU(True),]
        model4 += [nn.Conv2d(512, 512, kernel_size=3,
                             stride=1, padding=1, bias=True),]
        model4 += [nn.ReLU(True),]
        #model4 += [nn.Dropout(p=0.2)]
        model4 += [norm_layer(512),]

        model5 = [nn.Conv2d(512, 512, kernel_size=3,
                            dilation=2, stride=1, padding=2, bias=True),]
        model5 += [nn.ReLU(True),]
        model5 += [nn.Conv2d(512, 512, kernel_size=3,
                             dilation=2, stride=1, padding=2, bias=True),]
        model5 += [nn.ReLU(True),]
        model5 += [nn.Conv2d(512, 512, kernel_size=3,
                             dilation=2, stride=1, padding=2, bias=True),]
        model5 += [nn.ReLU(True),]
        #model5 += [nn.Dropout(p=0.2)]
        model5 += [norm_layer(512),]

        model6 = [nn.Conv2d(512, 512, kernel_size=3,
                            dilation=2, stride=1, padding=2, bias=True),]
        model6 += [nn.ReLU(True),]
        model6 += [nn.Conv2d(512, 512, kernel_size=3,
                             dilation=2, stride=1, padding=2, bias=True),]
        model6 += [nn.ReLU(True),]
        model6 += [nn.Conv2d(512, 512, kernel_size=3,
                             dilation=2, stride=1, padding=2, bias=True),]
        model6 += [nn.ReLU(True),]
        #model6 += [nn.Dropout(p=0.2)]
        model6 += [norm_layer(512),]

        model7 = [nn.Conv2d(512, 512, kernel_size=3,
                            stride=1, padding=1, bias=True),]
        model7 += [nn.ReLU(True),]
        model7 += [nn.Conv2d(512, 512, kernel_size=3,
                             stride=1, padding=1, bias=True),]
        model7 += [nn.ReLU(True),]
        model7 += [nn.Conv2d(512, 512, kernel_size=3,
                             stride=1, padding=1, bias=True),]
        model7 += [nn.ReLU(True),]
        model7 += [norm_layer(512),]
        # 128 x d512

        model8 = [nn.ConvTranspose2d(
            512, 256, kernel_size=4, stride=2, padding=1, bias=True),]
        model8 += [nn.ReLU(True),]
        # 256 x d256
        model8 += [nn.Conv2d(256, 256, kernel_size=3,
                             stride=1, padding=1, bias=True),]
        model8 += [nn.ReLU(True),]
        model8 += [norm_layer(256),]

        model8 += [nn.Conv2d(256, 128, kernel_size=3,
                             stride=1, padding=1, bias=True),]
        model8 += [nn.ReLU(True),]
        model8 += [nn.Conv2d(128, 1, kernel_size=1,
                             stride=1, padding=0, bias=True),]
        model8+=[nn.Tanh()]

        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model4 = nn.Sequential(*model4)
        self.model5 = nn.Sequential(*model5)
        self.model6 = nn.Sequential(*model6)
        self.model7 = nn.Sequential(*model7)
        self.model8 = nn.Sequential(*model8)

        # self.model_out = nn.Conv2d(256, 1, kernel_size=1, padding=0, dilation=1, stride=1, bias=False)

    def forward(self, input_ir):
        """Caculate grayscale image

        Args:
            input_ir (tensor): tensor of (N,1,W,H) of gracale image

        Returns:
            tensor: Output tensor of (N,256,W,H) of prob on brightness
        """
        conv1_2 = self.model1(input_ir)
        conv2_2 = self.model2(conv1_2)
        conv3_3 = self.model3(conv2_2)
        conv4_3 = self.model4(conv3_3)
        conv5_3 = self.model5(conv4_3)
        conv6_3 = self.model6(conv5_3)
        conv7_3 = self.model7(conv6_3)
        conv8_3 = self.model8(conv7_3)
        out_reg = conv8_3

        return out_reg


def tIR2bri():
    model = TIR2BRI()
    return model
