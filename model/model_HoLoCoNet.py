from model.cbam import *
from model.direction7 import *
from model.direction5 import *
from model.direction3 import *
import time
import numpy as np
from ptflops import get_model_complexity_info
from thop import profile


class Head(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Head, self).__init__()
        inter_channels = in_channels // 4
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(True),
            nn.Dropout(0.1),
            nn.Conv2d(inter_channels, out_channels, 1, 1, 0)
        )

    def forward(self, x):
        return self.block(x)


class M2AM(nn.Module):
    def __init__(self):
        super(M2AM, self).__init__()

        self.d711 = Conv_d711()
        self.d712 = Conv_d712()
        self.d713 = Conv_d713()
        self.d714 = Conv_d714()
        self.d715 = Conv_d715()
        self.d716 = Conv_d716()
        self.d717 = Conv_d717()
        self.d718 = Conv_d718()

        self.d511 = Conv_d511()
        self.d512 = Conv_d512()
        self.d513 = Conv_d513()
        self.d514 = Conv_d514()
        self.d515 = Conv_d515()
        self.d516 = Conv_d516()
        self.d517 = Conv_d517()
        self.d518 = Conv_d518()

        self.d311 = Conv_d311()
        self.d312 = Conv_d312()
        self.d313 = Conv_d313()
        self.d314 = Conv_d314()
        self.d315 = Conv_d315()
        self.d316 = Conv_d316()
        self.d317 = Conv_d317()
        self.d318 = Conv_d318()

    def forward(self, x):
        d711 = self.d711(x)
        d712 = self.d712(x)
        d713 = self.d713(x)
        d714 = self.d714(x)
        d715 = self.d715(x)
        d716 = self.d716(x)
        d717 = self.d717(x)
        d718 = self.d718(x)
        LoC7 = d711.mul(d715) + d712.mul(d716) + d713.mul(d717) + d714.mul(d718)

        d511 = self.d511(x)
        d512 = self.d512(x)
        d513 = self.d513(x)
        d514 = self.d514(x)
        d515 = self.d515(x)
        d516 = self.d516(x)
        d517 = self.d517(x)
        d518 = self.d518(x)
        LoC5 = d511.mul(d515) + d512.mul(d516) + d513.mul(d517) + d514.mul(d518)

        d311 = self.d311(x)
        d312 = self.d312(x)
        d313 = self.d313(x)
        d314 = self.d314(x)
        d315 = self.d315(x)
        d316 = self.d316(x)
        d317 = self.d317(x)
        d318 = self.d318(x)
        LoC3 = d311.mul(d315) + d312.mul(d316) + d313.mul(d317) + d314.mul(d318)

        mLoC = torch.max(LoC3, LoC5)
        mLoC = torch.max(mLoC, LoC7)
        mLoC = torch.sigmoid(mLoC)
        return mLoC


class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False):

        super(Conv2d_cd, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        out_normal = self.conv(x)

        kernel_diff1 = self.conv.weight.sum(2).sum(2)
        kernel_diff2 = kernel_diff1[:, :, None, None]
        out_diff = F.conv2d(input=x, weight=kernel_diff2, bias=self.conv.bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)

        return out_normal - out_diff


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, downsample):
        super(ResidualBlock, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),

            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        if downsample:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, 0, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = nn.Sequential()

    def forward(self, x):
        residual = x
        x = self.body(x)

        if self.downsample:
            residual = self.downsample(residual)

        out = F.relu(x+residual, True)
        return out


class D2CM(nn.Module):
    def __init__(self, in_channels):
        super(D2CM, self).__init__()

        out_channels = in_channels

        self.conv2 = nn.Sequential(
            Conv2d_cd(in_channels, out_channels, kernel_size=3, stride=1, padding=3, dilation=3),
        )
        self.conv3 = nn.Sequential(
            Conv2d_cd(in_channels, out_channels, kernel_size=3, stride=1, padding=5, dilation=5),
        )
        self.conv4 = nn.Sequential(
            Conv2d_cd(in_channels, out_channels, kernel_size=3, stride=1, padding=7, dilation=7),
        )

        self.dconv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=3, dilation=3),
        )
        self.dconv3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=5, dilation=5),
        )
        self.dconv4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=7, dilation=7),
        )

        self.conv_cat = nn.Sequential(
            nn.Conv2d(out_channels * 3, out_channels, 1, 1, 0),
            nn.BatchNorm2d(out_channels)
        )

        self.act = nn.ReLU(True)

    def forward(self, x):

        x2 = self.dconv2(self.conv2(x))
        x3 = self.dconv3(self.conv3(x))
        x4 = self.dconv4(self.conv4(x))

        x2p = x2
        x3p = x3 + x2
        x4p = x4 + x3 + x2

        x234p = torch.cat((x2p, x3p, x4p), dim=1)
        x_cat = self.conv_cat(x234p)

        output = self.act(x + x_cat)

        return output


class SEAM(nn.Module):
    def __init__(self, channels_high, channels_low):
        super(SEAM, self).__init__()

        self.dconv = nn.ConvTranspose2d(in_channels=channels_high, out_channels=channels_low, kernel_size=3, stride=2, padding=1)

        self.GAP = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels_low, channels_low // 4, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels_low // 4),
            nn.ReLU(True),
            nn.Conv2d(channels_low // 4, channels_low, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels_low)
        )

        self.AAP = nn.Sequential(
            nn.AdaptiveAvgPool2d(4),
            nn.Conv2d(channels_low, channels_low // 4, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels_low // 4),
            nn.ReLU(True),
            nn.Conv2d(channels_low // 4, channels_low, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels_low)
        )

        self.active = nn.Sigmoid()

    def forward(self, x_high, x_low):
        _, _, h, w = x_low.shape
        x_high = self.dconv(x_high, output_size=x_low.size())
        p_x_high = self.active((self.GAP(x_high) + self.AAP(x_high)))
        p_x_high = F.interpolate(p_x_high, scale_factor=h // 4, mode='nearest')
        output = x_low * p_x_high + x_high
        return output


class Backbone(nn.Module):

    def __init__(self, layer_blocks, channels):
        super(Backbone, self).__init__()

        self.layer1 = self._make_layer(block=ResidualBlock, block_num=layer_blocks[0],
                                       in_channels=channels[0], out_channels=channels[1], stride=1)
        self.layer2 = self._make_layer(block=ResidualBlock, block_num=layer_blocks[1],
                                       in_channels=channels[1], out_channels=channels[2], stride=2)
        self.layer3 = self._make_layer(block=ResidualBlock, block_num=layer_blocks[2],
                                       in_channels=channels[2], out_channels=channels[3], stride=2)

    def forward(self, inp_bchw):

        ls = []
        c1 = self.layer1(inp_bchw)
        ls.append(c1)
        c2 = self.layer2(c1)
        ls.append(c2)
        c3 = self.layer3(c2)
        ls.append(c3)
        return ls

    def _make_layer(self, block, block_num, in_channels, out_channels, stride):
        downsample = (in_channels != out_channels) or (stride != 1)
        layer = []
        layer.append(block(in_channels, out_channels, stride, downsample))
        for _ in range(block_num - 1):
            layer.append(block(out_channels, out_channels, 1, False))
        return nn.Sequential(*layer)


class Encoder(nn.Module):
    def __init__(self, layer_blocks=[4, 4, 4], channels=[16, 16, 32, 64]):
        super(Encoder, self).__init__()

        stem_width = channels[0]
        self.stem1 = nn.Sequential(
            nn.Conv2d(1, stem_width, 3, 1, 1, bias=False),
            nn.BatchNorm2d(stem_width),
            nn.ReLU(True)
        )

        self.stem2 = nn.Sequential(
            nn.Conv2d(stem_width, channels[0], 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(True)
        )

        self.firstloc = M2AM()

        self.layer1 = self._make_layer(block=ResidualBlock, block_num=layer_blocks[0],
                                       in_channels=channels[0], out_channels=channels[1], stride=1)
        self.layer2 = self._make_layer(block=ResidualBlock, block_num=layer_blocks[1],
                                       in_channels=channels[1], out_channels=channels[2], stride=2)
        self.layer3 = self._make_layer(block=ResidualBlock, block_num=layer_blocks[2],
                                       in_channels=channels[2], out_channels=channels[3], stride=2)

        self.refineloc1 = D2CM(in_channels=channels[1])
        self.refineloc2 = D2CM(in_channels=channels[2])
        self.refineloc3 = D2CM(in_channels=channels[3])

    def forward(self, x):
        mloc = self.firstloc(x)

        out1 = self.stem1(x)
        out2 = out1.mul(mloc)
        outstm = self.stem2(out1 + out2)

        c1 = self.layer1(outstm)
        c2 = self.layer2(c1)
        c3 = self.layer3(c2)

        rc3 = self.refineloc3(c3)
        rc2 = self.refineloc2(c2)
        rc1 = self.refineloc1(c1)

        return c3, c2, c1, rc3, rc2, rc1

    def _make_layer(self, block, block_num, in_channels, out_channels, stride):
        downsample = (in_channels != out_channels) or (stride != 1)
        layer = []
        layer.append(block(in_channels, out_channels, stride, downsample))
        for _ in range(block_num - 1):
            layer.append(block(out_channels, out_channels, 1, False))
        return nn.Sequential(*layer)


class Decoder(nn.Module):
    def __init__(self, channels=[16, 16, 32, 64]):
        super(Decoder, self).__init__()

        self.fuse23 = SEAM(channels_high=channels[3], channels_low=channels[2])
        self.fuse12 = SEAM(channels_high=channels[2], channels_low=channels[1])
        self.head = Head(channels[1], 1)

    def forward(self, rc3, rc2, rc1):

        rc23 = self.fuse23(rc3, rc2)
        rc123 = self.fuse12(rc23, rc1)
        pred = self.head(rc123)

        return pred


class HoLoCoNet(nn.Module):
    def __init__(self, layer_blocks=[4, 4, 4], channels=[16, 16, 32, 64]):
        super(HoLoCoNet, self).__init__()

        self.encoder = Encoder(layer_blocks=layer_blocks, channels=channels)

        self.decoder = Decoder(channels=channels)

    def forward(self, x):
        c3, c2, c1, rc3, rc2, rc1 = self.encoder(x)

        pred = self.decoder(rc3, rc2, rc1)

        return pred


class Y_HoLoCoNet(nn.Module):
    def __init__(self, layer_blocks=[4, 4, 4], channels=[16, 16, 32, 64]):
        super(Y_HoLoCoNet, self).__init__()

        self.encoder = Encoder(layer_blocks=layer_blocks, channels=channels)

        self.mask_decoder = Decoder(channels=channels)
        self.denoise_decoder = Decoder(channels=channels)

    def forward(self, x):
        c3, c2, c1, rc3, rc2, rc1 = self.encoder(x)

        mask_pred = self.mask_decoder(rc3, rc2, rc1)
        denoise_pred = self.denoise_decoder(rc3, rc2, rc1)

        return mask_pred, denoise_pred


if __name__ == '__main__':
    net = HoLoCoNet()
    net.eval()
    net = net.cuda()
    dummy_input = torch.rand(1, 1, 512, 512)
    dummy_input = dummy_input.cuda()
    with torch.no_grad():
        net(dummy_input)
        # print(criterion(a1, a3))

        for _ in range(100):
            net(dummy_input)
        print('=========Speed Testing=========')
        t_start = time.time()
        for _ in range(100):
            net(dummy_input)
        elapsed_time = time.time() - t_start

        speed_time = elapsed_time / 100 * 1000
        fps = 100 / elapsed_time

        macs, params = get_model_complexity_info(net, (1, 512, 512), as_strings=True, print_per_layer_stat=True, verbose=True)

        flops, params = profile(net, inputs=(dummy_input,))

        print(macs)

        print('Params: %2fM' % (params / 1e6))
        print('FLOPs: %2fGFLOPs' % (flops / 1e9))

        print('Elapsed Time: [%.2f s / %d iter]' % (elapsed_time, 100))
        print('Speed Time: %.2f ms / iter   FPS: %.2f' % (speed_time, fps))
        print("Number of parameters: ", sum([p.numel() for p in net.parameters() if p.requires_grad]))