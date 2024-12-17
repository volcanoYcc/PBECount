import torch
import torch.nn as nn
import torchvision

class ConvBlock(nn.Module):
    """
    Helper module that consists of a Conv -> BN -> ReLU
    """

    def __init__(self, in_channels, out_channels, padding=1, kernel_size=3, stride=1, with_nonlinearity=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, padding=padding, kernel_size=kernel_size, stride=stride)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.with_nonlinearity = with_nonlinearity

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.with_nonlinearity:
            x = self.relu(x)
        return x


class Bridge(nn.Module):
    """
    This is the middle layer of the UNet which just consists of some
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.bridge = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            ConvBlock(out_channels, out_channels)
        )

    def forward(self, x):
        return self.bridge(x)


class UpBlockForUNetWithResNet50(nn.Module):
    """
    Up block that encapsulates one up-sampling step which consists of Upsample -> ConvBlock -> ConvBlock
    """

    def __init__(self, in_channels, out_channels, up_conv_in_channels=None, up_conv_out_channels=None,
                 upsampling_method="conv_transpose"):
        super().__init__()

        if up_conv_in_channels == None:
            up_conv_in_channels = in_channels
        if up_conv_out_channels == None:
            up_conv_out_channels = out_channels

        if upsampling_method == "conv_transpose":
            self.upsample = nn.ConvTranspose2d(up_conv_in_channels, up_conv_out_channels, kernel_size=2, stride=2)
        elif upsampling_method == "bilinear":
            self.upsample = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
            )
        self.conv_block_1 = ConvBlock(in_channels, out_channels)
        self.conv_block_2 = ConvBlock(out_channels, out_channels)

    def forward(self, up_x, down_x):
        """
        :param up_x: this is the output from the previous up block
        :param down_x: this is the output from the down block
        :return: upsampled feature map
        """
        x = self.upsample(up_x)
        x = torch.cat([x, down_x], 1)
        x = self.conv_block_1(x)
        x = self.conv_block_2(x)
        return x


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.01)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


class Resnet34_Unet(nn.Module):
    DEPTH = 6

    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet.resnet34(pretrained=True)
        down_blocks = []
        self.input_block = nn.Sequential(*list(resnet.children()))[:3]
        self.input_block[0] = nn.Conv2d(4, 64, kernel_size=(7, 7), padding=(3, 3), stride=(2, 2), bias=False)
        self.input_pool = list(resnet.children())[3]
        for bottleneck in list(resnet.children()):
            if isinstance(bottleneck, nn.Sequential):
                down_blocks.append(bottleneck)
        self.down_blocks = nn.ModuleList(down_blocks)
        self.bridge = Bridge(512, 512)
        up_blocks = []
        up_blocks.append(UpBlockForUNetWithResNet50(512, 256))
        up_blocks.append(UpBlockForUNetWithResNet50(256, 128))
        up_blocks.append(UpBlockForUNetWithResNet50(128, 64))
        up_blocks.append(UpBlockForUNetWithResNet50(in_channels=64 + 32, out_channels=64,
                                                    up_conv_in_channels=64, up_conv_out_channels=32))
        up_blocks.append(UpBlockForUNetWithResNet50(in_channels=64 + 32, out_channels=64,
                                                    up_conv_in_channels=64, up_conv_out_channels=64))
        self.up_blocks = nn.ModuleList(up_blocks)

        up_blocks_1 = []
        up_blocks_1.append(UpBlockForUNetWithResNet50(256, 128))
        up_blocks_1.append(UpBlockForUNetWithResNet50(128, 64))
        up_blocks_1.append(UpBlockForUNetWithResNet50(in_channels=64 + 32, out_channels=64,
                                                    up_conv_in_channels=64, up_conv_out_channels=32))
        up_blocks_1.append(UpBlockForUNetWithResNet50(in_channels=32 + 4, out_channels=32,
                                                    up_conv_in_channels=64, up_conv_out_channels=32))
        self.up_blocks_1 = nn.ModuleList(up_blocks_1)

        up_blocks_2 = []
        up_blocks_2.append(UpBlockForUNetWithResNet50(128, 64))
        up_blocks_2.append(UpBlockForUNetWithResNet50(in_channels=64 + 32, out_channels=64,
                                                    up_conv_in_channels=64, up_conv_out_channels=32))
        up_blocks_2.append(UpBlockForUNetWithResNet50(in_channels=32 + 32, out_channels=32,
                                                    up_conv_in_channels=64, up_conv_out_channels=32))
        self.up_blocks_2 = nn.ModuleList(up_blocks_2)

        up_blocks_3 = []
        up_blocks_3.append(UpBlockForUNetWithResNet50(in_channels=64 + 32, out_channels=64,
                                                    up_conv_in_channels=64, up_conv_out_channels=32))
        up_blocks_3.append(UpBlockForUNetWithResNet50(in_channels=32 + 32, out_channels=32,
                                                    up_conv_in_channels=64, up_conv_out_channels=32))
        self.up_blocks_3 = nn.ModuleList(up_blocks_3)

        up_blocks_4 = []
        up_blocks_4.append(UpBlockForUNetWithResNet50(in_channels=32 + 32, out_channels=32,
                                                    up_conv_in_channels=64, up_conv_out_channels=32))
        self.up_blocks_4 = nn.ModuleList(up_blocks_4)

        initialize_weights(self.input_block[0])
        initialize_weights(self.bridge)
        initialize_weights(self.up_blocks)
        initialize_weights(self.up_blocks_1)
        initialize_weights(self.up_blocks_2)
        initialize_weights(self.up_blocks_3)
        initialize_weights(self.up_blocks_4)

        self.out_probmap = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )
        initialize_weights(self.out_probmap)

        self.out_score1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=7, padding=3, stride=3, bias=False),
            nn.BatchNorm2d(32, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=5, padding=2, stride=2, bias=False),
            nn.BatchNorm2d(16, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, kernel_size=3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(8, momentum=0.1),
            nn.ReLU(inplace=True)
        )
        self.out_score2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(8*64*64, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 1)
        )
        initialize_weights(self.out_score1)
        initialize_weights(self.out_score2)


    def forward(self, x):
        pre_pools = dict()
        pre_pools[f"layer_0"] = x
        x = self.input_block(x)
        pre_pools[f"layer_1"] = x
        x = self.input_pool(x)

        for i, block in enumerate(self.down_blocks, 2):
            x = block(x)
            if i == (Resnet34_Unet.DEPTH - 1):
                continue
            pre_pools[f"layer_{i}"] = x

        x = self.bridge(x)

        for i, block in enumerate(self.up_blocks_1, 1):
            key = f"layer_{Resnet34_Unet.DEPTH - 2 - i}"
            key1 = f"layer_{Resnet34_Unet.DEPTH - 1 - i}"
            pre_pools[key] = block(pre_pools[key1], pre_pools[key])

        for i, block in enumerate(self.up_blocks_2, 1):
            key = f"layer_{Resnet34_Unet.DEPTH - 3 - i}"
            key1 = f"layer_{Resnet34_Unet.DEPTH - 2 - i}"
            pre_pools[key] = block(pre_pools[key1], pre_pools[key])
        
        for i, block in enumerate(self.up_blocks_3, 1):
            key = f"layer_{Resnet34_Unet.DEPTH - 4 - i}"
            key1 = f"layer_{Resnet34_Unet.DEPTH - 3 - i}"
            pre_pools[key] = block(pre_pools[key1], pre_pools[key])

        for i, block in enumerate(self.up_blocks_4, 1):
            key = f"layer_{Resnet34_Unet.DEPTH - 5 - i}"
            key1 = f"layer_{Resnet34_Unet.DEPTH - 4 - i}"
            pre_pools[key] = block(pre_pools[key1], pre_pools[key])
        
        for i, block in enumerate(self.up_blocks, 1):
            key = f"layer_{Resnet34_Unet.DEPTH - 1 - i}"
            x = block(x, pre_pools[key])

        output_dict = {}
        output_probmap = self.out_probmap(x)
        output_dict['probmap'] = output_probmap

        output_score = self.out_score1(x)
        output_score = torchvision.ops.roi_align(output_score,torch.tensor([[0,0,0,output_score.shape[3]-1,output_score.shape[2]-1]]).to(output_score.device).to(torch.float32),(64,64))
        output_score = self.out_score2(output_score)
        output_dict['score'] = output_score

        return output_dict

if __name__=='__main__':
    model = Resnet34_Unet().cuda()

    num = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.5fM" % (num / 1e6))

    inp = torch.rand((1, 4, 512, 512)).cuda()
    output_dist = model(inp)
    for key,value in output_dist.items():
        print(key,value.shape)
    
    
    