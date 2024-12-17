import torch
import torch.nn as nn
from model34 import Resnet34_Unet

def _set_module(model, submodule_key, module):
    tokens = submodule_key.split('.')
    sub_tokens = tokens[:-1]
    cur_mod = model
    for s in sub_tokens:
        cur_mod = getattr(cur_mod, s)
    setattr(cur_mod, tokens[-1], module)

def change_module_norm(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            _set_module(model, name, nn.GroupNorm(num_channels=module.state_dict()['weight'].size()[0],num_groups=int(min(32,module.state_dict()['weight'].size()[0]/2))))
    for module in model.modules():
        if isinstance(module, nn.GroupNorm):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)

def change_module_acti(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.ReLU) or isinstance(module, nn.ReLU6):
            _set_module(model, name, nn.LeakyReLU())

def init_model(group=True,leaky=True):
    model = Resnet34_Unet()
    if group:
        change_module_norm(model)
    if leaky:
        change_module_acti(model)

    return model

if __name__=='__main__':
    model = init_model().cuda()

    num = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.5fM" % (num / 1e6))

    inp = torch.rand((1, 4, 512, 512)).cuda()
    output_dist = model(inp)
    for key,value in output_dist.items():
        print(key,value.shape)

