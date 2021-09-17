from models.backbones import ResNet, ResNetD, ShuffleNetV2, TResNet, Mobilenetv3, EfficientNet, RegNet, ResNest, ReXNet


def load_backbone(config, norm_layer=None):
    num_classes = 1000
    is_pretrained = config['basenet']['pretrained']
    is_last_dilation = config['basenet']['dilation']
    if config['basenet']['type'] == 'resnet':
        if config['basenet']['arch'] == 'resnet18':
            net = ResNet.resnet18(pretrained=is_pretrained, progress=False, num_classes=num_classes,
                                  replace_stride_with_dilation=[False, False, is_last_dilation],
                                  norm_layer=norm_layer)
        elif config['basenet']['arch'] == 'resnet50':
            net = ResNet.resnet50(pretrained=is_pretrained, progress=False, num_classes=num_classes,
                                  replace_stride_with_dilation=[False, False, is_last_dilation],
                                  norm_layer=norm_layer)
        elif config['basenet']['arch'] == 'resnext50':
            net = ResNet.resnext50_32x4d(pretrained=is_pretrained, progress=False, num_classes=num_classes,
                                         replace_stride_with_dilation=[False, False, is_last_dilation],
                                         norm_layer=norm_layer)
        elif config['basenet']['arch'] == 'resnet50d':
            net = ResNetD.resnet50d(pretrained=is_pretrained, progress=False, num_classes=num_classes,
                                    replace_stride_with_dilation=[False, False, is_last_dilation],
                                    norm_layer=norm_layer)
        else:
            raise ValueError('Unsupported architecture: ' + str(config['basenet']['arch']))
    elif config['basenet']['type'] == 'tresnet':
        if config['basenet']['arch'] == 'tresnetm':
            net = TResNet.TResnetM(num_classes=num_classes)
        elif config['basenet']['arch'] == 'tresnetl':
            net = TResNet.TResnetL(num_classes=num_classes)
        elif config['basenet']['arch'] == 'tresnetxl':
            net = TResNet.TResnetXL(num_classes=num_classes)
        else:
            raise ValueError('Unsupported architecture: ' + str(config['basenet']['arch']))
    elif config['basenet']['type'] == 'regnet':
        regnet_config = dict()
        if config['basenet']['arch'] == 'regnetx-200mf':
            regnet_config['depth'] = 13
            regnet_config['w0'] = 24
            regnet_config['wa'] = 36.44
            regnet_config['wm'] = 2.49
            regnet_config['group_w'] = 8
            regnet_config['se_on'] = False
            regnet_config['num_classes'] = num_classes

            net = RegNet.RegNet(regnet_config)
        elif config['basenet']['arch'] == 'regnetx-600mf':
            regnet_config['depth'] = 16
            regnet_config['w0'] = 48
            regnet_config['wa'] = 36.97
            regnet_config['wm'] = 2.24
            regnet_config['group_w'] = 24
            regnet_config['se_on'] = False
            regnet_config['num_classes'] = num_classes

            net = RegNet.RegNet(regnet_config)
        elif config['basenet']['arch'] == 'regnetx-4.0gf':
            regnet_config['depth'] = 23
            regnet_config['w0'] = 96
            regnet_config['wa'] = 38.65
            regnet_config['wm'] = 2.43
            regnet_config['group_w'] = 40
            regnet_config['se_on'] = False
            regnet_config['num_classes'] = num_classes

            net = RegNet.RegNet(regnet_config)
        elif config['basenet']['arch'] == 'regnetx-6.4gf':
            regnet_config['depth'] = 17
            regnet_config['w0'] = 184
            regnet_config['wa'] = 60.83
            regnet_config['wm'] = 2.07
            regnet_config['group_w'] = 56
            regnet_config['se_on'] = False
            regnet_config['num_classes'] = num_classes

            net = RegNet.RegNet(regnet_config)
        elif config['basenet']['arch'] == 'regnety-200mf':
            regnet_config['depth'] = 13
            regnet_config['w0'] = 24
            regnet_config['wa'] = 36.44
            regnet_config['wm'] = 2.49
            regnet_config['group_w'] = 8
            regnet_config['se_on'] = True
            regnet_config['num_classes'] = num_classes

            net = RegNet.RegNet(regnet_config)
        elif config['basenet']['arch'] == 'regnety-600mf':
            regnet_config['depth'] = 15
            regnet_config['w0'] = 48
            regnet_config['wa'] = 32.54
            regnet_config['wm'] = 2.32
            regnet_config['group_w'] = 16
            regnet_config['se_on'] = True
            regnet_config['num_classes'] = num_classes

            net = RegNet.RegNet(regnet_config)
        elif config['basenet']['arch'] == 'regnety-4.0gf':
            regnet_config['depth'] = 22
            regnet_config['w0'] = 96
            regnet_config['wa'] = 31.41
            regnet_config['wm'] = 2.24
            regnet_config['group_w'] = 64
            regnet_config['se_on'] = True
            regnet_config['num_classes'] = num_classes

            net = RegNet.RegNet(regnet_config)
        elif config['basenet']['arch'] == 'regnety-6.4gf':
            regnet_config['depth'] = 25
            regnet_config['w0'] = 112
            regnet_config['wa'] = 33.22
            regnet_config['wm'] = 2.27
            regnet_config['group_w'] = 72
            regnet_config['se_on'] = True
            regnet_config['num_classes'] = num_classes

            net = RegNet.RegNet(regnet_config)
        else:
            raise ValueError('Unsupported architecture: ' + str(config['basenet']['arch']))
    elif config['basenet']['type'] == 'resnest':
        if config['basenet']['arch'] == 'resnest50':
            net = ResNest.resnest50(pretrained=is_pretrained, num_classes=num_classes)
        elif config['basenet']['arch'] == 'resnest101':
            net = ResNest.resnest101(pretrained=is_pretrained, num_classes=num_classes)
        else:
            raise ValueError('Unsupported architecture: ' + str(config['basenet']['arch']))
    elif config['basenet']['type'] == 'efficient':
        if config['basenet']['arch'] == 'b0':
            net = EfficientNet.efficientnet_b0(pretrained=is_pretrained, num_classes=num_classes)
        elif config['basenet']['arch'] == 'b1':
            net = EfficientNet.efficientnet_b1(pretrained=is_pretrained, num_classes=num_classes)
        elif config['basenet']['arch'] == 'b2':
            net = EfficientNet.efficientnet_b2(pretrained=is_pretrained, num_classes=num_classes)
        elif config['basenet']['arch'] == 'b3':
            net = EfficientNet.efficientnet_b3(pretrained=is_pretrained, num_classes=num_classes)
        elif config['basenet']['arch'] == 'b4':
            net = EfficientNet.efficientnet_b4(pretrained=is_pretrained, num_classes=num_classes)
        elif config['basenet']['arch'] == 'b5':
            net = EfficientNet.efficientnet_b5(pretrained=is_pretrained, num_classes=num_classes)
        elif config['basenet']['arch'] == 'b6':
            net = EfficientNet.efficientnet_b6(pretrained=is_pretrained, num_classes=num_classes)
        else:
            raise ValueError('Unsupported architecture: ' + str(config['basenet']['arch']))
    elif config['basenet']['type'] == 'assembled':
        pass
    elif config['basenet']['type'] == 'shufflenet':
        if config['basenet']['arch'] == 'v2_x0_5':
            net = ShuffleNetV2.shufflenet_v2_x0_5(pretrained=is_pretrained, progress=False, num_classes=num_classes)
        elif config['basenet']['arch'] == 'v2_x1_0':
            net = ShuffleNetV2.shufflenet_v2_x1_0(pretrained=is_pretrained, progress=False, num_classes=num_classes)
        elif config['basenet']['arch'] == 'v2_x1_5':
            net = ShuffleNetV2.shufflenet_v2_x1_5(pretrained=is_pretrained, progress=False, num_classes=num_classes)
        elif config['basenet']['arch'] == 'v2_x2_0':
            net = ShuffleNetV2.shufflenet_v2_x2_0(pretrained=is_pretrained, progress=False, num_classes=num_classes)
        else:
            raise ValueError('Unsupported architecture: ' + str(config['basenet']['arch']))
    elif config['basenet']['type'] == 'mobilenet':
        if config['basenet']['arch'] == 'small_075':
            net = Mobilenetv3.mobilenetv3_small_075(pretrained=is_pretrained, num_classes=num_classes)
        elif config['basenet']['arch'] == 'small_100':
            net = Mobilenetv3.mobilenetv3_small_100(pretrained=is_pretrained, num_classes=num_classes)
        elif config['basenet']['arch'] == 'large_075':
            net = Mobilenetv3.mobilenetv3_large_075(pretrained=is_pretrained, num_classes=num_classes)
        elif config['basenet']['arch'] == 'large_100':
            net = Mobilenetv3.mobilenetv3_large_100(pretrained=is_pretrained, num_classes=num_classes)
        else:
            raise ValueError('Unsupported architecture: ' + str(config['basenet']['arch']))
    elif config['basenet']['type'] == 'rexnet':
        if config['basenet']['arch'] == 'rexnet1.0x':
            net = ReXNet.rexnet(num_classes=num_classes, width_multi=1.0)
        elif config['basenet']['arch'] == 'rexnet1.5x':
            net = ReXNet.rexnet(num_classes=num_classes, width_multi=1.5)
        elif config['basenet']['arch'] == 'rexnet2.0x':
            net = ReXNet.rexnet(num_classes=num_classes, width_multi=2.0)
        else:
            raise ValueError('Unsupported architecture: ' + str(config['basenet']['arch']))
    else:
        raise ValueError('Unsupported architecture: ' + str(config['basenet']['type']))

    return net
