from models.wresnet import *
from models.resnet import *
import os

def select_model(dataset,
                 model_name,
                 pretrained=False,
                 pretrained_models_path=None,
                 n_classes=10):

    assert model_name in ['WRN-16-1', 'WRN-16-2', 'WRN-40-1', 'WRN-40-2', 'ResNet34', 'WRN-10-2', 'WRN-10-1']
    if model_name=='WRN-16-1':
        model = WideResNet(depth=16, num_classes=n_classes, widen_factor=1, dropRate=0)
    elif model_name=='WRN-16-2':
        model = WideResNet(depth=16, num_classes=n_classes, widen_factor=2, dropRate=0)
    elif model_name=='WRN-40-1':
        model = WideResNet(depth=40, num_classes=n_classes, widen_factor=1, dropRate=0)
    elif model_name=='WRN-40-2':
        model = WideResNet(depth=40, num_classes=n_classes, widen_factor=2, dropRate=0)
    elif model_name == 'WRN-10-2':
        model = WideResNet(depth=10, num_classes=n_classes, widen_factor=2, dropRate=0)
    elif model_name == 'WRN-10-1':
        model = WideResNet(depth=10, num_classes=n_classes, widen_factor=1, dropRate=0)
    elif model_name=='ResNet34':
        model = resnet(depth=32, num_classes=n_classes)
    else:
        raise NotImplementedError

    if pretrained:
        model_path = os.path.join(pretrained_models_path)
        print('Loading Model from {}'.format(model_path))
        checkpoint = torch.load(model_path, map_location='cpu')
        print(checkpoint.keys())
        model.load_state_dict(checkpoint['state_dict'])

        #print("=> loaded checkpoint '{}' (epoch {}) (accuracy {})".format(model_path, checkpoint['epoch'], checkpoint['best_prec']))
        print("=> loaded checkpoint '{}' (epoch {}) ".format(model_path, checkpoint['epoch']))


    return model

if __name__ == '__main__':

    import torch
    from torchsummary import summary
    import random
    import time

    random.seed(1234)  # torch transforms use this seed
    torch.manual_seed(1234)
    torch.cuda.manual_seed(1234)

    support_x_task = torch.autograd.Variable(torch.FloatTensor(64, 3, 32, 32).uniform_(0, 1))

    t0 = time.time()
    model = select_model('CIFAR10', model_name='WRN-16-2')
    output, act = model(support_x_task)
    print("Time taken for forward pass: {} s".format(time.time() - t0))
    print("\nOUTPUT SHAPE: ", output.shape)
    summary(model, (3, 32, 32))