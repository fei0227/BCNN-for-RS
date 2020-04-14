import torch
import torch.nn as nn
import torchvision


class BCNN(torch.nn.Module):
    """B-CNN for RS datasets.
    The B-CNN model is illustrated as follows.
    conv1^2 (64) -> pool1 -> conv2^2 (128) -> pool2 -> conv3^3 (256) -> pool3
    -> conv4^3 (512) -> pool4 -> conv5^3 (512) -> bilinear pooling
    -> sqrt-normalize -> L2-normalize -> fc (200).
    The network accepts a 3*224*224 input, and the pool5 activation has shape
    512*14*14 since we down-sample 5 times.
    Attributes:
        features, torch.nn.Module: Convolution and pooling layers.
        fc, torch.nn.Module: class_num.
    """
    def __init__(self, class_num=200):
        """Declare all needed layers."""
        torch.nn.Module.__init__(self)
        self.class_num = class_num
        # Convolution and pooling layers of VGG-16.
        self.features = torchvision.models.vgg16(pretrained=False).features
        self.features = torch.nn.Sequential(*list(self.features.children())
                                            [:-1])  # Remove pool5.
        # Linear classifier.
        self.fc = torch.nn.Linear(512**2, class_num)

    def forward(self, X):
        """Forward pass of the network.
        Args:
            x, torch.autograd.Variable of shape N*3*224*224.
        Returns:
            Score, torch.autograd.Variable of shape N*class_num.
        """
        # feature extraction
        batch = X.size()[0]
        assert X.size() == (batch, 3, 224, 224)
        X = self.features(X)
        assert X.size() == (batch, 512, 14, 14)

        # Billinear pooling
        X = X.view(batch, 512, 14**2)
        X = torch.bmm(X, torch.transpose(X, 1, 2)) / (14**2)  # Bilinear
        assert X.size() == (batch, 512, 512)
        X = X.view(batch, 512**2)
        X = torch.sqrt(X + 1e-5)
        X = torch.nn.functional.normalize(X)

        # classification
        X = self.fc(X)
        assert X.size() == (batch, self.class_num)

        return X

def demo():

    # load pretrained model
    model = BCNN(class_num=20)

    # for k, v in model.named_parameters():
    #     print(k)
    #     if not "branch" in k and not "param" in k:
    #         v.requires_grad = False
    paras = dict(model.named_parameters())
    paras_new = []
    for k, v in paras.items():
        if 'feature' in k:
            paras_new += [{'params': [v], 'lr': 0.001}]
        else:
            paras_new += [{'params': [v], 'lr': 0.01}]
    optimizer = torch.optim.SGD(paras_new, momentum=0.9, weight_decay=0.0005)

    lr = 1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    for p in optimizer.param_groups:
        outputs = ''
        for k, v in p.items():
            print(k)
            if k is 'params':
                outputs += (k + ': ' + str(v[0].shape).ljust(30) + ' ')
            else:
                outputs += (k + ': ' + str(v).ljust(10) + ' ')
        print(outputs)

    # define loss
    criterion = torch.nn.CrossEntropyLoss()
    # define input
    input = torch.autograd.Variable(torch.ones(2, 3, 224, 224))
    target = torch.autograd.Variable(torch.ones(2)).long()

    # compute output
    output = model.forward(input)
    print(output)

    # backward
    loss = criterion(output, target)
    loss.backward()


if __name__ == '__main__':
    demo()
