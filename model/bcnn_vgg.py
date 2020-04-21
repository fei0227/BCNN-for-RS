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
    def __init__(self, class_num=200, pretrained=None):
        """Declare all needed layers."""
        torch.nn.Module.__init__(self)
        self.class_num = class_num
        # Convolution and pooling layers of VGG-16.
        vgg16 = torchvision.models.vgg16()
        if pretrained is not None:
            print("=> loading pretrained model '{}'".format(pretrained))
            checkpoint = torch.load(pretrained)
            vgg16.load_state_dict(checkpoint)
        self.features = torch.nn.Sequential(*list(vgg16.features.children())
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

    # define loss
    criterion = torch.nn.CrossEntropyLoss()
    # define input
    input = torch.autograd.Variable(torch.ones(2, 3, 224, 224))
    target = torch.autograd.Variable(torch.ones(2)).long()

    # compute output
    output = model.forward(input)
    print(output)
    print(output.argmax(dim=1))

    # backward
    loss = criterion(output, target)
    loss.backward()


if __name__ == '__main__':
    demo()
