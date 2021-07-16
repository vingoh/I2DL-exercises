"""Models for facial keypoint detection"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F


# ResNet


class KeypointModel(pl.LightningModule):
    """Facial keypoint detection model"""

    def __init__(self, hparams):
        """
        Initialize your model from a given dict containing all your hparams
        Warning: Don't change the method declaration (i.e. by adding more
            arguments), otherwise it might not work on the submission server
        """
        # super(KeypointModel, self).__init__()
        super().__init__()
        self.hparams = hparams
        ########################################################################
        # TODO: Define all the layers of your CNN, the only requirements are:  #
        # 1. The network takes in a batch of images of shape (Nx1x96x96)       #
        # 2. It ends with a linear layer that represents the keypoints.        #
        # Thus, the output layer needs to have shape (Nx30),                   #
        # with 2 values representing each of the 15 keypoint (x, y) pairs      #
        #                                                                      #
        # Some layers you might consider including:                            #
        # maxpooling layers, multiple conv layers, fully-connected layers,     #
        # and other layers (such as dropout or batch normalization) to avoid   #
        # overfitting.                                                         #
        ########################################################################
        n = 5

        self.resnet = nn.Sequential(  # pic size 32*32
            Lambda(lambda x: x.view(-1, 1, 96, 96)),
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16, track_running_stats=False),
            nn.ReLU(),
            ResidualStack(16, 16, stride=1, num_blocks=n),  # pic size 16*16
            ResidualStack(16, 32, stride=2, num_blocks=n),  # pic size 8*8
            ResidualStack(32, 64, stride=2, num_blocks=n),
            nn.AdaptiveAvgPool2d(1),
            Lambda(lambda x: x.squeeze()),
            nn.Linear(64, 30),  # nn.Softmax(dim=1),
        )
        self.resnet.apply(init_weights)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, x):
        ########################################################################
        # TODO: Define the forward pass behavior of your model                 #
        # for an input image x, forward(x) should return the                   #
        # corresponding predicted keypoints                                    #
        ########################################################################

        x = self.resnet(x)
        # print("x1:", x.shape)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return x

    def training_step(self, batch, batch_idx):
        # print("batch", batch)
        images, targets = batch['image'], batch['keypoints']
        # print('images', images.shape)

        # forward pass
        out = self.forward(images)
        out = out.view(out.shape[0], -1, 2)

        # loss
        loss = F.mse_loss(out, targets)

        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch['image'], batch['keypoints']

        # Perform a forward pass on the network with inputs
        out = self.forward(images)
        out = out.view(out.shape[0], -1, 2)

        # calculate the loss with the network predictions and ground truth targets
        loss = F.mse_loss(out, targets)
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        # Average the loss over the entire validation data from it's mini-batches
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        # Log the validation accuracy and loss values to the tensorboard
        print("validation loss", avg_loss)
        self.log('val_loss', avg_loss)

    def configure_optimizers(self):
        # can return multiple optimizers and learning_rate schedulers
        return torch.optim.Adam(self.resnet.parameters())


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)


class DataModule(pl.LightningDataModule):
    def __init__(self, batch_size, train, val):
        super().__init__()
        self.batch_size = batch_size
        self.train_set = train
        self.val_set = val

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_set, batch_size=self.batch_size)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        if stride > 1 or in_channels != out_channels:
            # Add strides in the skip connection and zeros for the new channels.
            self.skip = Lambda(lambda x: F.pad(x[:, :, ::stride, ::stride],
                                               (0, 0, 0, 0, 0, out_channels - in_channels),
                                               mode="constant", value=0))
        else:
            self.skip = nn.Sequential()

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, bias=False, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, bias=False, padding=1)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, input):

        residual = self.conv1(input)
        residual = self.batchnorm(residual)
        residual = self.relu(residual)
        residual = self.conv2(residual)
        residual = self.batchnorm(residual)
        output = self.relu(residual + self.skip(input))
        return output


class ResidualStack(nn.Module):

    def __init__(self, in_channels, out_channels, stride, num_blocks):
        super().__init__()

        self.stack = nn.ModuleList([ResidualBlock(in_channels, out_channels, stride=1)])
        if num_blocks > 1:
            self.stack.extend([ResidualBlock(out_channels, out_channels, stride=1)])

    def forward(self, input):
        pass
        for m in self.stack:
            input = m(input)

        return input


class Lambda(nn.Module):
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)

class DummyKeypointModel(pl.LightningModule):
    """Dummy model always predicting the keypoints of the first train sample"""

    def __init__(self):
        super().__init__()
        self.prediction = torch.tensor([[
            0.4685, -0.2319,
            -0.4253, -0.1953,
            0.2908, -0.2214,
            0.5992, -0.2214,
            -0.2685, -0.2109,
            -0.5873, -0.1900,
            0.1967, -0.3827,
            0.7656, -0.4295,
            -0.2035, -0.3758,
            -0.7389, -0.3573,
            0.0086, 0.2333,
            0.4163, 0.6620,
            -0.3521, 0.6985,
            0.0138, 0.6045,
            0.0190, 0.9076,
        ]])

    def forward(self, x):
        return self.prediction.repeat(x.size()[0], 1, 1, 1)