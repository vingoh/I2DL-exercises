"""Models for facial keypoint detection"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F


class KeypointModel(pl.LightningModule):
    """Facial keypoint detection model"""

    def __init__(self, hparams, train_set=None, val_set=None):
        """
        Initialize your model from a given dict containing all your hparams
        Warning: Don't change the method declaration (i.e. by adding more
            arguments), otherwise it might not work on the submission server
        """
        # super(KeypointModel, self).__init__()
        super().__init__()
        self.hparams = hparams
        self.train_set = train_set
        self.val_set = val_set
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

        self.KPModel = nn.Sequential(

            nn.Conv2d(1, 32, kernel_size=4),
            nn.MaxPool2d(2, stride=2, padding=0),
            nn.Dropout(0.1),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.MaxPool2d(2, stride=2, padding=0),
            nn.Dropout(0.2),
            nn.ELU(),
            nn.Conv2d(64, 128, kernel_size=2),
            nn.MaxPool2d(2, stride=2, padding=0),
            nn.Dropout(0.3),
            nn.ELU(),
            nn.MaxPool2d(2, stride=2, padding=0),
            nn.Flatten(),
            nn.Linear(3200, 1000),
            nn.ELU(),
            nn.Dropout(0.5),
            nn.Linear(1000, 30)

        )
        self.KPModel.apply(init_weights)

        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, x):
        ########################################################################
        # TODO: Define the forward pass behavior of your model                 #
        # for an input image x, forward(x) should return the                   #
        # corresponding predicted keypoints                                    #
        ########################################################################
        x = x.view(-1, 1, 96, 96)
        print("x1:", x.shape)
        x = self.KPModel(x)


        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        return x

    def training_step(self, batch, batch_idx):
        #print("batch", batch)
        images, targets = batch['image'], batch['keypoints']
        #print('images', images.shape)

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

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_set, shuffle=True, batch_size=self.hparams['batch_size'])

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_set, batch_size=self.hparams['batch_size'])

    def configure_optimizers(self):
        # can return multiple optimizers and learning_rate schedulers
        return torch.optim.Adam(self.KPModel.parameters())


def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)


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


class KeypointBlock(pl.LightningModule):
    def __init__(self, in_channels, out_channels, conv_ks, drop_prob):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=conv_ks)
        self.maxpool2d = nn.MaxPool2d(2, stride=2, padding=0)
        self.dropout = nn.Dropout(drop_prob)
        self.elu = nn.ELU()

    def forward(self, x):
        #print("x:", x.shape)
        x = self.conv(x)
        x = self.elu(x)
        x = self.maxpool2d(x)
        x = self.dropout(x)

        return x


class KeypointStack(pl.LightningModule):
    def __init__(self):
        self.stack = nn.ModuleList(
            KeypointBlock(in_channels=1, out_channels=32, conv_ks=4, drop_prob=0.1),
            KeypointBlock(in_channels=32, out_channels=64, conv_ks=3, drop_prob=0.2),
            KeypointBlock(in_channels=64, out_channels=128, conv_ks=2, drop_prob=0.3),
        )

    def forward(self, x):
        for md in self.stack:
            x = md(x)
        return x


