"""SegmentationNN"""
import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchvision.models as models
import numpy as np

class SegmentationNN(pl.LightningModule):

    def __init__(self, num_classes=23, hparams=None):
        super().__init__()
        self.hparams = hparams
        self.classes = num_classes
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################
        
        pass
        self.vgg11_feat = models.vgg11(pretrained=True).features[:15]
        print(self.vgg11_feat)
        self.classifier = nn.Sequential(
            nn.Linear(30, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),

        )
        self.upsample = nn.Upsample(size=(240, 240))

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

    def forward(self, x):
        """
        Forward pass of the convolutional neural network. Should not be called
        manually but by calling a model instance directly.

        Inputs:
        - x: PyTorch input Variable
        """
        #######################################################################
        #                             YOUR CODE                               #
        #######################################################################

        pass
        x = self.vgg11_feat(x)
        x = self.classifier(x)
        x = x.view([x.shape[0], self.classes, 128, -1])
        x = self.upsample(x)

        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################

        return x

    def training_step(self, batch, batch_idx):
        images, targets = batch

        # forward pass
        out = self.forward(images)
        #print('train', out.shape, targets.shape)

        # loss
        loss_func = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')
        loss = loss_func(out, targets)

        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        #print(images.shape)

        # Perform a forward pass on the network with inputs
        out = self.forward(images)
        #print(out.shape, targets.shape)

        # calculate the loss with the network predictions and ground truth targets
        loss_func = torch.nn.CrossEntropyLoss(ignore_index=-1, reduction='mean')
        loss = loss_func(out, targets)

        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        # Average the loss over the entire validation data from it's mini-batches
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()

        # Log the validation accuracy and loss values to the tensorboard
        print("validation loss", avg_loss)
        self.log('val_loss', avg_loss)

    def configure_optimizers(self):
        # can return multiple optimizers and learning_rate schedulers
        return torch.optim.Adam(
            [{'params': self.classifier.parameters()},
             {'params': self.upsample.parameters()},
             {'params': self.vgg11_feat.parameters()}],
            lr=self.hparams['lr']
        )

    @property
    def is_cuda(self):
        """
        Check if model parameters are allocated on the GPU.
        """
        return next(self.parameters()).is_cuda

    def save(self, path):
        """
        Save model with its parameters to the given path. Conventionally the
        path should end with "*.model".

        Inputs:
        - path: path string
        """
        print('Saving model... %s' % path)
        torch.save(self, path)

        
class DummySegmentationModel(pl.LightningModule):

    def __init__(self, target_image):
        super().__init__()
        def _to_one_hot(y, num_classes):
            scatter_dim = len(y.size())
            y_tensor = y.view(*y.size(), -1)
            zeros = torch.zeros(*y.size(), num_classes, dtype=y.dtype)

            return zeros.scatter(scatter_dim, y_tensor, 1)

        target_image[target_image == -1] = 1

        self.prediction = _to_one_hot(target_image, 23).permute(2, 0, 1).unsqueeze(0)

    def forward(self, x):
        return self.prediction.float()
