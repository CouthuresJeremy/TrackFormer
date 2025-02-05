import lightning as L
from torch import nn, optim
from src.my_model.utils.modules import TransformerEncoder, BaseModel
import torch


class TrackFormer(BaseModel):
    """
    A Transformer-based model for track fitting.

    The TrackFormer is designed to fit tracks from a set of hits as input features.
    It takes in a sequence of track-related  hits (B, seqL, 3 or 2) and outputs a sequence of
    track parameter predictions, such as the track position, momentum, and other relevant quantities.

    Args (int):
        input_dim : Dimensionality of hits .
        model_dim : Hidden dimensionality to use inside the Transformer.
        num_classes : Number of track parameters to predict per sequence element.
        num_heads : Number of attention heads to use in the Multi-Head Attention blocks.
        num_layers : Number of Transformer encoder blocks to use.
    """

    def __init__(
        self,
        input_dim,
        model_dim,
        num_classes,
        num_heads,
        num_layers,
        criterion,
        warmup,
        lr,
        use_scheduler=True,
        dropout=0.0,
        input_dropout=0.0,
    ):
        self.save_hyperparameters()
        super().__init__()
        self._create_model()

    def _create_model(self):
        #  Embedding
        self.embedding = nn.Sequential(
            nn.Dropout(self.hparams.input_dropout),
            nn.Linear(self.hparams.input_dim, self.hparams.model_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self.hparams.model_dim, self.hparams.model_dim),
        )

        # Transformer
        self.transformer = TransformerEncoder(
            num_layers=self.hparams.num_layers,
            input_dim=self.hparams.model_dim,
            dim_feedforward=2 * self.hparams.model_dim,
            num_heads=self.hparams.num_heads,
            dropout=self.hparams.dropout,
        )

        # regression head
        self.regression_head = nn.Sequential(
            nn.Linear(self.hparams.model_dim, 64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64, self.hparams.num_classes),
        )

    def forward(self, x, mask=None):
        """
        Inputs:
            x - Input features [Batch, SeqLen, input_dim]
            mask - Mask to apply on the attention outputs
        """
        x = self.embedding(x)
        x = self.transformer(x, mask=mask)
        x = x.mean(dim=1)
        x = self.regression_head(x)
        return x

    @torch.no_grad()
    def get_attention_maps(self, x):
        """
        Function for extracting the attention matrices
        """
        x = self.embedding(x)
        attention_maps = self.transformer.get_attention_maps(x)
        return attention_maps
