import lightning as L
from lightning.pytorch.cli import SaveConfigCallback
from torch import nn, optim
from src.my_model.utils.modules import TransformerEncoder, BaseModel
import torch
import torch.nn.functional as F
import math
import yaml


class PositionalEncoding(nn.Module):
    def __init__(self, model_dim, max_len=5000, mode="sinusoidal"):
        super().__init__()
        self.model_dim = model_dim
        self.mode = mode

        if mode == "sinusoidal":
            pe = torch.zeros(max_len, model_dim)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, model_dim, 2).float() * (-math.log(10000.0) / model_dim)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            self.register_buffer("pe", pe)
        elif mode == "learnable":
            self.pe = nn.Parameter(torch.randn(1, max_len, model_dim))
        else:
            self.pe = None

    def forward(self, x):
        if self.pe is not None:
            return x + self.pe[:, : x.size(1), :]
        return x


class TrackFormer(BaseModel):
    """
    A Transformer-based model for track fitting with optional angular outputs.

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
        min_lr=0.0,
        use_scheduler=True,
        dropout=0.0,
        input_dropout=0.0,
        metric=None,
        positional_encoding=None,
        norm_loss=None,
        aggregate_loss="mean",
        # indices of outputs that are angles in radians
        angle_indices=None,
        # normalize predicted (cos, sin) pairs to unit circle
        enforce_unit_circle: bool = True,
    ):
        # Save all hyperparameters
        self.save_hyperparameters()
        super().__init__()

        # Normalize / store angle index helpers
        if self.hparams.angle_indices is None:
            self.hparams.angle_indices = []
        self.angle_indices = sorted(set(self.hparams.angle_indices))
        self.num_angles = len(self.angle_indices)
        self.angle_index_set = set(self.angle_indices)

        # Indices that are not angles, preserving original order
        self.scalar_indices = [
            i for i in range(self.hparams.num_classes) if i not in self.angle_index_set
        ]

        # Augmented head size: +1 per angle index (because 2 outputs replace 1)
        self.augmented_out = self.hparams.num_classes + self.num_angles

        # Expose toggle for unit-circle normalization
        self.enforce_unit_circle = bool(self.hparams.enforce_unit_circle)

        self._create_model()

    def _create_model(self):
        # Embedding
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
            use_rope=self.hparams.positional_encoding == "rope",
        )

        # Positional encoding (RoPE is inside attention if selected)
        if (
            self.hparams.positional_encoding is None
            or self.hparams.positional_encoding == "rope"
        ):
            self.positional_encoding = None
        else:
            self.positional_encoding = PositionalEncoding(
                self.hparams.model_dim, mode=self.hparams.positional_encoding
            )

        # --- Regression head (augmented size) ---
        # Layout of y_aug:
        #   [ scalars in original order of scalar_indices,
        #     (cos,sin) pairs for each angle, in order of angle_indices ]
        self.regression_head = nn.Sequential(
            nn.Linear(self.hparams.model_dim, 64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64, self.augmented_out),
        )

    def _pool_sequence(self, x, mask):
        # Average pooling over the sequence length dimension (dim=1)
        # If padding is used, this will be impacted by the padding
        if mask is not None:
            # If a mask is provided, we need to average only over the unmasked elements
            reversed_mask = ~mask
            x = x.masked_fill(reversed_mask.unsqueeze(-1), 0.0)
            # Calculate the unmasked count for each sequence in the batch
            unmasked_count = mask.sum(dim=1, keepdim=True).clamp(min=1)
            # Make sure the unmasked count is an integer
            assert (
                unmasked_count.dtype == torch.int64
            ), "Unmasked count should be of type int64"
            x = x.sum(dim=1) / unmasked_count
        else:
            # If no mask is provided, simply average over the sequence length dimension
            x = x.mean(dim=1)
        return x

    def _normalize_angle_pairs(self, angle_block: torch.Tensor) -> torch.Tensor:
        """
        Normalize each (cos, sin) pair to unit length to stay on the unit circle.
        angle_block: [Batch, 2 * num_angles]
        """
        if self.num_angles == 0:
            return angle_block
        BatchSize = angle_block.size(0)
        pairs = angle_block.view(BatchSize, self.num_angles, 2)
        pairs = F.normalize(pairs, dim=-1, eps=1e-6)
        return pairs.view(BatchSize, 2 * self.num_angles)

    def decode_outputs(self, y_aug: torch.Tensor) -> torch.Tensor:
        """
        Convert augmented outputs to original layout:
          - copy scalar outputs back to their original indices
          - convert each (cos,sin) pair to angle in [-pi, pi] using atan2(sin, cos)
        Args:
            y_aug: [Batch, augmented_out]
        Returns:
            y: [Batch, num_classes] with angles in [-pi, pi]
        """
        BatchSize = y_aug.size(0)
        device = y_aug.device
        y = torch.empty(
            BatchSize, self.hparams.num_classes, device=device, dtype=y_aug.dtype
        )

        n_scalars = len(self.scalar_indices)
        # Split augmented vector
        scalars = y_aug[:, :n_scalars]
        angle_block = y_aug[:, n_scalars:]  # [Batch, 2 * num_angles]

        # Place scalars back to their original indices
        if n_scalars > 0:
            y[:, self.scalar_indices] = scalars

        # Decode angle pairs
        if self.num_angles > 0:
            pairs = angle_block.view(BatchSize, self.num_angles, 2)
            cos = pairs[..., 0]
            sin = pairs[..., 1]
            angles = torch.atan2(sin, cos)  # (-pi, pi]
            y[:, self.angle_indices] = angles

        return y

    def forward(self, x, mask=None, decode: bool = True):
        """
        Inputs:
            x - Input features [Batch, SeqLen, input_dim]
            mask - Bool mask [Batch, SeqLen] to apply on the attention outputs
            decode - if True, return [Batch, num_classes] with angles in [-pi, pi]
                     else return augmented outputs [Batch, augmented_out]
        """
        x = self.embedding(x)
        if self.positional_encoding is not None:
            x = self.positional_encoding(x)  # Apply positional encoding if not RoPE
        x = self.transformer(x, mask=mask)
        x = self._pool_sequence(x, mask)  # Pool over sequence length

        y_aug = self.regression_head(x)  # [Batch, augmented_out]

        # Keep (cos, sin) on the unit circle if requested
        if self.num_angles > 0 and self.enforce_unit_circle:
            n_scalars = len(self.scalar_indices)
            scalars = y_aug[:, :n_scalars]
            angle_block = y_aug[:, n_scalars:]
            angle_block = self._normalize_angle_pairs(angle_block)
            y_aug = torch.cat([scalars, angle_block], dim=-1)

        if decode:
            return self.decode_outputs(y_aug)
        return y_aug

    @torch.no_grad()
    def get_attention_maps(self, x):
        """
        Function for extracting the attention matrices
        """
        x = self.embedding(x)
        if self.positional_encoding is not None:
            x = self.positional_encoding(x)
        attention_maps = self.transformer.get_attention_maps(x)
        return attention_maps

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        # Find the SaveConfigCallback instance
        cfg_cb = next(
            (c for c in self.trainer.callbacks if isinstance(c, SaveConfigCallback)),
            None,
        )
        if cfg_cb is None:
            self.print("[warning] no SaveConfigCallback, config not saved")
            return

        # Use the SaveConfigCallback parser to dump the in-memory config to YAML
        yaml_str = cfg_cb.parser.dump(cfg_cb.config, skip_none=False)

        # Parse the YAML back into a dict and embed
        checkpoint["config"] = yaml.safe_load(yaml_str)
