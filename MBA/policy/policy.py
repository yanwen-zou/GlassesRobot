import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms

from policy.tokenizer import Sparse3DEncoder
from policy.transformer import Transformer
from policy.diffusion import DiffusionUNetPolicy


class RISE(nn.Module):
    def __init__(
        self, 
        num_action = 20,
        input_dim = 6,
        obs_feature_dim = 512, 
        action_dim = 10, 
        hidden_dim = 512,
        nheads = 8, 
        num_encoder_layers = 4, 
        num_decoder_layers = 1, 
        dim_feedforward = 2048, 
        dropout = 0.1,
        enable_mba = False,
        obj_dim = 10,
    ):
        super().__init__()
        num_obs = 1
        self.sparse_encoder = Sparse3DEncoder(input_dim, obs_feature_dim)
        self.transformer = Transformer(hidden_dim, nheads, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)
        self.enable_mba = enable_mba
        self.action_decoder = DiffusionUNetPolicy(action_dim, 
                                                  num_action, 
                                                  num_obs, 
                                                  obs_feature_dim, 
                                                  enable_mba=enable_mba,
                                                  obj_dim=obj_dim,
                                                  rot_smooth_lambda=0.05,
                                                  cond_extra_dim=obj_dim)
        self.readout_embed = nn.Embedding(1, hidden_dim)

    def forward(self, cloud, actions = None, batch_size = 24, actions_obj = None, sample_mba = False, current_obj = None):
        src, pos, src_padding_mask = self.sparse_encoder(cloud, batch_size=batch_size)
        readout = self.transformer(src, src_padding_mask, self.readout_embed.weight, pos)[-1]
        readout = readout[:, 0]
        if actions is not None:
            if actions_obj is not None:
                if not sample_mba:
                    loss = self.action_decoder.compute_obj_loss(readout, actions_obj, extra_cond=current_obj)
                else:
                    loss = self.action_decoder.compute_loss(readout, actions, extra_cond=current_obj)
            else:
                loss = self.action_decoder.compute_loss(readout, actions, extra_cond=current_obj)
            return loss
        else:
            outputs = {}
            if self.enable_mba:
                with torch.no_grad():
                    obj_pred = self.action_decoder.predict_obj(readout, extra_cond=current_obj)
                outputs["obj_pred"] = obj_pred
            with torch.no_grad():
                action_pred = self.action_decoder.predict_action(readout, extra_cond=current_obj)
            outputs["action_pred"] = action_pred
            return outputs
