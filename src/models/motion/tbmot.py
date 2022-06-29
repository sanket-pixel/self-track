from .transformer import PositionalEncoding, Transformer, MLP, TransformerEncoder, TransformerEncoderLayer
from torch import nn
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MotionModel(nn.Module):
    def __init__(self, arch="lstm", feature_size=128, num_layers=1, nhead=8, bbox_dim=4, warp_dim=6, use_warp=False,
                 mode="train"):
        super(MotionModel, self).__init__()
        self.arch = arch
        if arch == "transformer":
            self.pos_encoder = PositionalEncoding(feature_size)
            self.begin_sign = nn.Parameter(torch.randn(feature_size), requires_grad=True)
            self.encoder = Transformer(d_model=feature_size,nhead=nhead,num_encoder_layers=num_layers,
                                       num_decoder_layers=num_layers)
        elif arch == "lstm":
            self.encoder = nn.LSTM(input_size=bbox_dim, hidden_size=feature_size, num_layers=num_layers,batch_first=False)
        elif arch == "gru":
            self.encoder = nn.GRU(input_size=feature_size, hidden_size=feature_size, num_layers=num_layers)

        if use_warp:
            self.encode_bbox = MLP(bbox_dim, int(feature_size / 2), int(feature_size / 2), 2)
            self.encode_warp = MLP(warp_dim, int(feature_size / 2), int(feature_size / 2), 2)
        else:
            self.encode_bbox = MLP(bbox_dim, int(feature_size), int(feature_size), 2)
        self.decoder = MLP(feature_size, feature_size, bbox_dim, 2)
        self.feature_size = feature_size
        self.use_warp = use_warp

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask.to(device)

    def forward(self, input_bbox, input_warp=None, added_padding_mask=None):
        # input_emb = self.encode_bbox(input_bbox)
        input_emb = input_bbox
        if self.use_warp:
            input_emb_shape = input_emb.shape[1]
            if input_warp.shape[1] != input_emb_shape:
                warp_emb = self.encode_warp(input_warp).repeat_interleave(input_emb_shape,1)
            else:
                warp_emb = self.encode_warp(input_warp)
            input_emb = torch.cat([input_emb, warp_emb], -1)
        if self.arch == "transformer":
            input_emb = self.pos_encoder(input_emb)
            batch_size = input_emb.shape[1]
            begin_signed_batched = self.begin_sign.unsqueeze(0).repeat_interleave(batch_size, dim=0).unsqueeze(0)
            # because True means ignore and False means consider in attention
            src_key_padding_mask = ~added_padding_mask.bool()
            output = self.encoder(input_emb, begin_signed_batched, src_key_padding_mask=src_key_padding_mask)
        elif self.arch == "lstm":
            input_emb = nn.utils.rnn.pack_padded_sequence(input_emb, added_padding_mask.sum(1).cpu().int(),
                                                          enforce_sorted=False)
            _, (output, ct) = self.encoder(input_emb)
        elif self.arch == "gru":
            input_emb = nn.utils.rnn.pack_padded_sequence(input_emb, added_padding_mask.sum(1).cpu().int(),
                                                          enforce_sorted=False)
            _, output = self.encoder(input_emb)
        predicted_bbox = self.decoder(output[-1:, :, :])
        return predicted_bbox
