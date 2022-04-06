import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from numpy import pi

class Decoder(nn.Module):
    ''' Decoder class.

    Predicts volume density and color from 3D location, viewing
    direction, and latent code z.

    Args:
        hidden_size (int): hidden size of Decoder network
        n_blocks (int): number of layers
        n_blocks_view (int): number of view-dep layers
        skips (list): where to add a skip connection
        use_viewdirs: (bool): whether to use viewing directions
        n_freq_posenc (int), max freq for positional encoding (3D location)
        n_freq_posenc_views (int), max freq for positional encoding (
            viewing direction)
        dim (int): input dimension
        z_dim (int): dimension of latent code z
        rgb_out_dim (int): output dimension of feature / rgb prediction
        final_sigmoid_activation (bool): whether to apply a sigmoid activation
            to the feature / rgb output
        downscale_by (float): downscale factor for input points before applying
            the positional encoding
        positional_encoding (str): type of positional encoding
        gauss_dim_pos (int): dim for Gauss. positional encoding (position)
        gauss_dim_view (int): dim for Gauss. positional encoding (
            viewing direction)
        gauss_std (int): std for Gauss. positional encoding
    '''
    def __init__(self, hidden_size=128, n_blocks=8, n_blocks_view=1,
                 skips=[4], use_viewdirs=True, n_freq_posenc=10,
                 n_freq_posenc_views=4,
                 z_id_dim=100, z_exp_dim=50, z_exp_joints_dim=3, z_albedo_dim=50, z_detail_dim=128,
                 rgb_out_dim=128, final_sigmoid_activation=False,
                 downscale_p_by=2., positional_encoding="normal",
                 gauss_dim_pos=10, gauss_dim_view=4, gauss_std=4.,
                 **kwargs):
        super().__init__()
        self.use_viewdirs = use_viewdirs
        self.n_freq_posenc = n_freq_posenc
        self.n_freq_posenc_views = n_freq_posenc_views
        self.skips = skips
        self.downscale_p_by = downscale_p_by
        self.z_id_dim = z_id_dim
        self.z_exp_dim = z_exp_dim
        self.final_sigmoid_activation = final_sigmoid_activation
        self.n_blocks = n_blocks
        self.n_blocks_view = n_blocks_view

        assert(positional_encoding in ('normal', 'gauss'))
        self.positional_encoding = positional_encoding
        if positional_encoding == 'gauss':
            np.random.seed(42)
            # remove * 2 because of cos and sin
            self.B_pos = gauss_std * \
                torch.from_numpy(np.random.randn(
                    1,  gauss_dim_pos * 3, 3)).float().cuda()
            self.B_view = gauss_std * \
                torch.from_numpy(np.random.randn(
                    1,  gauss_dim_view * 3, 3)).float().cuda()
            dim_embed = 3 * gauss_dim_pos * 2
            dim_embed_view = 3 * gauss_dim_view * 2
        else:
            dim_embed = 3 * self.n_freq_posenc * 2
            dim_embed_view = 3 * self.n_freq_posenc_views * 2

        # Activation function
        self.activation = F.relu

        # Density prediction layers
        n_density_layers = 8
        self.id_skip = 5  # layer idx into which to skip connect the z_id

        self.fc_posencoded_in = nn.Linear(dim_embed, hidden_size)
        self.fc_z_in = nn.Linear(z_id_dim + z_exp_joints_dim + z_exp_dim, hidden_size)

        self.fc_density_layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for i in range(n_density_layers-1)]
        )

        self.fc_z_skip = nn.Linear(z_id_dim, hidden_size)

        self.sigma_out = nn.Linear(hidden_size, 1)

        # Appearance prediction layers
        n_feat_layers = 2
        dim_feat_out = 128
        self.fc_appearance_in = nn.Linear(z_albedo_dim + z_detail_dim, hidden_size)
        self.fc_feature_layers = nn.ModuleList(
            [nn.Linear(hidden_size, hidden_size) for i in range(n_feat_layers - 1)]
        )
        self.fc_feature_layers.append(
            nn.Linear(hidden_size, dim_feat_out)
        )


    def transform_points(self, p, views=False):
        # Positional encoding
        # normalize p between [-1, 1]
        p = p / self.downscale_p_by

        # we consider points up to [-1, 1]
        # so no scaling required here
        if self.positional_encoding == 'gauss':
            B = self.B_view if views else self.B_pos
            p_transformed = (B @ (pi * p.permute(0, 2, 1))).permute(0, 2, 1)
            p_transformed = torch.cat(
                [torch.sin(p_transformed), torch.cos(p_transformed)], dim=-1)
        else:
            L = self.n_freq_posenc_views if views else self.n_freq_posenc
            p_transformed = torch.cat([torch.cat(
                [torch.sin((2 ** i) * pi * p),
                 torch.cos((2 ** i) * pi * p)],
                dim=-1) for i in range(L)], dim=-1)
        return p_transformed

    def forward(self, p_in, ray_d, z_id=None, z_exp=None, z_exp_joints=None, z_albedo=None, z_detail=None):

        z_geom = torch.cat((z_id, z_exp, z_exp_joints), dim=1)
        z_app = torch.cat((z_albedo, z_detail), dim=1)

        p = self.transform_points(p_in)

        # Density (sigma) prediction
        p_encoded = self.fc_posencoded_in(p)
        z_encoded = self.fc_z_in(z_geom)

        net = p_encoded + z_encoded.unsqueeze(1)
        net = self.activation(net)

        for idx, layer in enumerate(self.fc_density_layers):
            net = self.activation(layer(net))

            # Adding skip connection for z_id
            if (idx + 1) == self.id_skip:
                net = net + self.fc_z_skip(z_id).unsqueeze(1)
        sigma_out = self.sigma_out(net).squeeze(-1)

        # Feature prediction
        net = net + self.fc_appearance_in(z_app).unsqueeze(1)
        for idx, layer in enumerate(self.fc_feature_layers):
            net = self.activation(layer(net))

        # TODO: GIRAFFE implementation uses viewing directions here.
        # Unsure if stylnerf/headnerf do the same or not.

        feat_out = net

        if self.final_sigmoid_activation:
            feat_out = torch.sigmoid(feat_out)

        return feat_out, sigma_out
