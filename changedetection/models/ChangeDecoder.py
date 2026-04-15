import torch
import torch.nn as nn
import torch.nn.functional as F
from classification.models.vmamba import VSSBlock, Permute

from module.DIEFEN import MultiHeadAttention

from module.FM import FusionModule

from module.TIF import ChannelExchange


class ChangeDecoder(nn.Module):
    def __init__(self, encoder_dims, channel_first, norm_layer, ssm_act_layer, mlp_act_layer, **kwargs):
        super(ChangeDecoder, self).__init__()

        # Define the VSS Block for Spatio-temporal relationship modelling
        self.st_block_41 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=2432, out_channels=128),           #  1920   1536   2048
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                     ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'],
                     ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                     ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'],
                     ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                     forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer,
                     mlp_drop_rate=kwargs['mlp_drop_rate'],
                     gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        self.st_block_42 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-1], out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                     ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'],
                     ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                     ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'],
                     ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                     forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer,
                     mlp_drop_rate=kwargs['mlp_drop_rate'],
                     gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),

        )
        self.st_block_43 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-1], out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                     ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'],
                     ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                     ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'],
                     ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                     forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer,
                     mlp_drop_rate=kwargs['mlp_drop_rate'],
                     gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )

        self.st_block_31 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=1280, out_channels=128),            #  960   768   1088
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                     ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'],
                     ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                     ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'],
                     ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                     forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer,
                     mlp_drop_rate=kwargs['mlp_drop_rate'],
                     gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        self.st_block_32 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=384, out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                     ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'],
                     ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                     ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'],
                     ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                     forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer,
                     mlp_drop_rate=kwargs['mlp_drop_rate'],
                     gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        self.st_block_33 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-2], out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                     ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'],
                     ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                     ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'],
                     ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                     forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer,
                     mlp_drop_rate=kwargs['mlp_drop_rate'],
                     gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )

        self.st_block_21 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=704, out_channels=128),            #  480  384   608
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                     ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'],
                     ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                     ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'],
                     ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                     forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer,
                     mlp_drop_rate=kwargs['mlp_drop_rate'],
                     gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        self.st_block_22 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-3], out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                     ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'],
                     ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                     ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'],
                     ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                     forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer,
                     mlp_drop_rate=kwargs['mlp_drop_rate'],
                     gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        self.st_block_23 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-3], out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                     ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'],
                     ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                     ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'],
                     ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                     forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer,
                     mlp_drop_rate=kwargs['mlp_drop_rate'],
                     gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )

        self.st_block_11 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=416, out_channels=128),                #   240    192   368
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                     ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'],
                     ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                     ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'],
                     ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                     forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer,
                     mlp_drop_rate=kwargs['mlp_drop_rate'],
                     gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        self.st_block_12 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-4], out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                     ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'],
                     ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                     ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'],
                     ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                     forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer,
                     mlp_drop_rate=kwargs['mlp_drop_rate'],
                     gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )
        self.st_block_13 = nn.Sequential(
            nn.Conv2d(kernel_size=1, in_channels=encoder_dims[-4], out_channels=128),
            Permute(0, 2, 3, 1) if not channel_first else nn.Identity(),
            VSSBlock(hidden_dim=128, drop_path=0.1, norm_layer=norm_layer, channel_first=channel_first,
                     ssm_d_state=kwargs['ssm_d_state'], ssm_ratio=kwargs['ssm_ratio'],
                     ssm_dt_rank=kwargs['ssm_dt_rank'], ssm_act_layer=ssm_act_layer,
                     ssm_conv=kwargs['ssm_conv'], ssm_conv_bias=kwargs['ssm_conv_bias'],
                     ssm_drop_rate=kwargs['ssm_drop_rate'], ssm_init=kwargs['ssm_init'],
                     forward_type=kwargs['forward_type'], mlp_ratio=kwargs['mlp_ratio'], mlp_act_layer=mlp_act_layer,
                     mlp_drop_rate=kwargs['mlp_drop_rate'],
                     gmlp=kwargs['gmlp'], use_checkpoint=kwargs['use_checkpoint']),
            Permute(0, 3, 1, 2) if not channel_first else nn.Identity(),
        )

        # Fuse layer
        self.fuse_layer_4 = nn.Sequential(nn.Conv2d(kernel_size=1, in_channels=128 * 5, out_channels=128),
                                          nn.BatchNorm2d(128), nn.ReLU(inplace=True))  # ReLU
        self.fuse_layer_3 = nn.Sequential(nn.Conv2d(kernel_size=1, in_channels=128 * 5, out_channels=128),
                                          nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.fuse_layer_2 = nn.Sequential(nn.Conv2d(kernel_size=1, in_channels=128 * 5, out_channels=128),
                                          nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.fuse_layer_1 = nn.Sequential(nn.Conv2d(kernel_size=1, in_channels=128 * 5, out_channels=128),
                                          nn.BatchNorm2d(128), nn.ReLU(inplace=True))

        # Smooth layer
        # self.smooth_layer_3 = ResBlock(128, 128)
        # self.smooth_layer_2 = ResBlock(128, 128)
        # self.smooth_layer_1 = ResBlock(128, 128)

        self.smooth_layer_3 = C2f_Edge(128, 128)
        self.smooth_layer_2 = C2f_Edge(128, 128)
        self.smooth_layer_1 = C2f_Edge(128, 128)


        self.tif = ChannelExchange()
        #self.ltif = LearnableChannelExchange()
        #self.ltif2 = LearnableChannelExchange()
        self.fm = FusionModule()


        self.diff_fuse4 = MultiHeadAttention(768)
        self.diff_fuse3 = MultiHeadAttention(384)
        self.diff_fuse2 = MultiHeadAttention(192)
        self.diff_fuse1 = MultiHeadAttention(96)


    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear') + y

    def forward(self, pre_features, post_features):
        pre_feat_1, pre_feat_2, pre_feat_3, pre_feat_4 = pre_features

        post_feat_1, post_feat_2, post_feat_3, post_feat_4 = post_features

        pr1, pr2, pr3, pr4 = self.fm(pre_features, post_features)

        # diff1 = pre_feat_1 - post_feat_1
        # diff2 = pre_feat_2 - post_feat_2
        # diff3 = pre_feat_3 - post_feat_3
        # diff4 = pre_feat_4 - post_feat_4

        diff4 = self.diff_fuse4(pre_feat_4, post_feat_4)
        diff3 = self.diff_fuse3(pre_feat_3, post_feat_3)
        diff2 = self.diff_fuse2(pre_feat_2, post_feat_2)
        diff1 = self.diff_fuse1(pre_feat_1, post_feat_1)



        # print('p1',pre_feat_1.shape)
        # print('po1',post_feat_1.shape)

        pre_feat_1_tif, post_feat_1_tif = self.tif(pre_feat_1, post_feat_1)
        pre_feat_2_tif, post_feat_2_tif = self.tif(pre_feat_2, post_feat_2)
        # pre_feat_3_tif, post_feat_3_tif = self.tif(pre_feat_3, post_feat_3)
        # pre_feat_1_tif, post_feat_4_tif = self.tif(pre_feat_1, post_feat_1)

        '''
            Stage I
        '''

        # SPPF
        #print(pre_feat_4.shape, post_feat_4.shape, diff4.shape, pr4.shape)
        p41 = torch.cat([pre_feat_4, post_feat_4, pr4, diff4], 1)
        # p41 = torch.cat([pre_feat_4, post_feat_4], 1)

        # p41 = self.cv1(p41)
        # p41 = self.sppf(p41)

        # p41 = self.c2psa(p41)
        # p41 = self.cv2(p41)
        p41 = self.st_block_41(p41)

        # p41 = self.st_block_41(torch.cat([pre_feat_4, post_feat_4], dim=1))

        B, C, H, W = pre_feat_4.size()
        # Create an empty tensor of the correct shape (B, C, H, 2*W)
        ct_tensor_42 = torch.empty(B, C, H, 2 * W).cuda()
        # Fill in odd columns with A and even columns with B
        ct_tensor_42[:, :, :, ::2] = pre_feat_4  # Odd columns
        ct_tensor_42[:, :, :, 1::2] = post_feat_4  # Even columns
        p42 = self.st_block_42(ct_tensor_42)

        ct_tensor_43 = torch.empty(B, C, H, 2 * W).cuda()
        ct_tensor_43[:, :, :, 0:W] = pre_feat_4
        ct_tensor_43[:, :, :, W:] = post_feat_4
        p43 = self.st_block_43(ct_tensor_43)

        p4 = self.fuse_layer_4(
            torch.cat([p41, p42[:, :, :, ::2], p42[:, :, :, 1::2], p43[:, :, :, 0:W], p43[:, :, :, W:]], dim=1))

        '''
            Stage II
        '''
        p31 = self.st_block_31(torch.cat([pre_feat_3, post_feat_3, pr3, diff3], dim=1))
        # p31 = self.st_block_31(torch.cat([pre_feat_3, post_feat_3], dim=1))

        B, C, H, W = pre_feat_3.size()
        # Create an empty tensor of the correct shape (B, C, H, 2*W)
        ct_tensor_32 = torch.empty(B, C, H, 2 * W).cuda()
        # Fill in odd columns with A and even columns with B
        ct_tensor_32[:, :, :, ::2] = pre_feat_3  # Odd columns
        ct_tensor_32[:, :, :, 1::2] = post_feat_3  # Even columns
        p32 = self.st_block_32(ct_tensor_32)

        ct_tensor_33 = torch.empty(B, C, H, 2 * W).cuda()
        ct_tensor_33[:, :, :, 0:W] = pre_feat_3
        ct_tensor_33[:, :, :, W:] = post_feat_3
        p33 = self.st_block_33(ct_tensor_33)

        p3 = self.fuse_layer_3(
            torch.cat([p31, p32[:, :, :, ::2], p32[:, :, :, 1::2], p33[:, :, :, 0:W], p33[:, :, :, W:]], dim=1))
        p3 = self._upsample_add(p4, p3)
        p3 = self.smooth_layer_3(p3)

        '''
            Stage III
        '''
        p21 = self.st_block_21(torch.cat([pre_feat_2_tif, post_feat_2_tif, pr2, diff2], dim=1))
        # p21 = self.st_block_21(torch.cat([pre_feat_2_tif, post_feat_2_tif], dim=1))
        # p21 = self.st_block_21(torch.cat([pre_feat_2, post_feat_2], dim=1))

        B, C, H, W = pre_feat_2.size()
        # Create an empty tensor of the correct shape (B, C, H, 2*W)
        ct_tensor_22 = torch.empty(B, C, H, 2 * W).cuda()
        # Fill in odd columns with A and even columns with B
        ct_tensor_22[:, :, :, ::2] = pre_feat_2  # Odd columns
        ct_tensor_22[:, :, :, 1::2] = post_feat_2  # Even columns
        p22 = self.st_block_22(ct_tensor_22)

        ct_tensor_23 = torch.empty(B, C, H, 2 * W).cuda()
        ct_tensor_23[:, :, :, 0:W] = pre_feat_2
        ct_tensor_23[:, :, :, W:] = post_feat_2
        p23 = self.st_block_23(ct_tensor_23)

        p2 = self.fuse_layer_2(
            torch.cat([p21, p22[:, :, :, ::2], p22[:, :, :, 1::2], p23[:, :, :, 0:W], p23[:, :, :, W:]], dim=1))
        p2 = self._upsample_add(p3, p2)
        p2 = self.smooth_layer_2(p2)
        '''
            Stage IV
        '''
        p11 = self.st_block_11(torch.cat([pre_feat_1_tif, post_feat_1_tif, pr1, diff1], dim=1))
        # p11 = self.st_block_11(torch.cat([pre_feat_1_tif, post_feat_1_tif], dim=1))
        # p11 = self.st_block_11(torch.cat([pre_feat_1, post_feat_1], dim=1))

        B, C, H, W = pre_feat_1.size()
        # Create an empty tensor of the correct shape (B, C, H, 2*W)
        ct_tensor_12 = torch.empty(B, C, H, 2 * W).cuda()
        # Fill in odd columns with A and even columns with B
        ct_tensor_12[:, :, :, ::2] = pre_feat_1  # Odd columns
        ct_tensor_12[:, :, :, 1::2] = post_feat_1  # Even columns
        p12 = self.st_block_12(ct_tensor_12)

        ct_tensor_13 = torch.empty(B, C, H, 2 * W).cuda()
        ct_tensor_13[:, :, :, 0:W] = pre_feat_1
        ct_tensor_13[:, :, :, W:] = post_feat_1
        p13 = self.st_block_13(ct_tensor_13)

        p1 = self.fuse_layer_1(
            torch.cat([p11, p12[:, :, :, ::2], p12[:, :, :, 1::2], p13[:, :, :, 0:W], p13[:, :, :, W:]], dim=1))

        p1 = self._upsample_add(p2, p1)
        p1 = self.smooth_layer_1(p1)

        return p1


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        # print("SPPF",x)
        y = [self.cv1(x)]
        # print("ok")c
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))


def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        # print(x.shape)
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积：提升效率与特征提取能力"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class SEBlock(nn.Module):
    """轻量级通道注意力模块 (Squeeze-and-Excitation)"""
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class Block(nn.Module):
    """
    改进版特征提取与融合模块
    - 分离拼接结构
    - 深度可分离卷积
    - 多尺度空洞卷积
    - 通道注意力(SE)
    - 多层融合
    """
    def __init__(self, in_channels, out_channels, split_ratio=0.5, dilation=2):
        super().__init__()
        split_c1 = int(in_channels * split_ratio)
        split_c2 = in_channels - split_c1

        # ===== 分支1：标准深度可分离卷积 =====
        self.branch1 = DepthwiseSeparableConv(split_c1, out_channels // 2)

        # ===== 分支2：空洞卷积分支，扩大感受野 =====
        self.branch2 = nn.Sequential(
            nn.Conv2d(split_c2, out_channels // 2, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )

        # ===== 融合后再进行通道注意力增强 =====
        self.se = SEBlock(out_channels)

        # ===== 全局上下文增强 =====
        self.context_pool = nn.AdaptiveAvgPool2d(1)
        self.context_conv = nn.Conv2d(out_channels, out_channels, 1, bias=False)

        # ===== 最终输出整合 =====
        self.fuse = nn.Sequential(
            nn.Conv2d(out_channels * 2, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 分离输入通道
        c1 = int(x.size(1) * 0.5)
        x1, x2 = torch.split(x, [c1, x.size(1) - c1], dim=1)

        # 分支提取
        f1 = self.branch1(x1)
        f2 = self.branch2(x2)

        # 拼接融合
        fused = torch.cat([f1, f2], dim=1)

        # 通道注意力
        fused = self.se(fused)

        # 上下文特征
        global_feat = self.context_pool(fused)
        global_feat = self.context_conv(global_feat)
        global_feat = F.interpolate(global_feat, size=fused.size()[2:], mode='bilinear', align_corners=False)

        # 最终融合输出
        out = torch.cat([fused, global_feat], dim=1)
        out = self.fuse(out)
        return out


class C2f01(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        #self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.block = Block(self.c, self.c)

    def forward(self, x):
        """前向传播通过C2f层"""
        # 通过第一个卷积层
        cv1_out = self.cv1(x)

        # 分成两份
        part1, part2 = cv1_out.chunk(2, dim=1)

        # 处理第二份特征
        processed_features = [part1, part2]  # 初始的两份特征

        part2 = self.block(part2)  # 通过模块处理
        processed_features.append(part2)  # 添加处理结果

        # 拼接所有特征并通过第二个卷积层
        return self.cv2(torch.cat(processed_features, dim=1))


class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class EdgeGradient(nn.Module):
    """提取边缘梯度特征，并进行卷积融合"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Sobel算子卷积核
        sobel_x = torch.tensor([[[-1, 0, 1],
                                 [-2, 0, 2],
                                 [-1, 0, 1]]], dtype=torch.float32)
        sobel_y = torch.tensor([[[-1, -2, -1],
                                 [0, 0, 0],
                                 [1, 2, 1]]], dtype=torch.float32)
        self.register_buffer("weight_x", sobel_x.unsqueeze(0))
        self.register_buffer("weight_y", sobel_y.unsqueeze(0))
        self.conv = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()
        self.alpha = nn.Parameter(torch.tensor(0.5))  # 可学习权重

    def forward(self, x):
        # Sobel边缘响应
        grad_x = F.conv2d(x, self.weight_x.expand(x.size(1), -1, -1, -1), padding=1, groups=x.size(1))
        grad_y = F.conv2d(x, self.weight_y.expand(x.size(1), -1, -1, -1), padding=1, groups=x.size(1))
        edge = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-6)
        edge = self.act(self.bn(self.conv(edge)))
        # 与原特征融合
        return x + self.alpha * edge

# -----------------------------
# C2f + EdgeGradient 模块
# -----------------------------
class C2f_Edge(nn.Module):
    """C2f模块 + Edge Gradient增强"""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))
        self.edge_grad = EdgeGradient(c2, c2)  # Edge增强模块

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        out = self.cv2(torch.cat(y, 1))
        out = self.edge_grad(out)
        return out



class C3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


class C3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n)
        )


class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))


