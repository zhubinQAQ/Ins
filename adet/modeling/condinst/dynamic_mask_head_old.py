import torch
from torch.nn import functional as F
from torch import nn

from adet.utils.comm import compute_locations, aligned_bilinear


def dice_coefficient(x, target):
    eps = 1e-5
    n_inst = x.size(0)
    x = x.reshape(n_inst, -1)
    target = target.reshape(n_inst, -1)
    intersection = (x * target).sum(dim=1)
    union = (x ** 2.0).sum(dim=1) + (target ** 2.0).sum(dim=1) + eps
    loss = 1. - (2 * intersection / union)
    return loss


def parse_dynamic_params(params, channels, weight_nums, bias_nums):
    # params (n, 169)
    assert params.dim() == 2
    assert len(weight_nums) == len(bias_nums)
    assert params.size(1) == sum(weight_nums) + sum(bias_nums)

    num_insts = params.size(0)
    num_layers = len(weight_nums)
    # weight: [80, 64, 8]
    # bias: [8, 8, 1]
    # 152 + 17 = 169
    params_splits = list(torch.split_with_sizes(
        params, weight_nums + bias_nums, dim=1
    ))
    # torch.Size([n, 169])[88, 72, 9]
    # params_splits [(n, 88), (n, 72), (n, 9)]
    # [torch.Size([421, 80]), torch.Size([421, 64]), torch.Size([421, 8]),
    #  torch.Size([421, 8]), torch.Size([421, 8]), torch.Size([421, 1])]
    # [torch.Size([421, 80]), torch.Size([421, 64]), torch.Size([421, 8])]
    # [torch.Size([421, 8]), torch.Size([421, 8]), torch.Size([421, 1])]

    weight_splits = params_splits[:num_layers]
    bias_splits = params_splits[num_layers:]

    for l in range(num_layers):
        if l < num_layers - 1:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(num_insts * channels, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_insts * channels)
        else:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(num_insts * 1, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_insts)
    # [torch.Size([3368, 10, 1, 1]), torch.Size([3368, 8, 1, 1]), torch.Size([421, 8, 1, 1])]
    # [torch.Size([3368]), torch.Size([3368]), torch.Size([421])]

    return weight_splits, bias_splits


def build_dynamic_mask_head(cfg):
    return DynamicMaskHead(cfg)


class DynamicMaskHead(nn.Module):
    def __init__(self, cfg):
        super(DynamicMaskHead, self).__init__()
        self.num_layers = cfg.MODEL.CONDINST.MASK_HEAD.NUM_LAYERS  # 3
        self.channels = cfg.MODEL.CONDINST.MASK_HEAD.CHANNELS  # 8
        self.in_channels = cfg.MODEL.CONDINST.MASK_BRANCH.OUT_CHANNELS  # 8
        self.mask_out_stride = cfg.MODEL.CONDINST.MASK_OUT_STRIDE  # 4
        self.disable_rel_coords = cfg.MODEL.CONDINST.MASK_HEAD.DISABLE_REL_COORDS  # False

        soi = cfg.MODEL.FCOS.SIZES_OF_INTEREST  # [64, 128, 256, 512]
        self.register_buffer("sizes_of_interest", torch.tensor(soi + [soi[-1] * 2]))

        weight_nums, bias_nums = [], []
        # weight: [80, 64, 8]
        # bias: [8, 8, 1]
        # 152 + 17 = 169
        for l in range(self.num_layers):
            if l == 0:
                if not self.disable_rel_coords:
                    weight_nums.append((self.in_channels + 2) * self.channels)
                else:
                    weight_nums.append(self.in_channels * self.channels)
                bias_nums.append(self.channels)
            elif l == self.num_layers - 1:
                weight_nums.append(self.channels * 1)
                bias_nums.append(1)
            else:
                weight_nums.append(self.channels * self.channels)
                bias_nums.append(self.channels)

        self.weight_nums = weight_nums
        self.bias_nums = bias_nums
        self.num_gen_params = sum(weight_nums) + sum(bias_nums)

    def mask_heads_forward(self, features, weights, biases, num_insts):
        '''
        :param features
        :param weights: [w0, w1, ...]
        :param bias: [b0, b1, ...]
        :return:
        '''
        assert features.dim() == 4
        n_layers = len(weights)
        x = features
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv2d(
                x, w, bias=b,
                stride=1, padding=0,
                groups=num_insts
            )
            if i < n_layers - 1:
                x = F.relu(x)
        return x

    def mask_heads_forward_with_coords(
            self, mask_feats, mask_feat_stride, instances
    ):
        # 1/8 P3 对应的原图真实坐标
        locations = compute_locations(
            mask_feats.size(2), mask_feats.size(3),
            stride=mask_feat_stride, device=mask_feats.device
        )
        n_inst = len(instances)

        # 在fcos每个点采样时，会记录所属img id，最后根据分类pos id筛选，
        # 最后在实例分割这块根据ins个数复制相应image出来的mask feature
        # 即(2, n, h, w) -> (ins, n, hxw)
        im_inds = instances.im_inds
        mask_head_params = instances.mask_head_params

        N, _, H, W = mask_feats.size()

        if not self.disable_rel_coords:
            # 之前在fcos记录了每个pos location的中心位置，在这里生成相对坐标
            # 即target所在的点变成0，其余的变成和它的相对距离
            instance_locations = instances.locations
            # (39, 1, 2) - (1, hxw, 2)
            relative_coords = instance_locations.reshape(-1, 1, 2) - locations.reshape(1, -1, 2)
            # (39, hxw, 2) --> (39, 2, hxw)
            relative_coords = relative_coords.permute(0, 2, 1).float()
            # 给每个相对距离乘以一个衰减系数，如果instance越大，即来自高层特征，
            # 则会给它的相对距离更大的衰减因子
            soi = self.sizes_of_interest.float()[instances.fpn_levels]
            relative_coords = relative_coords / soi.reshape(-1, 1, 1)
            relative_coords = relative_coords.to(dtype=mask_feats.dtype)

            mask_head_inputs = torch.cat([
                relative_coords, mask_feats[im_inds].reshape(n_inst, self.in_channels, H * W)
            ], dim=1)
            # torch.Size([39, 10, 12880]) torch.Size([2, 8, 92, 140]) torch.Size([39, 2, 12880])
            # print(mask_head_inputs.shape, mask_feats.shape, relative_coords.shape)
        else:
            mask_head_inputs = mask_feats[im_inds].reshape(n_inst, self.in_channels, H * W)

        mask_head_inputs = mask_head_inputs.reshape(1, -1, H, W)

        weights, biases = parse_dynamic_params(
            mask_head_params, self.channels,
            self.weight_nums, self.bias_nums
        )

        # 现在的mask_head_inputs的每个instance包含了全图的mask feature
        # 以及所预测instance中心点的相对距离信息
        # torch.Size([1, 580, 100, 136])
        # [torch.Size([464, 10, 1, 1]), torch.Size([464, 8, 1, 1]), torch.Size([58, 8, 1, 1])]
        # [torch.Size([464]), torch.Size([464]), torch.Size([58])]
        mask_logits = self.mask_heads_forward(mask_head_inputs, weights, biases, n_inst)
        mask_logits = mask_logits.reshape(-1, 1, H, W)

        assert mask_feat_stride >= self.mask_out_stride
        assert mask_feat_stride % self.mask_out_stride == 0
        mask_logits = aligned_bilinear(mask_logits, int(mask_feat_stride / self.mask_out_stride))

        return mask_logits.sigmoid()

    def __call__(self, mask_feats, mask_feat_stride, pred_instances, gt_instances=None):
        if self.training:
            # mask_feats: [2, 8, 96, 148]
            # mask_feat_stride: 8
            # pred: labels/reg_targets/gt_ctrs/locations/fpn_levels/
            #       logits_pred/reg_pred/ctrness_pred/
            #       pos_inds/mask_head_params
            gt_inds = pred_instances.gt_inds
            gt_bitmasks = torch.cat([per_im.gt_bitmasks for per_im in gt_instances])
            gt_bitmasks = gt_bitmasks[gt_inds].unsqueeze(dim=1).to(dtype=mask_feats.dtype)

            if len(pred_instances) == 0:
                loss_mask = mask_feats.sum() * 0 + pred_instances.mask_head_params.sum() * 0
            else:
                mask_scores = self.mask_heads_forward_with_coords(
                    mask_feats, mask_feat_stride, pred_instances
                )
                # 全图大小的loss， 1/4下采样
                # torch.Size([42, 1, 200, 296]) torch.Size([42, 1, 200, 296])
                self.collect(mask_scores, gt_bitmasks)
                mask_losses = dice_coefficient(mask_scores, gt_bitmasks)
                loss_mask = mask_losses.mean()

            return loss_mask.float()
        else:
            if len(pred_instances) > 0:
                mask_scores = self.mask_heads_forward_with_coords(
                    mask_feats, mask_feat_stride, pred_instances
                )
                pred_instances.pred_global_masks = mask_scores.float()

            return pred_instances

    def collect(self, mask_scores, gt_bitmasks):
        self.iter += 1
        if dist.get_rank() == 0:
            if self.iter % 100 == 0:
                import cv2
                sample_mask_scores = mask_scores[0, 0].clone()
                sample_gt_bitmasks = gt_bitmasks[0, 0].clone()
                img = torch.cat([sample_mask_scores, sample_gt_bitmasks]).detach().cpu().numpy()
                cv2.imwrite('pngs_ori/test_pred_gt_{}.png'.format(str(self.iter).zfill(6)), img * 255)
            if self.iter % 10000 == 0:
                print(zhubin)