import torch
from torch.nn import functional as F
from torch import nn

from adet.utils.comm import compute_locations, aligned_bilinear
import torch.distributed as dist


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
    assert params.dim() == 2
    assert len(weight_nums) == len(bias_nums)
    assert params.size(1) == sum(weight_nums) + sum(bias_nums)

    num_insts = params.size(0)
    num_layers = len(weight_nums)

    params_splits = list(torch.split_with_sizes(
        params, weight_nums + bias_nums, dim=1
    ))

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

    return weight_splits, bias_splits


def build_dynamic_mask_head(cfg):
    return DynamicMaskHead(cfg)


class DynamicMaskHead(nn.Module):
    def __init__(self, cfg):
        super(DynamicMaskHead, self).__init__()
        self.num_layers = cfg.MODEL.CONDINST.MASK_HEAD.NUM_LAYERS
        self.channels = cfg.MODEL.CONDINST.MASK_HEAD.CHANNELS
        self.in_channels = cfg.MODEL.CONDINST.MASK_BRANCH.OUT_CHANNELS
        self.mask_out_stride = cfg.MODEL.CONDINST.MASK_OUT_STRIDE
        self.disable_rel_coords = cfg.MODEL.CONDINST.MASK_HEAD.DISABLE_REL_COORDS

        soi = cfg.MODEL.FCOS.SIZES_OF_INTEREST
        self.register_buffer("sizes_of_interest", torch.tensor(soi + [soi[-1] * 2]))
        self.mapping_ratio = cfg.MODEL.CONDINST.MASK_HEAD.MAPPING_RATIO
        self.grid_num = cfg.MODEL.CONDINST.MASK_HEAD.GRID_NUM[0]

        weight_nums, bias_nums = [], []
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

    def mask_heads_forward(self, features, weights, biases, num_insts, locations_ind):
        '''
        :param features
        :param weights: [w0, w1, ...]
        :param bias: [b0, b1, ...]
        :return:
        '''
        assert features.dim() == 4
        n_layers = len(weights)
        x = features  # 1/8 (N, 10, h, w)
        n_ins, c, h, w = x.shape
        device = x.device
        assert h % self.grid_num == 0, (h, self.grid_num)
        assert w % self.grid_num == 0, (w, self.grid_num)

        i_h = int(h / self.grid_num)
        i_w = int(w / self.grid_num)

        x = x.reshape(n_ins, c, self.grid_num, i_h, self.grid_num, i_w).permute(0, 1, 2, 4, 3, 5)
        x = x.reshape(n_ins, c, self.grid_num ** 2, i_h, i_w).permute(2, 0, 1, 3, 4)
        ins_ind = torch.arange(0, n_ins).cuda(device=device)

        assert locations_ind.max() < x.shape[0], (locations_ind, x.shape)
        x = x[locations_ind, ins_ind]
        x = x.reshape(1, -1, i_h, i_w)
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
            self, mask_feats, mask_feat_stride, instances, out_size
    ):
        locations = compute_locations(
            mask_feats.size(2), mask_feats.size(3),
            stride=mask_feat_stride, device=mask_feats.device
        )
        n_inst = len(instances)

        im_inds = instances.im_inds
        mask_head_params = instances.mask_head_params

        N, _, H, W = mask_feats.size()

        instance_locations = instances.locations
        new_locations = instance_locations / 8.  # n, 2(x, y)
        new_locations_x = new_locations[:, 0].clamp(min=0, max=W - 1) // int((W / self.grid_num))
        new_locations_y = new_locations[:, 1].clamp(min=0, max=H - 1) // int((H / self.grid_num))
        new_locations_ind = (new_locations_x + self.grid_num * new_locations_y).to(torch.int64)

        if not self.disable_rel_coords:
            relative_coords = instance_locations.reshape(-1, 1, 2) - locations.reshape(1, -1, 2)
            relative_coords = relative_coords.permute(0, 2, 1).float()
            soi = self.sizes_of_interest.float()[instances.fpn_levels]
            relative_coords = relative_coords / soi.reshape(-1, 1, 1)
            relative_coords = relative_coords.to(dtype=mask_feats.dtype)

            mask_head_inputs = torch.cat([
                relative_coords, mask_feats[im_inds].reshape(n_inst, self.in_channels, H * W)
            ], dim=1)
            mask_head_inputs = mask_head_inputs.reshape(n_inst, 10, H, W)
        else:
            mask_head_inputs = mask_feats[im_inds].reshape(n_inst, self.in_channels, H * W)
            mask_head_inputs = mask_head_inputs.reshape(n_inst, self.in_channels, H, W)

        weights, biases = parse_dynamic_params(
            mask_head_params, self.channels,
            self.weight_nums, self.bias_nums
        )
        mask_logits = self.mask_heads_forward(mask_head_inputs, weights, biases, n_inst, new_locations_ind)
        mask_logits = mask_logits.reshape(-1, 1, int(H / self.grid_num), int(W / self.grid_num))

        assert mask_feat_stride >= self.mask_out_stride
        assert mask_feat_stride % self.mask_out_stride == 0
        mask_logits = F.interpolate(
            mask_logits, size=out_size,
            mode='bilinear',
            align_corners=True
        )

        return mask_logits.sigmoid()

    def __call__(self, mask_feats, mask_feat_stride, pred_instances, gt_instances=None):
        if self.training:
            gt_inds = pred_instances.gt_inds
            gt_bitmasks = self.crop_and_expand(gt_instances)
            gt_bitmasks = gt_bitmasks[gt_inds].unsqueeze(dim=1).to(dtype=mask_feats.dtype)

            if len(pred_instances) == 0:
                loss_mask = mask_feats.sum() * 0 + pred_instances.mask_head_params.sum() * 0
            else:
                _, _, h, w = gt_bitmasks.shape
                mask_scores = self.mask_heads_forward_with_coords(
                    mask_feats, mask_feat_stride, pred_instances, (h, w)
                )
                mask_losses = dice_coefficient(mask_scores, gt_bitmasks)
                loss_mask = mask_losses.mean()

            return loss_mask.float()
        else:
            if len(pred_instances) > 0:
                mapping_ratio = self.mapping_ratio
                _, _, f_h, f_w = mask_feats.shape
                h = int((mapping_ratio / 2. * f_h) + 0.5)
                w = int((mapping_ratio / 2. * f_w) + 0.5)
                mask_scores = self.mask_heads_forward_with_coords(
                    mask_feats, mask_feat_stride, pred_instances, (h, w)
                )
                mask_scores = self.recover_ins2all(mask_scores, pred_instances, (f_h, f_w))
                pred_instances.pred_global_masks = mask_scores.float()

            return pred_instances

    def crop_and_expand(self, gt_instances):
        gt_bitmasks = torch.cat([per_im.gt_bitmasks for per_im in gt_instances])  # 1/4
        gt_boxes = torch.cat([per_im.gt_boxes.tensor for per_im in gt_instances])  # 1
        resized_gt_boxes = gt_boxes / 4.
        center_x_gt_boxes = ((resized_gt_boxes[:, 2] + resized_gt_boxes[:, 0]) / 2.0).squeeze().tolist()
        center_y_gt_boxes = ((resized_gt_boxes[:, 3] + resized_gt_boxes[:, 1]) / 2.0).squeeze().tolist()
        n, h, w = gt_bitmasks.shape
        h_i = int(h / (2 * self.grid_num))  # 4: pad 25 2: pad 50
        w_i = int(w / (2 * self.grid_num))
        pad_gt_bitmasks = F.pad(gt_bitmasks, [w_i, w_i, h_i, h_i], mode='constant', value=0)
        expand_gt_bitmasks = []
        for idx, (c_x, c_y) in enumerate(zip(center_x_gt_boxes, center_y_gt_boxes)):
            i = c_x // (w / self.grid_num)
            j = c_y // (h / self.grid_num)
            assert i < self.grid_num, i
            assert j < self.grid_num, j
            per_c_x = w_i + (i * 2 + 1) * w_i
            per_c_y = h_i + (j * 2 + 1) * h_i
            x1 = int(per_c_x - self.mapping_ratio * w_i)
            x2 = int(per_c_x + self.mapping_ratio * w_i)
            y1 = int(per_c_y - self.mapping_ratio * h_i)
            y2 = int(per_c_y + self.mapping_ratio * h_i)
            expand_gt_bitmasks.append(pad_gt_bitmasks[idx, y1:y2, x1:x2].unsqueeze(0))
        expand_gt_bitmasks = torch.cat(expand_gt_bitmasks)

        return expand_gt_bitmasks

    def recover_ins2all(self, mask_scores, pred_instances, size):
        # 2grid 1map: 50x50 i_w: 100
        # 2grid 2map: 100x100
        n, _, h, w = mask_scores.shape
        i_h = h / self.mapping_ratio
        i_w = w / self.mapping_ratio
        pred_boxes = pred_instances.pred_boxes.tensor  # 1
        resized_pred_boxes = pred_boxes / (16 / self.grid_num)
        # resized_pred_boxes = pred_boxes / 4.
        assert len(resized_pred_boxes.shape) == 2, resized_pred_boxes
        # s = self.grid_num / self.mapping_ratio
        center_x_pred_boxes = ((resized_pred_boxes[:, 2] + resized_pred_boxes[:, 0]) / 2.0).clamp(min=0, max= w - 1)
        center_y_pred_boxes = ((resized_pred_boxes[:, 3] + resized_pred_boxes[:, 1]) / 2.0).clamp(min=0, max= h - 1)
        center_x_pred_boxes = center_x_pred_boxes.tolist()
        center_y_pred_boxes = center_y_pred_boxes.tolist()
        assert len(center_x_pred_boxes) >= 1
        assert len(center_y_pred_boxes) >= 1

        recover_mask_scores = []
        for idx, (c_x, c_y) in enumerate(zip(center_x_pred_boxes, center_y_pred_boxes)):
            i = c_x // i_w
            j = c_y // i_h
            assert i < self.grid_num, (i, c_x, i_w, w)
            assert j < self.grid_num, (j, c_y, i_h, h)
            w_l = int(i * i_w)  # 25x
            w_r = int((self.grid_num - i - 1) * i_w)
            h_u = int(j * i_h)
            h_d = int((self.grid_num - j - 1) * i_h)
            recover_mask = F.pad(mask_scores[idx], [w_l, w_r, h_u, h_d], mode='constant', value=0)
            _, _h, _w = recover_mask.shape
            x_s = (self.mapping_ratio - 1) * (i_w / 2)
            y_s = (self.mapping_ratio - 1) * (i_h / 2)
            recover_mask = recover_mask[:, int(y_s):int(_h - y_s), int(x_s):int(_w - x_s)]
            recover_mask = F.interpolate(
                recover_mask.unsqueeze(0), size=size,
                mode='bilinear',
                align_corners=True
            )
            recover_mask_scores.append(recover_mask)
        recover_mask_scores = torch.cat(recover_mask_scores)

        return recover_mask_scores

    # def crop_and_expand20(self, gt_instances):
    #     area_num = self.grid_num
    #     gt_bitmasks = torch.cat([per_im.gt_bitmasks for per_im in gt_instances])  # 1/4
    #     gt_boxes = torch.cat([per_im.gt_boxes.tensor for per_im in gt_instances])  # 1
    #     resized_gt_boxes = gt_boxes / 4.
    #     center_x_gt_boxes = ((resized_gt_boxes[:, 2] + resized_gt_boxes[:, 0]) / 2.0).squeeze().tolist()
    #     center_y_gt_boxes = ((resized_gt_boxes[:, 3] + resized_gt_boxes[:, 1]) / 2.0).squeeze().tolist()
    #     n, h, w = gt_bitmasks.shape
    #     h_i = int(h / 8)
    #     w_i = int(w / 8)
    #     pad_gt_bitmasks = F.pad(gt_bitmasks, [w_i, w_i, h_i, h_i], mode='constant', value=0)
    #     expand_gt_bitmasks = []
    #     for idx, (c_x, c_y) in enumerate(zip(center_x_gt_boxes, center_y_gt_boxes)):
    #         i = c_x // (w / 4)
    #         j = c_y // (h / 4)
    #         assert i < area_num, i
    #         assert j < area_num, j
    #         per_c_x = w_i + (i * 2 + 1) * w_i
    #         per_c_y = h_i + (j * 2 + 1) * h_i
    #         x1 = int(per_c_x - w / 4)
    #         x2 = int(per_c_x + w / 4)
    #         y1 = int(per_c_y - h / 4)
    #         y2 = int(per_c_y + h / 4)
    #
    #         y_l = int(h / 2)
    #         x_l = int(w / 2)
    #         assert x2 - x1 == x_l
    #         assert y2 - y1 == y_l
    #         expand_gt_bitmasks.append(pad_gt_bitmasks[idx, y1:(y1 + y_l), x1:(x1 + x_l)].unsqueeze(0))
    #     expand_gt_bitmasks = torch.cat(expand_gt_bitmasks)
    #
    #     return expand_gt_bitmasks
    #
    # def crop_and_expand15(self, gt_instances):
    #     area_num = self.grid_num
    #     gt_bitmasks = torch.cat([per_im.gt_bitmasks for per_im in gt_instances])  # 1/4
    #     gt_boxes = torch.cat([per_im.gt_boxes.tensor for per_im in gt_instances])  # 1
    #     resized_gt_boxes = gt_boxes / 4.
    #     center_x_gt_boxes = ((resized_gt_boxes[:, 2] + resized_gt_boxes[:, 0]) / 2.0).squeeze().tolist()
    #     center_y_gt_boxes = ((resized_gt_boxes[:, 3] + resized_gt_boxes[:, 1]) / 2.0).squeeze().tolist()
    #     n, h, w = gt_bitmasks.shape  # 200， 200
    #     h_i = int(h / 8)
    #     w_i = int(w / 8)
    #     pad_gt_bitmasks = F.pad(gt_bitmasks, [w_i, w_i, h_i, h_i], mode='constant', value=0)
    #     expand_gt_bitmasks = []
    #     for idx, (c_x, c_y) in enumerate(zip(center_x_gt_boxes, center_y_gt_boxes)):
    #         i = c_x // (w / 4)
    #         j = c_y // (h / 4)
    #         assert i < area_num, i
    #         assert j < area_num, j
    #         per_c_x = w_i + (i * 2 + 1) * w_i
    #         per_c_y = h_i + (j * 2 + 1) * h_i
    #         x1 = int(per_c_x - 1.5 * w_i)
    #         x2 = int(per_c_x + 1.5 * w_i)
    #         y1 = int(per_c_y - 1.5 * h_i)
    #         y2 = int(per_c_y + 1.5 * h_i)
    #
    #         y_l = int(3 * h_i)
    #         x_l = int(3 * w_i)
    #         assert x2 - x1 == x_l
    #         assert y2 - y1 == y_l
    #         expand_gt_bitmasks.append(pad_gt_bitmasks[idx, y1:(y1 + y_l), x1:(x1 + x_l)].unsqueeze(0))
    #     expand_gt_bitmasks = torch.cat(expand_gt_bitmasks)
    #
    #     return expand_gt_bitmasks

    def recover_ins2all_1x(self, mask_scores, pred_instances, size):
        # mask scores shape: 50x50
        # return mask shape: 100x100
        n, _, h, w = mask_scores.shape
        mapping_ratio = 1
        i_h = h / mapping_ratio
        i_w = w / mapping_ratio
        area_num = 4
        pred_boxes = pred_instances.pred_boxes.tensor  # 1
        resized_pred_boxes = pred_boxes / 4.
        assert len(resized_pred_boxes.shape) == 2, resized_pred_boxes
        center_x_pred_boxes = ((resized_pred_boxes[:, 2] + resized_pred_boxes[:, 0]) / 2.0).clamp(min=0, max=4 * w - 1)
        center_y_pred_boxes = ((resized_pred_boxes[:, 3] + resized_pred_boxes[:, 1]) / 2.0).clamp(min=0, max=4 * h - 1)
        center_x_pred_boxes = center_x_pred_boxes.tolist()
        center_y_pred_boxes = center_y_pred_boxes.tolist()
        assert len(center_x_pred_boxes) >= 1
        assert len(center_y_pred_boxes) >= 1

        recover_mask_scores = []
        for idx, (c_x, c_y) in enumerate(zip(center_x_pred_boxes, center_y_pred_boxes)):
            i = c_x // i_w
            j = c_y // i_h
            assert i < area_num, i
            assert j < area_num, j
            l = (area_num + 1) * 2 - 2 * mapping_ratio  # 8
            w_l = int(i * i_w)  # 25x
            w_r = int((3 - i) * i_w)
            h_u = int(j * i_h)
            h_d = int((3 - j) * i_h)
            # print(pred_boxes[0], c_x, c_y, i_w, i_h, w_l, w_r, h_u, h_d)
            # print(zhubin)
            recover_mask = F.pad(mask_scores[idx], [w_l, w_r, h_u, h_d], mode='constant', value=0)
            recover_mask = F.interpolate(
                recover_mask.unsqueeze(0), size=size,
                mode='bilinear',
                align_corners=True
            )
            recover_mask_scores.append(recover_mask)
        recover_mask_scores = torch.cat(recover_mask_scores)

        return recover_mask_scores

    def recover_ins2all_15x(self, mask_scores, pred_instances, size):
        # mask scores shape: 75x75
        # return mask shape: 100x100
        n, _, h, w = mask_scores.shape
        mapping_ratio = 1.5
        i_h = h / mapping_ratio
        i_w = w / mapping_ratio
        area_num = 4
        pred_boxes = pred_instances.pred_boxes.tensor  # 1
        resized_pred_boxes = pred_boxes / 4.
        assert len(resized_pred_boxes.shape) == 2, resized_pred_boxes
        center_x_pred_boxes = ((resized_pred_boxes[:, 2] + resized_pred_boxes[:, 0]) / 2.0).clamp(min=0,
                                                                                                  max=2.67 * w - 1)
        center_y_pred_boxes = ((resized_pred_boxes[:, 3] + resized_pred_boxes[:, 1]) / 2.0).clamp(min=0,
                                                                                                  max=2.67 * h - 1)
        center_x_pred_boxes = center_x_pred_boxes.tolist()
        center_y_pred_boxes = center_y_pred_boxes.tolist()
        assert len(center_x_pred_boxes) >= 1
        assert len(center_y_pred_boxes) >= 1

        recover_mask_scores = []
        for idx, (c_x, c_y) in enumerate(zip(center_x_pred_boxes, center_y_pred_boxes)):
            i = c_x // i_w
            j = c_y // i_h
            assert i < area_num, i
            assert j < area_num, j
            w_l = int(i * i_w)  # 25x
            w_r = int((3 - i) * i_w)
            h_u = int(j * i_h)
            h_d = int((3 - j) * i_h)
            recover_mask = F.pad(mask_scores[idx], [w_l, w_r, h_u, h_d], mode='constant', value=0)
            _, _h, _w = recover_mask.shape
            recover_mask = recover_mask[:, int(i_h / 4):int(_h - i_h / 4), int(i_w / 4):int(_w - i_w / 4)]
            recover_mask = F.interpolate(
                recover_mask.unsqueeze(0), size=size,
                mode='bilinear',
                align_corners=True
            )
            recover_mask_scores.append(recover_mask)
        recover_mask_scores = torch.cat(recover_mask_scores)

        return recover_mask_scores

    def recover_ins2all_2x(self, mask_scores, pred_instances, size):
        # mask scores shape: 100x100
        # return mask shape: 100x100
        n, _, h, w = mask_scores.shape
        mapping_ratio = 2
        i_h = h / mapping_ratio
        i_w = w / mapping_ratio
        area_num = 4
        pred_boxes = pred_instances.pred_boxes.tensor  # 1
        resized_pred_boxes = pred_boxes / 4.
        assert len(resized_pred_boxes.shape) == 2, resized_pred_boxes
        center_x_pred_boxes = ((resized_pred_boxes[:, 2] + resized_pred_boxes[:, 0]) / 2.0).clamp(min=0, max=2 * w - 1)
        center_y_pred_boxes = ((resized_pred_boxes[:, 3] + resized_pred_boxes[:, 1]) / 2.0).clamp(min=0, max=2 * h - 1)
        center_x_pred_boxes = center_x_pred_boxes.tolist()
        center_y_pred_boxes = center_y_pred_boxes.tolist()
        assert len(center_x_pred_boxes) >= 1
        assert len(center_y_pred_boxes) >= 1

        recover_mask_scores = []
        for idx, (c_x, c_y) in enumerate(zip(center_x_pred_boxes, center_y_pred_boxes)):
            i = c_x // i_w
            j = c_y // i_h
            assert i < area_num, i
            assert j < area_num, j
            w_l = int(i * i_w)  # 25x
            w_r = int((3 - i) * i_w)
            h_u = int(j * i_h)
            h_d = int((3 - j) * i_h)
            recover_mask = F.pad(mask_scores[idx], [w_l, w_r, h_u, h_d], mode='constant', value=0)
            _, _h, _w = recover_mask.shape
            recover_mask = recover_mask[:, int(i_h / 2):int(_h - i_h / 2), int(i_w / 2):int(_w - i_w / 2)]
            recover_mask = F.interpolate(
                recover_mask.unsqueeze(0), size=size,
                mode='bilinear',
                align_corners=True
            )
            recover_mask_scores.append(recover_mask)
        recover_mask_scores = torch.cat(recover_mask_scores)

        return recover_mask_scores

    def recover_ins2all_2x_old(self, mask_scores, pred_instances):
        n, _, h, w = mask_scores.shape
        i_h = h / 2
        i_w = w / 2
        area_num = 4
        mapping_ratio = 2
        pred_boxes = pred_instances.pred_boxes.tensor  # 1
        resized_pred_boxes = pred_boxes / 4.
        assert len(resized_pred_boxes.shape) == 2, resized_pred_boxes
        # assert resized_pred_boxes.max() < 2 * max(h, w), (resized_pred_boxes.max(), h, w)
        center_x_pred_boxes = ((resized_pred_boxes[:, 2] + resized_pred_boxes[:, 0]) / 2.0).clamp(min=0, max=2 * w - 1)
        center_y_pred_boxes = ((resized_pred_boxes[:, 3] + resized_pred_boxes[:, 1]) / 2.0).clamp(min=0, max=2 * h - 1)
        center_x_pred_boxes = center_x_pred_boxes.tolist()
        center_y_pred_boxes = center_y_pred_boxes.tolist()
        assert len(center_x_pred_boxes) >= 1
        assert len(center_y_pred_boxes) >= 1

        recover_mask_scores = []
        for idx, (c_x, c_y) in enumerate(zip(center_x_pred_boxes, center_y_pred_boxes)):
            i = c_x // i_w
            j = c_y // i_h
            assert i < area_num, i
            assert j < area_num, j
            l = (area_num + 1) * 2 - 2 * mapping_ratio
            w_l = int((i * 2) * (i_w / 2))
            w_r = int((l - i * 2) * (i_w / 2))
            h_u = int((j * 2) * (i_h / 2))
            h_d = int((l - j * 2) * (i_h / 2))
            recover_mask = F.pad(mask_scores[idx], [w_l, w_r, h_u, h_d], mode='constant', value=0)
            recover_mask = recover_mask[:, int(i_h / 2):int(i_h / 2 + 2 * h), int(i_w / 2):int(i_w / 2 + 2 * w)]
            recover_mask_scores.append(recover_mask.unsqueeze(0))
        recover_mask_scores = torch.cat(recover_mask_scores)

        return recover_mask_scores

    def recover_ins2all_test(self, mask_scores, pred_instances):
        mask_scores = aligned_bilinear(mask_scores, 2)
        return mask_scores

    def collect(self, mask_scores, gt_bitmasks):
        self.iter += 1
        if dist.get_rank() == 0:
            if self.iter % 100 == 0:
                import cv2
                sample_mask_scores = mask_scores[0, 0].clone()
                sample_gt_bitmasks = gt_bitmasks[0, 0].clone()
                img = torch.cat([sample_mask_scores, sample_gt_bitmasks]).detach().cpu().numpy()
                cv2.imwrite('pngs/test_pred_gt_{}.png'.format(str(self.iter).zfill(6)), img * 255)
