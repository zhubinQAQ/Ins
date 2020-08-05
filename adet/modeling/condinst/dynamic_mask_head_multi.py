import torch
from torch.nn import functional as F
from torch import nn
import numpy as np

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


def get_grid_map(grid=3):
    grid_dict = {}
    for i in range(grid**2):
        x = i % grid
        y = i // grid
        move = [[0, ], [0, ]]
        if x == 0:
            move[0].append(-1)
            move[1].append(0)
            if y == 0:
                move[0].extend([-1, 0])
                move[1].extend([-1, -1])
            if y == grid-1:
                move[0].extend([0, -1])
                move[1].extend([1, 1])
        elif x == grid-1:
            if y == 0:
                move[0].extend([0, 1])
                move[1].extend([-1, -1])
                move[0].append(1)
                move[1].append(0)
            elif y == grid-1:
                move[0].append(1)
                move[1].append(0)
                move[0].extend([1, 0])
                move[1].extend([1, 1])
            else:
                move[0].append(1)
                move[1].append(0)
        elif y == 0:
            move[0].append(0)
            move[1].append(-1)
        elif y == grid-1:
            move[0].append(0)
            move[1].append(1)
        grid_dict[i] = move
    grid_tensor = np.ones([grid**2, 9, 2]) * 100
    grid_tensor[0, [0, 1, 2, 3]] = np.array(grid_dict[0]).T
    grid_tensor[1, [0, 3]] = np.array(grid_dict[1]).T
    grid_tensor[2, [0, 3, 4, 5]] = np.array(grid_dict[2]).T
    grid_tensor[3, [0, 1]] = np.array(grid_dict[3]).T
    grid_tensor[4, [0, ]] = np.array(grid_dict[4]).T
    grid_tensor[5, [0, 5]] = np.array(grid_dict[5]).T
    grid_tensor[6, [0, 1, 7, 8]] = np.array(grid_dict[6]).T
    grid_tensor[7, [0, 7]] = np.array(grid_dict[7]).T
    grid_tensor[8, [0, 5, 6, 7]] = np.array(grid_dict[8]).T
    return grid_tensor


def parse_dynamic_params(params, channels, weight_nums, bias_nums, inds, concat=False):
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

    multi_weight_splits = [[] for _ in inds]
    multi_bias_splits = [[] for _ in inds]

    for l in range(num_layers):
        if l < num_layers - 1:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(num_insts, channels, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_insts, channels)
            for idx, ind in enumerate(inds):
                weight_splits_per_ind = weight_splits[l][ind]
                bias_splits_per_ind = bias_splits[l][ind]
                n, c, _, _, _ = weight_splits_per_ind.shape
                if n > 0:
                    if concat and idx:
                        multi_weight_splits[idx].append(weight_splits_per_ind)
                        multi_bias_splits[idx].append(bias_splits_per_ind)
                    else:
                        multi_weight_splits[idx].append(weight_splits_per_ind.reshape(n * c, -1, 1, 1))
                        multi_bias_splits[idx].append(bias_splits_per_ind.reshape(n * c))
                else:
                    multi_weight_splits[idx].append([])
                    multi_bias_splits[idx].append([])
        else:
            # out_channels x in_channels x 1 x 1
            weight_splits[l] = weight_splits[l].reshape(num_insts, -1, 1, 1)
            bias_splits[l] = bias_splits[l].reshape(num_insts)
            for idx, ind in enumerate(inds):
                weight_splits_per_ind = weight_splits[l][ind]
                bias_splits_per_ind = bias_splits[l][ind]
                n, _, _, _ = weight_splits_per_ind.shape
                if n > 0:
                    multi_weight_splits[idx].append(weight_splits_per_ind)
                    multi_bias_splits[idx].append(bias_splits_per_ind)
                else:
                    multi_weight_splits[idx].append([])
                    multi_bias_splits[idx].append([])

    return multi_weight_splits, multi_bias_splits


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
        self.grid_num = cfg.MODEL.CONDINST.MASK_HEAD.GRID_NUM
        self.split = cfg.MODEL.CONDINST.MASK_HEAD.SPLIT
        self.concat = cfg.MODEL.CONDINST.MASK_HEAD.CONCAT
        self.loss_weights = cfg.MODEL.CONDINST.MASK_HEAD.LOSS_WEIGHTS

        assert isinstance(self.grid_num, list)

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

    def mask_heads_forward(self, features, out_size, weights, biases, num_insts, locations_ind, grid_num=1):
        '''
        :param features
        :param weights: [w0, w1, ...]
        :param bias: [b0, b1, ...]
        :return:
        '''
        if not len(weights[0]) or not len(biases[0]):
            return None
        assert num_insts > 0
        N, _, H, W = features.shape
        assert grid_num == 1 or len(locations_ind), (grid_num, locations_ind)
        if grid_num > 1:
            assert features.dim() == 4
            n_layers = len(weights)
            x = features  # 1/8 (N, 10, h, w)
            n_ins, c, h, w = x.shape
            device = x.device
            assert h % grid_num == 0, (h, grid_num)
            assert w % grid_num == 0, (w, grid_num)

            i_h = int(h / grid_num)
            i_w = int(w / grid_num)

            x = x.reshape(n_ins, c, grid_num, i_h, grid_num, i_w).permute(0, 1, 2, 4, 3, 5)
            x = x.reshape(n_ins, c, grid_num ** 2, i_h, i_w).permute(2, 0, 1, 3, 4)
            ins_ind = torch.arange(0, n_ins).cuda(device=device)

            assert locations_ind.max() < x.shape[0], (locations_ind, x.shape)
            x = x[locations_ind, ins_ind]
            x = x.reshape(1, -1, i_h, i_w)
        else:
            assert features.dim() == 4
            n_layers = len(weights)
            x = features.reshape(1, -1, H, W)
        for i, (w, b) in enumerate(zip(weights, biases)):
            x = F.conv2d(
                x, w, bias=b,
                stride=1, padding=0,
                groups=num_insts
            )
            if i < n_layers - 1:
                x = F.relu(x)
        x = x.reshape(-1, 1, int(H / grid_num), int(W / grid_num))
        assert out_size is not None, num_insts
        if self.training:
            # x = aligned_bilinear(x, 2 * grid_num)
            x = F.interpolate(
                x, size=out_size,
                mode='bilinear',
                align_corners=True
            )
        else:
            # x = aligned_bilinear(x, 2)
            x = F.interpolate(
                x, scale_factor=2,
                mode='bilinear',
                align_corners=True
            )
        return x

    def add_locaions_info(self, locations, instances, mask_feats):
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

        mask_feats = torch.cat([
            relative_coords, mask_feats
        ], dim=1)
        return mask_feats

    def mask_heads_forward_with_coords(
            self, mask_feats, mask_feat_stride, instances, gt_instances=None
    ):
        n_inst = len(instances)

        im_inds = instances.im_inds
        mask_head_params = instances.mask_head_params
        instance_locations = instances.locations
        levels = instances.fpn_levels
        # 0, 1, 2, 3, 4 => P3/P4/P5/P6/P7
        # (3, 4) ==> 1
        # (1, 2) ==> 2
        # 0 ==> 4
        ind1 = (levels > self.split[2]) & (levels <= self.split[3])
        ind2 = (levels > self.split[1]) & (levels <= self.split[2])
        ind4 = (levels > self.split[0]) & (levels <= self.split[1])

        weights, biases = parse_dynamic_params(
            mask_head_params, self.channels,
            self.weight_nums, self.bias_nums, [ind1, ind2, ind4], self.concat
        )

        N, _, H, W = mask_feats.size()
        num_layers = self.num_layers
        mask_head_inputs = mask_feats[im_inds].reshape(n_inst, self.in_channels, H * W)
        if not self.disable_rel_coords:
            base_locations = compute_locations(
                mask_feats.size(2), mask_feats.size(3),
                stride=mask_feat_stride, device=mask_feats.device
            )
            mask_head_inputs = self.add_locaions_info(base_locations, instances, mask_head_inputs)
            mask_head_inputs = mask_head_inputs.reshape(n_inst, self.in_channels + 2, H, W)
        else:
            mask_head_inputs = mask_head_inputs.reshape(n_inst, self.in_channels, H, W)

        def get_loaction_weights_bias(level_ind, locations, i_weight, i_bias, grid_num=4, grid_inside_num=3):
            new_locations = locations / 4.  # n, 2(x, y)
            new_locations[:, 0] = new_locations[:, 0].clamp(min=0, max=2*W - 1)
            new_locations[:, 1] = new_locations[:, 1].clamp(min=0, max=2*H - 1)
            new_locations_x = new_locations[:, 0] // int((2*W / grid_num))
            new_locations_y = new_locations[:, 1] // int((2*H / grid_num))
            new_locations_ind = (new_locations_x + grid_num * new_locations_y).to(torch.int64)
            if not self.concat or not len(locations):
                return level_ind, new_locations_ind, i_weight, i_bias, None
            new_inside_locations = torch.zeros_like(new_locations)
            new_inside_locations[:, 0] = new_locations[:, 0] % int((2*W / grid_num))
            new_inside_locations[:, 1] = new_locations[:, 1] % int((2*H / grid_num))
            new_locations_inside_x = new_inside_locations[:, 0] // round((2*W / (grid_num * grid_inside_num)) + 0.5)
            new_locations_inside_y = new_inside_locations[:, 1] // round((2*H / (grid_num * grid_inside_num)) + 0.5)
            relative_inside_location = (new_locations_inside_x + grid_inside_num * new_locations_inside_y).to(
                torch.int64)
            assert (relative_inside_location < grid_inside_num ** 2).all(), \
                (W, H, new_inside_locations, new_locations_inside_x, new_locations_inside_y, relative_inside_location)
            # import time
            # time1 = time.time()
            # cate part
            grid_map = torch.from_numpy(get_grid_map(grid_inside_num)).to(device=mask_feats.device)
            new_locations_ind = new_locations_ind.cpu().numpy()
            maps = grid_map[relative_inside_location].clone()  # n, 9, 2
            n, _, _ = maps.shape
            maps[:, :, 0] = maps[:, :, 0] + new_locations_x.repeat(9).reshape(9, -1).T
            maps[:, :, 1] = maps[:, :, 1] + new_locations_y.repeat(9).reshape(9, -1).T
            maps = maps.reshape(-1, 2)
            ind_valid = (maps[:, 0] >= 0) & (maps[:, 0] < grid_num) & (maps[:, 1] >= 0) & (maps[:, 1] < grid_num)
            maps = maps[ind_valid]
            final_locations_ind = (maps[:, 0] + grid_num * maps[:, 1]).to(dtype=torch.int64, device=mask_feats.device)
            param_ind = ind_valid.reshape(n, 9)
            param_ind = param_ind.T*torch.arange(1, n+1).to(device=mask_feats.device)
            param_ind = param_ind.T.flatten()
            param_ind = param_ind[param_ind > 0] - 1
            param_ind = param_ind.to(dtype=torch.int64, device=mask_feats.device)
            gt_ind = torch.arange(0, n*9)[ind_valid].to(dtype=torch.int64, device=mask_feats.device)
            if not len(param_ind):
                return level_ind, new_locations_ind, i_weight, i_bias, None
            for l in range(num_layers):
                assert len(param_ind), param_ind
                i_weight[l] = i_weight[l][param_ind]
                i_bias[l] = i_bias[l][param_ind]
                if l < num_layers - 1:
                    n, c, _, _, _ = i_weight[l].shape
                    i_weight[l] = i_weight[l].reshape(n * c, -1, 1, 1)
                    i_bias[l] = i_bias[l].reshape(n * c)
            return param_ind, final_locations_ind, i_weight, i_bias, gt_ind

        param_ind_2, new_ind2, new_weights_2, new_biases_2, gt_ind_2 = get_loaction_weights_bias(
            ind2, instance_locations[ind2], weights[1], biases[1], grid_num=self.grid_num[1]
        )
        param_ind_4, new_ind4, new_weights_4, new_biases_4, gt_ind_4 = get_loaction_weights_bias(
            ind4, instance_locations[ind4], weights[2], biases[2], grid_num=self.grid_num[2]
        )

        mask_head_inputs1 = mask_head_inputs[ind1]
        if not self.concat:
            mask_head_inputs2 = mask_head_inputs[ind2]
            mask_head_inputs4 = mask_head_inputs[ind4]
            inds_list = [ind1, ind2, ind4]
        else:
            mask_head_inputs2 = mask_head_inputs[param_ind_2]
            mask_head_inputs4 = mask_head_inputs[param_ind_4]
            inds_list = [(ind1, ind1), (ind2, param_ind_2), (ind4, param_ind_4)]

        n_inst1 = mask_head_inputs1.shape[0]
        n_inst2 = mask_head_inputs2.shape[0]
        n_inst4 = mask_head_inputs4.shape[0]

        if gt_instances is not None:
            gt_inds = instances.gt_inds
            gt_bitmasks = torch.cat([per_im.gt_bitmasks for per_im in gt_instances])  # 1/4
            gt_boxes = torch.cat([per_im.gt_boxes.tensor for per_im in gt_instances])  # 1

            if not self.concat:
                gt_inds1 = gt_inds[ind1]
                crop_gt_bitmasks1, out_size1 = self.crop_and_expand(gt_bitmasks, gt_boxes, grid_num=self.grid_num[0])
                gt_bitmasks1 = crop_gt_bitmasks1[gt_inds1].unsqueeze(dim=1).to(dtype=mask_feats.dtype)
                gt_inds2 = gt_inds[ind2]
                crop_gt_bitmasks2, out_size2 = self.crop_and_expand(gt_bitmasks, gt_boxes, grid_num=self.grid_num[1])
                gt_inds4 = gt_inds[ind4]
                crop_gt_bitmasks4, out_size4 = self.crop_and_expand(gt_bitmasks, gt_boxes, grid_num=self.grid_num[2])
                gt_bitmasks2 = crop_gt_bitmasks2[gt_inds2]
                gt_bitmasks4 = crop_gt_bitmasks4[gt_inds4]
            else:
                gt_inds1 = gt_inds[ind1]
                crop_gt_bitmasks1, out_size1 = self.crop_and_expand_concate(gt_bitmasks, gt_boxes, grid_num=self.grid_num[0])
                gt_bitmasks1 = crop_gt_bitmasks1[gt_inds1].unsqueeze(dim=1).to(dtype=mask_feats.dtype)
                crop_gt_bitmasks2 = self.crop_and_expand_concate(gt_bitmasks, gt_boxes, grid_num=self.grid_num[1])
                crop_gt_bitmasks4 = self.crop_and_expand_concate(gt_bitmasks, gt_boxes, grid_num=self.grid_num[2])
                gt_inds2 = gt_inds[ind2]
                gt_inds4 = gt_inds[ind4]
                gt_bitmasks2 = [crop_gt_bitmasks2[ind] for ind in gt_inds2.tolist()]
                gt_bitmasks4 = [crop_gt_bitmasks4[ind] for ind in gt_inds4.tolist()]
                gt_bitmasks2 = torch.cat(gt_bitmasks2)[gt_ind_2] if len(gt_inds2) else None
                gt_bitmasks4 = torch.cat(gt_bitmasks4)[gt_ind_4] if len(gt_inds4) else None
                out_size2 = gt_bitmasks2.shape[1:] if len(gt_inds2) else None
                out_size4 = gt_bitmasks4.shape[1:] if len(gt_inds4) else None
            gt_bitmasks2 = gt_bitmasks2.unsqueeze(dim=1).to(dtype=mask_feats.dtype) if len(gt_inds2) else None
            gt_bitmasks4 = gt_bitmasks4.unsqueeze(dim=1).to(dtype=mask_feats.dtype) if len(gt_inds4) else None
            gt_bitmasks_list = [gt_bitmasks1, gt_bitmasks2, gt_bitmasks4]
        else:
            gt_bitmasks_list = []
            out_size1 = (int(H * 2), int(W * 2))
            out_size2 = (H, W)
            out_size4 = (int(H / 2), int(W / 2))
            # out_size1 = (int(H * 2), int(W * 2))
            # out_size2 = (int(H * 2), int(W * 2))
            # out_size4 = (int(H * 2), int(W * 2))
        out_size = {1: out_size1, 2: out_size2, 4: out_size4}

        mask_logits1 = self.mask_heads_forward(
            mask_head_inputs1, out_size[self.grid_num[0]], weights[0], biases[0], n_inst1, [], self.grid_num[0])
        mask_logits2 = self.mask_heads_forward(
            mask_head_inputs2, out_size[self.grid_num[1]], weights[1], biases[1], n_inst2, new_ind2, self.grid_num[1])
        mask_logits4 = self.mask_heads_forward(
            mask_head_inputs4, out_size[self.grid_num[2]], weights[2], biases[2], n_inst4, new_ind4, self.grid_num[2])

        mask_logits_list = [mask_logits1, mask_logits2, mask_logits4]

        return mask_logits_list, gt_bitmasks_list, inds_list, [None, gt_ind_2, gt_ind_4]

    def __call__(self, mask_feats, mask_feat_stride, pred_instances, gt_instances=None):
        if self.training:
            if len(pred_instances) == 0:
                loss_mask = mask_feats.sum() * 0 + pred_instances.mask_head_params.sum() * 0
            else:
                mask_scores, gt_bitmasks, _, _ = self.mask_heads_forward_with_coords(
                    mask_feats, mask_feat_stride, pred_instances, gt_instances
                )
                losses = []
                norm = 0
                for i, (mask_scores_per_level, gt_bitmasks_per_level) in enumerate(zip(mask_scores, gt_bitmasks)):
                    if mask_scores_per_level is not None:
                        assert gt_bitmasks_per_level is not None
                        w = self.loss_weights[i]
                        norm += w
                        mask_scores_per_level = mask_scores_per_level.sigmoid()
                        mask_losses = dice_coefficient(mask_scores_per_level, gt_bitmasks_per_level)
                        loss_mask = w * mask_losses.mean()
                        losses.append(loss_mask)
                assert norm > 0
                loss_mask = sum(losses) / norm
            return loss_mask.float()
        else:
            if len(pred_instances) > 0:
                mask_scores, _, inds, gt_inds = self.mask_heads_forward_with_coords(
                    mask_feats, mask_feat_stride, pred_instances
                )
                if self.concat:
                    pred_instances = self.recover_ins2all_concat(mask_feats, mask_scores, pred_instances, inds, gt_inds)
                else:
                    pred_instances = self.recover_ins2all(mask_feats, mask_scores, pred_instances, inds)

            return pred_instances

    def crop_and_expand(self, gt_bitmasks, gt_boxes, grid_num=4, mapping_ratio=1.0):
        if grid_num == 1:
            return gt_bitmasks, gt_bitmasks.shape[1:]
        resized_gt_boxes = gt_boxes / 4.
        center_x_gt_boxes = ((resized_gt_boxes[:, 2] + resized_gt_boxes[:, 0]) / 2.0).squeeze().tolist()
        center_y_gt_boxes = ((resized_gt_boxes[:, 3] + resized_gt_boxes[:, 1]) / 2.0).squeeze().tolist()
        n, h, w = gt_bitmasks.shape
        h_i = int(h / (2 * grid_num))  # 4: pad 25 2: pad 50
        w_i = int(w / (2 * grid_num))
        pad_gt_bitmasks = F.pad(gt_bitmasks, [w_i, w_i, h_i, h_i], mode='constant', value=0)
        expand_gt_bitmasks = []
        for idx, (c_x, c_y) in enumerate(zip(center_x_gt_boxes, center_y_gt_boxes)):
            i = c_x // (w / grid_num)
            j = c_y // (h / grid_num)
            assert i < grid_num, i
            assert j < grid_num, j
            per_c_x = w_i + (i * 2 + 1) * w_i
            per_c_y = h_i + (j * 2 + 1) * h_i
            x1 = int(per_c_x - mapping_ratio * w_i)
            x2 = int(per_c_x + mapping_ratio * w_i)
            y1 = int(per_c_y - mapping_ratio * h_i)
            y2 = int(per_c_y + mapping_ratio * h_i)
            expand_gt_bitmasks.append(pad_gt_bitmasks[idx, y1:y2, x1:x2].unsqueeze(0))
        expand_gt_bitmasks = torch.cat(expand_gt_bitmasks)

        return expand_gt_bitmasks, expand_gt_bitmasks.shape[1:]

    def crop_and_expand_concate(self, gt_bitmasks, gt_boxes, grid_num=4, grid_inside_num=3, mapping_ratio=1.0):
        if self.mask_out_stride > 1:
            if grid_num == 1:
                return gt_bitmasks, gt_bitmasks.shape[1:]
            resized_gt_boxes = gt_boxes / 4.
            center_x_gt_boxes = ((resized_gt_boxes[:, 2] + resized_gt_boxes[:, 0]) / 2.0).squeeze().cpu().numpy()
            center_y_gt_boxes = ((resized_gt_boxes[:, 3] + resized_gt_boxes[:, 1]) / 2.0).squeeze().cpu().numpy()
            n, h, w = gt_bitmasks.shape
            device = gt_bitmasks.device
            h_i = int(h / (2 * grid_num))  # 4: pad 25 2: pad 50
            w_i = int(w / (2 * grid_num))
            pad_gt_bitmasks = F.pad(gt_bitmasks, [w_i, w_i, h_i, h_i], mode='constant', value=0)
            expand_gt_bitmasks = []
            # grid_map = torch.from_numpy(get_grid_map(grid_inside_num)).to(device=device)
            grid_map = get_grid_map(grid_inside_num)
            for idx, (c_x, c_y) in enumerate(zip(center_x_gt_boxes, center_y_gt_boxes)):
                i = c_x // (w / grid_num)
                j = c_y // (h / grid_num)
                i_list = []
                j_list = []
                inside_c_x = c_x % (w / grid_num)
                inside_c_y = c_y % (h / grid_num)
                inside_i = inside_c_x // round((w / (grid_num * grid_inside_num)) + 0.5)
                inside_j = inside_c_y // round((h / (grid_num * grid_inside_num)) + 0.5)
                inside_ind = int(inside_i + grid_inside_num * inside_j)
                map = grid_map[inside_ind]
                # map[:, 0] = map[:, 0] + i
                # map[:, 1] = map[:, 1] + j
                # valid_ind = (map[:, 0] >=0) & (map[:, 0] < grid_num) & (map[:, 1] >=0) & (map[:, 1] < grid_num)
                # map = map[valid_ind]
                # inside = torch.arange(0, 9)[valid_ind].tolist()
                inside = []
                for i_inside_ind, (x, y) in enumerate(map):
                    if x < 100:
                        _x = i + x
                        _y = j + y
                        if _x >= 0 and _y >= 0 and _x < grid_num and _y < grid_num:
                            i_list.append(_x)
                            j_list.append(_y)
                            inside.append(i_inside_ind)
                expand_gt_bitmasks_per_ins = torch.zeros(9, 2*h_i, 2*w_i).to(dtype=torch.float32, device=device)
                # per_c_x = w_i + (map[:, 0] * 2 + 1) * w_i
                # per_c_y = h_i + (map[:, 1] * 2 + 1) * h_i
                # x1 = per_c_x - mapping_ratio * w_i
                # x2 = per_c_x + mapping_ratio * w_i
                # y1 = per_c_y - mapping_ratio * h_i
                # y2 = per_c_y + mapping_ratio * h_i
                # for per, per_inside in enumerate(inside):
                #     expand_gt_bitmasks_per_ins[per_inside] = \
                #         pad_gt_bitmasks[idx, int(y1[per]):int(y2[per]), int(x1[per]):int(x2[per])]
                for per_i, per_j, per_inside in zip(i_list, j_list, inside):
                    assert per_i < grid_num, per_i
                    assert per_j < grid_num, per_j
                    per_c_x = w_i + (per_i * 2 + 1) * w_i
                    per_c_y = h_i + (per_j * 2 + 1) * h_i
                    x1 = int(per_c_x - mapping_ratio * w_i)
                    x2 = int(per_c_x + mapping_ratio * w_i)
                    y1 = int(per_c_y - mapping_ratio * h_i)
                    y2 = int(per_c_y + mapping_ratio * h_i)
                    expand_gt_bitmasks_per_ins[per_inside] = pad_gt_bitmasks[idx, y1:y2, x1:x2]
                expand_gt_bitmasks.append(expand_gt_bitmasks_per_ins)
            return expand_gt_bitmasks
        else:
            gt_bitmasks = F.interpolate(
                gt_bitmasks.unsqueeze(0), scale_factor=0.25 * grid_num,
                mode='bilinear',
                align_corners=True
            ).squeeze(0)
            # stride = int(4 / grid_num)
            # start = int(stride // 2)
            # gt_bitmasks = gt_bitmasks[:, start::stride, start::stride]
            if grid_num == 1:
                return gt_bitmasks, gt_bitmasks.shape[1:]
            # 2: 0.5  4: 1
            resized_gt_boxes = gt_boxes * (0.25 * grid_num)
            center_x_gt_boxes = ((resized_gt_boxes[:, 2] + resized_gt_boxes[:, 0]) / 2.0).squeeze().cpu().numpy()
            center_y_gt_boxes = ((resized_gt_boxes[:, 3] + resized_gt_boxes[:, 1]) / 2.0).squeeze().cpu().numpy()
            n, h, w = gt_bitmasks.shape
            device = gt_bitmasks.device
            h_i = int(h / (2 * grid_num))  # 4: pad 25 2: pad 50
            w_i = int(w / (2 * grid_num))
            pad_gt_bitmasks = F.pad(gt_bitmasks, [w_i, w_i, h_i, h_i], mode='constant', value=0)
            expand_gt_bitmasks = []
            grid_map = get_grid_map(grid_inside_num)
            for idx, (c_x, c_y) in enumerate(zip(center_x_gt_boxes, center_y_gt_boxes)):
                i = c_x // (w / grid_num)
                j = c_y // (h / grid_num)
                i_list = []
                j_list = []
                inside_c_x = c_x % (w / grid_num)
                inside_c_y = c_y % (h / grid_num)
                inside_i = inside_c_x // round((w / (grid_num * grid_inside_num)) + 0.5)
                inside_j = inside_c_y // round((h / (grid_num * grid_inside_num)) + 0.5)
                inside_ind = int(inside_i + grid_inside_num * inside_j)
                map = grid_map[inside_ind]
                inside = []
                for i_inside_ind, (x, y) in enumerate(map):
                    if x < 100:
                        _x = i + x
                        _y = j + y
                        if _x >= 0 and _y >= 0 and _x < grid_num and _y < grid_num:
                            i_list.append(_x)
                            j_list.append(_y)
                            inside.append(i_inside_ind)
                expand_gt_bitmasks_per_ins = torch.zeros(9, 2 * h_i, 2 * w_i).to(dtype=torch.float32, device=device)
                for per_i, per_j, per_inside in zip(i_list, j_list, inside):
                    assert per_i < grid_num, per_i
                    assert per_j < grid_num, per_j
                    per_c_x = w_i + (per_i * 2 + 1) * w_i
                    per_c_y = h_i + (per_j * 2 + 1) * h_i
                    x1 = int(per_c_x - mapping_ratio * w_i)
                    x2 = int(per_c_x + mapping_ratio * w_i)
                    y1 = int(per_c_y - mapping_ratio * h_i)
                    y2 = int(per_c_y + mapping_ratio * h_i)
                    expand_gt_bitmasks_per_ins[per_inside] = pad_gt_bitmasks[idx, y1:y2, x1:x2]
                expand_gt_bitmasks.append(expand_gt_bitmasks_per_ins)
            return expand_gt_bitmasks

    def recover_ins2all(self, mask_feats, mask_scores, pred_instances, inds):
        _, _, H, W = mask_feats.shape
        device = mask_feats.device
        N = pred_instances.pred_boxes.tensor.shape[0]
        locations = pred_instances.locations
        mask_scores_all = []
        # 1: 200 -> 800
        # 2: 200 -> 400 -> 800
        # 4: 200 -> 800 -> 800
        for idx_all, mask_scores_per_level in enumerate(mask_scores):
            if mask_scores_per_level is None:
                mask_scores_all.append(None)
                continue
            if idx_all == 0 or self.grid_num[idx_all] == 1:
                mask_scores_all.append(mask_scores_per_level)
                continue
            n, _, h, w = mask_scores_per_level.shape
            if idx_all == 1:
                i_h = H
                i_w = W
            elif idx_all == 2:
                i_h = int(H / 2)
                i_w = int(W / 2)
            else:
                return 0

            recover_mask_scores = []
            per_locations = locations[inds[idx_all]].cpu().numpy() / 4.
            for idx, center in enumerate(per_locations):
                c_x, c_y = center
                i = (c_x - 0.5) // i_w
                j = (c_y - 0.5) // i_h
                assert i < self.grid_num[idx_all], (i, c_x, i_w, w)
                assert j < self.grid_num[idx_all], (j, c_y, i_h, h)
                w_l = int(i * i_w)  # 25x
                w_r = int((self.grid_num[idx_all] - i - 1) * i_w)
                h_u = int(j * i_h)
                h_d = int((self.grid_num[idx_all] - j - 1) * i_h)
                recover_mask = F.pad(mask_scores_per_level[idx], [w_l, w_r, h_u, h_d], mode='constant', value=0)
                recover_mask_scores.append(recover_mask)
            recover_mask_scores = torch.cat(recover_mask_scores)
            mask_scores_all.append(recover_mask_scores)

        pred_global_masks = torch.zeros([N, 1, 2 * H, 2 * W]).to(device=device)
        for ind, final_mask_scores_per_level in zip(inds, mask_scores_all):
            if final_mask_scores_per_level is not None:
                pred_global_masks[ind] = final_mask_scores_per_level.float()

        pred_instances.pred_global_masks = pred_global_masks
        return pred_instances

    def recover_ins2all_concat(self, mask_feats, mask_scores, pred_instances, inds, gt_inds, grid_inside_num=3):
        _, _, H, W = mask_feats.shape
        device = mask_feats.device
        N = pred_instances.pred_boxes.tensor.shape[0]
        locations = pred_instances.locations
        mask_scores_all = []
        # 1: 200 -> 800
        # 2: 200 -> 400 -> 800
        # 2: 200 -> 800 -> 800
        for idx_all, mask_scores_per_level in enumerate(mask_scores):
            if mask_scores_per_level is None:
                mask_scores_all.append(None)
                continue
            if idx_all == 0:
                # mask_scores_per_level = F.interpolate(
                #     mask_scores_per_level, scale_factor=int(4 / self.grid_num[idx_all]),
                #     mode='bilinear',
                #     align_corners=True
                # )
                # mask_scores_per_level = aligned_bilinear(mask_scores_per_level, 4)
                mask_scores_all.append(mask_scores_per_level)
                continue
            n, _, h, w = mask_scores_per_level.shape
            per_locations = locations[inds[idx_all][0]][inds[idx_all][1]].cpu().numpy() / 4.
            pred_boxes = pred_instances.pred_boxes.tensor[inds[idx_all][0]][
                             inds[idx_all][1]].cpu().numpy() / 4.  # 1
            if idx_all == 1:
                i_h = H
                i_w = W
            elif idx_all == 2:
                i_h = int(H / 2)
                i_w = int(W / 2)
            else:
                return 0
            # if idx_all == 1:
            #     per_locations = locations[inds[idx_all][0]][inds[idx_all][1]].cpu().numpy() / 2.
            #     pred_boxes = pred_instances.pred_boxes.tensor[inds[idx_all][0]][
            #                      inds[idx_all][1]].cpu().numpy() / 2.  # 1
            #     i_h = 2 * H
            #     i_w = 2 * W
            # elif idx_all == 2:
            #     per_locations = locations[inds[idx_all][0]][inds[idx_all][1]].cpu().numpy()
            #     pred_boxes = pred_instances.pred_boxes.tensor[inds[idx_all][0]][
            #                      inds[idx_all][1]].cpu().numpy()
            #     i_h = 2 * H
            #     i_w = 2 * W
            # else:
            #     return 0

            loc_map = {0: (0, 0),
                       1: (-1, 0),
                       2: (-1, -1),
                       3: (0, -1),
                       4: (1, -1),
                       5: (1, 0),
                       6: (1, 1),
                       7: (0, 1),
                       8: (-1, 1)}

            recover_mask_scores = []
            for idx, (box, center, gt_ind_per_ins) in enumerate(zip(pred_boxes, per_locations, gt_inds[idx_all])):
                c_x, c_y = center
                i = c_x // i_w
                j = c_y // i_h
                ins_id = int(gt_ind_per_ins // 9) + 1
                loc_id = int(gt_ind_per_ins % 9)
                delata = loc_map[loc_id]
                i = i + delata[0]
                j = j + delata[1]
                assert i < self.grid_num[idx_all], (idx_all, idx, i, c_x, i_w, w, delata[0])
                assert j < self.grid_num[idx_all], (idx_all, idx, j, c_y, i_h, h, delata[1])
                recover_mask = torch.zeros([1, 1, 2 * H, 2 * W]).to(device=device)
                _, r_h, r_w = mask_scores_per_level[idx].shape
                recover_mask[:, :, int(j * i_h):int(j * i_h + r_h), int(i * i_w):int(i * i_w + r_w)] = mask_scores_per_level[idx]
                clean_mask = torch.zeros_like(recover_mask)
                _, _, r_h, r_w = clean_mask.shape
                delta = 0
                x1 = max(0, box[0] - delta)
                x2 = min(r_w - 1, box[2] + delta)
                y1 = max(0, box[1] - delta)
                y2 = min(r_h - 1, box[3] + delta)
                clean_mask[:, :, int(y1 + 0.5):int(y2 + 0.5), int(x1 + 0.5):int(x2 + 0.5)] = 1
                recover_mask = recover_mask * clean_mask
                if ins_id > len(recover_mask_scores):
                    recover_mask_scores.append(recover_mask)
                else:
                    recover_mask_scores[-1] = recover_mask_scores[-1] + recover_mask
            recover_mask_scores = torch.cat(recover_mask_scores)
            # if idx_all == 1:
            #     recover_mask_scores = F.interpolate(
            #         recover_mask_scores, scale_factor=2,
            #         mode='bilinear',
            #         align_corners=True
            #     )
                # recover_mask_scores = aligned_bilinear(recover_mask_scores, 2)
            mask_scores_all.append(recover_mask_scores)

        pred_global_masks = torch.zeros([N, 1, 2 * H, 2 * W]).to(device=device)
        n_ins = 0
        for ind, final_mask_scores_per_level in zip(inds, mask_scores_all):
            if final_mask_scores_per_level is not None:
                n_ins += final_mask_scores_per_level.shape[0]
                pred_global_masks[ind[0]] = final_mask_scores_per_level.float()
        assert n_ins == N, (N, n_ins)

        pred_instances.pred_global_masks = pred_global_masks
        return pred_instances

    def recover_ins2all_concat_old(self, mask_feats, mask_scores, pred_instances, inds, gt_inds, grid_inside_num=3):
        # 2grid 1map: 50x50 i_w: 100
        # 2grid 2map: 100x100
        _, _, H, W = mask_feats.shape
        device = mask_feats.device
        N = pred_instances.pred_boxes.tensor.shape[0]
        locations = pred_instances.locations
        mask_scores_all = []
        for idx_all, mask_scores_per_level in enumerate(mask_scores):
            if mask_scores_per_level is None:
                mask_scores_all.append(None)
                continue
            if idx_all == 0:
                mask_scores_all.append(mask_scores_per_level)
                continue
            n, _, h, w = mask_scores_per_level.shape
            if idx_all == 1:
                i_h = H
                i_w = W
            elif idx_all == 2:
                i_h = int(H / 2)
                i_w = int(W / 2)
            else:
                return 0

            # loc_map = torch.tensor([[0, 0],
            #                         [-1, 0],
            #                         [-1, -1],
            #                         [0, -1],
            #                         [1, -1],
            #                         [1, 0],
            #                         [1, 1],
            #                         [0, 1],
            #                         [-1, 1]]).to(device=device)
            # per_locations = locations[inds[idx_all][0]][inds[idx_all][1]] / 4.
            # per_locations_i = per_locations[:, 0] // i_w
            # per_locations_j = per_locations[:, 1] // i_h
            # ins_ids = 1 + (gt_inds[idx_all] // 9)
            # loc_ids = gt_inds[idx_all] % 9 # n, 1
            # delatas = loc_map[loc_ids]
            # per_locations_i = per_locations_i + delatas[:, 0]
            # per_locations_j = per_locations_j + delatas[:, 1]
            # w_l = per_locations_i * i_w
            # w_r = (self.grid_num[idx_all] - per_locations_i - 1) * i_w
            # h_u = per_locations_j * i_h
            # h_d = (self.grid_num[idx_all] - per_locations_j - 1) * i_h
            # recover_mask_scores = []
            # for i in range(per_locations.shape[0]):
            #     recover_mask = F.pad(
            #         mask_scores_per_level[i],
            #         [int(w_l[i]), int(w_r[i]), int(h_u[i]), int(h_d[i])],
            #         mode='constant',
            #         value=0
            #     ).unsqueeze(0)
            #     if ins_ids[i] > len(recover_mask_scores):
            #         recover_mask_scores.append(recover_mask)
            #     else:
            #         recover_mask_scores[-1] = recover_mask_scores[-1] + recover_mask
            # assert locations[inds[idx_all][0]].shape[0] == len(recover_mask_scores), \
            #     (locations[inds[idx_all][0]].shape[0], len(recover_mask_scores))
            # recover_mask_scores = torch.cat(recover_mask_scores)
            # mask_scores_all.append(recover_mask_scores)
            loc_map = {0: (0, 0),
                       1: (-1, 0),
                       2: (-1, -1),
                       3: (0, -1),
                       4: (1, -1),
                       5: (1, 0),
                       6: (1, 1),
                       7: (0, 1),
                       8: (-1, 1)}
            per_locations = locations[inds[idx_all][0]][inds[idx_all][1]].cpu().numpy() / 4.
            pred_boxes = pred_instances.pred_boxes.tensor[inds[idx_all][0]][inds[idx_all][1]].cpu().numpy() / 4.  # 1

            recover_mask_scores = []
            for idx, (box, center, gt_ind_per_ins) in enumerate(zip(pred_boxes, per_locations, gt_inds[idx_all])):
                c_x, c_y = center
                i = c_x // i_w
                j = c_y // i_h
                ins_id = int(gt_ind_per_ins // 9) + 1
                loc_id = int(gt_ind_per_ins % 9)
                delata = loc_map[loc_id]
                i = i + delata[0]
                j = j + delata[1]
                assert i < self.grid_num[idx_all], (idx_all, idx, i, c_x, i_w, w, delata[0])
                assert j < self.grid_num[idx_all], (idx_all, idx, j, c_y, i_h, h, delata[1])
                w_l = int(i * i_w)  # 25x
                w_r = int((self.grid_num[idx_all] - i - 1) * i_w)
                h_u = int(j * i_h)
                h_d = int((self.grid_num[idx_all] - j - 1) * i_h)
                recover_mask = F.pad(mask_scores_per_level[idx], [w_l, w_r, h_u, h_d], mode='constant', value=0).unsqueeze(0)
                clean_mask = torch.zeros_like(recover_mask)
                clean_mask[:, :, int(box[1]):int(box[3]), int(box[0]):int(box[2])] = 1
                recover_mask = recover_mask * clean_mask
                if ins_id > len(recover_mask_scores):
                    recover_mask_scores.append(recover_mask)
                else:
                    recover_mask_scores[-1] = recover_mask_scores[-1] + recover_mask
            recover_mask_scores = torch.cat(recover_mask_scores)
            mask_scores_all.append(recover_mask_scores)
        pred_global_masks = torch.zeros([N, 1, 2 * H, 2 * W]).to(device=device)
        n_ins = 0
        for ind, final_mask_scores_per_level in zip(inds, mask_scores_all):
            if final_mask_scores_per_level is not None:
                n_ins += final_mask_scores_per_level.shape[0]
                pred_global_masks[ind[0]] = final_mask_scores_per_level.float()
        assert n_ins == N, (N, n_ins)

        pred_instances.pred_global_masks = pred_global_masks
        return pred_instances

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
