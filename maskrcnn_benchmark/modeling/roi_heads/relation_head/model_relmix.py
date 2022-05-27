import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from maskrcnn_benchmark.modeling.utils import cat
RG = np.random.default_rng()
import logging


class Cumix_head(nn.Module):
    def __init__(self, in_channels, config):
        super(Cumix_head, self).__init__()
        self.cfg = config

    def manual_CE(self, predictions, labels):
        loss = -torch.mean(torch.sum(labels * torch.log_softmax(predictions,dim=1),dim=1))
        return loss

    def add_hubness_loss(self, cls_scores):
        # xp_yall_prob   (batch_size, num_classes)
        # xp_yall_prob.T (num_classes, batch_size
        # xp_yall_prob.expand(0, 1, -1, 1)
        # xp_yall_probT_average_reshape = xp_yall_probT_reshaped.mean(axis=2)
        # hubness_dist = xp_yall_probT_average_reshape - hubness_blob
        # hubness_dist_sqr = hubness_dist.pow(2)
        # hubness_dist_sqr_scaled = hubness_dist_sqr * cfg.TRAIN.HUBNESS_SCALiE
        cls_scores = F.softmax(cls_scores, dim=1)
        hubness_blob = 1./cls_scores.size(1)
        cls_scores_T = cls_scores.transpose(0, 1)
        cls_scores_T = cls_scores_T.unsqueeze(1).unsqueeze(3).expand(-1, 1, -1, 1)
        cls_scores_T = cls_scores_T.mean(dim=2, keepdim=True)
        hubness_dist = cls_scores_T - hubness_blob
        hubness_dist = hubness_dist.pow(2) * self.cfg.MODEL.HUBNESS_SCALE
        hubness_loss = hubness_dist.mean()
        return hubness_loss

    # spo_feat is concatenation of SPO
    def forward(self, sub, sub_l, obj, obj_l, prd_vis_embeddings, prd_labels, prd_weights):
        device_id = prd_vis_embeddings.get_device()
        mixed_prd_cls_scores = None
        mixed_prd_labels = None

        if self.cfg.MODEL.CUMIX and self.training:
            bs = prd_vis_embeddings.shape[0]
            probs = prd_weights[prd_labels].cpu().numpy()
            indices = torch.from_numpy(np.argsort(-probs, kind='quicksort')).cuda(device_id)    ### in decreasing order, means tail classes are ahead
            if self.cfg.MODEL.AUG_PERCENT == 50:
                partition_num = bs // 2
                indices_1 = indices[0:partition_num]
                indices_2 = np.random.permutation(indices_1)
                indices_3 = indices[partition_num: partition_num*2]
            elif self.cfg.MODEL.AUG_PERCENT == 70:
                partition_num = int(bs * 0.7)
                indices_1 = indices[0:partition_num]
                indices_2 = np.random.permutation(indices_1)
                indices_3 = np.random.permutation(indices_1)
            elif self.cfg.MODEL.AUG_PERCENT == 60:
                partition_num = int(bs * 0.6)
                indices_1 = indices[0:partition_num]
                indices_2 = np.random.permutation(indices_1)
                indices_3 = np.random.permutation(indices_1)
            else:
                partition_num = bs // 3
                indices_1 = indices[0:partition_num]
                indices_2 = indices[partition_num : partition_num*2]
                indices_3 = indices[partition_num*2 : partition_num*3]

            mix_sub_embs, mix_sub_labels, mix_obj_embs, mix_obj_labels, mixed_prd_embeddings, mixed_prd_labels  = \
                cumix(sub, sub_l, obj, obj_l, prd_vis_embeddings, prd_labels, indices_1, indices_2, indices_3, device_id, self.cfg)
        return mix_sub_embs, mix_sub_labels, mix_obj_embs, mix_obj_labels, mixed_prd_embeddings, mixed_prd_labels


def create_one_hot(y, classes, device_id):
    y_onehot = torch.FloatTensor(y.size(0), classes).cuda(device_id)
    y_onehot.zero_()
    y_onehot.scatter_(1, y.view(-1, 1), 1)
    return y_onehot


def cumix(sbj_vis_embeddings, sbj_labels, obj_vis_embeddings, obj_labels, prd_vis_embeddings, prd_labels, indices_1, indices_2, indices_3, device_id, cfg):
    if cfg.MODEL.RANDOM_LAMBDA:
        lamda = torch.from_numpy(RG.beta(0.8, 0.8, [indices_1.shape[0], 1])).float().cuda(device_id)
    else:
        lamda = 0.65

    alpha1 = torch.randint(0, 2, [indices_1.shape[0], 1]).cuda(device_id)
    # alpha2 = torch.randint(0, 2, [indices_1.shape[0]]).cuda(device_id)
    # alpha3 = torch.randint(0, 2, [indices_1.shape[0]]).cuda(device_id)

    if cfg.MODEL.MIXUP:
        #mixed_sbj_embeddings = lamda * sbj_vis_embeddings[indices_1] + (1 - lamda)*(sbj_vis_embeddings[indices_2])
        #mixed_obj_embeddings = lamda * obj_vis_embeddings[indices_1] + (1 - lamda)*(obj_vis_embeddings[indices_2])
        mixed_prd_embeddings = lamda * prd_vis_embeddings[indices_1] + (1 - lamda)*(prd_vis_embeddings[indices_2])

        #sbj_one_hot_labels = create_one_hot(sbj_labels, cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES -1, device_id)
        prd_one_hot_labels = create_one_hot(prd_labels, cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES, device_id)
        #obj_one_hot_labels = create_one_hot(obj_labels, cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES -1, device_id)

        #mixed_sbj_labels = lamda * sbj_one_hot_labels[indices_1] + (1 - lamda)*(sbj_one_hot_labels[indices_2])
        #mixed_obj_labels = lamda * obj_one_hot_labels[indices_1] + (1 - lamda)*(obj_one_hot_labels[indices_2])
        mixed_prd_labels = lamda * prd_one_hot_labels[indices_1] + (1 - lamda)*(prd_one_hot_labels[indices_2])
    else:
        mixed_sbj_embeddings = lamda * sbj_vis_embeddings[indices_1] + (1 - lamda)*(alpha1 * sbj_vis_embeddings[indices_2] + \
              (1 - alpha1) * sbj_vis_embeddings[indices_3])
        mixed_obj_embeddings = lamda * obj_vis_embeddings[indices_1] + (1 - lamda)*(alpha1 * obj_vis_embeddings[indices_2] + \
              (1 - alpha1) * obj_vis_embeddings[indices_3])
        mixed_prd_embeddings = lamda * prd_vis_embeddings[indices_1] + (1 - lamda)*(alpha1 * prd_vis_embeddings[indices_2] + \
               (1 - alpha1) * prd_vis_embeddings[indices_3])

        sbj_one_hot_labels = create_one_hot(sbj_labels, cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES, device_id)
        prd_one_hot_labels = create_one_hot(prd_labels, cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES, device_id)
        obj_one_hot_labels = create_one_hot(obj_labels, cfg.MODEL.ROI_BOX_HEAD.NUM_CLASSES, device_id)

        mixed_sbj_labels = lamda * sbj_one_hot_labels[indices_1] + (1 - lamda)*(alpha1 * sbj_one_hot_labels[indices_2] + \
               (1 - alpha1) * sbj_one_hot_labels[indices_3])
        mixed_obj_labels = lamda * obj_one_hot_labels[indices_1] + (1 - lamda)*(alpha1 * obj_one_hot_labels[indices_2] + \
               (1 - alpha1) * obj_one_hot_labels[indices_3])
        mixed_prd_labels = lamda * prd_one_hot_labels[indices_1] + (1 - lamda)*(alpha1 * prd_one_hot_labels[indices_2] + \
                (1 - alpha1) * prd_one_hot_labels[indices_3])
    
    # return mixed_prd_embeddings, mixed_prd_labels
    return mixed_sbj_embeddings, mixed_sbj_labels, mixed_obj_embeddings, mixed_obj_labels, mixed_prd_embeddings, mixed_prd_labels


def manual_CE(predictions, labels):
    loss = -torch.mean(torch.sum(labels * torch.log_softmax(predictions,dim=1),dim=1))
    return loss

def manual_log_softmax(pred, weight):
    e_x = torch.exp(pred - torch.max(pred))
    # print('!! e_x shape !! ', e_x.shape)
    # print('!! weight shape !! ', weight.shape)
    return e_x / torch.sum(weight * e_x, 1).unsqueeze(1)

def add_hubness_loss(cls_scores, cfg):
    # xp_yall_prob   (batch_size, num_classes)
    # xp_yall_prob.T (num_classes, batch_size
    # xp_yall_prob.expand(0, 1, -1, 1)
    # xp_yall_probT_average_reshape = xp_yall_probT_reshaped.mean(axis=2)
    # hubness_dist = xp_yall_probT_average_reshape - hubness_blob
    # hubness_dist_sqr = hubness_dist.pow(2)
    # hubness_dist_sqr_scaled = hubness_dist_sqr * cfg.TRAIN.HUBNESS_SCALE
    cls_scores = F.softmax(cls_scores, dim=1)
    hubness_blob = 1./cls_scores.size(1)
    cls_scores_T = cls_scores.transpose(0, 1)
    cls_scores_T = cls_scores_T.unsqueeze(1).unsqueeze(3).expand(-1, 1, -1, 1)
    cls_scores_T = cls_scores_T.mean(dim=2, keepdim=True)
    hubness_dist = cls_scores_T - hubness_blob
    hubness_dist = hubness_dist.pow(2) * cfg.TRAIN.HUBNESS_SCALE
    hubness_loss = hubness_dist.mean()
    return hubness_loss


def reldn_losses(prd_cls_scores, prd_labels_int32, cfg, fg_only=False, weight=None):
    device_id = prd_cls_scores.get_device()
    prd_labels = Variable(torch.from_numpy(prd_labels_int32.astype('int64'))).cuda(device_id)
    if cfg.MODEL.LOSS == 'weighted_cross_entropy' or cfg.MODEL.LOSS == 'weighted_focal':
        weight = Variable(torch.from_numpy(weight)).cuda(device_id)
    loss_cls_prd = add_cls_loss(prd_cls_scores, prd_labels, weight=weight)
    # class accuracy
    prd_cls_preds = prd_cls_scores.max(dim=1)[1].type_as(prd_labels)
    accuracy_cls_prd = prd_cls_preds.eq(prd_labels).float().mean(dim=0)

    return loss_cls_prd, accuracy_cls_prd


def reldn_so_losses(sbj_cls_scores, obj_cls_scores, sbj_labels_int32, obj_labels_int32, cfg):
    device_id = sbj_cls_scores.get_device()

    sbj_labels = Variable(torch.from_numpy(sbj_labels_int32.astype('int64'))).cuda(device_id)
    loss_cls_sbj = add_cls_loss(sbj_cls_scores, sbj_labels)
    sbj_cls_preds = sbj_cls_scores.max(dim=1)[1].type_as(sbj_labels)
    accuracy_cls_sbj = sbj_cls_preds.eq(sbj_labels).float().mean(dim=0)
    
    obj_labels = Variable(torch.from_numpy(obj_labels_int32.astype('int64'))).cuda(device_id)
    loss_cls_obj = add_cls_loss(obj_cls_scores, obj_labels)
    obj_cls_preds = obj_cls_scores.max(dim=1)[1].type_as(obj_labels)
    accuracy_cls_obj = obj_cls_preds.eq(obj_labels).float().mean(dim=0)
    
    return loss_cls_sbj, loss_cls_obj, accuracy_cls_sbj, accuracy_cls_obj

def get_freq_from_dict(freq_dict, categories):
    freqs = np.zeros(len(categories))
    for i, cat in enumerate(categories):
        if cat in freq_dict.keys():
            freqs[i] = freq_dict[cat]
        else:
            freqs[i] = 0
    return freqs

# for ii in range(5):
#     bgnn = torch.load(a[ii])
#     c = ['labels', 'attributes', 'pred_labels', 'pred_scores', 'rel_pair_idxs', 'pred_rel_scores', 'pred_rel_labels']
#     preds = bgnn['predictions']

# import torch
# a  = torch.load('eval_results.pytorch')
# c = ['labels', 'attributes', 'pred_labels', 'pred_scores', 'rel_pair_idxs', 'pred_rel_scores', 'pred_rel_labels']

# for x in range(len(a['predictions'])):
#     for y in a['predictions'][x].__dict__['extra_fields'].keys():
#         if y in c:
#             continue
#         else:
#             a['predictions'][x].__dict__['extra_fields'][y]=None



# torch.save(a, '25000_35000.pytorch')