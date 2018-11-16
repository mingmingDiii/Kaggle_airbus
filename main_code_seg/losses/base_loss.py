import torch.nn as nn
import torch.nn.functional as F
import torch
from main_code_seg.losses import lovasz_losses
# class CrossEntropyLoss2d(nn.Module):
#     def __init__(self, weight=None, size_average=True, ignore_index=-1):
#         super(CrossEntropyLoss2d, self).__init__()
#         self.nll_loss = nn.NLLLoss(weight, size_average, ignore_index)
#
#     def forward(self, inputs, targets):
#         return self.nll_loss(F.log_softmax(inputs, dim=1), targets)


class LossBinary:
    """
     Implementation from  https://github.com/ternaus/robot-surgery-segmentation
    """

    def __init__(self, jaccard_weight=0):
        self.nll_loss = nn.BCEWithLogitsLoss()
        self.jaccard_weight = jaccard_weight

    def __call__(self, outputs, targets):
        loss = self.nll_loss(outputs, targets)

        if self.jaccard_weight:
            eps = 1e-15
            jaccard_target = (targets == 1.0).float()
            jaccard_output = F.sigmoid(outputs)

            intersection = (jaccard_output * jaccard_target).sum()
            union = jaccard_output.sum() + jaccard_target.sum()

            loss -= self.jaccard_weight * torch.log((intersection + eps) / (union - intersection + eps))
        return loss



def lov_criterion(logit,truth):
    logit = logit.squeeze(1)
    truth = truth.squeeze(1)
    loss = lovasz_losses.lovasz_hinge(logit,truth,per_image=True,ignore=None)
    return loss

def slov_criterion(logit,truth):
    logit = logit.squeeze(1)
    truth = truth.squeeze(1)
    loss1 = lovasz_losses.lovasz_hinge(logit,truth,per_image=True,ignore=None)
    loss2 = lovasz_losses.lovasz_hinge(-logit,1.0-truth,per_image=True,ignore=None)

    loss = (loss1+loss2)/2.0

    return loss


class LovLoss(nn.Module):
    def __init__(self):
        super(LovLoss, self).__init__()

    def forward(self, logits,targets):
        loss = lov_criterion(logits,targets)

        return loss

class SLovLoss(nn.Module):
    def __init__(self):
        super(SLovLoss, self).__init__()

    def forward(self, logits,targets):
        loss = slov_criterion(logits,targets)
        return loss








class SoftDiceLoss(nn.Module):
    def __init__(self,smooth = 1.0,eps = 1e-7):
        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth
        self.eps = eps
        
    def forward(self, logits,targets):
        bts = targets.size(0)
        probs = F.sigmoid(logits)

        pred_flat = probs.view(bts,-1)
        target_flat = targets.view(bts,-1)
        intersection = (pred_flat*target_flat)


        score = 2.0* (intersection.sum(1)+self.smooth)/(pred_flat.sum(1)+target_flat.sum(1)+self.smooth+self.eps)

        score = 1.0-score.mean()

        return score


class Mix_softDice_bce(nn.Module):
    def __init__(self,dice_w=0.5,bce_w=0.5):
        super(Mix_softDice_bce, self).__init__()

        self.dice_w = dice_w
        self.bce_w = bce_w

    def forward(self,logits,targets):

        softDice_loss = SoftDiceLoss()(logits=logits,targets=targets)
        bce_loss = nn.BCEWithLogitsLoss()(logits,targets)

        all_loss = self.dice_w*softDice_loss+self.bce_w*bce_loss


        return all_loss



class Weighted_bce(nn.Module):
    def __init__(self):
        super(Weighted_bce, self).__init__()

    def forward(self, logits,targets,weights):

        bce_loss = nn.BCEWithLogitsLoss(reduce=False,reduction=None)(logits,targets)

        weighted_loss = bce_loss*weights

        weighted_loss = torch.mean(weighted_loss)

        return weighted_loss



class Mix_softDice_bce_lov(nn.Module):
    def __init__(self,dice_w=0.3,bce_w=0.3,lov_w = 0.4):
        super(Mix_softDice_bce_lov, self).__init__()

        self.dice_w = dice_w
        self.bce_w = bce_w
        self.lov_w = lov_w

    def forward(self,logits,targets):

        softDice_loss = SoftDiceLoss()(logits=logits,targets=targets)
        bce_loss = nn.BCEWithLogitsLoss()(logits,targets)
        lov_loss = LovLoss()(logits,targets)

        all_loss = self.dice_w*softDice_loss+self.bce_w*bce_loss+self.lov_w*lov_loss


        return all_loss


class Mix_softDice_wbce_lov(nn.Module):
    def __init__(self,dice_w=0.3,bce_w=0.3,lov_w = 0.4):
        super(Mix_softDice_wbce_lov, self).__init__()

        self.dice_w = dice_w
        self.bce_w = bce_w
        self.lov_w = lov_w

    def forward(self,logits,targets,weights):

        softDice_loss = SoftDiceLoss()(logits=logits,targets=targets)
        wbce_loss = Weighted_bce()(logits,targets,weights)
        lov_loss = LovLoss()(logits,targets)

        all_loss = self.dice_w*softDice_loss+self.bce_w*wbce_loss+self.lov_w*lov_loss


        return all_loss


class Mix_softDice_bce_slov(nn.Module):
    def __init__(self,dice_w=0.3,bce_w=0.3,lov_w = 0.4):
        super(Mix_softDice_bce_slov, self).__init__()

        self.dice_w = dice_w
        self.bce_w = bce_w
        self.lov_w = lov_w

    def forward(self,logits,targets):

        softDice_loss = SoftDiceLoss()(logits=logits,targets=targets)
        bce_loss = nn.BCEWithLogitsLoss()(logits,targets)
        lov_loss = SLovLoss()(logits,targets)

        all_loss = self.dice_w*softDice_loss+self.bce_w*bce_loss+self.lov_w*lov_loss


        return all_loss