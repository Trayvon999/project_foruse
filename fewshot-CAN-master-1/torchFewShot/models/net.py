import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchFewShot.models.resnet12 import resnet12
from torchFewShot.models.cam import MultiHeadCrossAttention as CAM


def one_hot(labels_train):
    labels_train = labels_train.cpu()
    nKnovel = 5
    labels_train_1hot_size = list(labels_train.size()) + [nKnovel]
    labels_train_unsqueeze = labels_train.unsqueeze(dim=labels_train.dim())
    labels_train_1hot = torch.zeros(labels_train_1hot_size).scatter_(
        len(labels_train_1hot_size) - 1, labels_train_unsqueeze, 1
    )
    return labels_train_1hot


class Model(nn.Module):
    def __init__(self, scale_cls, iter_num_prob=35.0 / 75, num_classes=64):
        super(Model, self).__init__()
        self.scale_cls = scale_cls
        self.iter_num_prob = iter_num_prob

        self.base = resnet12()
        self.cam = CAM()

        self.nFeat = self.base.nFeat
        self.clasifier = nn.Conv2d(self.nFeat, num_classes, kernel_size=1)

    def test(self, ftrain_vec, ftest_vec):
        ftest = F.normalize(ftest_vec, p=2, dim=-1)
        ftrain = F.normalize(ftrain_vec, p=2, dim=-1)
        scores = self.scale_cls * torch.sum(ftest * ftrain, dim=-1)
        return scores

    def forward(self, xtrain, xtest, ytrain, ytest):

        batch_size, num_train = xtrain.size(0), xtrain.size(1)
        num_test = xtest.size(1)
        K = ytrain.size(2)

        ytrain = ytrain.transpose(1, 2)   # [B,5,K]

        # ---------------------------------------------------------
        # Encode images
        # ---------------------------------------------------------
        xtrain = xtrain.view(-1, 3, 84, 84)
        xtest = xtest.view(-1, 3, 84, 84)
        x = torch.cat((xtrain, xtest), 0)
        f = self.base(x)    # [B*(5+N),512,6,6]

        # Support set reshape
        ftrain = f[: batch_size * num_train]
        ftrain = ftrain.view(batch_size, num_train, *ftrain.shape[1:])  # [B,5,512,6,6]

        # Class prototype averaging
        ftrain_flat = ftrain.view(batch_size, num_train, -1)
        f_proto = torch.bmm(ytrain, ftrain_flat)
        f_proto = f_proto / ytrain.sum(2).unsqueeze(-1)
        f_proto = f_proto.view(batch_size, -1, ftrain.size(2), ftrain.size(3), ftrain.size(4))

        ftrain = f_proto.clone()

        # Query reshape
        ftest = f[batch_size * num_train:].view(
            batch_size, num_test, *ftrain.shape[2:]
        )

        # ---------------------------------------------------------
        # CAM
        # ---------------------------------------------------------
        ftrain, ftest = self.cam(ftrain, ftest)

        # ---------------------------------------------------------
        # Generate pooled vectors for prototype matching
        # ---------------------------------------------------------
        ftrain_vec = ftrain.mean(dim=[3, 4])   # [B,5,512]
        ftest_vec  = ftest.mean(dim=[3, 4])    # [B,N,512]

        # ---------------------------------------------------------
        # Testing mode: return only matching scores
        # ---------------------------------------------------------
        if not self.training:
            return self.test(ftrain_vec, ftest_vec)

        # ---------------------------------------------------------
        # Training classification (prototype matching)
        # ---------------------------------------------------------
        ftrain_norm = F.normalize(ftrain_vec, p=2, dim=-1)
        ftest_norm  = F.normalize(ftest_vec,  p=2, dim=-1)

        ftrain_exp = ftrain_norm.unsqueeze(1)   # [B,1,5,512]
        ftest_exp  = ftest_norm.unsqueeze(2)    # [B,N,1,512]

        cls_scores = torch.sum(ftest_exp * ftrain_exp, dim=3)   # [B,N,5]
        cls_scores = self.scale_cls * cls_scores
        cls_scores = cls_scores.view(batch_size * num_test, -1)

        # ---------------------------------------------------------
        # CNN classification head (requires spatial maps)
        # ---------------------------------------------------------
        ftest_map = ftest.view(batch_size * num_test, 512, 6, 6)
        ytest_out = self.clasifier(ftest_map)     # [B*N,64,6,6]

        return ytest_out, cls_scores

    # ======================================================================
    # Helper for Transductive testing
    # ======================================================================
    def helper(self, ftrain, ftest, ytrain):
        b, n, c, h, w = ftrain.size()
        k = ytrain.size(2)

        ytrain_t = ytrain.transpose(1, 2)

        ftrain_flat = ftrain.view(b, n, -1)
        f_proto = torch.bmm(ytrain_t, ftrain_flat)
        f_proto = f_proto.div(ytrain_t.sum(2, keepdim=True))
        f_proto = f_proto.view(b, k, c, h, w)

        ftrain, ftest = self.cam(f_proto, ftest)

        ftrain_vec = ftrain.mean(dim=[3, 4])
        ftest_vec  = ftest.mean(dim=[3, 4])

        ftest_vec = F.normalize(ftest_vec, p=2, dim=-1)
        ftrain_vec = F.normalize(ftrain_vec, p=2, dim=-1)

        scores = self.scale_cls * torch.sum(ftest_vec * ftrain_vec, dim=-1)
        return scores

    # ======================================================================
    # Transductive test
    # ======================================================================
    def test_transductive(self, xtrain, xtest, ytrain, ytest):

        iter_num_prob = self.iter_num_prob
        batch_size, num_train = xtrain.size(0), xtrain.size(1)
        num_test = xtest.size(1)
        K = ytrain.size(2)

        xtrain = xtrain.view(-1, 3, 84, 84)
        xtest = xtest.view(-1, 3, 84, 84)
        x = torch.cat((xtrain, xtest), 0)
        f = self.base(x)

        ftrain = f[: batch_size * num_train].view(batch_size, num_train, *f.size()[1:])
        ftest  = f[batch_size * num_train:].view(batch_size, num_test, *f.size()[1:])

        cls_scores = self.helper(ftrain, ftest, ytrain)

        num_images_per_iter = int(num_test * iter_num_prob)
        num_iter = num_test // num_images_per_iter

        for i in range(num_iter):

            max_scores, preds = torch.max(cls_scores, 2)

            top_idx = torch.argsort(max_scores.view(-1), descending=True)
            top_idx = top_idx[: num_images_per_iter * (i + 1)]

            ftest_iter = ftest[0, top_idx].unsqueeze(0)
            preds_iter = preds[0, top_idx].unsqueeze(0)
            preds_iter = one_hot(preds_iter).cuda()

            ftrain_iter = torch.cat((ftrain, ftest_iter), 1)
            ytrain_iter = torch.cat((ytrain, preds_iter), 1)

            cls_scores = self.helper(ftrain_iter, ftest, ytrain_iter)

        return cls_scores
