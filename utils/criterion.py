import torch
from torch import nn
from torch.nn.modules.loss import _Loss
from sklearn.metrics import f1_score


class Criterion(_Loss):
    def __init__(self, opt):
        way = opt.ways
        shot = opt.shots
        super(Criterion, self).__init__()
        self.amount = way * shot

    def forward(self, probs, target,num_support=None):  # (Q,C) (Q)
        if num_support is None:
            num_support = self.amount
        target = target[num_support:]
        target_onehot = torch.zeros_like(probs)
        target_onehot = target_onehot.scatter(1, target.reshape(-1, 1), 1)
        loss = torch.mean((probs - target_onehot) ** 2)
        pred = torch.argmax(probs, dim=1)
        assert target.shape[0] == pred.shape[0], "target len != pred len"
        acc = torch.sum(target == pred).float() / target.shape[0]
        f1 = f1_score(target.data.cpu().numpy(), pred.data.cpu().numpy(), average='macro')
        return pred, loss, acc, f1


class CrossEntropyCriterion(_Loss):
    def __init__(self, opt):
        way = opt.ways
        shot = opt.shots
        super(CrossEntropyCriterion, self).__init__()
        self.amount = way * shot
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, probs, target,num_support=None):  # (Q,C) (Q)
        if num_support is None:
            num_support = self.amount
        target = target[num_support:]
        loss = self.ce_loss(target,probs)
        pred = torch.argmax(probs, dim=1)
        assert target.shape[0] == pred.shape[0], "target len != pred len"
        acc = torch.sum(target == pred).float() / target.shape[0]
        f1 = f1_score(target.data.cpu().numpy(), pred.data.cpu().numpy(), average='macro')
        return pred, loss, acc, f1


class TraditionCriterion(_Loss):
    def __init__(self, opt):
        super(TraditionCriterion, self).__init__()
        self.amount = opt.batch_size
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, probs, target):  # (B,C) (B)
        loss = self.ce_loss(probs, target)
        pred = torch.argmax(probs, dim=1)
        assert target.shape[0] == pred.shape[0], "target len != pred len"
        acc = torch.sum(target == pred).float() / target.shape[0]
        f1 = f1_score(target.data.cpu().numpy(), pred.data.cpu().numpy(), average='macro')
        return loss


class InfoNCE(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(InfoNCE, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().add(0.0000001).to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))


        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask_pos = mask * logits_mask
        mask_neg = (torch.ones_like(mask)-mask) * logits_mask

        similarity = torch.exp(torch.mm(anchor_feature, contrast_feature.t()) / self.temperature)

        pos = torch.sum(similarity * mask_pos, 1)
        neg = torch.sum(similarity * mask_neg, 1)
        loss = -(torch.mean(torch.log(pos / (pos + neg))))

        return loss

class Stance_loss(nn.Module):
    def __init__(self, temperature, contrast_mode='all',
                 base_temperature=0.07):
        super(Stance_loss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().add(0.0000001).to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        mask = mask.repeat(anchor_count, contrast_count)

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask_pos = mask * logits_mask
        mask_neg = (torch.ones_like(mask)-mask) * logits_mask

        similarity = torch.exp(torch.mm(anchor_feature, contrast_feature.t()) / self.temperature)

        pos = torch.sum(similarity * mask_pos, 1)
        neg = torch.sum(similarity * mask_neg, 1)
        loss = -(torch.mean(torch.log(pos / (pos + neg))))

        return loss



class OurContrastive_loss(nn.Module):
    def __init__(self, opt, contrast_mode='all',
                 base_temperature=0.07):
        super(OurContrastive_loss, self).__init__()
        self.temperature = opt.temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.alpha = opt.alpha
        self.beta = opt.beta

    def forward(self, features, labels=None, target = None, mask=None):

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask_labels = torch.eq(labels, labels.T).float().to(device)
        target = target.contiguous().view(-1, 1)
        if target.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask_target = torch.eq(target, target.T).float().to(device)
        mask_dif_target = torch.ones_like(mask_target) - mask_target
        mask_a = (mask_labels * mask_target) * self.alpha
        mask_b = (mask_labels * mask_dif_target) * self.beta
        mask_both = (mask_a + mask_b).add(0.0000001)
        mask = mask_labels
        mask = mask.add(0.0000001)
        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        mask = mask.repeat(anchor_count, contrast_count)
        mask_both = mask_both.repeat(anchor_count, contrast_count)

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask_pos = mask_both * logits_mask
        mask_neg = (torch.ones_like(mask) - mask) * logits_mask

        similarity = torch.exp(torch.mm(anchor_feature, contrast_feature.t()) / self.temperature)

        pos = torch.sum(similarity * mask_pos, 1)
        neg = torch.sum(similarity * mask_neg, 1)
        loss = -(torch.mean(torch.log(pos / (pos + neg))))
        return loss

class Target_loss(nn.Module):

    def __init__(self, opt,contrast_mode='all',
                 base_temperature=0.07):
        super(Target_loss, self).__init__()
        self.temperature = opt.temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.alpha = nn.Parameter(torch.Tensor([opt.alpha]))
        self.beta = nn.Parameter(torch.Tensor([opt.beta]))


    def forward(self, features, labels=None, target = None, mask=None):

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))


        batch_size = features.shape[0]

        labels = labels.contiguous().view(-1, 1)
        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask_labels = torch.eq(labels, labels.T).float().to(device)

        target = target.contiguous().view(-1, 1)
        if target.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask_traget = torch.eq(target, target.T).float().to(device)
        alpha = torch.sigmoid(self.alpha).to(device)
        beta = 1 + torch.sigmoid(self.beta).to(device)


        mask = mask_labels* mask_traget
        mask = (mask_labels - mask)*beta+(mask)*alpha


        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))



        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask_pos = mask * logits_mask
        mask_neg = (torch.ones_like(mask_labels) - mask_labels) * logits_mask
        # mask_neg = logits_mask

        similarity = torch.exp(torch.mm(anchor_feature, contrast_feature.t()) / self.temperature)

        pos = torch.sum(similarity * mask_pos, 1)
        neg = torch.sum(similarity * mask_neg, 1)
        loss = -(torch.mean(torch.log(pos / (pos + neg))))
        if torch.isinf(loss) or torch.isnan(loss):
            loss = torch.zeros_like(loss).to(device)

        return loss


class Prototype_loss(nn.Module):
    def __init__(self,  contrast_mode='all',
                 base_temperature=0.07):
        super(Prototype_loss, self).__init__()

        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, prototype, temp_proto):

        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        mask = torch.eye(features.shape[0], prototype.shape[0],dtype=torch.float32).add(0.0000001).to(device)

        mask_pos = mask
        mask_neg = torch.ones_like(mask)-mask
        similarity = torch.mm(features, prototype.t())
        similarity = torch.exp(similarity / temp_proto)

        pos = torch.sum(similarity * mask_pos, 1)
        neg = torch.sum(similarity * mask_neg, 1)
        loss = -(torch.mean(torch.log(pos / (pos + neg))))

        return loss


if __name__=='__main__':
    loss = Prototype_loss(0.07)
    feature = torch.randn([5,100])
    prototype = torch.randn([10,100])
    temp_proto = torch.randn([10])

    loss(feature,prototype,temp_proto)