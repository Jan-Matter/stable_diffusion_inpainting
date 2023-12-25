import torch


class ContrastiveLoss:

    def __init__(self, direction_count, latent_diff_dim):
        self.direction_count = direction_count
        self.latent_diff_dim = latent_diff_dim



    def prepare_features(self, y_direction1, y_direction2):

        features = []

        for i in range(y_direction1.shape[0]):

            for j in range(y_direction1.shape[1]):
                # 1. get features
                feats_1 = y_direction1[i][j]
                feats_2 = y_direction2[i][j]
                # 2. project features to a space where the
                # contrastive loss is applied
                projected_feats_1 = feats_1
                projected_feats_2 = feats_2
                # 3. subtract with original features
                diff = projected_feats_1 - projected_feats_2
                # 4. normalization
                diff = diff / torch.norm(diff)

                features.append(diff)

        features = torch.cat(features, dim=0)

        return features

    def contrastive_loss(self, y_direction1, y_direction2):
        """
        Args:
            y_direction1: torch.tensor(batch_size, direction_count, latent_diff_dim)
            y_direction2: torch.tensor(batch_size, direction_count, latent_diff_dim)
        Returns:
            loss: torch.tensor(batch_size, ) (1d tensor of batch_size length)
        """
        """
        """
        ABS = True
        TEMP = 0.5
        K = y_direction1.shape[1]
        REDUCE = ['mean', 'sum'][0]

        out = self.prepare_features(y_direction1, y_direction2)

        n_samples = len(out)

        # similarity matrix
        sim = torch.mm(out, out.t().contiguous())

        if ABS:
            sim = torch.abs(sim)

        sim = torch.exp(sim * TEMP)

        # mask for pairs
        mask = torch.zeros((n_samples, n_samples), device=sim.device).bool()
        for i in range(K):
            start, end = i * (n_samples // K), (i + 1) * (n_samples // K)
            mask[start:end, start:end] = 1

        diag_mask = ~(torch.eye(n_samples, device=sim.device).bool())

        # pos and neg similarity and remove self similarity for pos
        pos = sim.masked_select(mask * diag_mask).view(n_samples, -1)
        neg = sim.masked_select(~mask).view(n_samples, -1)

        if REDUCE == "mean":
            pos = pos.mean(dim=-1)
            neg = neg.mean(dim=-1)
        elif REDUCE == "sum":
            pos = pos.sum(dim=-1)
            neg = neg.sum(dim=-1)
        else:
            raise ValueError("Only mean and sum is supported for reduce method")

        acc = (pos > neg).float().mean()
        loss = -torch.log(pos / neg).mean()

        return acc, loss
