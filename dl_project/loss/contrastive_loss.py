import torch

class ContrastiveLoss:

    def __init__(self, generator=None, projector=None, latent_diff_dim=None, model=None, ABS=True, TEMP=0.5, K=None, REDUCE='mean'):

        self.abs = ABS
        self.temp = TEMP
        self.k = K
        self.reduce = REDUCE

        self.model = model
        self.generator = generator
        self.projector = projector

    def prepare_features(self, y_direction, y_original):

        features = []

        for i in range(y_direction.shape[0]):

            for j in range(y_direction.shape[1]):
                # 1. get features
                feats_1 = y_direction[i][j]
                feats_2 = y_original[i][j]
                # 2. project features to a space where the
                # contrastive loss is applied
                projected_feats_1 = feats_1
                projected_feats_2 = feats_2
                # 3. subtract with original features
                diff = projected_feats_1 - projected_feats_2
                # 4. normalization
                diff = diff / torch.norm(diff)

                features.append(diff)

        features = torch.stack(features)

        return features

    def contrastive_loss_without_generator(self, y_direction, y_original):
        """
        Args:
            y_direction1: torch.tensor(batch_size, direction_count, K, latent_diff_dim)
            y_direction2: torch.tensor(batch_size, direction_count, K, latent_diff_dim)
        Returns:
            loss: torch.tensor(batch_size, direction_count) (1d tensor of batch_size length)
        """
        ABS = True
        TEMP = 0.5
        K = y_direction.shape[2]
        REDUCE = ['mean', 'sum'][0]

        out = self.prepare_features(y_direction, y_original)

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

        acc = (pos > neg).float()
        loss = -torch.log(pos / neg)

        return acc, loss

    def contrastive_loss(self, z):
        """
        Make sure to call optimizer.zero_grad() before calling this function
        """

        z_orig = z

        # Original features
        with torch.no_grad():
            orig_feats = self.generator.get_features(z)
            orig_feats = self.projector(orig_feats)

        # Apply directions
        z = self.model(z)

        # reorganize z
        reorganized_z = []
        for i in range(z.shape[1]):
            reorganized_z.append(z[:, i, :])
        reorganized_z = torch.cat(reorganized_z)
        z = reorganized_z

        # forward
        batch_size = z_orig.shape[0]
        features = []
        for j in range(z.shape[0] // batch_size):
            # Prepare batch
            start, end = j * batch_size, (j + 1) * batch_size
            z_batch = z[start:end, ...]

            # Get features
            feats = self.generator.get_features(z_batch)
            feats = self.projector(feats)

            # Take feature divergence
            feats = feats - orig_feats
            feats = feats / torch.reshape(torch.norm(feats, dim=1), (-1, 1))

            features.append(feats)
        features = torch.cat(features, dim=0)

        # Loss
        out = features
        n_samples = len(out)
        assert (n_samples % self.k) == 0, "Batch size is not divisible by given k!"

        # similarity matrix
        sim = torch.mm(out, out.t().contiguous())

        if self.abs:
            sim = torch.abs(sim)

        sim = torch.exp(sim * self.temp)

        # mask for pairs
        mask = torch.zeros((n_samples, n_samples), device=sim.device).bool()
        for i in range(self.k):
            start, end = i * (n_samples // self.k), (i + 1) * (
                        n_samples // self.k)
            mask[start:end, start:end] = 1

        diag_mask = ~(torch.eye(n_samples, device=sim.device).bool())

        # pos and neg similarity and remove self similarity for pos
        pos = sim.masked_select(mask * diag_mask).view(n_samples, -1)
        neg = sim.masked_select(~mask).view(n_samples, -1)

        if self.reduce == "mean":
            pos = pos.mean(dim=-1)
            neg = neg.mean(dim=-1)
        elif self.reduce == "sum":
            pos = pos.sum(dim=-1)
            neg = neg.sum(dim=-1)
        else:
            raise ValueError("Only mean and sum is supported for reduce method")

        acc = (pos > neg).float().mean()
        loss = -torch.log(pos / neg).mean()
        return acc, loss

