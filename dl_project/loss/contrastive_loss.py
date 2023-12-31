import torch

class ContrastiveLoss:

    def __init__(self, abs=True, temp=0.5):

        self.abs = abs
        self.temp = temp

    def __call__(self, features, group_indices, reduce='mean'):
        return self.forward(features, group_indices, reduce)

    def forward(self, features, group_indices, reduce='mean'):
        """
        Compute the contrastive loss. Two features from the same
        group (indicated by group indices) are considered a positive pair.
        Two features from two different groups are considered a negative pair.
        This contrastive loss aims to make a positive pair has a high
        similarity and make a negative pair has a
        low similarity.

        Args:
            features: tensor of shape [num_features, feature+length]
            group_indices: tensor of shape [num_features]
            reduce: 'mean', 'sum', or 'none'. If 'none', a loss matrix of
                shape [num_features, num_features] will be returned with
                diagonal entries being zeros.

        Returns: Contrastive loss as a value (when reduce is 'sum' or 'mean') or
            a loss matrix (when reduce is 'none').

            Examples::

                >>> # Generate features and group indices
                >>> num_features = 8
                >>> feature_length = 128
                >>> group_indices = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2], dtype=torch.int)
                >>> features = torch.randn(num_features, feature_length)
                >>> ct_loss = ContrastiveLoss()

                >>> # Example of computing a loss matrix
                >>> loss_matrix = ct_loss(features, group_indices, reduce='none')

                >>> # Example of computing an averaged loss value
                >>> loss_value = ct_loss(features, group_indices, reduce='mean')
        """

        num_features = features.shape[0]
        device = features.device

        # create the similarity matrix and masks
        sim = torch.zeros(num_features, num_features, device=device)
        positive_mask = torch.zeros(num_features, num_features, dtype=torch.bool, device=device)
        negative_mask = torch.zeros(num_features, num_features, dtype=torch.bool, device=device)
        for row in range(num_features):
            for col in range(num_features):
                # compute similarity for current pair
                u = features[row]
                v = features[col]
                norm_u = torch.norm(u)
                norm_v = torch.norm(v)
                current_sim = (u @ v) / (norm_u * norm_v)
                sim[row][col] = current_sim

                # determine if current pair is a positive pair,
                # a negative pair, or none
                group_u = group_indices[row]
                group_v = group_indices[col]
                is_positive_pair = group_u == group_v
                on_diagonal = row == col
                positive_mask[row][col] = is_positive_pair and not on_diagonal
                negative_mask[row][col] = not is_positive_pair and not on_diagonal

        # take the absolute value of the similarity matrix
        if self.abs:
            sim = torch.abs(sim)

        # numerator of contrastive loss: temperature and exp
        numerator = torch.exp(sim * self.temp)

        # compute loss
        loss_matrix = torch.zeros(num_features, num_features, device=device)
        for row in range(num_features):
            for col in range(num_features):
                if row != col:
                    current_numerator = numerator[row][col]
                    current_denominator = (numerator[row].masked_select(negative_mask[row])).sum()
                    epsilon = 1e-8
                    current_loss = -torch.log(current_numerator / (current_denominator + epsilon))
                    loss_matrix[row][col] = current_loss

        # return the loss
        if reduce == 'none':
            return loss_matrix
        elif reduce == 'mean':
            return loss_matrix.masked_select(~(torch.eye(loss_matrix.shape[0], dtype=torch.bool))).mean()
        elif reduce == 'sum':
            return loss_matrix.masked_select(~(torch.eye(loss_matrix.shape[0], dtype=torch.bool))).sum()
        else:
            raise ValueError("Incorrect reduce method")

    # TODO: check this vectorized version
    def vectorized_forward(self, features, group_indices, reduce='mean'):
        num_features = features.shape[0]
        device = features.device

        # compute similarity matrix
        norms = torch.norm(features, dim=1).unsqueeze(1)
        sim = torch.matmul(features, features.t()) / torch.matmul(norms,
                                                                  norms.t() + 1e-8)
        if self.abs:
            sim = torch.abs(sim)

        # create positive and negative masks
        group_eq = (group_indices.unsqueeze(1) == group_indices.unsqueeze(
            0))
        diag_mask = torch.eye(num_features, dtype=torch.bool, device=device)
        positive_mask = group_eq & ~diag_mask
        negative_mask = ~group_eq & ~diag_mask

        # numerator of contrastive loss: temperature and exp
        numerator = torch.exp(sim * self.temp)

        # compute loss
        masked_negatives = numerator * negative_mask.float()
        denominator = masked_negatives.sum(dim=1)

        epsilon = 1e-8
        loss_matrix = -torch.log(
            numerator / (denominator.unsqueeze(1) + epsilon))
        loss_matrix[
            torch.eye(num_features, dtype=torch.bool, device=device)] = 0

        # reduce and return the loss
        if reduce == 'none':
            return loss_matrix
        elif reduce == 'mean':
            return loss_matrix[~diag_mask].mean()
        elif reduce == 'sum':
            return loss_matrix[~diag_mask].sum()
        else:
            raise ValueError("Incorrect reduce method")
