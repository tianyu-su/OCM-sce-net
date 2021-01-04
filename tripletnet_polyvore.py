r"""
update the condition weight branch by https://github.com/rxtan2/Learning-Similarity-Conditions/issues/7
The original version don't work on the Polyvore Outfit dataset.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class ConceptBranch(nn.Module):
    def __init__(self, out_dim, embedding_dim):
        super(ConceptBranch, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(embedding_dim, 32), nn.ReLU(inplace=True))
        self.fc2 = nn.Sequential(nn.Linear(32, out_dim), nn.Softmax(dim=-1))

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class CS_Tripletnet(nn.Module):
    def __init__(self, embeddingnet, num_concepts):
        super(CS_Tripletnet, self).__init__()
        self.embeddingnet = embeddingnet
        self.num_concepts = num_concepts
        self.concept_branch = ConceptBranch(self.num_concepts, 64 * 2)

    def forward(self, x, y, z, c):
        """ x: Anchor image,
            y: Distant (negative) image,
            z: Close (positive) image,
            c: Integer indicating according to which notion of similarity images are compared"""

        general_x = self.embeddingnet.embeddingnet(x)
        general_y = self.embeddingnet.embeddingnet(y)
        general_z = self.embeddingnet.embeddingnet(z)

        # calculate image similarity loss on the general embedding
        # ref: https://github.com/mvasil/fashion-compatibility/blob/299b426e38b92b4441534e025bf84caa0ea3155b/tripletnet.py#L97
        sim_i_disti_p = F.pairwise_distance(general_y, general_z, 2)
        sim_i_disti_n1 = F.pairwise_distance(general_y, general_x, 2)
        sim_i_disti_n2 = F.pairwise_distance(general_z, general_x, 2)


        # l2-normalize embeddings
        # norm = torch.norm(general_x, p=2, dim=1) + 1e-10
        # general_x = general_x / norm.expand_as(general_x)
        # norm = torch.norm(general_y, p=2, dim=1) + 1e-10
        # general_y = general_y / norm.expand_as(general_y)
        # norm = torch.norm(general_z, p=2, dim=1) + 1e-10
        # general_z = general_z / norm.expand_as(general_z)
        general_x = F.normalize(general_x, p=2, dim=1)
        general_y = F.normalize(general_y, p=2, dim=1)
        general_z = F.normalize(general_z, p=2, dim=1)

        anchor_far = torch.cat([general_x, general_y], 1)
        anchor_close = torch.cat([general_x, general_z], 1)

        weights_xy = self.concept_branch(anchor_far)
        weights_xz = self.concept_branch(anchor_close)

        embedded_x_far = None
        embedded_x_close = None
        embedded_y = None
        embedded_z = None
        mask_norm = []
        for idx in range(self.num_concepts):
            concept_idx = np.zeros((len(x),), dtype=int)
            concept_idx += idx
            concept_idx = torch.from_numpy(concept_idx)
            concept_idx = concept_idx.cuda()
            concept_idx = Variable(concept_idx)

            tmp_embedded_x, masknorm_norm_x, embed_norm_x, tot_embed_norm_x = self.embeddingnet(x, concept_idx)
            tmp_embedded_y, masknorm_norm_y, embed_norm_y, tot_embed_norm_y = self.embeddingnet(y, concept_idx)
            tmp_embedded_z, masknorm_norm_z, embed_norm_z, tot_embed_norm_z = self.embeddingnet(z, concept_idx)

            mask_norm.append(masknorm_norm_x)
            # if mask_norm is None:
            #     mask_norm = masknorm_norm_x
            # else:
            #     mask_norm += masknorm_norm_x

            weights_far = weights_xy[:, idx].unsqueeze(1)
            weights_close = weights_xz[:, idx].unsqueeze(1)

            if embedded_x_far is None:
                embedded_x_far = tmp_embedded_x * weights_far.expand_as(tmp_embedded_x)
                embedded_x_close = tmp_embedded_x * weights_close.expand_as(tmp_embedded_x)
                embedded_y = tmp_embedded_y * weights_far.expand_as(tmp_embedded_y)
                embedded_z = tmp_embedded_z * weights_close.expand_as(tmp_embedded_z)
            else:
                embedded_x_far += tmp_embedded_x * weights_far.expand_as(tmp_embedded_x)
                embedded_x_close += tmp_embedded_x * weights_close.expand_as(tmp_embedded_x)
                embedded_y += tmp_embedded_y * weights_far.expand_as(tmp_embedded_y)
                embedded_z += tmp_embedded_z * weights_close.expand_as(tmp_embedded_z)

        mask_norm = torch.stack(mask_norm).sum() / self.num_concepts
        embed_norm = (embed_norm_x + embed_norm_y + embed_norm_z) / 3
        mask_embed_norm = (tot_embed_norm_x + tot_embed_norm_y + tot_embed_norm_z) / 3
        dist_a = F.pairwise_distance(embedded_x_far, embedded_y, 2)
        dist_b = F.pairwise_distance(embedded_x_close, embedded_z, 2)
        return dist_a, dist_b, mask_norm, embed_norm, mask_embed_norm, sim_i_disti_p,sim_i_disti_n1,sim_i_disti_n2
