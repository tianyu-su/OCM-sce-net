"""
@inproceedings{VasilevaECCV18FasionCompatibility,
Author = {Mariya I. Vasileva and Bryan A. Plummer and Krishna Dusad and Shreya Rajpal and Ranjitha Kumar and David Forsyth},
Title = {Learning Type-Aware Embeddings for Fashion Compatibility},
booktitle = {ECCV},
Year = {2018}
}
https://github.com/mvasil/fashion-compatibility/blob/master/polyvore_outfits.py
"""

import json
import os
import os.path
import pickle

import numpy as np
import torch
import torch.utils.data
from PIL import Image
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from metrics import MRR_HR


def default_image_loader(path):
    return Image.open(path).convert('RGB')


def parse_iminfo(question, im2index, id2im, gt=None):
    """ Maps the questions from the FITB and compatibility tasks back to
        their index in the precomputed matrix of features

        question: List of images to measure compatibility between
        im2index: Dictionary mapping an image name to its location in a
                  precomputed matrix of features
        gt: optional, the ground truth outfit set this item belongs to
    """
    questions = []
    is_correct = np.zeros(len(question), np.bool)
    for index, im_id in enumerate(question):
        set_id = im_id.split('_')[0]
        if gt is None:
            gt = set_id

        im = id2im[im_id]
        questions.append((im2index[im], im))
        is_correct[index] = set_id == gt

    return questions, is_correct, gt


def load_typespaces(rootdir, rand_typespaces, num_rand_embed):
    """ loads a mapping of pairs of types to the embedding used to
        compare them

        rand_typespaces: Boolean indicator of randomly assigning type
                         specific spaces to their embedding
        num_rand_embed: number of embeddings to use when
                        rand_typespaces is true
    """
    typespace_fn = os.path.join(rootdir, 'typespaces.p')
    typespaces = pickle.load(open(typespace_fn, 'rb'))
    if not rand_typespaces:
        ts = {}
        for index, t in enumerate(typespaces):
            ts[t] = index

        typespaces = ts
        return typespaces

    # load a previously created random typespace or create one
    # if none exist
    width = 0
    fn = os.path.join(rootdir, 'typespaces_rand_%i.p') % num_rand_embed
    if os.path.isfile(fn):
        typespaces = pickle.load(open(fn, 'rb'))
    else:
        spaces = np.random.permutation(len(typespaces))
        width = np.ceil(len(spaces) / float(num_rand_embed))
        ts = {}
        for index, t in enumerate(spaces):
            ts[typespaces[t]] = int(np.floor(index / width))

        typespaces = ts
        pickle.dump(typespaces, open(fn, 'wb'))

    return typespaces


def load_compatibility_questions(fn, im2index, id2im):
    """ Returns the list of compatibility questions for the
        split """
    with open(fn, 'r') as f:
        lines = f.readlines()

    compatibility_questions = []
    for line in lines:
        data = line.strip().split()
        compat_question, _, _ = parse_iminfo(data[1:], im2index, id2im)
        compatibility_questions.append((compat_question, int(data[0])))

    return compatibility_questions


def load_fitb_questions(fn, im2index, id2im):
    """ Returns the list of fill in the blank questions for the
        split """
    data = json.load(open(fn, 'r'))
    questions = []
    for item in data:
        question = item['question']
        q_index, _, gt = parse_iminfo(question, im2index, id2im)
        answer = item['answers']
        a_index, is_correct, _ = parse_iminfo(answer, im2index, id2im, gt)
        questions.append((q_index, a_index, is_correct))

    return questions


def load_retrieval_questions(fn, im2index, id2im):
    """ Returns the list of fill in the blank questions for the split
         [[P,N,N,N], [P,N,N,N] ]
         P:(items,label)
         N:(items,label)
         """
    with open(fn, 'r') as f:
        data = json.load(f)
    questions = []
    print('extract from disk ...')
    for item in tqdm(data):
        question = item['question']
        q_index, _, gt = parse_iminfo(question, im2index, id2im)
        answer = [item["right"]] + item['candidate']
        a_index, is_correct, _ = parse_iminfo(answer, im2index, id2im, gt)
        questions.append((q_index, a_index, is_correct))

    return questions


class TripletImageLoader(torch.utils.data.Dataset):
    def __init__(self, args, split, meta_data, text_dim=None, transform=None, loader=default_image_loader,
                 random_type="same_type"):
        rootdir = os.path.join(args.datadir, 'polyvore_outfits', args.polyvore_split)
        self.impath = os.path.join(args.datadir, 'polyvore_outfits', 'images')
        self.is_train = split == 'train'
        data_json = os.path.join(rootdir, '%s.json' % split)
        outfit_data = json.load(open(data_json, 'r'))

        # get list of images and make a mapping used to quickly organize the data
        im2type = {}
        category2ims = {}
        imnames = set()
        id2im = {}
        for outfit in outfit_data:
            outfit_id = outfit['set_id']
            for item in outfit['items']:
                im = item['item_id']
                category = meta_data[im]['semantic_category']
                im2type[im] = category

                if category not in category2ims:
                    category2ims[category] = {}

                if outfit_id not in category2ims[category]:
                    category2ims[category][outfit_id] = []

                category2ims[category][outfit_id].append(im)
                id2im['%s_%i' % (outfit_id, item['index'])] = im
                imnames.add(im)

        imnames = list(imnames)
        im2index = {}
        for index, im in enumerate(imnames):
            im2index[im] = index

        self.data = outfit_data
        self.imnames = imnames
        self.im2type = im2type
        self.typespaces = load_typespaces(rootdir, args.rand_typespaces, args.num_rand_embed)
        self.transform = transform
        self.loader = loader
        self.split = split

        if self.is_train:

            # At train time we pull the list of outfits and enumerate the pairwise
            # comparisons between them to train with.  Negatives are pulled by the
            # __get_item__ function
            pos_pairs = []
            max_items = 0
            for outfit in outfit_data:
                items = outfit['items']
                cnt = len(items)
                max_items = max(cnt, max_items)
                outfit_id = outfit['set_id']
                for j in range(cnt - 1):
                    for k in range(j + 1, cnt):
                        pos_pairs.append([outfit_id, items[j]['item_id'], items[k]['item_id']])

            self.pos_pairs = pos_pairs
            self.category2ims = category2ims
            self.max_items = max_items
        else:
            # pull the two task's questions for test and val splits
            fn = os.path.join(rootdir, 'fill_in_blank_%s.json' % split)
            self.fitb_questions = load_fitb_questions(fn, im2index, id2im)
            fn = os.path.join(rootdir, 'compatibility_%s.txt' % split)
            self.compatibility_questions = load_compatibility_questions(fn, im2index, id2im)
            fn = os.path.join(args.datadir, 'polyvore_outfits', 'retrival_data', random_type,
                              "{}.json".format(args.polyvore_split))

            self.retrieval_questions = load_retrieval_questions(fn, im2index, id2im)

    def load_train_item(self, image_id):
        """ Returns a single item in the triplet and its data
        """
        imfn = os.path.join(self.impath, '%s.jpg' % image_id)
        img = self.loader(imfn)
        if self.transform is not None:
            img = self.transform(img)

        item_type = self.im2type[image_id]
        return img, item_type

    def sample_negative(self, outfit_id, item_id, item_type):
        """ Returns a randomly sampled item from a different set
            than the outfit at data_index, but of the same type as
            item_type
        
            data_index: index in self.data where the positive pair
                        of items was pulled from
            item_type: the coarse type of the item that the item
                       that was paired with the anchor
        """
        item_out = item_id
        candidate_sets = list(self.category2ims[item_type].keys())
        attempts = 0
        while item_out == item_id and attempts < 100:
            choice = np.random.choice(candidate_sets)
            items = self.category2ims[item_type][choice]
            item_index = np.random.choice(range(len(items)))
            item_out = items[item_index]
            attempts += 1

        return item_out

    def get_typespace(self, anchor, pair):
        """ Returns the index of the type specific embedding
            for the pair of item types provided as input
        """
        query = (anchor, pair)
        if query not in self.typespaces:
            query = (pair, anchor)

        return self.typespaces[query]

    def test_compatibility(self, tnet, embeds, img_norm_embeds):
        """ Returns the area under a roc curve for the compatibility
            task

            embeds: precomputed embedding features used to score
                    each compatibility question
            metric: a function used to score the elementwise product
                    of a pair of embeddings, if None euclidean
                    distance is used
        """
        scores = []
        labels = np.zeros(len(self.compatibility_questions), np.int32)
        for index, (outfit, label) in enumerate(self.compatibility_questions):
            labels[index] = label
            n_items = len(outfit)
            outfit_score = 0.0
            num_comparisons = 0.0
            for i in range(n_items - 1):
                item1, img1 = outfit[i]
                for j in range(i + 1, n_items):
                    item2, img2 = outfit[j]

                    feat = torch.cat([img_norm_embeds[item1], img_norm_embeds[item2]])
                    weights = tnet.concept_branch(feat.unsqueeze(0))  # (1,num_condition)

                    embed1 = torch.matmul(weights, embeds[item1])  # (1,num_condition) * (num_condition,dim)
                    embed2 = torch.matmul(weights, embeds[item2])

                    outfit_score += torch.nn.functional.pairwise_distance(embed1, embed2, 2)

                    num_comparisons += 1.

            outfit_score /= num_comparisons
            scores.append(outfit_score)

        scores = torch.cat(scores).squeeze().cpu().numpy()
        # scores = np.load('feats.npy')
        # print(scores)
        # assert(False)
        # np.save('feats.npy', scores)
        auc = roc_auc_score(labels, 1 - scores)
        return auc

    def test_fitb(self, tnet, embeds, img_norm_embeds):
        """ Returns the accuracy of the fill in the blank task

            embeds: precomputed embedding features used to score
                    each compatibility question
            metric: a function used to score the elementwise product
                    of a pair of embeddings, if None euclidean
                    distance is used
        """
        correct = 0.
        n_questions = 0.
        for q_index, (questions, answers, is_correct) in enumerate(self.fitb_questions):
            answer_score = np.zeros(len(answers), dtype=np.float32)
            for index, (answer, img1) in enumerate(answers):
                score = 0.0
                for question, img2 in questions:
                    feat = torch.cat([img_norm_embeds[question], img_norm_embeds[answer]])
                    weights = tnet.concept_branch(feat.unsqueeze(0))  # (1,num_condition)

                    embed1 = torch.matmul(weights, embeds[question])  # (1,num_condition) * (num_condition,dim)
                    embed2 = torch.matmul(weights, embeds[answer])

                    score += torch.nn.functional.pairwise_distance(embed1, embed2, 2)

                answer_score[index] = score.squeeze().cpu().numpy()

            correct += is_correct[np.argmin(answer_score)]
            n_questions += 1

        # scores are based on distances so need to convert them so higher is better
        acc = correct / n_questions
        return acc

    def test_retrieval(self, tnet, embeds, img_norm_embeds):
        """ Returns the accuracy of the fill in the blank task

            embeds: precomputed embedding features used to score
                    each compatibility question
            metric: a function used to score the elementwise product
                    of a pair of embeddings, if None euclidean
                    distance is used
        """
        print("inference....")
        # HR_ks = (1, 10, 100, 200)
        HR_ks = (1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 150, 200)

        total_score = []
        for (questions, answers, is_correct) in tqdm(self.retrieval_questions):
            answer_score = np.zeros(len(answers), dtype=np.float32)
            for index, (answer, img1) in enumerate(answers):
                score = 0.0
                for question, img2 in questions:
                    feat = torch.cat([img_norm_embeds[question], img_norm_embeds[answer]])
                    weights = tnet.concept_branch(feat.unsqueeze(0))  # (1,num_condition)

                    embed1 = torch.matmul(weights, embeds[question])  # (1,num_condition) * (num_condition,dim)
                    embed2 = torch.matmul(weights, embeds[answer])

                    score += torch.nn.functional.pairwise_distance(embed1, embed2, 2)

                answer_score[index] = score.squeeze().cpu().numpy()

            assert is_correct[0] == True, "dont't metric"
            total_score.append(np.expand_dims(answer_score, axis=0))

        # scores are based on distances so need to convert them so higher is better
        total_score = np.concatenate(total_score, axis=0)
        res_metrics = MRR_HR(-total_score, HR_ks=HR_ks)

        print("MRR:{:.3F} ".format(res_metrics[0]), end=' ')
        for idx, kk in enumerate(HR_ks):
            print("HR@{}:{:.3f} ".format(kk, res_metrics[1][idx]), end=' ')

    def __getitem__(self, index):
        # anchor, far, close
        if self.is_train:
            outfit_id, anchor_im, pos_im = self.pos_pairs[index]
            img1, anchor_type = self.load_train_item(anchor_im)
            img2, item_type = self.load_train_item(pos_im)

            neg_im = self.sample_negative(outfit_id, pos_im, item_type)
            img3, _ = self.load_train_item(neg_im)
            condition = self.get_typespace(anchor_type, item_type)
            return img1, img3, img2, condition

        anchor = self.imnames[index]
        img1 = self.loader(os.path.join(self.impath, '%s.jpg' % anchor))
        if self.transform is not None:
            img1 = self.transform(img1)

        return img1

    def shuffle(self):
        np.random.shuffle(self.pos_pairs)

    def __len__(self):
        if self.is_train:
            return len(self.pos_pairs)

        return len(self.imnames)
