import numpy as np
from sklearn.neighbors import NearestNeighbors
from utils.extractor import Extractor
import torch as t


def euclidean_distances(x, y, squared=True):
    x_square = np.expand_dims(np.einsum('ij, ij->i', x, x), axis=1)
    y_square = np.expand_dims(np.einsum('ij, ij->i', y, y), axis=0)

    distances = np.dot(x, y.T)
    distances *= -2
    distances += x_square
    distances += y_square
    np.maximum(distances, 0, distances)
    np.sqrt(distances, distances)
    return distances


def cosine_distances(x, y):
    x_square = np.expand_dims(np.einsum('ij, ij->i', x, x), axis=1) + 1e-16
    y_square = np.expand_dims(np.einsum('ij, ij->i', y, y), axis=0) + 1e-16

    distances = np.dot(x, y.T)
    np.sqrt(x_square, x_square)
    np.sqrt(y_square, y_square)
    distances /= (x_square * y_square + 1e-8)
    np.minimum(distances, 1., distances)
    np.maximum(distances, -1., distances)
    return 1. - distances


def partition_arg_topK(matrix, K, axis=0):
    a_part = np.argpartition(matrix, K, axis=axis)
    if axis ==0 :
        row_index = np.arange(matrix.shape[1-axis])
        a_sec_argsort_K = np.argsort(matrix[a_part[0:K, :], row_index], axis=axis)
        return a_part[0:K, :][a_sec_argsort_K, row_index]
    else:
        column_index = np.arange(matrix.shape[1-axis])[:, None]
        a_sec_argsort_K = np.argsort(matrix[column_index, a_part[:, 0:K]], axis=axis)
        return a_part[:, 0:K][column_index, a_sec_argsort_K]


class Tester(object):

    def __init__(self, opt):

        # self.vis = opt.vis
        self.test_opt = opt

        self.photo_net = opt.photo_net
        self.sketch_net = opt.sketch_net

        self.photo_test = opt.photo_test
        self.sketch_test = opt.sketch_test

        self.eps = 1e-8

    @t.no_grad()
    def _extract_feature(self):
        with t.no_grad():
            self.photo_net.eval()
            self.sketch_net.eval()

            extractor = Extractor(e_model=self.photo_net, opt=self.test_opt, vis=False, dataloader=True)
            photo_data = extractor.extract(self.photo_test)

            extractor.reload_model(self.sketch_net)
            sketch_data = extractor.extract(self.sketch_test)

            photo_name = photo_data['name']
            photo_feature = photo_data['feature']

            sketch_name = sketch_data['name']
            sketch_feature = sketch_data['feature']

        return photo_name, photo_feature, sketch_name, sketch_feature

    @t.no_grad()
    def _extract_feature_embedding(self):
        with t.no_grad():
            self.photo_net.eval()
            self.sketch_net.eval()

            extractor = Extractor(e_model=self.photo_net, opt=self.test_opt, cat_info=False, vis=False, dataloader=True)
            photo_data = extractor.extract(self.photo_test)

            extractor.reload_model(self.sketch_net)
            sketch_data = extractor.extract(self.sketch_test)

            photo_name = photo_data['name']
            photo_feature = photo_data['feature']

            sketch_name = sketch_data['name']
            sketch_feature = sketch_data['feature']

        return photo_name, photo_feature, sketch_name, sketch_feature

    @staticmethod
    def compute_distance(sketch, photo, metric='euclidean'):
        return euclidean_distances(sketch, photo) if metric=='euclidean' else cosine_distances(sketch, photo)


    @t.no_grad()
    def test_category_recall(self, metric='euclidean'):

        photo_name, photo_feature, sketch_name, sketch_feature = self._extract_feature()
        gt_class = []
        index = 0
        first_class = photo_name[0].split('/')[0]
        i = 1
        gt_class.append([0])
        while i < len(photo_name):
            c_name = photo_name[i].split('/')[0]
            if c_name != first_class:
                index += 1
                gt_class.append([i])
            else:
                gt_class[index].append(i)
            first_class = c_name
            i += 1

        gt_list = []
        index = 0
        first_class = sketch_name[0].split('/')[0]
        i = 1
        gt_list.append(gt_class[index])
        while i < len(sketch_name):
            s_name = sketch_name[i].split('/')[0]
            if s_name != first_class:
                index += 1
            gt_list.append(gt_class[index])
            first_class = s_name
            i += 1

        gt_list = np.asarray(gt_list)
        gt_list = np.reshape(gt_list, (gt_list.shape[0], 1, gt_list.shape[1]))

        distances = euclidean_distances(sketch_feature, photo_feature) if metric == 'euclidean' \
            else cosine_distances(sketch_feature, photo_feature)
        topK = partition_arg_topK(distances, 10, axis=1)

        recall_1 = np.sum(np.any(gt_list == topK[:, 0, None, None], axis=2)) / gt_list.shape[0]
        recall_5 = np.sum(np.any(gt_list == topK[:, :5, None], axis=(1, 2))) / gt_list.shape[0]
        recall_10 = np.sum(np.any(gt_list == topK[:, :10, None], axis=(1, 2))) / gt_list.shape[0]

        print('recall@1 :', recall_1, '    recall@5 :', recall_5, '     recall@10 :', recall_10)
        return {'recall@1': recall_1, 'recall@5': recall_5, 'recall@10': recall_10}

    @t.no_grad()
    def test_item_recall(self, metric='euclidean'):
        photo_name, photo_feature, sketch_name, sketch_feature = self._extract_feature()
        items = len(photo_name)
        gt_list = np.arange(items)
        gt_list = np.reshape(gt_list, (items, 1))

        K = 10
        distances = self.compute_distance(sketch_feature, photo_feature, metric=metric)
        topK = partition_arg_topK(distances, K, axis=1)
        recall_1 = np.sum(topK[:, 0, None] == gt_list) / items
        recall_5 = np.sum(np.any(topK[:, :5] == gt_list, axis=1)) / items
        recall_10 = np.sum(np.any(topK[:, :10] == gt_list, axis=1)) / items

        print('recall@1 :', recall_1, '    recall@5 :', recall_5, '     recall@10 :', recall_10)
        return {'recall@1': recall_1, 'recall@5': recall_5, 'recall@10': recall_10}

    @t.no_grad()
    def test_instance_recall(self, metric='euclidean'):
        photo_name, photo_feature, sketch_name, sketch_feature = self._extract_feature()
        gt_list = []
        index = 0
        for i, s_name in enumerate(sketch_name):
            query_name = s_name.split('/')[-1]
            query_name = query_name.split('-')[0]
            pk_name = photo_name[index]
            p_name = pk_name.split('/')[-1]
            p_name = p_name.split('.')[0]
            if query_name != p_name:
                index += 1
            gt_list.append(index)

        test_item = len(gt_list)

        gt_list = np.asarray(gt_list)
        gt_list = np.reshape(gt_list, (test_item, 1))

        K = 10

        distances = euclidean_distances(sketch_feature, photo_feature) if metric == 'euclidean' \
            else cosine_distances(sketch_feature, photo_feature)
        topK = partition_arg_topK(distances, K, axis=1)
        recall_1 = np.sum(topK[:, 0, None] == gt_list) / test_item
        recall_5 = np.sum(np.any(topK[:, :5] == gt_list, axis=1)) / test_item
        recall_10 = np.sum(np.any(topK[:, :10] == gt_list, axis=1)) / test_item

        print('recall@1 :', recall_1, '    recall@5 :', recall_5, '     recall@10 :', recall_10)
        return {'recall@1': recall_1, 'recall@5': recall_5, 'recall@10': recall_10}

    @t.no_grad()
    def test_instance_reall_wo_class(self, metric='euclidean', K=10):
        photo_name, photo_feature, sketch_name, sketch_feature = self._extract_feature()
        photo_name = list(map(lambda x: x.split('/')[-1], photo_name))
        sketch_name = list(map(lambda x: x.split('/')[-1], sketch_name))

        gt_list = []
        index = 0
        for i, s_name in enumerate(sketch_name):
            query_name = s_name.split('_')[0]

            pk_name = photo_name[index]
            p_name = pk_name.split('.')[0]
            if query_name != p_name:
                index += 1
            gt_list.append(index)

        gt_list = np.asarray(gt_list)
        gt_list = np.reshape(gt_list, (gt_list.shape[0], 1))

        distances = euclidean_distances(sketch_feature, photo_feature) if metric == 'euclidean' \
            else cosine_distances(sketch_feature, photo_feature)
        # print(np.shape(distances))
        topK = partition_arg_topK(distances, K, axis=1)
        print(np.shape(topK))

        gt_len = gt_list.shape[0]

        recall_1 = np.sum(topK[:, 0, None] == gt_list) / gt_len
        recall_5 = np.sum(np.any(topK[:, :5] == gt_list, axis=1)) / gt_len
        recall_10 = np.sum(np.any(topK[:, :K] == gt_list, axis=1)) / gt_len

        print('recall@1 :', recall_1, '    recall@5 :', recall_5, '     recall@10 :', recall_10)
        return {'recall@1': recall_1, 'recall@5': recall_5, 'recall@10': recall_10}

    def _map_change(self, inputArr):
        dup = np.copy(inputArr)
        for idx in range(inputArr.shape[1]):
            if idx != 0:
                dup[:, idx] = dup[:, idx - 1] + dup[:, idx]

        return np.multiply(dup, inputArr)

    @t.no_grad()
    def test_quickdraw(self, index):
        return_result = dict()
        photo_name, photo_feature, sketch_name, sketch_features = self._extract_feature()

        nbrs = NearestNeighbors(n_neighbors=np.size(photo_feature, 0),
                                algorithm='brute', metric='euclidean').fit(photo_feature)

        test_sketch_class = []
        for s_name in sketch_name:
            s_class = s_name.split('/')[0]
            test_sketch_class.append(s_class)

        retrieve_class = []
        for p_name in photo_name:
            p_class = p_name.split('/')[0]
            retrieve_class.append(p_class)

        test_sketch_class = np.array(test_sketch_class)
        retrieve_class = np.array(retrieve_class)

        map_all_total = 0
        precision_all_total = 0

        map_index_total = 0
        precision_index_total = 0
        sketch_features = np.vsplit(sketch_features, 10)
        for sketch_feature in sketch_features:

            # sketch_feature = np.reshape(sketch_feature, [1, np.shape(sketch_feature)[0]])

            distance, indices = nbrs.kneighbors(sketch_feature)

            # print('test_sketch_class.shape:', np.shape(test_sketch_class))
            # print('retrieve_class.shape:', np.shape(retrieve_class))
            # print('indices.shape:', np.shape(indices))

            # indices = indices[:, :200]
            # print('indices.shape:', np.shape(indices))
            # print('indices.shape:', np.shape(indices))

            retrieved_class = retrieve_class[indices]
            results = np.zeros(retrieved_class.shape)
            # print('retrieved_class.shape:', np.shape(retrieved_class))
            # print('results.shape:', np.shape(results))
            for idx in range(results.shape[0]):
                results[idx] = (retrieved_class[idx] == test_sketch_class[idx])

            precision = np.mean(results, axis=1)

            pre_count = np.sum(results, axis=1) + self.eps
            temp = [np.arange(results.shape[1]) for ii in range(results.shape[0])]
            mAP_term = 1.0 / (np.stack(temp, axis=0) + 1)
            mAP = np.sum(np.multiply(self._map_change(results), mAP_term), axis=1)

            mAP = mAP / pre_count
            # print('mAP : ', np.mean(mAP))

            map_all_total += np.mean(mAP)
            precision_all_total += np.mean(precision)

            # print('The mean precision@200 for test sketches is ' + str(np.mean(precision)))
            # print('The mAP for test_sketches is ' + str(np.mean(mAP)))

            # return_result['precision@all'] = np.mean(precision)
            # return_result['mAP@all'] = np.mean(mAP)

            # calculate mAP@index
            '''
            indices = indices[:, :index]

            retrieved_class = retrieve_class[indices]
            results = np.zeros(retrieved_class.shape)
            # print('retrieved_class.shape:', np.shape(retrieved_class))
            # print('results.shape:', np.shape(results))
            for idx in range(results.shape[0]):
                results[idx] = (retrieved_class[idx] == test_sketch_class[idx])
            '''
            results = results[:, :index]
            precision = np.mean(results, axis=1)

            pre_count = np.sum(results, axis=1) + self.eps
            temp = [np.arange(index) for ii in range(results.shape[0])]
            mAP_term = 1.0 / (np.stack(temp, axis=0) + 1)
            mAP = np.sum(np.multiply(self._map_change(results), mAP_term), axis=1)
            mAP = mAP / pre_count

            map_index_total += np.mean(mAP)
            precision_index_total += np.mean(precision)

            # print('The mean precision@200 for test sketches is ' + str(np.mean(precision)))
            # print('The mAP for test_sketches is ' + str(np.mean(mAP)))

            # return_result['precision@{0}'.format(index)] = np.mean(precision)
            # return_result['mAP@{0]'.format(index)] = np.mean(mAP)

        print('The mean presision@all for test sketches is ' + str(precision_all_total / len(sketch_features)))
        print('The mAP@all for test_sketches is ' + str(map_all_total / len(sketch_features)))

        print('The mean precsion@{0} for test sketches is'.format(index) + str(
            precision_index_total / len(sketch_features)))
        print('The mAP@{0} for test sketches is'.format(index) + str(map_index_total / len(sketch_features)))

        return_result['precsion@all'] = precision_all_total / len(sketch_features)
        return_result['mAP@all'] = map_all_total / len(sketch_features)
        return_result['precsion@{0}'.format(index)] = precision_index_total / len(sketch_features)
        return_result['mAP@{0}'.format(index)] = map_index_total / len(sketch_features)

        return return_result

    def test(self, dbname):
        if dbname == 'sketchydb':
            return self.test_instance_recall()
        elif dbname == 'shoev2':
            return self.test_instance_reall_wo_class()
        else:
            return self.test_item_recall()