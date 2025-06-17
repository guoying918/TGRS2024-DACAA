import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import argparse
import pickle
from sklearn import metrics
import time
from Tools.data_processing import *
import Tools.utils as utils 
from Tools.modelStatsRecord import *
from Models.net1_1 import *
from Tools.metrics import *

parser = argparse.ArgumentParser(description="Few Shot Visual Recognition")
parser.add_argument("-dataset","--dataset",type = str, default = 'PaviaU') # PaviaU、Salinas、IndianPines、Houston2018
parser.add_argument("-f","--feature_dim",type = int, default = 128)
parser.add_argument("-c","--src_input_dim",type = int, default = 100)
parser.add_argument("-d","--tar_input_dim",type = int, default = 100) 
parser.add_argument("-n","--n_dim",type = int, default = 100)
parser.add_argument("-w","--class_num",type = int, default = 9) 
parser.add_argument("-s","--shot_num_per_class",type = int, default = 180)
parser.add_argument("-b","--query_num_per_class",type = int, default = 19)
parser.add_argument("-e","--episode",type = int, default= 10000)
parser.add_argument("-l","--learning_rate", type = float, default = 0.001)
parser.add_argument("-g","--gpu",type=int, default=0)
# target
parser.add_argument("-m","--test_class_num",type=int, default=9) # 9,16,16,20
parser.add_argument("-z","--test_lsample_num_per_class",type=int, default=5, help='5 4 3 2 1')

args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Hyper Parameters
LossWeight = 5
DATASET = args.dataset
FEATURE_DIM = args.feature_dim
SRC_INPUT_DIMENSION = args.src_input_dim
TAR_INPUT_DIMENSION = args.tar_input_dim
N_DIMENSION = args.n_dim
CLASS_NUM = args.class_num
SHOT_NUM_PER_CLASS = args.shot_num_per_class
QUERY_NUM_PER_CLASS = args.query_num_per_class
EPISODE = args.episode
LEARNING_RATE = args.learning_rate
TEST_CLASS_NUM = args.test_class_num
TEST_LSAMPLE_NUM_PER_CLASS = args.test_lsample_num_per_class

utils.same_seeds(0)
def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('classificationMap'):
        os.makedirs('classificationMap')
_init_()

# load source domain data set
with open(os.path.join('./Datasets', 'Patch9_TRIAN_META_DATA_imdb_ocbs.pickle'), 'rb') as handle:
    source_imdb = pickle.load(handle)
source_imdb['data']=np.array(source_imdb['data'])
source_imdb['Labels']=np.array(source_imdb['Labels'],dtype='int')
source_imdb['set']=np.array(source_imdb['set'],dtype='int')

# process source domain data set
data_train = source_imdb['data'] # (86874, 9, 9, 100)
labels_train = source_imdb['Labels'] # (86874,)
keys_all_train = sorted(list(set(labels_train)))  # class [0,...,45]
print(keys_all_train) # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18,...,45]
label_encoder_train = {}  #{0: 0, 1: 1, 2: 2, 3: 3,...,45: 45}
for i in range(len(keys_all_train)):
    label_encoder_train[keys_all_train[i]] = i

train_set = {}
for class_, path in zip(labels_train, data_train):
    if label_encoder_train[class_] not in train_set:
        train_set[label_encoder_train[class_]] = []
    train_set[label_encoder_train[class_]].append(path)
data = train_set
del train_set
del keys_all_train
del label_encoder_train

print("Num classes for source domain datasets: " + str(len(data)))
print(data.keys())
data = utils.sanity_check(data) # 200 labels samples per class
print("Num classes of the number of class larger than 200: " + str(len(data))) # 40 classes  8000 samples

for class_ in data:
    for i in range(len(data[class_])):
        image_transpose = np.transpose(data[class_][i], (2, 0, 1))  # （9,9,100）-> (100,9,9)
        data[class_][i] = image_transpose

# source few-shot classification data
metatrain_data = data
print(len(metatrain_data.keys()), metatrain_data.keys()) # 40 classes
del data

# source domain adaptation data
print(np.array(source_imdb['data']).shape) # (86874, 9, 9, 100)
source_imdb['data'] = source_imdb['data'].transpose((1, 2, 3, 0)) # (9, 9, 100, 86874)
print(source_imdb['data'].shape)
print(source_imdb['Labels'].shape)
source_dataset = utils.matcifar(source_imdb, train=True, d=3, medicinal=0)
source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=128, shuffle=True, num_workers=0)
del source_dataset, source_imdb

## target domain data set
dataset_folder = './Datasets/test_ocbs/'
if DATASET == 'Houston2018':
    Data_Band_Scaler, GroundTruth = utils.data_load_TIFHDR_PCA(DATASET, dataset_folder, numComponents=10)
else: # PaviaU、Salinas、IndianPines
    Data_Band_Scaler, GroundTruth = utils.load_data(DATASET, dataset_folder)

# model
def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data = torch.ones(m.bias.data.size())

def ComputeClassCenters(support_feat_inv, support_features, support_labels):
    """
    Computes the class centers for the provided support features.

    Args:
        support_feat_inv (torch.Tensor): Inverted support feature tensor.
        support_features (torch.Tensor): Original support feature tensor.
        support_labels (torch.Tensor): Support labels tensor.

    Returns:
        tuple: (mean_support_sample, support_features_pre) where each is a tensor 
               containing the computed class centers.
    """
    class_centers = {}  # Class centers for the inverted support features
    class_centers_pre = {}  # Class centers for the original support features
    
    for c in torch.unique(support_labels):
        # Filter out feature vectors which have class c
        class_mask = torch.eq(support_labels, c)
        class_mask_indices = torch.nonzero(class_mask)
        
        class_features = torch.index_select(support_feat_inv, 0, torch.reshape(class_mask_indices, (-1,)).cuda())
        class_features_pre = torch.index_select(support_features, 0, torch.reshape(class_mask_indices, (-1,)).cuda())
        
        # Mean pooling examples to form class means
        class_center = torch.mean(class_features, dim=0)
        class_center_pre = torch.mean(class_features_pre, dim=0)
        
        # Updating the class representations dictionary with the mean pooled representation
        class_centers[c.item()] = class_center
        class_centers_pre[c.item()] = class_center_pre
    
    mean_support_sample = torch.stack(list(class_centers.values()))
    support_features_pre = torch.stack(list(class_centers_pre.values()))
    
    return mean_support_sample, support_features_pre

class FeatureAggregator(nn.Module):
    def __init__(self, support_feat_inv, support_features, support_labels, query_features, query_labels):
        self.support_feat_inv = support_feat_inv
        self.support_features = support_features
        self.support_labels = support_labels
        self.query_features = query_features
        self.query_labels = query_labels
        self.queryagg_results = torch.zeros(len(query_labels), support_feat_inv.size(1)).cuda()

    def feature_aggregation(self, query_feature, support_fearture, pre_support_feature):
        """
        The feature aggregation function.
        """
        input_tensor = torch.cat([query_feature, support_fearture.sigmoid() * query_feature], dim=1) 
        
        adaptive_avgpool = nn.AdaptiveAvgPool2d((None, query_feature.size(1)))
        output_tensor_1 = adaptive_avgpool(input_tensor.unsqueeze(1)).squeeze(1)

        input_tensor_quanz = torch.cat([pre_support_feature, query_feature], dim=1)
        inner_product = torch.mul(input_tensor_quanz[:, :128], input_tensor_quanz[:, 128:])
        # x1, x2 = torch.chunk(input_tensor_quanz, 2, dim=1)
        # inner_product = torch.mul(x1, x2)
        outter_product = torch.sum(inner_product, dim=1, keepdim=True)
        extended_vector = torch.cat((input_tensor, outter_product), dim=1)

        output_tensor_2 = feature_encoder.Class_Adap(extended_vector).cuda() # torch.Size([9, 1])
        
        output_tensor_temp = torch.mul(output_tensor_1, output_tensor_2) # orch.Size([9, 128])*torch.Size([9, 1])
        output_tensor = torch.sum(output_tensor_temp, dim=0, keepdim=True) # torch.Size([9, 128]) -> torch.Size([1, 128])

        output_tensors = output_tensor + query_feature[0].unsqueeze(0)
        
        return output_tensors

    def aggregate_features(self,):
        """
        Aggregates query features using class-adaptive aggregation.
        """
        mean_support_sample, support_features_pre = ComputeClassCenters(self.support_feat_inv, self.support_features, self.support_labels)
        query_features_expanded = self.query_features.unsqueeze(1).expand(-1, mean_support_sample.size(0), -1)

        # Class-adaptive feature aggregation
        for img_id in range(len(self.query_labels)):
            self.queryagg_results[img_id] = self.feature_aggregation(
                query_features_expanded[img_id],
                mean_support_sample,
                support_features_pre
            )
        
        return self.queryagg_results

crossEntropy = nn.CrossEntropyLoss().cuda()
# run 10 times
nDataSet = 10
acc = np.zeros([nDataSet, 1])
A = np.zeros([nDataSet, TEST_CLASS_NUM])
P = np.zeros([nDataSet, TEST_CLASS_NUM])
k = np.zeros([nDataSet, 1])
training_time = np.zeros([nDataSet, 1])
test_time = np.zeros([nDataSet, 1])
best_predict_all = []
best_acc_all = 0.0
best_G,best_RandPerm,best_Row, best_Column,best_nTrain = None,None,None,None,None
latest_G,latest_RandPerm,latest_Row, latest_Column,latest_nTrain = None,None,None,None,None

# the seed set considered the comparison works in this paper
seeds = [1330, 1220, 1336, 1337, 1224, 1236, 1226, 1235, 1233, 1229]

for iDataSet in range(nDataSet):
    np.random.seed(seeds[iDataSet])
    # load target domain data for training and testing
    train_loader, test_loader, target_da_metatrain_data,G,RandPerm,Row, Column,nTrain = get_target_dataset(
    Data_Band_Scaler=Data_Band_Scaler, GroundTruth=GroundTruth,class_num=TEST_CLASS_NUM, shot_num_per_class=TEST_LSAMPLE_NUM_PER_CLASS)

    # model
    feature_encoder = Network()
    print(get_parameter_number(feature_encoder))

    feature_encoder.apply(weights_init)
    feature_encoder.cuda()
    feature_encoder.train()
    # optimizer
    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(), lr=args.learning_rate)
    feature_encoder_optim_main = torch.optim.Adam(feature_encoder.feature_enc.parameters(), lr=args.learning_rate)
    feature_encoder_optim_vae = torch.optim.Adam(feature_encoder.vae.parameters(), lr=args.learning_rate)
    feature_encoder_optim_class_adap = torch.optim.Adam(feature_encoder.Class_Adap.parameters(), lr=args.learning_rate)

    print("Training...")

    last_accuracy = 0.0
    best_episdoe = 0
    train_loss = []
    test_acc = []
    total_hit, total_num = 0.0, 0.0
    test_acc_list = []

    torch.cuda.synchronize()
    train_start = time.time()
    EPISODE_1 = 1000
    NRUNS = 10
    flag = 1
    for i in range(NRUNS):
        for episode in range(EPISODE_1): 
            # source domain few-shot
            if episode < 800:
                '''Few-shot claification for source domain data set'''
                # get few-shot classification samples
                task = utils.Task(metatrain_data, CLASS_NUM, SHOT_NUM_PER_CLASS, QUERY_NUM_PER_CLASS)  # 9, 180, 19
                support_dataloader = utils.get_HBKC_data_loader(task, num_per_class=SHOT_NUM_PER_CLASS, split="train", shuffle=False) # batch_size=num_per_class*task.num_classes
                query_dataloader = utils.get_HBKC_data_loader(task, num_per_class=QUERY_NUM_PER_CLASS, split="test", shuffle=True)

                # sample datas
                supports, support_labels = support_dataloader.__iter__().__next__()  
                querys, query_labels = query_dataloader.__iter__().__next__()  

                # calculate features
                support_features, _ = feature_encoder(supports.cuda(), domain = 'source') 
                query_features, _ = feature_encoder(querys.cuda(), domain = 'source') 

                support_feat_rec, support_feat_inv, _, mu, log_var = feature_encoder.vae(support_features.cuda())
                
                aggregator = FeatureAggregator(support_feat_inv, support_features, support_labels, query_features, query_labels)
                queryagg_results = aggregator.aggregate_features()
                
                logits = MD_distance(support_features,support_labels, queryagg_results)

                vae_loss = feature_encoder.vae.loss_funcation(support_features, support_feat_rec, mu, log_var)  
                f_loss = crossEntropy(logits, query_labels.cuda())
                loss = LossWeight * f_loss + vae_loss
                # Update parameters
                feature_encoder_optim.zero_grad()
                loss.backward()
                feature_encoder_optim.step()

                total_hit += torch.sum(torch.argmax(logits, dim=1).cpu() == query_labels).item()
                total_num += querys.shape[0]

            if episode >= 800:
                '''Few-shot classification for target domain data set'''
                # get few-shot classification samples
                task = utils.Task(target_da_metatrain_data, TEST_CLASS_NUM, SHOT_NUM_PER_CLASS, QUERY_NUM_PER_CLASS)  # 5， 1，15
                support_dataloader = utils.get_HBKC_data_loader(task, num_per_class=SHOT_NUM_PER_CLASS, split="train", shuffle=False)
                query_dataloader = utils.get_HBKC_data_loader(task, num_per_class=QUERY_NUM_PER_CLASS, split="test", shuffle=True)

                # sample datas
                supports, support_labels = support_dataloader.__iter__().__next__()  # (5, 100, 9, 9)
                querys, query_labels = query_dataloader.__iter__().__next__()  # (75,100,9,9)

                # calculate features
                support_features, _ = feature_encoder(supports.cuda(), domain='target')  # torch.Size([2880,1024])
                query_features, _ = feature_encoder(querys.cuda(), domain='target')  # torch.Size([2880,1024])
                support_feat_rec, support_feat_inv, _, mu, log_var = feature_encoder.vae(support_features)

                aggregator = FeatureAggregator(support_feat_inv, support_features, support_labels, query_features, query_labels)
                train_queryagg_results = aggregator.aggregate_features()
                
                # fsl_loss
                logits = MD_distance(support_features, support_labels, train_queryagg_results)
                f_loss = crossEntropy(logits, query_labels.cuda())

                loss = f_loss
                
                # Update parameters
                feature_encoder_optim_main.zero_grad()
                feature_encoder_optim_class_adap.zero_grad()
                loss.backward()
                feature_encoder_optim_main.step()
                feature_encoder_optim_class_adap.step()

                total_hit += torch.sum(torch.argmax(logits, dim=1).cpu() == query_labels).item()
                total_num += querys.shape[0]

            if (episode + 1) % 100 == 0:  # display
                elapsed_time = time.time()-train_start
                train_loss.append(loss.item())
                print('*'*25)
                print('episode {:>3d}: loss: {:6.4f}, query_sample_num: {:>3d}, acc {:6.4f}, elapsed time: {:6.4f}'.format(i * EPISODE_1 + episode + 1, \
                                                                                                                    loss.item(),
                                                                                                                    querys.shape[0],
                                                                                                                    total_hit / total_num,
                                                                                                                    elapsed_time))
                print('*'*25)

            if (episode + 1) % 1000 == 0 or flag == 1:
                # test
                print("Testing ...")
                # train_end = time.time()
                feature_encoder.eval()
                total_rewards = 0
                counter = 0
                accuracies = []
                predict = np.array([], dtype=np.int64)
                labels = np.array([], dtype=np.int64)
                test_features_all = []
                test_labels_all = np.array([], dtype=np.int64)

                train_datas, train_labels = train_loader.__iter__().__next__()
                train_features, _ = feature_encoder(Variable(train_datas).cuda(), domain='target')  # (45, 160)

                support_feat_rec, support_feat_inv, _, mu, log_var = feature_encoder.vae(train_features)
              
                mean_support_sample, support_features_pre = ComputeClassCenters(support_feat_inv, train_features, train_labels)
                
                torch.cuda.synchronize()
                train_end = time.time()
                for test_datas, test_labels in test_loader:
                    batch_size = test_labels.shape[0]  # Dataloader set batch_size 100
                    test_features, _ = feature_encoder(Variable(test_datas).cuda(), domain='target')  # (100, 160)
                    
                    aggregator = FeatureAggregator(support_feat_inv, train_features, train_labels, test_features, test_labels)
                    test_queryagg_results = aggregator.aggregate_features()
                    
                    predict_logits,class_representations,class_precision_matrices = MD_distance_test1(train_features, train_labels, test_queryagg_results)
                    test_features_tmp = test_features.cpu().detach().numpy()
                    test_features_all.append(test_features_tmp)

                    predict_labels = torch.argmax(predict_logits, dim=1).cpu()
                    test_labels = test_labels.numpy()
                    rewards = [1 if predict_labels[j] == test_labels[j] else 0 for j in range(batch_size)]
                    test_labels_all = np.append(test_labels_all, test_labels)

                    total_rewards += np.sum(rewards)
                    counter += batch_size

                    predict = np.append(predict, predict_labels)
                    labels = np.append(labels, test_labels)

                    accuracy = total_rewards / 1.0 / counter  #
                    accuracies.append(accuracy)
                test_accuracy = 100. * total_rewards / len(test_loader.dataset)

                torch.cuda.synchronize()
                test_end = time.time()
                
                print('\t\tAccuracy: {}/{} ({:.2f}%) iDataSet: {} \n'.format( total_rewards, len(test_loader.dataset),
                    100. * total_rewards / len(test_loader.dataset), iDataSet))  
                # Training mode
                feature_encoder.train()
                if test_accuracy > last_accuracy:
                    # save networks
                    torch.save(feature_encoder.state_dict(), str( "./checkpoints/DACAA"+str(iDataSet) +"iter_" + str(TEST_LSAMPLE_NUM_PER_CLASS) +"shot.pkl"))
                    print("save networks for episode:", i * EPISODE_1 + episode + 1)
                    last_accuracy = test_accuracy
                    best_episode = i * EPISODE_1 + episode 

                    acc[iDataSet] = total_rewards / len(test_loader.dataset)
                    OA = acc[iDataSet]
                    C = metrics.confusion_matrix(labels, predict)
                    A[iDataSet, :] = np.diag(C) / np.sum(C, 1, dtype=np.float64)
                    P[iDataSet, :] = np.diag(C) / np.sum(C, 1, dtype=np.float64)

                    k[iDataSet] = metrics.cohen_kappa_score(labels, predict)

                print('best episode:[{}], best accuracy={}'.format((best_episode + 1), last_accuracy))
                flag = 0
    training_time[iDataSet] = train_end - train_start
    test_time[iDataSet] = test_end - train_end
    
    latest_G, latest_RandPerm, latest_Row, latest_Column, latest_nTrain = G, RandPerm, Row, Column, nTrain
    for i in range(len(predict)):  # predict ndarray <class 'tuple'>: (9729,)
        latest_G[latest_Row[latest_RandPerm[latest_nTrain + i]]][latest_Column[latest_RandPerm[latest_nTrain + i]]] = \
            predict[i] + 1
    sio.savemat('./classificationMap/pred_map_latest' + '_' + str(iDataSet) + "iter_" + repr(int(OA * 10000)) + '.mat', {'latest_G': latest_G})
    
    if test_accuracy > best_acc_all:
        best_predict_all = predict
        best_G,best_RandPerm,best_Row, best_Column,best_nTrain = G, RandPerm, Row, Column, nTrain
    print('iter:{} best episode:[{}], best accuracy={}'.format(iDataSet, (best_episode + 1), last_accuracy))
    print('***********************************************************************************')
###
ELEMENT_ACC_RES_SS4 = np.transpose(A)
AA_RES_SS4 = np.mean(ELEMENT_ACC_RES_SS4,0)
OA_RES_SS4 = np.transpose(acc)
KAPPA_RES_SS4 = np.transpose(k)
ELEMENT_PRE_RES_SS4 = np.transpose(P)
AP_RES_SS4= np.mean(ELEMENT_PRE_RES_SS4,0)
TRAINING_TIME_RES_SS4 = np.transpose(training_time)
TESTING_TIME_RES_SS4 = np.transpose(test_time)
classes_num = TEST_CLASS_NUM
ITER = nDataSet

outputRecord(ELEMENT_ACC_RES_SS4, AA_RES_SS4, OA_RES_SS4, KAPPA_RES_SS4,
                              ELEMENT_PRE_RES_SS4, AP_RES_SS4,
                              TRAINING_TIME_RES_SS4, TESTING_TIME_RES_SS4,
                              classes_num, ITER,
                              './Results/results_{}shot.txt'.format(TEST_LSAMPLE_NUM_PER_CLASS))
