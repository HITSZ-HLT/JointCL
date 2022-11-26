import argparse
from tqdm import tqdm
import faiss
import os
import torch
from torch import optim
import random
import numpy as np
from criterion import TraditionCriterion, Stance_loss
from torch.utils.data import RandomSampler, DataLoader
from tensorboardX import SummaryWriter
from sklearn.metrics import f1_score, accuracy_score
from data_utils import  Tokenizer4Bert,ZeroshotDataset
from transformers import BertModel

from models.bert_scl_prototype_graph import BERT_SCL_Proto_Graph
import pickle
from time import strftime,localtime

from sklearn.metrics import classification_report
gpu_id = 3
torch.cuda.set_device(gpu_id)
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

class Instructor(object):
    def __init__(self,opt):
        self.opt = opt
        tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name)
        bert_proto = BertModel.from_pretrained(opt.pretrained_bert_name)
        # bert_class = BertModel.from_pretrained(opt.pretrained_bert_name)
        self.model = opt.model_class(opt,bert_proto, ).to(opt.device)
        print("using model: ",opt.model_name)
        print("running dataset: ", opt.dataset)
        print("output_dir: ", opt.output_dir)
        train_file_name = './vast_train1.dat'
        dev_file_name = './vast_dev1.dat'
        test_file_name = './vast_test1.dat'
        try:
            self.trainset = pickle.load(open(train_file_name, 'rb'))
            self.valset = pickle.load(open(dev_file_name, 'rb'))
            self.testset = pickle.load(open(test_file_name, 'rb'))
        except:

            self.trainset = ZeroshotDataset(data_dir=self.opt.train_dir,tokenizer=tokenizer, opt=self.opt,data_type = 'train')
            self.valset = ZeroshotDataset(data_dir=self.opt.dev_dir, tokenizer=tokenizer, opt=self.opt,data_type = 'dev')
            self.testset = ZeroshotDataset(data_dir=self.opt.test_dir,tokenizer=tokenizer, opt=self.opt,data_type = 'test')
            pickle.dump(self.trainset, open(train_file_name, 'wb'))
            pickle.dump(self.valset, open(dev_file_name, 'wb'))
            pickle.dump(self.valset, open(test_file_name, 'wb'))

        if 'scl' in self.opt.model_name:

            # self.prototype_criterion = Stance_loss(opt.temperature).to(opt.device)
            self.stance_criterion = Stance_loss(opt.temperature).to(opt.device)
            self.target_criterion = Stance_loss(opt.temperature).to(opt.device)
            self.logits_criterion = TraditionCriterion(opt)
            params = ([p for p in self.model.parameters()] + [p for p in self.target_criterion.parameters()])
        else:
            self.criterion = TraditionCriterion(opt)
            params = ([p for p in self.model.parameters()])
        self.optimizer = self.opt.optim_class(params, lr=self.opt.lr)


    def run_tradition(self):
        best_acc, best_f1 = self.train_traditon()
        # opt.output_dir = '/home/zhuqinglin/Pycharm/EMNLP2021/Zero-Shot-Contrastive-v2/bert-scl-vast/bert-scl-prototype/zeroshot/2021-09-12 01-57-03/'
        state_dict_dir = opt.output_dir + "/state_dict"
        print("\n\nReload the best model with best acc {} from path {}\n\n".format(best_acc, state_dict_dir))
        ckpt = torch.load(os.path.join(state_dict_dir, "best_acc_model.bin"))
        self.model.load_state_dict(ckpt)
        acc,f1 = self.test_tradition()

        print("\n\nReload the best model with best f1 {} from path {}\n\n".format(best_f1, state_dict_dir))
        ckpt = torch.load(os.path.join(state_dict_dir, "best_f1_model.bin"))
        self.model.load_state_dict(ckpt)
        acc,f1 = self.test_tradition()

        return acc,f1


    def compute_features(self,train_loader):
        print('Computing features...')
        self.model.eval()
        features = torch.zeros(len(train_loader.dataset),self.opt.bert_dim).cuda()
        for  batch in tqdm(train_loader):
            input_features = [batch[feat_name].to(self.opt.device) for feat_name in self.opt.input_features]
            index = batch['index']
            with torch.no_grad():

                feature = self.model.prototype_encode(input_features)
                feature = feature.squeeze(dim=1)
                features[index] = feature

        return features.cpu()


    def run_kmeans(self, x):
        """
        Args:
            x: data to be clustered
        """

        print('performing kmeans clustering')
        results = {'im2cluster':[],'centroids':[],'density':[]}

        for seed, num_cluster in enumerate(self.opt.num_cluster):
            # intialize faiss clustering parameters
            d = x.shape[1]
            k = int(num_cluster)
            clus = faiss.Clustering(d, k)
            clus.verbose = True
            clus.niter = 20
            clus.nredo = 5
            clus.seed = seed
            clus.max_points_per_centroid = 1000
            clus.min_points_per_centroid = 10

            res = faiss.StandardGpuResources()
            cfg = faiss.GpuIndexFlatConfig()
            cfg.useFloat16 = False
            cfg.device = gpu_id
            index = faiss.GpuIndexFlatL2(res, d, cfg)

            clus.train(x, index)

            D, I = index.search(x, 1) # for each sample, find cluster distance and assignments
            im2cluster = [int(n[0]) for n in I]

            # get cluster centroids
            centroids = faiss.vector_to_array(clus.centroids).reshape(k,d)

            # sample-to-centroid distances for each cluster
            Dcluster = [[] for c in range(k)]
            for im,i in enumerate(im2cluster):
                Dcluster[i].append(D[im][0])

            # concentration estimation (phi)
            density = np.zeros(k)
            for i,dist in enumerate(Dcluster):
                if len(dist)>1:
                    d = (np.asarray(dist)**0.5).mean()/np.log(len(dist)+10)
                    density[i] = d

                    #if cluster only has one point, use the max to estimate its concentration
            dmax = density.max()
            for i,dist in enumerate(Dcluster):
                if len(dist)<=1:
                    density[i] = dmax

            density = density.clip(np.percentile(density,10),np.percentile(density,90)) #clamp extreme values for stability
            density = self.opt.temperature*density/density.mean()  #scale the mean to temperature

            # convert to cuda Tensors for broadcast
            centroids = torch.Tensor(centroids).cuda()
            centroids = torch.nn.functional.normalize(centroids, p=2, dim=1)

            im2cluster = torch.LongTensor(im2cluster).cuda()
            density = torch.Tensor(density).cuda()

            results['centroids'].append(centroids)
            results['density'].append(density)
            results['im2cluster'].append(im2cluster)

        return results


    def run_prototype(self,train_loader):
        # ----------------------- Run-Kmeans -----------------------

        self.opt.warmup_epoch = 0
        self.opt.num_cluster = [100]
        cluster_result = None

        # compute momentum features for center-cropped images
        features = self.compute_features(train_loader)
        # pickle.dump(features, open('try_features.dat', 'wb'))
        # features = pickle.load(open('try_features.dat', 'rb'))

        # placeholder for clustering result
        cluster_result = {'im2cluster':[],'centroids':[],'density':[]}
        for num_cluster in self.opt.num_cluster:
            cluster_result['im2cluster'].append(torch.zeros(len(train_loader.dataset),dtype=torch.long).cuda())
            cluster_result['centroids'].append(torch.zeros(int(num_cluster),self.opt.bert_dim).cuda())
            cluster_result['density'].append(torch.zeros(int(num_cluster)).cuda())

        features = features.numpy()
        cluster_result = self.run_kmeans(features)  # run kmeans clustering on master node
        # save the clustering result
        # torch.save(cluster_result,os.path.join(args.exp_dir, 'clusters_%d'%epoch))
        return cluster_result




    def train_traditon(self):
        sampler = RandomSampler(self.trainset)
        train_loader = DataLoader(self.trainset, batch_size=self.opt.batch_size, sampler=sampler)
        train_loader_prototype = DataLoader(self.trainset, batch_size=self.opt.batch_size, sampler=sampler)
        print("Train loader length: {}".format(len(train_loader)))
        optimizer = self.optimizer
        best_acc = 0
        best_f1 = 0
        cnt = 0

        for i_epoch in range(self.opt.epochs):


            #  ----------------------- train -----------------------

            print('>' * 20, 'epoch:{}'.format(i_epoch), '<'*20)
            n_correct, n_total, loss_total = 0, 0, 0
            self.model.train()
            for i_batch, batch in enumerate(train_loader):

                if i_batch % int(len(train_loader)/self.opt.cluster_times) == 0:
                    cluster_result = self.run_prototype(train_loader_prototype)

                    # pickle.dump(features, open('try_features.dat', 'wb'))
                    # features = pickle.load(open('try_features.dat', 'rb'))



                input_features = [batch[feat_name].to(self.opt.device) for feat_name in self.opt.input_features]
                index = batch['index']
                true_stance = batch['polarity']
                # true_targets = batch['polarity']
                if opt.n_gpus > 0:
                    true_stance = true_stance.to(self.opt.device)
                if 'scl' in self.opt.model_name:
                    true_targets = batch['topic_index']
                    s_t_list= [str(i+j) for i, j in zip(true_stance,true_targets)]
                    s_t_list_drop = list(set(s_t_list))

                    polarity2label = { polarity:idx for idx, polarity in enumerate(s_t_list_drop)}

                    s_t = torch.Tensor([polarity2label[i] for i in s_t_list]).cuda()



                    feature = self.model.prototype_encode(input_features)
                    logits, node_for_con = self.model(input_features+[cluster_result['centroids']])
                    self.cluster_result = [cluster_result['centroids']]



                    if cluster_result is not None:

                        for n, (im2cluster,prototypes,density) in enumerate(zip(cluster_result['im2cluster'],cluster_result['centroids'],cluster_result['density'])):
                            # get positive prototypes

                            prototype_loss = self.target_criterion(node_for_con,s_t)
                            stance_loss = self.stance_criterion(feature,true_stance)
                    else:
                        prototype_loss = 0.0

                    logits_loss = self.logits_criterion(logits, true_stance)

                    # target_loss = self.target_criterion(feature, true_stance,true_targets)
                    loss = logits_loss + stance_loss * self.opt.stance_loss_weight + prototype_loss * self.opt.prototype_loss_weight
                else:
                    logits = self.model(input_features)
                    loss = self.criterion(logits, true_stance)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                n_correct += (torch.argmax(logits, -1) == true_stance).sum().item()
                n_total += len(logits)
                loss_total += loss.item() * len(logits)
                if cnt % self.opt.log_step == 0:
                    train_acc = n_correct / n_total
                    train_loss = loss_total / n_total
                    if 'scl' in self.opt.model_name:
                        print("Train step: {} acc:{:.5} total_loss: {:.5}  loss:{:.5},stance loss: {:.5} ,prototype loss: {:.5} ".
                              format(cnt, train_acc, train_loss, loss,stance_loss,prototype_loss))
                    else:
                        print("Train step: {} acc:{} loss: {}".format(cnt, train_acc, train_loss))

                if cnt != 0 and cnt % self.opt.eval_steps == 0 and i_epoch>0:
                    eval_acc, eval_f1 = self.dev_tradition()
                    if eval_acc > best_acc:
                        print('Better ACC! Saving model!')
                        best_acc = eval_acc
                        print("Saving model of best acc: {}".format(best_acc))
                        state_dict_dir = opt.output_dir + "/state_dict"
                        if not os.path.exists(state_dict_dir):
                            os.makedirs(state_dict_dir)
                        torch.save(self.model.state_dict(), os.path.join(state_dict_dir, "best_acc_model.bin"))
                    if eval_f1 > best_f1:
                        print('Better F1! Saving model!')
                        best_f1 = eval_f1
                        print("Saving model of best f1: {}".format(best_f1))
                        state_dict_dir = opt.output_dir + "/state_dict"
                        if not os.path.exists(state_dict_dir):
                            os.makedirs(state_dict_dir)
                        torch.save(self.model.state_dict(), os.path.join(state_dict_dir, "best_f1_model.bin"))
                cnt += 1
        print("Training finished.")
        return best_acc, best_f1

    def dev_tradition(self):
        self.model.eval()
        sampler = RandomSampler(self.valset)
        dev_loader = DataLoader(dataset=self.valset, batch_size=self.opt.eval_batch_size, sampler=sampler)
        all_labels = []
        all_logits = []
        eval_loss = 0
        cnt = 0
        for i_batch, batch in enumerate(dev_loader):
            input_features = [batch[feat_name].to(self.opt.device) for feat_name in self.opt.input_features]
            true_stance = batch['polarity']
            if opt.n_gpus > 0:
                true_stance = true_stance.to(self.opt.device)
            with torch.no_grad():
                if 'scl' in self.opt.model_name:
                    try:
                        pickle.dump(self.cluster_result, open(opt.output_dir + '/cluster_result', 'wb'))
                    except:
                        self.cluster_result = pickle.load(open(opt.output_dir +'/cluster_result', 'rb'))
                    logits,_ = self.model(input_features+self.cluster_result)
                    loss = self.logits_criterion(logits, true_stance)
                else:
                    logits = self.model(input_features)
                    loss = self.criterion(logits, true_stance)
            if self.opt.n_gpus > 1:
                loss = loss.mean().item()
            else:
                loss = loss.item()
            eval_loss += loss
            labels = true_stance.detach().cpu().numpy()
            logits = logits.detach().cpu().numpy()
            all_labels.append(labels)
            all_logits.append(logits)
            cnt = cnt + 1
        all_labels = np.concatenate(all_labels, axis=0)
        all_logits = np.concatenate(all_logits, axis=0)
        preds = all_logits.argmax(axis=1)

        acc = accuracy_score(y_true=all_labels, y_pred=preds)
        f1 = f1_score(all_labels, preds, average='macro')
        self.model.train()
        return acc, f1

    def test_tradition(self):
        self.model.eval()
        sampler = RandomSampler(self.testset)
        test_loader = DataLoader(dataset=self.testset, batch_size=self.opt.eval_batch_size, sampler=sampler)
        all_labels = []
        all_logits = []
        eval_loss = 0
        cnt = 0
        for i_batch, batch in enumerate(test_loader):
            input_features = [batch[feat_name].to(self.opt.device) for feat_name in self.opt.input_features]
            true_stance = batch['polarity']
            if opt.n_gpus > 0:
                true_stance = true_stance.to(self.opt.device)
            with torch.no_grad():
                if 'scl' in self.opt.model_name:
                    try:
                        pickle.dump(self.cluster_result, open(opt.output_dir + '/cluster_result', 'wb'))
                    except:
                        self.cluster_result = pickle.load(open(opt.output_dir +'/cluster_result', 'rb'))
                    logits,_ = self.model(input_features+self.cluster_result)
                    loss = self.logits_criterion(logits, true_stance)
                else:
                    logits = self.model(input_features)
                    loss = self.criterion(logits, true_stance)
            if self.opt.n_gpus > 1:
                loss = loss.mean().item()
            else:
                loss = loss.item()
            eval_loss += loss
            labels = true_stance.detach().cpu().numpy()
            logits = logits.detach().cpu().numpy()
            all_labels.append(labels)
            all_logits.append(logits)
            cnt = cnt + 1
        all_labels = np.concatenate(all_labels, axis=0)
        all_logits = np.concatenate(all_logits, axis=0)
        preds = all_logits.argmax(axis=1)
        acc = accuracy_score(y_true=all_labels, y_pred=preds)
        f1 = f1_score(all_labels, preds, average='macro')
        print(classification_report(all_labels, preds, digits=6))
        print("Test Acc: {} F1:{}".format(acc, f1))
        self.model.train()
        return acc,f1

    def save_evaluation_result(self,f1,acc,score_dict,file_name):
        result_path = os.path.join(self.opt.output_dir, file_name)
        with open(result_path, 'w', encoding='utf-8') as out_file:
            out_file.write('Test Acc: {} F1: {}\nReport:\n'.format(acc, f1))
            for k in sorted(score_dict.keys()):
                scores = score_dict[k]
                for meas_name, value in scores.items():
                    out_file.write("{} {}: {}\n".format(k, meas_name, value))
                out_file.write("\n")

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    # config
    parser = argparse.ArgumentParser()

    parser.add_argument('--model_name', default='bert-scl-prototype-graph', type=str,required=False)
    parser.add_argument('--type', default=0, help='2 for all,0 for zero shot ,1 for few shot',type=str, required=False)
    parser.add_argument('--dataset', default='zeroshot', type=str,required=False)
    parser.add_argument('--output_par_dir',default='bert-scl-vast1',type=str)
    parser.add_argument('--polarities', default=["pro","con","neutral"], nargs='+', help="if just two polarity switch to ['positive', 'negtive']",required=False)
    parser.add_argument('--optimizer', default='adam', type=str,required=False)
    parser.add_argument('--temperature', default=0.07, type=float,required=False)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str,required=False)
    parser.add_argument('--lr', default=5e-6, type=float, help='try 5e-5, 2e-5, 1e-3 for others',required=False)
    parser.add_argument('--dropout', default=0.1, type=float,required=False)
    parser.add_argument('--l2reg', default=0.001, type=float,required=False)
    parser.add_argument('--log_step', default=10, type=int,required=False)
    parser.add_argument('--log_path', default="./log", type=str,required=False)
    parser.add_argument('--embed_dim', default=300, type=int,required=False)
    parser.add_argument('--hidden_dim', default=128, type=int,required=False,help="lstm encoder hidden size")
    parser.add_argument('--feature_dim', default=2*128, type=int,required=False,help="feature dim after encoder depends on encoder")
    parser.add_argument('--output_dim', default=64, type=int,required=False)
    parser.add_argument('--relation_dim',default=100,type=int,required=False)
    parser.add_argument('--bert_dim', default=768, type=int,required=False)
    parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str,required=False)
    parser.add_argument('--max_seq_len', default=200, type=int,required=False)
    parser.add_argument('--train_dir', default='./datasets/VAST/vast_train.csv', type=str,required=False)
    parser.add_argument('--dev_dir', default='./datasets/VAST/vast_dev.csv', type=str,required=False)
    parser.add_argument('--test_dir', default='./datasets/VAST/vast_test.csv', type=str,required=False)
    parser.add_argument('--stance_loss_weight',default=1,type=float,required=False)
    parser.add_argument('--prototype_loss_weight',default=0.01,type=float,required=False)
    parser.add_argument('--alpha', default=0.8, type=float,required=False)
    parser.add_argument('--beta', default=1.2, type=float,required=False)

    parser.add_argument('--device', default=None, type=str, help='e.g. cuda:0',required=False)
    parser.add_argument('--seed', default=0, type=int, help='set seed for reproducibility')

    parser.add_argument("--batch_size", default=16, type=int, required=False)
    parser.add_argument("--eval_batch_size", default=16, type=int, required=False)
    parser.add_argument("--epochs", default=15, type=int, required=False)
    parser.add_argument("--eval_steps", default=50, type=int, required=False)
    parser.add_argument("--cluster_times", default=5, type=int, required=False)


    # graph para
    parser.add_argument('--gnn_dims', default='192,192', type=str,required=False)
    parser.add_argument('--att_heads', default='4,4', type=str,required=False)
    parser.add_argument('--dp', default=0.1, type=float)


    opt = parser.parse_args()

    if opt.seed:
        set_seed(opt.seed)
    model_classes = {

        'bert-scl-prototype-graph': BERT_SCL_Proto_Graph,
    }
    input_features = {

        'bert-scl-prototype-graph':['concat_bert_indices', 'concat_segments_indices'],

    }

    optimizers = {
        'adam':optim.Adam,
    }

    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt.n_gpus = torch.cuda.device_count()
    opt.model_class = model_classes[opt.model_name]
    opt.optim_class = optimizers[opt.optimizer]
    opt.input_features = input_features[opt.model_name]
    opt.output_dir = os.path.join(opt.output_par_dir,opt.model_name,opt.dataset,strftime("%Y-%m-%d %H-%M-%S", localtime())) ##get output directory to save results

    opt.num_labels = len(opt.polarities)
    if not os.path.exists(opt.output_dir):
        os.makedirs(opt.output_dir)
    writer = SummaryWriter(opt.log_path)
    print(opt)
    ins = Instructor(opt)
    acc,f1 = ins.run_tradition()
    print('#' * 20, 'result : Acc {}, F1 {}'.format(acc, f1) )
    writer.close()
