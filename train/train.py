import copy
import torch
import argparse
from datasets import MNIST75sp, SPMotif
from torch_geometric.data import DataLoader
from torch import Tensor
from torch_geometric.nn import VGAE as PyGVGAE
from torch.utils.data import random_split
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphConv, BatchNorm, global_mean_pool
from torch_geometric.utils import softmax, degree
from utils.mask import set_masks, clear_masks
from tqdm import tqdm
import os
import random
import numpy as np
import os.path as osp
from torch.autograd import grad
from utils.logger import Logger
from datetime import datetime
from utils.helper import random_partition, args_print
from utils.get_subgraph import split_graph, relabel
from train.nets import *
from train.utils_mp import *
from collections import defaultdict
from gnn import MolHivNet, GINVirtual_node, LEGNN, kGNN, ARMAGNN


class InnerProductDecoder_Domain(torch.nn.Module):

    def forward(self, z: Tensor, edge_index: Tensor, domain_embs: Tensor =None,
                          sigmoid: bool = False) -> Tensor:
        #z=torch.nn.functional.normalize(z,dim=-1)
        if domain_embs!=None:
            z=z*domain_embs
        value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
        return torch.sigmoid(value) if sigmoid else value



class CausalAttNet(nn.Module):
    
    def __init__(self, causal_ratio):
        super(CausalAttNet, self).__init__()
        self.conv1 = GraphConv(in_channels=5, out_channels=args.channels)
        self.conv2 = GraphConv(in_channels=args.channels, out_channels=args.channels)
        self.mlp = nn.Sequential(
            nn.Linear(args.channels*2, args.channels*4),
            nn.ReLU(),
            nn.Linear(args.channels*4, 1)
        )
        self.ratio = causal_ratio
    def forward(self, data):
        x = F.relu(self.conv1(data.x, data.edge_index, data.edge_attr.view(-1)))
        x = self.conv2(x, data.edge_index, data.edge_attr.view(-1))

        row, col = data.edge_index
        edge_rep = torch.cat([x[row], x[col]], dim=-1)
        edge_score = self.mlp(edge_rep).view(-1)

        (causal_edge_index, causal_edge_attr, causal_edge_weight), \
        (conf_edge_index, conf_edge_attr, conf_edge_weight) = split_graph(data,edge_score, self.ratio)

        causal_x, causal_edge_index, causal_batch, _ = relabel(x, causal_edge_index, data.batch)
        conf_x, conf_edge_index, conf_batch, _ = relabel(x, conf_edge_index, data.batch)

        return (causal_x, causal_edge_index, causal_edge_attr, causal_edge_weight, causal_batch),\
                (conf_x, conf_edge_index, conf_edge_attr, conf_edge_weight, conf_batch),\
                edge_score

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False #True


if __name__ == "__main__":
    # Arguments
    parser = argparse.ArgumentParser(description='Training for Causal Feature Learning')
    parser.add_argument('--dataset', default='SPMotif', type=str, help='datasets.')
    parser.add_argument('--bias', default='0.333', type=str, help='select bias extend')
    parser.add_argument('--cuda', default=0, type=int, help='cuda device')
    parser.add_argument('--datadir', default='data/', type=str, help='directory for datasets.')
    parser.add_argument('--epoch', default=2000, type=int, help='training iterations')
    parser.add_argument('--reg', default=1, type=int)
    parser.add_argument('--seed',  nargs='?', default='[1,2,3]', help='random seed')
    parser.add_argument('--channels', default=32, type=int, help='width of network')
    parser.add_argument('--commit', default='', type=str, help='experiment name')
    parser.add_argument('--hidden_channels', default=128, type=int)
    parser.add_argument('--decode_size', default=5, type=int, help='decode size')
    parser.add_argument('--dropout', default=0.1, type=float, help='dropout')
    parser.add_argument('--pretrain', default=10, type=int, help='pretrain epoch')
    parser.add_argument('--use_id_graph', default=False, type=bool, help='pretrain epoch')



    # basic
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--net_lr', default=1e-3, type=float, help='learning rate for the predictor')
    args = parser.parse_args()
    args.seed = eval(args.seed)

    # dataset
    if args.dataset=='MNIST':
        num_classes = 10
        args.input_size=5
        n_train_data, n_val_data = 20000, 5000
        train_val = MNIST75sp(osp.join(args.datadir, 'MNISTSP/'), mode='train')
        perm_idx = torch.randperm(len(train_val), generator=torch.Generator().manual_seed(0))
        train_val = train_val[perm_idx]
        train_dataset, val_dataset = train_val[:n_train_data], train_val[-n_val_data:]
        test_dataset = MNIST75sp(osp.join(args.datadir, 'MNISTSP/'), mode='test')
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        n_test_data = float(len(test_loader.dataset))

        color_noises = torch.load(osp.join(args.datadir, 'MNISTSP/raw/mnist_75sp_color_noise.pt')).view(-1,3)
    elif args.dataset=='SPMotif':
        num_classes = 3
        args.input_size = 4
        train_dataset = SPMotif(osp.join(args.datadir, f'SPMotif-{args.bias}/'), mode='train')
        val_dataset = SPMotif(osp.join(args.datadir, f'SPMotif-{args.bias}/'), mode='val')
        test_dataset = SPMotif(osp.join(args.datadir, f'SPMotif-{args.bias}/'), mode='test')
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
        n_train_data, n_val_data = len(train_dataset), len(val_dataset)
        n_test_data = float(len(test_dataset))
    elif args.dataset=='SST2':
        from datasets.graphsst2_dataset import get_dataset, get_dataloader
        num_classes = 2
        args.input_size = 768
        dataset = get_dataset(dataset_dir='data/', dataset_name='Graph_SST2', task=None)
        dataloader = get_dataloader(dataset,
                                    batch_size=args.batch_size,
                                    degree_bias=True,
                                    seed=args.seed)
        train_loader = dataloader['train']
        val_loader = dataloader['eval']
        test_loader = dataloader['test']
        n_train_data, n_val_data = len(train_loader.dataset), len(val_loader.dataset)
        n_test_data = float(len(test_loader.dataset))
    elif args.dataset=='Molhiv':
        from ogb.graphproppred import PygGraphPropPredDataset, Evaluator

        num_classes = 1
        args.input_size = 9
        dataset = PygGraphPropPredDataset(name='ogbg-molhiv')
        split_idx = dataset.get_idx_split()
        train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(dataset[split_idx["test"]], batch_size=args.batch_size, shuffle=False)
        n_train_data, n_val_data, n_test_data = len(train_loader.dataset), len(val_loader.dataset), float(
            len(test_loader.dataset))
        evaluator = Evaluator('ogbg-molhiv')

    # logger
    datetime_now = datetime.now().strftime("%Y%m%d-%H%M%S")
    all_info = defaultdict(list)
    experiment_name = f'mnistsp.{bool(args.reg)}.{args.commit}.netlr_{args.net_lr}.batch_{args.batch_size}'\
                      f'.channels_{args.channels}.seed_{args.seed}.{datetime_now}'
    exp_dir = osp.join('local/', experiment_name)
    os.makedirs(exp_dir, exist_ok=True)
    logger = Logger.init_logger(filename=exp_dir + '/_output_.log')
    args_print(args, logger)

    device = torch.device('cuda:%d' % args.cuda if torch.cuda.is_available() else 'cpu')





    if args.dataset=='MNIST' :
        gnn=GCN(in_channels=args.hidden_channels,
                                hidden_channels=args.hidden_channels,
                                out_channels=num_classes,
                                num_layers=3,
                                dropout=args.dropout,use_bn=False).cuda()
        encoder = kGNN(in_channels=args.input_size, hid_channels=args.hidden_channels, num_unit=4).cuda()

    elif args.dataset=='Molhiv':
        #from gnn import MolHivNet, GINVirtual_node
        #gnn = GINVirtual_node(num_layers=2, emb_dim=300, dropout=0).cuda()
        gnn=GCN(in_channels=300,
                                hidden_channels=args.hidden_channels,
                                out_channels=num_classes,
                                num_layers=3,
                                dropout=args.dropout,use_bn=False).cuda()
        encoder= GINVirtual_node(num_layers=2, emb_dim=300, dropout=0).cuda()

    elif args.dataset=='SPMotif':
        args.hidden_channels=64
        args.decode_size=5
        args.dropout=0.1
        args.net_lr=1e-3
        gnn=GCN(in_channels=args.hidden_channels,
                                hidden_channels=args.hidden_channels,
                                out_channels=num_classes,
                                num_layers=2,
                                dropout=args.dropout,use_bn=True).cuda()
        encoder= LEGNN(in_channels=args.input_size, hid_channels=args.hidden_channels, num_unit=4).cuda()

    elif args.dataset=='SST2':
        gnn=GCN(in_channels=args.hidden_channels,
                                hidden_channels=args.hidden_channels,
                                out_channels=num_classes,
                                num_layers=3,
                                dropout=args.dropout,use_bn=False).cuda()
        encoder = ARMAGNN(num_layers=2, in_channels=args.input_size, hid_channels=args.hidden_channels).cuda()




    batch_gnn_enocder=batch_gnn(in_channels=args.hidden_channels*2, hidden_channels=args.hidden_channels,
                                encoder_mu=encoder, encoder_logstd=encoder,
                                output_type='hidden').cuda()

    vgae1 = PyGVGAE(batch_gnn_enocder)
    vgae2 = PyGVGAE(batch_gnn_enocder)


    decoder=InnerProductDecoder_Domain()





    #gnn_output=batch_gnn(in_channels=args.hidden_channels, hidden_channels=args.hidden_channels,
    #          encoder_mu=gnn, output_type='pool').cuda()

    pool=Pool(in_channels=args.hidden_channels)
    decode_size=args.decode_size #10 #5


    for seed in args.seed:
        set_seed(seed)
        # models and optimizers
        model_optimizer = torch.optim.Adam(
            list(vgae1.parameters())+list(vgae2.parameters())+list(gnn.parameters()),lr=args.net_lr)
            #list(gnn_output.parameters()),



        CELoss = nn.CrossEntropyLoss(reduction="mean")
        BCELoss=nn.BCEWithLogitsLoss(reduction="mean")

        def train_mode():
            gnn.train();vgae1.train();vgae2.train()
            
        def val_mode():
            gnn.eval();vgae1.eval();vgae2.eval()


        def test(loader):
            acc=0
            out_len=0
            n_samples = 0
            y_true = []
            y_pred = []
            for j, graph in enumerate(loader):
                graph.to(device)

                if args.dataset=='MNIST':
                    noise_level = 0.4
                    noise = color_noises[n_samples:n_samples + graph.x.size(0), :].to(device) * noise_level
                    graph.x[:, :3] = graph.x[:, :3] + noise
                    n_samples += graph.x.size(0)

                mu, logstd = vgae1.encoder(graph.x.float(), graph.edge_index, graph.edge_attr, graph.batch)  # [batch, args.hidden]
                mu, logstd = pool(mu, None, graph.batch), pool(logstd, None,
                                                               graph.batch)  # [num_graph, args.hidden]

                embs, edge_index= vgae_generate(vgae1, mu, logstd)



                output_total=[gnn(emb, edge).mean(0) for emb, edge in zip(embs, edge_index)]


                if args.dataset!='Molhiv':
                    out = torch.stack(output_total, 0)
                    out_len+=out.argmax(-1).view(-1).shape[0]
                    acc += torch.sum(out.argmax(-1).view(-1) == graph.y.view(-1))
                else:
                    y_true.append(graph.y.view(-1).detach().cpu())
                    y_pred.append(torch.stack(output_total, 0).view(-1).detach().cpu())

            if args.dataset!='Molhiv':
                acc = float(acc) / len(loader.dataset)
                return acc
            else:
                y_true = torch.cat(y_true, dim=0).numpy().reshape(-1, 1)
                y_pred = torch.cat(y_pred, dim=0).numpy().reshape(-1, 1)
                input_dict = {"y_true": y_true, "y_pred": y_pred}
                return evaluator.eval(input_dict)['rocauc']



        def vgae_generate(vgae, mu, logstd):

            vgae.__mu__, vgae.__logstd__ = mu, logstd
            # decode_size = args.decode_size
            decode_edge_index = torch.ones([decode_size, decode_size]).nonzero().t().contiguous().cuda()

            embs=[]
            edge_index=[]
            for i in range(mu.shape[0]):
                mu_repeated = mu[i].repeat(decode_size, 1)
                logstd_repeated = logstd[i].repeat(decode_size, 1)

                #emb = vgae.reparametrize(mu_repeated, logstd_repeated.clamp(max=10))
                emb = vgae.reparametrize(mu_repeated, logstd_repeated)
                edge_prob = decoder(emb, decode_edge_index, domain_embs=None, sigmoid=True).view(-1, 1)

                #if vgae.training==True:
                #    print(edge_prob)

                logits = torch.cat([torch.log(1 - edge_prob + 1e-9), torch.log(edge_prob + 1e-9)], dim=1)

                mask = F.gumbel_softmax(logits=logits, hard=True).bool()[:, 1]

                #mask=torch.zeros(mask.shape).bool().cuda()



                row, col = decode_edge_index
                row, col = torch.masked_select(row, mask), torch.masked_select(col, mask)

                # row, col = row[mask], col[mask]

                new_edge_index = torch.stack([row, col], dim=0)
                xs = vgae.reparametrize(mu_repeated, logstd_repeated.clamp(max=10))

                #embs.append(emb)
                embs.append(emb)
                edge_index.append(new_edge_index)

            return embs, edge_index



        cnt, last_val_acc = 0, 0
        best_acc=0

        for epoch in tqdm(range(args.epoch)):
            n_bw=0
            all_loss=[]
            train_acc = []
            train_mode()
            for step, graph in enumerate(train_loader):

                n_bw += 1
                graph.to(device)
                N = graph.num_graphs
                #print(N)
                #print(graph)
                #print(graph.edge_index.shape)

                mu, logstd = vgae1.encoder(graph.x.float(), graph.edge_index, graph.edge_attr, graph.batch)  # [batch, args.hidden]

                print(mu[0])
                print(logstd[0])

                mu, logstd= pool(mu,None, graph.batch), pool(logstd,None, graph.batch) #[num_graph, args.hidden]

                if args.use_id_graph:
                    mu2, logstd2 = vgae2.encoder(graph.x.float(), graph.edge_index, graph.edge_attr,
                                               graph.batch)  # [batch, args.hidden]
                    mu2, logstd2 = pool(mu2, None, graph.batch), pool(logstd2, None,
                                                                   graph.batch)  # [num_graph, args.hidden]
                    embs, edge_index= vgae_generate(vgae2, mu2, logstd2)
                    output_total=[gnn(emb, edge).mean(0) for emb, edge in zip(embs, edge_index)]




                embs, edge_index= vgae_generate(vgae1, mu, logstd)

                emb_only=False

                if emb_only:
                    output_total = [emb.mean(0) for emb, edge in zip(embs, edge_index)]
                else:

                    output_total=[gnn(emb, edge).mean(0) for emb, edge in zip(embs, edge_index)]



                out = torch.stack(output_total, 0)


                acc = torch.sum(out.argmax(-1).view(-1) == graph.y.view(-1))/mu.shape[0]


                train_acc.append(acc.item())


                if args.dataset!='Molhiv':
                    loss = CELoss(torch.stack(output_total,0), graph.y.view(-1))#+vgae1.kl_loss()
                else:
                    loss = BCELoss(torch.stack(output_total, 0).view(-1), graph.y.view(-1).float())  # +vgae.kl_loss()

                #if n_bw%100==0:
                #    tqdm.write('loss:{:.4f} ACC: {:.4f}'.format(loss.item(), acc.item()))

                

                # logger
                #all_loss+=loss

               # train_acc = float(train_acc) / len(train_loader.dataset)

                #all_loss /= n_bw
                model_optimizer.zero_grad()
                #all_loss.backward()
                loss.backward()
                model_optimizer.step()

                all_loss.append(loss.item())

            train_acc = np.mean(train_acc)

            val_mode()
            with torch.no_grad():

                #train_acc = test_acc(train_loader, att_net, g)
                #val_acc = test_acc(val_loader, att_net, g)
                # testing acc

                test_acc=test(test_loader)
                val_acc=test(val_loader)


                if test_acc>best_acc:
                    best_acc=test_acc

                        
            tqdm.write("Epoch [{:3d}/{:d}]  all_loss:{:2.3f} Train_ACC {:.3f} Val_ACC {:.3f} Test_ACC {:.3f} Best_ACC {:.4f}".format(
                    epoch, args.epoch,  np.mean(all_loss), train_acc,val_acc, test_acc, best_acc))


            # activate early stopping
            if epoch >= args.pretrain:
                if val_acc < last_val_acc:
                    cnt += 1
                else:
                    cnt = 0
                    last_val_acc = val_acc
            if cnt >= 100:
                logger.info("Early Stopping")
                break

        all_info['train_acc'].append(train_acc)
        all_info['val_acc'].append(val_acc)
        all_info['test_acc'].append(test_acc)

        #torch.save(g.cpu(), osp.join(exp_dir, 'predictor-%d.pt' % seed))
        #torch.save(att_net.cpu(), osp.join(exp_dir, 'attention_net-%d.pt' % seed))
        logger.info("=" * 100)

    logger.info("Train ACC:{:.4f}±{:.4f}".format(
                    torch.tensor(all_info['train_acc']).mean(), torch.tensor(all_info['train_acc']).std()
                ))
            