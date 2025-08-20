from tqdm import tqdm
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch_geometric.loader import DataLoader

from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

from utils import num_graphs

def train(dataset, split_idx, model, args, device):
    best_valid_auc, best_epoch, best_valid_auc_c = 0, 0, 0
    
    train_dataset = dataset[split_idx["train"]]
    valid_dataset = dataset[split_idx["valid"]]
    test_dataset = dataset[split_idx["test"]]
    
    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, args.batch_size, shuffle=False)

    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay)
    
    for epoch in tqdm(range(1, args.epochs + 1)):

        train_loss, loss_c, loss_cs = train_epoch(
            model, optimizer, train_loader, device, args)
        
        if args.eval_metric == 'auc':
            valid_auc, valid_auc_c = eval_auc(
                model, valid_loader, device, args)
        elif args.eval_metric == 'pr':
            valid_auc, valid_auc_c = eval_pr(
                model, valid_loader, device, args)
        else:
            raise Exception("metric option not valid")
        
        try:
            wandb.log({
                'train_loss': train_loss,
                'loss_c': loss_c,
                'loss_cs': loss_cs,
                'valid_auc': valid_auc,
                'valid_auc_c': valid_auc_c,
            })
        except: pass

        if (valid_auc >= best_valid_auc)&(epoch>=10):
            best_valid_auc = valid_auc
            best_epoch = epoch
            best_valid_auc_c = valid_auc_c
            best_model = copy.deepcopy(model)
        

    print("Dataset:[{}] Model:[{}] | Best Valid:[{:.2f}] at epoch [{}] | Test_c:[{:.2f}]"
            .format(args.dataset,
                    args.model,
                    best_valid_auc * 100, 
                    best_epoch,
                    best_valid_auc_c * 100))
    
    if args.eval_metric == 'auc':
        last_test_auc, last_test_auc_c = eval_auc(
            model, test_loader, device, args)
        test_auc, test_auc_c = eval_auc(
            best_model, test_loader, device, args)
    elif args.eval_metric == 'pr':
        last_test_auc, last_test_auc_c = eval_pr(
            model, test_loader, device, args)
        test_auc, test_auc_c = eval_pr(
            best_model, test_loader, device, args)
    else:
        raise Exception("metric option not valid")
    
    print("Dataset:[{}] Model:[{}] | Best Test:[{:.2f}] | Test_c:[{:.2f}]"
            .format(args.dataset,
                    args.model,
                    test_auc * 100, 
                    test_auc_c * 100))
    print("Dataset:[{}] Model:[{}] | Final Test:[{:.2f}] | Test_c:[{:.2f}]"
            .format(args.dataset,
                    args.model,
                    last_test_auc * 100, 
                    last_test_auc_c * 100))
    try:
        wandb.log({
            'test_auc': test_auc,
            'test_auc_c': test_auc_c,
            'last_test_auc': last_test_auc,
            'last_test_auc_c': last_test_auc_c
        })
    except: pass
    
    return test_auc, test_auc_c, best_valid_auc, best_model

def train_epoch(model, optimizer, train_loader, device, args):
    
    model.train()
    total_loss = 0
    total_loss_c = 0
    total_loss_cs = 0
    
    for it, data in enumerate(train_loader):
        
        optimizer.zero_grad()
        data = data.to(device)
        target = data.y.view(-1)
        mask = ~target.isnan()
        target = target[mask].to(torch.long)
        c_logit, cs_logit, sparsity_loss = model(data)
        
        c_loss = nn.CrossEntropyLoss()(c_logit.view(-1, args.num_classes)[mask], target)
        cs_loss = nn.CrossEntropyLoss()(cs_logit.view(-1, args.num_classes)[mask], target)
        loss = args.c * c_loss + args.cs * cs_loss + args.s * sparsity_loss


        loss.backward()
        total_loss += loss.item() * num_graphs(data)
        total_loss_c += c_loss.item() * num_graphs(data)
        total_loss_cs += cs_loss.item() * num_graphs(data)
        optimizer.step()
    
    num = len(train_loader.dataset)
    total_loss = total_loss / num
    total_loss_c = total_loss_c / num
    total_loss_cs = total_loss_cs / num

    return total_loss, total_loss_c, total_loss_cs 


def eval_auc(model, loader, device, args):
    
    model.eval()
    correct, correct_c = 0, 0
    pred, pred_c, y = torch.tensor([]), torch.tensor([]), torch.tensor([])
    for data in loader:
        data = data.to(device)
        y = torch.cat([y, data.y.view(-1, args.num_task).to('cpu')], dim=0)
        with torch.no_grad():
            c_logit, cs_logit, _ = model(data)
            pred = torch.cat([pred, cs_logit.detach().to('cpu')], dim=0)
            pred_c = torch.cat([pred_c, c_logit.detach().to('cpu')], dim=0)
    for i in range(args.num_task):
        mask = ~y[:,i].isnan()
        if args.num_classes == 2:
            correct += roc_auc_score(y[mask,i], F.softmax(pred[:,i*2:i*2+2], dim=-1)[mask,1])
            correct_c += roc_auc_score(y[mask,i], F.softmax(pred_c[:,i*2:i*2+2], dim=-1)[mask,1])
        else:
            correct += roc_auc_score(y[mask,i], F.softmax(pred[mask, i*args.num_classes:(i+1)*args.num_classes], dim=-1), multi_class='ovr')
            correct_c += roc_auc_score(y[mask,i], F.softmax(pred_c[mask, i*args.num_classes:(i+1)*args.num_classes], dim=-1), multi_class='ovr')

    acc_cs = correct / args.num_task
    acc_c = correct_c / args.num_task
    return acc_cs, acc_c


def eval_pr(model, loader, device, args):
    
    model.eval()
    correct, correct_c = 0, 0
    pred, pred_c, y = torch.tensor([]), torch.tensor([]), torch.tensor([])
    for data in loader:
        data = data.to(device)
        y = torch.cat([y, data.y.view(-1, args.num_task).to('cpu')], dim=0)
        with torch.no_grad():
            c_logit, cs_logit, _ = model(data)
            pred = torch.cat([pred, cs_logit.detach().to('cpu')], dim=0)
            pred_c = torch.cat([pred_c, c_logit.detach().to('cpu')], dim=0)
    for i in range(args.num_task):
        mask = ~y[:,i].isnan()
        if args.num_classes == 2:
            precision, recall, _ = precision_recall_curve(y[mask,i], F.softmax(pred[:,i*2:i*2+2], dim=-1)[mask,1])
            correct += auc(recall, precision)
            precision, recall, _ = precision_recall_curve(y[mask,i], F.softmax(pred_c[:,i*2:i*2+2], dim=-1)[mask,1])
            correct_c += auc(recall, precision)  
        else:
            raise Exception("num_classes is 2 only now")      


    acc_cs = correct / args.num_task
    acc_c = correct_c / args.num_task
    return acc_cs, acc_c