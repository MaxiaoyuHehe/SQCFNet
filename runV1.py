

import random
import os
import numpy as np
import scipy.io
from tqdm import tqdm
from dataloaders import DataLoaderIQA
from models.resnet_backbone import resnet50_backbone
from models.smrmV1 import SmRmSepNet, IQAL1L2Net, IQAL2L3Net, IQAL3L4Net, IQAMergeNet
import torch
from scipy import stats
import argparse
import json
import scipy.io as scio
import smtplib
from email.mime.text import MIMEText
from email.header import Header
import cv2




def getReuslts(len, type):
    L1plccs, L2plccs, L3plccs, Mplccs = 0.0, 0.0, 0.0, 0.0
    L1srccs, L2srccs, L3srccs, Msrccs  = 0.0, 0.0, 0.0, 0.0
    for t in range(len):
        d1 = scipy.io.loadmat('./results/test_gt%s_cnt%06d.mat' % (type, t))['gt']
        L1d2 = scipy.io.loadmat('./results/test_predL1%s_cnt%06d.mat' % (type, t))['pred']
        L2d2 = scipy.io.loadmat('./results/test_predL2%s_cnt%06d.mat' % (type, t))['pred']
        L3d2 = scipy.io.loadmat('./results/test_predL3%s_cnt%06d.mat' % (type, t))['pred']
        Md2 = scipy.io.loadmat('./results/test_predM%s_cnt%06d.mat' % (type, t))['pred']
        L1srcc_val_t, _ = stats.spearmanr(d1.squeeze(), L1d2.squeeze())
        L1plcc_val_t, _ = stats.pearsonr(d1.squeeze(), L1d2.squeeze())
        if L1plcc_val_t > L1plccs:
            L1plccs = L1plcc_val_t
        if L1srcc_val_t > L1srccs:
            L1srccs = L1srcc_val_t

        L2srcc_val_t, _ = stats.spearmanr(d1.squeeze(), L2d2.squeeze())
        L2plcc_val_t, _ = stats.pearsonr(d1.squeeze(), L2d2.squeeze())
        if L2plcc_val_t > L2plccs:
            L2plccs = L2plcc_val_t
        if L2srcc_val_t > L2srccs:
            L2srccs = L2srcc_val_t

        L3srcc_val_t, _ = stats.spearmanr(d1.squeeze(), L3d2.squeeze())
        L3plcc_val_t, _ = stats.pearsonr(d1.squeeze(), L3d2.squeeze())
        if L3plcc_val_t > L3plccs:
            L3plccs = L3plcc_val_t
        if L3srcc_val_t > L3srccs:
            L3srccs = L3srcc_val_t

        Msrcc_val_t, _ = stats.spearmanr(d1.squeeze(), Md2.squeeze())
        Mplcc_val_t, _ = stats.pearsonr(d1.squeeze(), Md2.squeeze())
        if Mplcc_val_t > Mplccs:
            Mplccs = Mplcc_val_t
        if Msrcc_val_t > Msrccs:
            Msrccs = Msrcc_val_t
    return L1plccs, L1srccs, L2plccs, L2srccs, L3plccs, L3srccs, Mplccs, Msrccs


class Config(dict):

    def __init__(self, *args, **kwargs):
        super(Config, self).__init__(*args, **kwargs)

    def __getattr__(self, key):
        value = self[key]
        if isinstance(value, dict):
            value = Config(value)
        return value


def DataSetup(root, batch_size, data_lens):
    scn_idxs = [x for x in range(data_lens[0])]
    np.random.shuffle(scn_idxs)
    scn_idxs_train = scn_idxs[:int(0.8 * data_lens[0])]
    scn_idxs_test = scn_idxs[int(0.8 * data_lens[0]):]

    loader_train = DataLoaderIQA('koniq', root, scn_idxs_train, batch_size=batch_size, istrain=True).get_data()
    loader_test = DataLoaderIQA('koniq', root, scn_idxs_test, batch_size=batch_size, istrain=False).get_data()
    return loader_train, loader_test


def test_model(models, loaders, config, cnt):
    torch.cuda.empty_cache()
    models.iqaL1.train(False)
    models.iqaL1.eval()
    models.iqaL2.train(False)
    models.iqaL2.eval()
    models.iqaL3.train(False)
    models.iqaL3.eval()
    models.iqaM.train(False)
    models.iqaM.eval()
    my_device = torch.device('cuda:0')
    pred_valsL1 = np.empty((0, 1))
    pred_valsL2 = np.empty((0, 1))
    pred_valsL3 = np.empty((0, 1))
    pred_valsM = np.empty((0, 1))
    gt_vals = np.empty((0, 1))
    bcnt = 0
    for inputs, labels in loaders.test:
        inputs, labels = inputs.float().to(my_device), labels.float().to(my_device)
        img_ft = {}
        _ = models.backbone(inputs)
        img_ft['f1'], img_ft['f2'], img_ft['f3'], img_ft['f4'] = models.saves.outputs[0], \
                                                                 models.saves.outputs[1], \
                                                                 models.saves.outputs[2], \
                                                                 models.saves.outputs[3]
        models.saves.outputs.clear()
        fts = models.iqab(img_ft)
        outL1 = models.iqaL1(fts)
        outL2 = models.iqaL2(fts)
        outL3 = models.iqaL3(fts)
        outM = models.iqaM(outL1['F'], outL2['F'], outL3['F'])
        predL1 = outL1['Q']
        predL2 = outL2['Q']
        predL3 = outL3['Q']
        predM = outM['Q']

        pred_valsL1 = np.append(pred_valsL1, predL1.detach().cpu().numpy(), axis=0)
        pred_valsL2 = np.append(pred_valsL2, predL2.detach().cpu().numpy(), axis=0)
        pred_valsL3 = np.append(pred_valsL3, predL3.detach().cpu().numpy(), axis=0)
        pred_valsM = np.append(pred_valsM, predM.detach().cpu().numpy(), axis=0)
        gt_vals = np.append(gt_vals, labels[:, None].detach().cpu().numpy(), axis=0)
        bcnt += 1

    scipy.io.savemat('./results/test_gt%s_cnt%06d.mat' % (config.type, cnt), {'gt': gt_vals})
    scipy.io.savemat('./results/test_predL1%s_cnt%06d.mat' % (config.type, cnt), {'pred': pred_valsL1})
    scipy.io.savemat('./results/test_predL2%s_cnt%06d.mat' % (config.type, cnt), {'pred': pred_valsL2})
    scipy.io.savemat('./results/test_predL3%s_cnt%06d.mat' % (config.type, cnt), {'pred': pred_valsL3})
    scipy.io.savemat('./results/test_predM%s_cnt%06d.mat' % (config.type, cnt), {'pred': pred_valsM})

    srcc_valL1, _ = stats.spearmanr(pred_valsL1.squeeze(), gt_vals.squeeze())
    plcc_valL1, _ = stats.pearsonr(pred_valsL1.squeeze(), gt_vals.squeeze())
    srcc_valL2, _ = stats.spearmanr(pred_valsL2.squeeze(), gt_vals.squeeze())
    plcc_valL2, _ = stats.pearsonr(pred_valsL2.squeeze(), gt_vals.squeeze())
    srcc_valL3, _ = stats.spearmanr(pred_valsL3.squeeze(), gt_vals.squeeze())
    plcc_valL3, _ = stats.pearsonr(pred_valsL3.squeeze(), gt_vals.squeeze())
    srcc_valM, _ = stats.spearmanr(pred_valsM.squeeze(), gt_vals.squeeze())
    plcc_valM, _ = stats.pearsonr(pred_valsM.squeeze(), gt_vals.squeeze())
	
	
	
    models.iqaL1.train(True)
    models.iqaL2.train(True)
    models.iqaL3.train(True)
    models.iqaM.train(True)
    return srcc_valL1, plcc_valL1, srcc_valL2, plcc_valL2, srcc_valL3, plcc_valL3, srcc_valM, plcc_valM


def train_model(models, loaders, optims, config):
    torch.cuda.empty_cache()
    models.iqaL1.train(True)
    models.iqaL2.train(True)
    models.iqaL3.train(True)
    models.iqaM.train(True)
    my_device = torch.device('cuda:0')

    for t in range(config.nepoch):
        pred_vals = np.empty((0, 1))
        gt_vals = np.empty((0, 1))
        epoch_loss = []

        for inputs, labels in tqdm(loaders.train):
            inputs, labels = inputs.float().to(my_device), labels.float().to(my_device)
            img_ft = {}
            _ = models.backbone(inputs)
            img_ft['f1'], img_ft['f2'], img_ft['f3'], img_ft['f4'] = models.saves.outputs[0], \
                                                                     models.saves.outputs[1], \
                                                                     models.saves.outputs[2], \
                                                                     models.saves.outputs[3]
            models.saves.outputs.clear()
            fts = models.iqab(img_ft)
            outL1 = models.iqaL1(fts)
            outL2 = models.iqaL2(fts)
            outL3 = models.iqaL3(fts)
            loss1 = optims.criterion(outL1['Q'].squeeze(), labels.detach().squeeze())
            loss2 = optims.criterion(outL2['Q'].squeeze(), labels.detach().squeeze())
            loss3 = optims.criterion(outL3['Q'].squeeze(), labels.detach().squeeze())

            optims.optimL1.zero_grad()
            loss1.backward()
            optims.optimL1.step()
            optims.schedL1.step()

            optims.optimL2.zero_grad()
            loss2.backward()
            optims.optimL2.step()
            optims.schedL2.step()

            optims.optimL3.zero_grad()
            loss3.backward()
            optims.optimL3.step()
            optims.schedL3.step()

            outM = models.iqaM(outL1['F'],outL2['F'],outL3['F'])
            lossM = optims.criterion(outM['Q'].squeeze(), labels.detach().squeeze())
            optims.optimM.zero_grad()
            lossM.backward()
            optims.optimM.step()
            optims.schedM.step()
            epoch_loss.append(lossM.item())
            pred_vals = np.append(pred_vals, outM['Q'].detach().cpu().numpy(), axis=0)
            gt_vals = np.append(gt_vals, labels[:, None].detach().cpu().numpy(), axis=0)

        print('testing....')
        srcc_val_t, _ = stats.spearmanr(pred_vals.squeeze(), gt_vals.squeeze())
        plcc_val_t, _ = stats.pearsonr(pred_vals.squeeze(), gt_vals.squeeze())

        srcc_valL1, plcc_valL1, srcc_valL2, plcc_valL2, srcc_valL3, plcc_valL3, srcc_valM, plcc_valM = test_model(models, loaders, config, t)
        print(
            'Test: %03d SRCCL1 : %.4f PLCCL1 : %.4f SRCCL2 : %.4f PLCCL2 : %.4f SRCCL3 : %.4f PLCCL3 : %.4f SRCCM : %.4f PLCCM : %.4f || Train Phase SRCC: %.4f PLCC %.4f RecLoss: %.4f\t' % (
                t, srcc_valL1, plcc_valL1, srcc_valL2, plcc_valL2, srcc_valL3, plcc_valL3, srcc_valM, plcc_valM, srcc_val_t, plcc_val_t, sum(epoch_loss) / len(epoch_loss)))
        # torch.save(cur_model, './results/model.pkl')
        torch.save({'merge':models.iqaM.state_dict(),
                    'L1': models.iqaL1.state_dict(),
                    'L2': models.iqaL2.state_dict(),
                    'L3': models.iqaL3.state_dict(),
                    }, './V1model_%02d.pth' % t)

        scipy.io.savemat('./results/train_gt%s_cnt%06d.mat' % (config.type, t), {'gt': gt_vals})
        scipy.io.savemat('./results/train_pred%s_cnt%06d.mat' % (config.type, t), {'pred': pred_vals})


def getHyperParams(sd):
    myconfigs = {
        'lrA': 2e-4,  # Up Block
        'lrB': 2e-4,  # Q Conv Block
        'lrC': 2e-4,  # Res Block
        'weight_decay': 5e-4,
        'T_MAX': 50,
        'eta_min': 0,
        'nepoch': 50,
        'batch_size': 12,
        'data_lens': (10073, 10073),
        'root': '/root/IQADatasets/koniq/',
        'type': 'KONIQ_L3L4SM_%04d' % sd
    }
    return Config(myconfigs)

class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)

    def clear(self):
        self.outputs = []
import torch
import torchvision

def main(sdcfg):
    myseed = sdcfg.sd
    random.seed(myseed)
    os.environ['PYTHONHASHSEED'] = str(myseed)
    np.random.seed(myseed)
    torch.manual_seed(myseed)
    torch.cuda.manual_seed_all(myseed)
    torch.backends.cudnn.deterministic = True
    myconfig = getHyperParams(myseed)

    myFCN = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).cuda()
    mymodelbackbone = myFCN.backbone.body
    save_output = SaveOutput()
    hook_handles = []
    for nn, layer in mymodelbackbone.named_modules():
        print(nn)
        if nn == 'layer1.2' or nn == 'layer2.3' or nn == 'layer3.5' or nn == 'layer4.2':
            handle = layer.register_forward_hook(save_output)
            hook_handles.append(handle)

    mymodelbackbone.train(False)
    mymodelbackbone.eval()

    mymodelIQA = SmRmSepNet().cuda()
    mymodelIQA.train(False)
    mymodelIQA.eval()

    mymodelL1L2 = IQAL1L2Net().cuda()
    mymodelL2L3 = IQAL2L3Net().cuda()
    mymodelL3L4 = IQAL3L4Net().cuda()
    mymodelMerge = IQAMergeNet().cuda()

    parasL1 = [
        {'params': mymodelL1L2.Asqueezea.parameters(), 'lr': myconfig['lrA']},
        {'params': mymodelL1L2.Asqueezeb.parameters(), 'lr': myconfig['lrA']},
        {'params': mymodelL1L2.Asqueezec.parameters(), 'lr': myconfig['lrA']},
        {'params': mymodelL1L2.Aconv.parameters(), 'lr': myconfig['lrA']},
        {'params': mymodelL1L2.Aqdense.parameters(), 'lr': myconfig['lrA']},
    ]

    parasL2 = [
        {'params': mymodelL2L3.Bsqueezea.parameters(), 'lr': myconfig['lrA']},
        {'params': mymodelL2L3.Bsqueezeb.parameters(), 'lr': myconfig['lrA']},
        {'params': mymodelL2L3.Bsqueezec.parameters(), 'lr': myconfig['lrA']},
        {'params': mymodelL2L3.Bconv.parameters(), 'lr': myconfig['lrA']},
        {'params': mymodelL2L3.Bqdense.parameters(), 'lr': myconfig['lrA']},
    ]
    parasL3 = [
        {'params': mymodelL3L4.Csqueezea.parameters(), 'lr': myconfig['lrA']},
        {'params': mymodelL3L4.Csqueezeb.parameters(), 'lr': myconfig['lrA']},
        {'params': mymodelL3L4.Csqueezec.parameters(), 'lr': myconfig['lrA']},
        {'params': mymodelL3L4.Cconv.parameters(), 'lr': myconfig['lrA']},
        {'params': mymodelL3L4.Cqdense.parameters(), 'lr': myconfig['lrA']},
    ]
    parasM = [
        {'params': mymodelMerge.qdense.parameters(), 'lr': myconfig['lrA']},
    ]

    optimizerL1 = torch.optim.Adam(parasL1, weight_decay=myconfig.weight_decay)
    optimizerL2 = torch.optim.Adam(parasL2, weight_decay=myconfig.weight_decay)
    optimizerL3 = torch.optim.Adam(parasL3, weight_decay=myconfig.weight_decay)
    optimizerM = torch.optim.Adam(parasM, weight_decay=myconfig.weight_decay)

    train_loader, test_loader = DataSetup(myconfig.root, myconfig.batch_size, myconfig.data_lens)
    schedulerL1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerL1, myconfig.T_MAX, myconfig.eta_min)
    schedulerL2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerL2, myconfig.T_MAX, myconfig.eta_min)
    schedulerL3 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerL3, myconfig.T_MAX, myconfig.eta_min)
    schedulerM = torch.optim.lr_scheduler.CosineAnnealingLR(optimizerM, myconfig.T_MAX, myconfig.eta_min)

    criterion = torch.nn.MSELoss()

    optim_params = Config({'criterion': criterion,
                           'optimL1': optimizerL1,
                           'schedL1': schedulerL1,
                           'optimL2': optimizerL2,
                           'schedL2': schedulerL2,
                           'optimL3': optimizerL3,
                           'schedL3': schedulerL3,
                           'optimM': optimizerM,
                           'schedM': schedulerM,
                           })
    models_params = Config({'backbone': mymodelbackbone,
                            'iqab': mymodelIQA,
                            'iqaL1': mymodelL1L2,
                            'iqaL2': mymodelL2L3,
                            'iqaL3': mymodelL3L4,
                            'iqaM': mymodelMerge,
                            'saves': save_output,
                            })
    data_loaders = Config({'train': train_loader,
                           'test': test_loader})

    train_model(models_params, data_loaders, optim_params, myconfig)

    L1plccs, L1srccs, L2plccs, L2srccs, L3plccs, L3srccs, Mplccs, Msrccs = getReuslts(myconfig.nepoch, myconfig.type)

    #SendEmail('The Training of Type %s is Finished The Max PLCCL1 is %0.4f | The Max SRCCL1 is %0.4f | The Max PLCCL2 is %0.4f | The Max #SRCCL2 is %0.4f | The Max PLCCL3 is %0.4f | The Max SRCCL3 is %0.4f | The Max PLCCM is %0.4f | The Max SRCCM is %0.4f' % (
    #    myconfig.type, L1plccs, L1srccs, L2plccs, L2srccs, L3plccs, L3srccs, Mplccs, Msrccs))

    print('OK..')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', dest='sd', type=int, default=3407, help='Random Seed')
    sdcfg = parser.parse_args()
    main(sdcfg)

