import sys
import os
import time

import torch
import torch.optim as optim

from dataloader import CA_Dataset
from model_init import init_model
from train_utils import train_one_epoch, eval_one_epoch, multi_save, save_checkpoint, load_checkpoint

config_train_stage1 = {
            'max_epoch':200,
            'dev':torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
            'lr_start':1e-4,
            'lr_finish':1e-5,
            'save_num':3,
            'save_type':['val','test'],
            'post_scores':[0.0,0.05,0.1,0.15,0.2],
            'pre_trained':None,
            'resume':False,
            'work':'train'
            }
config_train_stage2 = {
            'max_epoch':50,
            'dev':torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
            'lr_start':1e-5,
            'lr_finish':1e-5,
            'save_num':3,
            'save_type':['val','test'],
            'post_scores':[0.0,0.05,0.1,0.15,0.2],
            'pre_trained':'run/train/2024-12-16-15-46-39/train_data/best_mae_test_probmap_0.pth.tar',
            'resume':False,
            'work':'train'
            }
config_eval = {
            'dev':torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'),
            'eval_type':['val','test'],
            'post_scores':[0.05,0.1,0.15],
            'pre_trained':'run/model_paper/best1.pth.tar',
            'work':'eval'
            }

if __name__ == '__main__':
    config = config_eval
    crop_aug = True
    base_dir = sys.path[0]
    data_dir = os.path.join(os.path.dirname(sys.path[0]), 'FSC147_384_V2')
    train_dataset = CA_Dataset(data_dir, split = 'train', crop_aug = crop_aug)
    val_dataset = CA_Dataset(data_dir, split = 'val', train = False)
    test_dataset = CA_Dataset(data_dir, split = 'test', train = False)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    model = init_model().to(config['dev'])
    if config['pre_trained']!=None:
        model,pre_epoch = load_checkpoint(model,os.path.join(base_dir,config['pre_trained']))
    else:
        pre_epoch = 0

    if config['work'] == 'train':
        train_name = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        os.mkdir(os.path.join(base_dir, 'run', 'train',train_name))
        os.mkdir(os.path.join(base_dir, 'run', 'train',train_name,'train_data'))

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(params, config['lr_start'],weight_decay=1e-4)
        schedular = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['max_epoch'], eta_min=config['lr_finish'])
        if config['resume']:
            for _ in range(pre_epoch):
                schedular.step()
        else:
            pre_epoch = 0

        best_mae_val_probmap,best_mae_test_probmap,best_epoch_val_probmap,best_epoch_test_probmap = [],[],[],[]

        outputfile = open(os.path.join(base_dir,'run','train',train_name,'train_data',"log.txt"), 'w')
        outputfile.close()

        for epoch in range(pre_epoch,config['max_epoch']):
            outputfile = open(os.path.join(base_dir,'run','train',train_name,'train_data',"log.txt"), 'a')
            lr = optimizer.state_dict()['param_groups'][0]['lr']

            return_losses = train_one_epoch(model, train_loader, epoch, optimizer, schedular, config['dev'])

            text = 'epoch: ' + str(epoch) + ' lr: ' + str(lr)
            for key,value in return_losses.items():
                text = text + ' ' + key + ': ' + str(value)
            text = text + '\n'

            save_checkpoint(model,{'epoch':epoch,'base_root': os.path.join(base_dir,'run','train',train_name,'train_data')},'last')

            if 'val' in config['save_type']:
                val_result_dict = eval_one_epoch(model, val_loader, config['post_scores'], config['dev'])
                for key,value in val_result_dict.items():
                    if key != 'all':
                        text = text + 'val_' + key + ': ' + str(value) + '\n'
                best_mae_val_probmap, best_epoch_val_probmap = multi_save(best_mae_val_probmap,best_epoch_val_probmap,min(val_result_dict['all']['maes']),epoch,base_dir,train_name,'train_data','val_probmap',model,config)
            if 'test' in config['save_type']:
                test_result_dict = eval_one_epoch(model, test_loader, config['post_scores'], config['dev'])
                for key,value in test_result_dict.items():
                    if key != 'all':
                        text = text + 'test_' + key + ': ' + str(value) + '\n'
                best_mae_test_probmap, best_epoch_test_probmap = multi_save(best_mae_test_probmap,best_epoch_test_probmap,min(test_result_dict['all']['maes']),epoch,base_dir,train_name,'train_data','test_probmap',model,config)

            print(text)
            print(text,file=outputfile)
            outputfile.close()

        outputfile = open(os.path.join(base_dir,'run','train',train_name,'train_data',"log.txt"), 'a')
        text = ''
        text = text + 'best_val_weights_probmap:' + '\n'
        for i in range(len(best_mae_val_probmap)):
            text = text + str(best_epoch_val_probmap[i]) + ': ' + str(best_mae_val_probmap[i]) + ' '
        text = text + '\n'
        text = text + 'best_test_weights_probmap:'+ '\n'
        for i in range(len(best_mae_test_probmap)):
            text = text + str(best_epoch_test_probmap[i]) + ': ' + str(best_mae_test_probmap[i]) + ' '
        text = text + '\n'

        print(text,file=outputfile)
        outputfile.close()

    elif config['work'] == 'eval':
        text = ''
        if 'val' in config['eval_type']:
            val_result_dict = eval_one_epoch(model, val_loader, config['post_scores'], config['dev'])
            for key,value in val_result_dict.items():
                if key != 'all':
                    text = text + 'val_' + key + ': ' + str(value) + '\n'
        if 'test' in config['eval_type']:
            test_result_dict = eval_one_epoch(model, test_loader, config['post_scores'], config['dev'])
            for key,value in test_result_dict.items():
                if key != 'all':
                    text = text + 'test_' + key + ': ' + str(value) + '\n'
        print(text)