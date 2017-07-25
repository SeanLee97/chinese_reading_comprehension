# !/usr/bin/env python
# -*- coding: utf-8 -*-
import glob, os
from config.config import *

def load_previous_model(model, optimizer = None):
    f_list = glob.glob(os.path.join(MODEL_DIR, '/rd_epoch') + '-*.pt')
    start_epoch = 1
    if len(f_list) >= 1:
        epoch_list = [int(i.split('-')[-1].split('.')[0]) for i in f_list]
        last_checkpoint = f_list[np.argmax(epoch_list)]
        if os.path.exists(last_checkpoint):
            print('load from {}'.format(last_checkpoint))
            model_state_dict = torch.load(last_checkpoint, map_location=lambda storage, loc: storage)
            model.load_state_dict(model_state_dict['model'])
            if optimizer != None:
                optimizer.load_state_dict(model_state_dict['optimizer'])
            start_epoch = np.max(model_state_dict['epoch'])
    return model, optimizer, start_epoch
