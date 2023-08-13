import os
import random
import numpy as np
import torch
def dump_scores(result_dir, eval_res, pred_loss=None):
    with open(os.path.join(result_dir, "losses"), 'w') as wf:
        if pred_loss is not None:
            wf.write("Predictor Training...\n")
            wf.write('\n'.join(["Epoch {} -- {:.4f}".format(i+1, v) for i, v in enumerate(pred_loss)])+'\n')

    score_dir = '/'.join(result_dir.split('/')[:-1])
    with open(os.path.join(score_dir, "exp_scores.txt"), 'a+') as fw:
        fw.write('Exp '+result_dir.split('/')[-1]+':\n')
        try:
            fw.write("test res -- " + '\t'.join(["{}:{:.4f}".format(k, v) for k,v in eval_res.items()]))
        except:
            print(eval_res)
        fw.write('{}{}{}'.format('\n', '='*40, '\n'))


def seed_everything(seed=1234):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

import json
def json_pretty_dump(obj, filename):
    with open(filename, "w") as fw:
        json.dump(obj, fw, sort_keys=True, indent=4, separators=(",", ": "), ensure_ascii=False,)
            
import hashlib
import logging
class stringFilter(logging.Filter):
    def filter(self, record):
        if record.msg.find('timestamp') == -1 and record.msg.find('time-series to datasets') == -1:
            return True
        return False

from datetime import datetime, timedelta
import sys
def dump_configs(configs, dataset, mode="predict"):
    hash_id = hashlib.md5(str(sorted([(k, v) for k, v in configs.items()])).encode("utf-8")).hexdigest()[0:8]
    cur_time = (datetime.now()+timedelta(hours=8)).strftime("%d-%H-%M")
    
    model_dir = os.path.join("./model_save", mode, dataset, cur_time+'_'+hash_id)
    os.makedirs(model_dir, exist_ok=True)
    result_dir = os.path.join("./result", mode, dataset, cur_time+'_'+hash_id)
    os.makedirs(result_dir, exist_ok=True)

    json_pretty_dump(configs, os.path.join(result_dir, "configs.json"))

    log_file = os.path.join(result_dir, "running.log")
    # logs will not show in the file without the two lines.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    stream_handler = logging.StreamHandler(sys.stdout) #filter useless logging messages
    stream_handler.addFilter(stringFilter())
    file_handler = logging.FileHandler(log_file)
    file_handler.addFilter(stringFilter())

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s P%(process)d %(levelname)s %(message)s",
        handlers=[file_handler, stream_handler],
    )
    logging.info(json.dumps(configs, indent=4))
    return hash_id, model_dir, result_dir