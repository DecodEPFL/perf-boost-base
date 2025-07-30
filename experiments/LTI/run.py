
import sys, os, logging, torch
from datetime import datetime
import numpy as np
from torch.utils.data import DataLoader

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(BASE_DIR)
sys.path.insert(1, BASE_DIR)

from config import device
from arg_parser import argument_parser, print_args
from plants import LTIDataset
from utils.assistive_functions import WrapLogger


# ----- SET UP LOGGER -----
now = datetime.now().strftime("%m_%d_%H_%M_%S")
save_path = os.path.join(BASE_DIR, 'experiments', 'LTI', 'saved_results')
save_folder = os.path.join(save_path, 'perf_boost_'+now)
os.makedirs(save_folder)
logging.basicConfig(filename=os.path.join(save_folder, 'log'), format='%(asctime)s %(message)s', filemode='w')
logger = logging.getLogger('perf_boost_')
logger.setLevel(logging.DEBUG)
logger = WrapLogger(logger)

# ----- parse and set experiment arguments -----
args = argument_parser()
msg = print_args(args)
logger.info(msg)
torch.manual_seed(args.random_seed)

# ------------ 1. Dataset ------------
d_dist_v = 0.3*np.ones((args.state_dim, 1))
disturbance = {
    'type':'N biased',
    'mean':0.3*np.ones(args.state_dim),
    'cov':np.matmul(d_dist_v, np.transpose(d_dist_v))
}
dataset = LTIDataset(
    random_seed=args.random_seed, horizon=args.horizon,
    state_dim=args.state_dim, disturbance=disturbance
)
# divide to train and test
train_data_full, test_data = dataset.get_data(num_train_samples=args.num_rollouts, num_test_samples=500)
train_data_full, test_data = train_data_full.to(device), test_data.to(device)
# validation data
if args.early_stopping or args.return_best:
    valid_inds = torch.randperm(train_data_full.shape[0])[:int(args.validation_frac*train_data_full.shape[0])]
    train_inds = [ind for ind in range(train_data_full.shape[0]) if ind not in valid_inds]
    valid_data = train_data_full[valid_inds, :, :]
    train_data = train_data_full[train_inds, :, :]
else:
    valid_data = None
    train_data = train_data_full
# batch the data
train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
