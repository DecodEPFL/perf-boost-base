import os, sys
import logging
import torch
from datetime import datetime
from torch.utils.data import DataLoader

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(BASE_DIR)
sys.path.insert(1, BASE_DIR)

from config import device
from arg_parser import argument_parser, print_args
from plants import RobotsSystem, RobotsDataset
from plot_functions import plot_trajectories, plot_traj_vs_time
from controllers.PB_controller import PerfBoostController
from loss_functions import RobotsLoss
from assistive_functions import WrapLogger

# ----- Configuration -----
args = argument_parser()

# ----- SET UP LOGGER -----
now = datetime.now().strftime("%m_%d_%H_%M_%S")
save_path = os.path.join(BASE_DIR, 'experiments', 'minimal_example', 'saved_results')
save_folder = os.path.join(save_path, 'perf_boost_'+args.nn_type+'_'+now)
os.makedirs(save_folder)
logging.basicConfig(filename=os.path.join(save_folder, 'log'), format='%(asctime)s %(message)s', filemode='w')
logger = logging.getLogger('perf_boost_'+args.nn_type+'_')
logger.setLevel(logging.DEBUG)
logger = WrapLogger(logger)

# ----- parse and set experiment arguments -----
msg = print_args(args)
logger.info(msg)
torch.manual_seed(args.random_seed)

# ------------ 1. Dataset ------------
dataset = RobotsDataset(random_seed=args.random_seed, horizon=args.horizon, std_ini=args.std_init_plant, n_agents=2)
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
# data for plots
t_ext = args.horizon * 4
plot_data = torch.zeros(1, t_ext, train_data.shape[-1], device=device)
plot_data[:, 0, :] = (dataset.x0.detach() - dataset.xbar)
plot_data = plot_data.to(device)
# batch the data
train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

# ------------ 2. Plant ------------
plant_input_init = None     # all zero
plant_state_init = None     # same as xbar
sys = RobotsSystem(xbar=dataset.xbar,
                   x_init=plant_state_init,
                   u_init=plant_input_init,
                   linear_plant=args.linearize_plant,
                   k=args.spring_const
                   ).to(device)

# ------------ 3. Controller ------------
ctl = PerfBoostController(noiseless_forward=sys.noiseless_forward,
                          input_init=sys.x_init,
                          output_init=sys.u_init,
                          nn_type=args.nn_type,
                          dim_internal=args.dim_internal,
                          output_amplification=args.output_amplification,
                          # SSM properties
                          scaffolding_nonlin=args.scaffolding_nonlin,
                          dim_middle=args.dim_middle,
                          dim_scaffolding=args.dim_scaffolding,
                          rmin=args.rmin,
                          rmax=args.rmax,
                          max_phase=args.max_phase,
                          # REN properties
                          dim_nl=args.dim_nl,
                          initialization_std=args.cont_init_std,
                          #   pos_def_tol=args.pos_def_tol,
                          # contraction_rate_lb = args.contraction_rate_lb,
                          # ren_internal_state_init=None,  # None for random initialization
                          ).to(device)

logger.info('[INFO] The internal operator emme is of type ' + args.nn_type + ' with %i parameters.' % ctl.num_params)

# plot closed-loop trajectories before training the controller
logger.info('Plotting closed-loop trajectories before training the controller...')
x_log, _, u_log = sys.rollout(ctl, plot_data)
filename = os.path.join(save_folder, 'CL_init.pdf')
plot_trajectories(
    x_log[0, :, :],  # remove extra dim due to batching
    dataset.xbar, sys.n_agents, filename=filename, text="CL - before training", T=t_ext
)
plot_traj_vs_time(
    args.horizon, sys.n_agents, 
    x_log[0, :args.horizon, :], u_log[0, :args.horizon, :], 
    filename = os.path.join(save_folder, 'Traj_init.pdf'),
    text="Trajectories - initial controller"
)
total_n_params = sum(p.numel() for p in ctl.parameters() if p.requires_grad)
logger.info("[INFO] Number of parameters: %i" % total_n_params)

# ------------ 4. Loss ------------
Q = torch.kron(torch.eye(args.n_agents), torch.eye(4)).to(device)   # TODO: move to args and print info
loss_fn = RobotsLoss(
    Q=Q, alpha_u=args.alpha_u, xbar=dataset.xbar,
    loss_bound=None, sat_bound=None,
    alpha_col=args.alpha_col, alpha_obst=args.alpha_obst,
    min_dist=args.min_dist if args.col_av else None,
    n_agents=sys.n_agents if args.col_av else None,
)

# ------------ 5. Training ------------
ctl.fit(
    sys=sys, train_dataloader=train_dataloader, valid_data=valid_data, 
    lr=args.lr, loss_fn=loss_fn, epochs=args.epochs, log_epoch=args.log_epoch, 
    return_best=args.return_best, logger=logger, early_stopping=args.early_stopping, 
    n_logs_no_change=args.n_logs_no_change, tol_percentage=args.tol_percentage
)

# ------ 6. Save and evaluate the trained model ------
# save
res_dict = ctl.emme.state_dict()
res_dict['Q'] = Q  # Include loss function parameters for completeness
filename = os.path.join(save_folder, 'trained_controller'+'.pt')
torch.save(res_dict, filename)
logger.info('[INFO] saved trained model.')

# evaluate on the train data
logger.info('\n[INFO] evaluating the trained controller on %i training rollouts.' % train_data.shape[0])
with torch.no_grad():
    x_log, _, u_log = sys.rollout(
        controller=ctl, data=train_data, train=False,
    )   # use the entire train data, not a batch
    # evaluate losses
    loss = loss_fn.forward(x_log, u_log)
    msg = 'Loss: %.4f' % loss
# count collisions
if args.col_av:
    num_col = loss_fn.count_collisions(x_log)
    msg += ' -- Number of collisions = %i' % num_col
logger.info(msg)

# evaluate on the test data
logger.info('\n[INFO] evaluating the trained controller on %i test rollouts.' % test_data.shape[0])
with torch.no_grad():
    # simulate over horizon steps
    x_log, _, u_log = sys.rollout(
        controller=ctl, data=test_data, train=False,
    )
    # loss
    test_loss = loss_fn.forward(x_log, u_log).item()
    msg = "Loss: %.4f" % test_loss
# count collisions
if args.col_av:
    num_col = loss_fn.count_collisions(x_log)
    msg += ' -- Number of collisions = %i' % num_col
logger.info(msg)

# plot closed-loop trajectories using the trained controller
logger.info('Plotting closed-loop trajectories using the trained controller...')
x_log, _, u_log = sys.rollout(ctl, plot_data)
filename = os.path.join(save_folder, 'CL_trained.pdf')
plot_trajectories(
    x_log[0, :, :],  # remove extra dim due to batching
    dataset.xbar, sys.n_agents, filename=filename, text="CL - trained controller", T=t_ext,
    obstacle_centers=loss_fn.obstacle_centers,
    obstacle_covs=loss_fn.obstacle_covs
)
plot_traj_vs_time(
    args.horizon, 
    sys.n_agents, x_log[0, :args.horizon, :], u_log[0, :args.horizon, :], 
    filename = os.path.join(save_folder, 'Traj_trained.pdf'),
    text="Trajectories - trained controller"
)
