import os
import time
import torch
import argparse
import torch.nn as nn
import MinkowskiEngine as ME
import matplotlib.pyplot as plt
import torch.distributed as dist

from tqdm import tqdm
from copy import deepcopy
from pathlib import Path
from easydict import EasyDict as edict
from diffusers.optimization import get_cosine_schedule_with_warmup

from policy import RISE
from dataset.realworld import RealWorldDataset, collate_fn
from utils.training import set_seed, plot_history, sync_loss

default_args = edict({
    "data_path": "data",
    "aug": False,
    "aug_jitter": False,
    "num_action": 10,
    "voxel_size": 0.005,
    "obs_feature_dim": 512,
    "hidden_dim": 512,
    "nheads": 8,
    "num_encoder_layers": 4,
    "num_decoder_layers": 1,
    "dim_feedforward": 2048,
    "dropout": 0.1,
    "ckpt_dir": "logs/collect_pens",
    "resume_ckpt": None,
    "resume_epoch": -1,
    "lr": 3e-4,
    "batch_size": 240,
    "num_epochs": 1000,
    "save_epochs": 50,
    "num_workers": 24,
    "prefetch_factor": 4,
    "seed": 233,
    "enable_mba": True,
    "obj_dim": 10,
    "enable_headpose_head": True,
    "headpose_dim": 9,
    "obj_pose_mode": "abs",
    "add_curr_cond": True,
})


def train(args_override):
    # load default arguments
    args = deepcopy(default_args)
    for key, value in args_override.items():
        args[key] = value

    # prepare distributed training
    torch.multiprocessing.set_sharing_strategy('file_system')
    WORLD_SIZE = int(os.environ['WORLD_SIZE'])
    RANK = int(os.environ['RANK'])
    LOCAL_RANK = int(os.environ['LOCAL_RANK'])
    os.environ['NCCL_P2P_DISABLE'] = '1'
    dist.init_process_group(backend = 'nccl', init_method = 'env://', world_size = WORLD_SIZE, rank = RANK)

    # set up device
    set_seed(args.seed)
    torch.cuda.set_device(LOCAL_RANK)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.wandb_run_name is None:
        args.wandb_run_name = Path(args.ckpt_dir).resolve().name

    wandb_run = None
    if args.enable_wandb and RANK == 0:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name,
            dir=args.ckpt_dir,
            mode=args.wandb_mode,
            config=dict(args),
        )

    def _sync_scalar(value: float) -> float:
        t = torch.tensor([value], dtype=torch.float64, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= WORLD_SIZE
        return float(t.item())

    def _maybe_sync_cuda() -> None:
        return

    # load
    assert args.enable_mba, "train_obj.py requires --enable_mba to build object diffusion head."
    if RANK == 0:
        print("Training object trajectory diffusion head.")

    # dataset & dataloader
    if RANK == 0: print("Loading dataset ...")
    dataset = RealWorldDataset(
        path = args.data_path,
        split = 'train',
        num_obs = 1,
        num_action = args.num_action,
        voxel_size = args.voxel_size,
        aug = args.aug,
        aug_jitter = args.aug_jitter, 
        with_cloud = False,
        with_obj_action = True,
        with_headpose = args.enable_headpose_head,
        obj_pose_mode=args.obj_pose_mode,
    )
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, 
        num_replicas = WORLD_SIZE, 
        rank = RANK, 
        shuffle = True
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size = args.batch_size // WORLD_SIZE,
        num_workers = args.num_workers,
        prefetch_factor = args.prefetch_factor if args.num_workers > 0 else None,
        persistent_workers = bool(args.num_workers > 0),
        pin_memory = True,
        collate_fn = collate_fn,
        sampler = sampler
    )

    # policy
    if RANK == 0: print("Loading policy ...")
    policy = RISE(
        num_action = args.num_action,
        input_dim = 6,
        obs_feature_dim = args.obs_feature_dim,
        action_dim = 10,
        hidden_dim = args.hidden_dim,
        nheads = args.nheads,
        num_encoder_layers = args.num_encoder_layers,
        num_decoder_layers = args.num_decoder_layers,
        dropout = args.dropout,
        enable_mba = args.enable_mba,
        obj_dim = args.obj_dim,
        enable_headpose_head = args.enable_headpose_head,
        headpose_dim = args.headpose_dim,
        # obj_pose_mode = args.obj_pose_mode,
        add_curr_cond = args.add_curr_cond,
    ).to(device)
    if RANK == 0:
        n_parameters = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        print("Number of parameters: {:.2f}M".format(n_parameters / 1e6))
    policy = nn.parallel.DistributedDataParallel(
        policy, 
        device_ids = [LOCAL_RANK], 
        output_device = LOCAL_RANK, 
        find_unused_parameters = True
    )

    # load checkpoint
    if args.resume_ckpt is not None:
        policy.module.load_state_dict(torch.load(args.resume_ckpt, map_location = device), strict = False)
        if RANK == 0:
            print("Checkpoint {} loaded.".format(args.resume_ckpt))

    # ckpt path
    if RANK == 0 and not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)
    
    # optimizer and lr scheduler
    if RANK == 0: print("Loading optimizer and scheduler ...")
    optimizer = torch.optim.AdamW(policy.parameters(), lr = args.lr, betas = [0.95, 0.999], weight_decay = 1e-6)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer = optimizer,
        num_warmup_steps = 2000,
        num_training_steps = len(dataloader) * args.num_epochs
    )
    lr_scheduler.last_epoch = len(dataloader) * (args.resume_epoch + 1) - 1

    # training
    train_history = []
    global_step = 0

    policy.train()
    for epoch in range(args.resume_epoch + 1, args.num_epochs):
        if RANK == 0: print("Epoch {}".format(epoch)) 
        sampler.set_epoch(epoch)
        optimizer.zero_grad()
        num_steps = len(dataloader)
        pbar = tqdm(dataloader) if RANK == 0 else dataloader
        avg_loss = 0
        avg_obj_loss = 0
        avg_headpose_loss = 0
        avg_data_time = 0.0
        avg_prep_time = 0.0
        avg_forward_time = 0.0
        avg_backward_time = 0.0
        avg_optim_time = 0.0
        iter_end = time.perf_counter()

        for step_idx, data in enumerate(pbar):
            data_wait = time.perf_counter() - iter_end
            avg_data_time += data_wait

            _maybe_sync_cuda()
            t0 = time.perf_counter()
            # cloud data processing
            cloud_coords = data['input_coords_list']
            cloud_feats = data['input_feats_list']
            obj_data = data['action_obj_normalized']
            current_obj = data.get('current_obj_pose_normalized')
            headpose_data = data.get('action_headpose_normalized')
            current_headpose = data.get('current_headpose_normalized')
            cloud_feats = cloud_feats.to(device)
            cloud_coords = cloud_coords.to(device)
            obj_data = obj_data.to(device)
            if current_obj is not None:
                current_obj = current_obj.to(device)
            if headpose_data is not None:
                headpose_data = headpose_data.to(device)
            if current_headpose is not None:
                current_headpose = current_headpose.to(device)
            batch_size_cur = obj_data.shape[0]
            cloud_data = ME.SparseTensor(cloud_feats, cloud_coords)
            _maybe_sync_cuda()
            t1 = time.perf_counter()
            avg_prep_time += (t1 - t0)

            losses = policy(cloud = cloud_data,
                            batch_size = batch_size_cur,
                            actions_obj = obj_data,
                            sample_mba = False,
                            current_obj = current_obj,
                            headpose_data = headpose_data,
                            headpose_cond = current_headpose)
            _maybe_sync_cuda()
            t2 = time.perf_counter()
            avg_forward_time += (t2 - t1)

            if isinstance(losses, dict):
                loss = losses.get("obj_loss")
                headpose_loss = losses.get("headpose_loss")
                if loss is None:
                    raise RuntimeError("obj_loss is required for training in train_obj.")
                avg_obj_loss += loss.item()
                if headpose_loss is not None:
                    avg_headpose_loss += headpose_loss.item()
                    loss = loss + headpose_loss * args.headpose_loss_weight
            else:
                loss = losses

            loss.backward()
            _maybe_sync_cuda()
            t3 = time.perf_counter()
            avg_backward_time += (t3 - t2)

            optimizer.step()
            optimizer.zero_grad()
            lr_scheduler.step()
            _maybe_sync_cuda()
            t4 = time.perf_counter()
            avg_optim_time += (t4 - t3)

            avg_loss += loss.item()
            global_step += 1
            iter_end = time.perf_counter()

            if args.enable_wandb and RANK == 0 and (step_idx + 1) % args.wandb_log_interval == 0:
                wandb.log(
                    {
                        "train/step_loss": float(loss.item()),
                        "train/lr": float(optimizer.param_groups[0]["lr"]),
                        "time/data_wait_s": float(avg_data_time / (step_idx + 1)),
                        "time/prep_s": float(avg_prep_time / (step_idx + 1)),
                        "time/forward_s": float(avg_forward_time / (step_idx + 1)),
                        "time/backward_s": float(avg_backward_time / (step_idx + 1)),
                        "time/optim_s": float(avg_optim_time / (step_idx + 1)),
                    },
                    step=global_step,
                )

        avg_loss = avg_loss / num_steps
        avg_obj_loss = avg_obj_loss / num_steps
        avg_headpose_loss = avg_headpose_loss / num_steps
        avg_data_time = avg_data_time / num_steps
        avg_prep_time = avg_prep_time / num_steps
        avg_forward_time = avg_forward_time / num_steps
        avg_backward_time = avg_backward_time / num_steps
        avg_optim_time = avg_optim_time / num_steps

        avg_loss = _sync_scalar(avg_loss)
        avg_obj_loss = _sync_scalar(avg_obj_loss)
        avg_headpose_loss = _sync_scalar(avg_headpose_loss)
        avg_data_time = _sync_scalar(avg_data_time)
        avg_prep_time = _sync_scalar(avg_prep_time)
        avg_forward_time = _sync_scalar(avg_forward_time)
        avg_backward_time = _sync_scalar(avg_backward_time)
        avg_optim_time = _sync_scalar(avg_optim_time)
        train_history.append(avg_loss)

        if RANK == 0:
            timings = {
                "data_wait": avg_data_time,
                "prep": avg_prep_time,
                "forward": avg_forward_time,
                "backward": avg_backward_time,
                "optim": avg_optim_time,
            }
            step_total = sum(timings.values())
            bottleneck = max(timings, key=timings.get)
            print(
                "Train loss: {:.6f} (obj: {:.6f}, headpose: {:.6f})".format(
                    avg_loss, avg_obj_loss, avg_headpose_loss
                )
            )
            print(
                "Time/step (ms) | data {:.2f} | prep {:.2f} | fwd {:.2f} | bwd {:.2f} | optim {:.2f} | bottleneck={} ({:.1f}%)".format(
                    avg_data_time * 1000.0,
                    avg_prep_time * 1000.0,
                    avg_forward_time * 1000.0,
                    avg_backward_time * 1000.0,
                    avg_optim_time * 1000.0,
                    bottleneck,
                    100.0 * timings[bottleneck] / max(step_total, 1e-12),
                )
            )

            if args.enable_wandb:
                wandb.log(
                    {
                        "train/epoch": int(epoch),
                        "train/loss": float(avg_loss),
                        "train/obj_loss": float(avg_obj_loss),
                        "train/headpose_loss": float(avg_headpose_loss),
                        "train/lr": float(optimizer.param_groups[0]["lr"]),
                        "time/data_wait_s": float(avg_data_time),
                        "time/prep_s": float(avg_prep_time),
                        "time/forward_s": float(avg_forward_time),
                        "time/backward_s": float(avg_backward_time),
                        "time/optim_s": float(avg_optim_time),
                        "time/step_total_s": float(step_total),
                        "time/bottleneck_name": bottleneck,
                        "time/bottleneck_ratio": float(timings[bottleneck] / max(step_total, 1e-12)),
                    },
                    step=global_step,
                )

            if (epoch + 1) % args.save_epochs == 0:
                torch.save(
                    policy.module.state_dict(),
                    os.path.join(args.ckpt_dir, "policy_epoch_{}_seed_{}.ckpt".format(epoch + 1, args.seed))
                )
                plot_history(train_history, epoch, args.ckpt_dir, args.seed)

    if RANK == 0:
        torch.save(
            policy.module.state_dict(),
            os.path.join(args.ckpt_dir, "policy_last.ckpt")
        )
        if wandb_run is not None:
            wandb_run.finish()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', action = 'store', type = str, help = 'data path', required = True)
    parser.add_argument('--aug', action = 'store_true', help = 'whether to add 3D data augmentation')
    parser.add_argument('--aug_jitter', action = 'store_true', help = 'whether to add color jitter augmentation')
    parser.add_argument('--num_action', action = 'store', type = int, help = 'number of action steps', required = False, default = 10)
    parser.add_argument('--voxel_size', action = 'store', type = float, help = 'voxel size', required = False, default = 0.005)
    parser.add_argument('--obs_feature_dim', action = 'store', type = int, help = 'observation feature dimension', required = False, default = 512)
    parser.add_argument('--hidden_dim', action = 'store', type = int, help = 'hidden dimension', required = False, default = 512)
    parser.add_argument('--nheads', action = 'store', type = int, help = 'number of heads', required = False, default = 8)
    parser.add_argument('--num_encoder_layers', action = 'store', type = int, help = 'number of encoder layers', required = False, default = 4)
    parser.add_argument('--num_decoder_layers', action = 'store', type = int, help = 'number of decoder layers', required = False, default = 1)
    parser.add_argument('--dim_feedforward', action = 'store', type = int, help = 'feedforward dimension', required = False, default = 2048)
    parser.add_argument('--dropout', action = 'store', type = float, help = 'dropout ratio', required = False, default = 0.1)
    parser.add_argument('--ckpt_dir', action = 'store', type = str, help = 'checkpoint directory', required = True)
    parser.add_argument('--resume_ckpt', action = 'store', type = str, help = 'resume checkpoint file', required = False, default = None)
    parser.add_argument('--resume_epoch', action = 'store', type = int, help = 'resume from which epoch', required = False, default = -1)
    parser.add_argument('--lr', action = 'store', type = float, help = 'learning rate', required = False, default = 3e-4)
    parser.add_argument('--batch_size', action = 'store', type = int, help = 'batch size', required = False, default = 240)
    parser.add_argument('--num_epochs', action = 'store', type = int, help = 'training epochs', required = False, default = 1000)
    parser.add_argument('--save_epochs', action = 'store', type = int, help = 'saving epochs', required = False, default = 50)
    parser.add_argument('--num_workers', action = 'store', type = int, help = 'number of workers', required = False, default = 24)
    parser.add_argument('--prefetch_factor', action='store', type=int, help='dataloader prefetch factor per worker', required=False, default=4)
    parser.add_argument('--seed', action = 'store', type = int, help = 'seed', required = False, default = 233)

    parser.add_argument('--enable_mba', action = 'store_true', help = 'mba enabled / disabled')
    parser.add_argument('--obj_dim', action = 'store', type = int, help = 'hidden dimension', required = False, default = 10)
    parser.add_argument('--disable_headpose_head', action = 'store_false', dest = 'enable_headpose_head', help = 'disable headpose diffusion head')
    parser.add_argument('--headpose_dim', action = 'store', type = int, help = 'headpose action dimension', required = False, default = 9)
    parser.add_argument('--obj_pose_mode', action='store', type=str, choices=['abs', 'delta'], required=False, default='delta', help='object pose prediction target type')
    parser.add_argument('--headpose_loss_weight', action = 'store', type = float, help = 'headpose loss weight', required = False, default = 0.3)
    parser.add_argument('--add_curr_cond', action = 'store_true', help = 'add curr obj pose as extra cond for diffusion head')
    parser.add_argument('--disable_wandb', action='store_false', dest='enable_wandb', help='disable Weights & Biases logging')
    parser.set_defaults(enable_wandb=True)
    parser.add_argument('--wandb_project', action='store', type=str, required=False, default='ActiveGlass')
    parser.add_argument('--wandb_entity', action='store', type=str, required=False, default=None)
    parser.add_argument('--wandb_run_name', action='store', type=str, required=False, default=None)
    parser.add_argument('--wandb_mode', action='store', type=str, required=False, default='online', choices=['online', 'offline', 'disabled'])
    parser.add_argument('--wandb_log_interval', action='store', type=int, required=False, default=20, help='log every N steps to wandb')
    train(vars(parser.parse_args()))
