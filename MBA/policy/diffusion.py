from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import reduce
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix

from policy.diffusion_modules.conditional_unet1d import ConditionalUnet1D
from policy.diffusion_modules.mask_generator import LowdimMaskGenerator

ROT_DIM = 6

class DiffusionUNetPolicy(nn.Module):
    def __init__(self,
            action_dim,
            horizon, 
            n_obs_steps,
            obs_feature_dim,
            num_inference_steps=20,
            diffusion_step_embed_dim=256,
            down_dims=(256,512),
            kernel_size=5,
            n_groups=8,
            cond_predict_scale=True,
            # load parameters
            enable_mba=False,
            obj_dim=10,
            rot_smooth_lambda=0.0,
            cond_extra_dim=0,
            obj_pose_mode="abs",
            # parameters passed to step
            **kwargs):
        super().__init__()

        # create diffusion model
        input_dim = action_dim
        global_cond_dim = obs_feature_dim * n_obs_steps

        self.model = ConditionalUnet1D(
            input_dim=input_dim,
            local_cond_dim=None,
            global_cond_dim=global_cond_dim,
            diffusion_step_embed_dim=diffusion_step_embed_dim,
            down_dims=down_dims,
            kernel_size=kernel_size,
            n_groups=n_groups,
            cond_predict_scale=cond_predict_scale
        )

        self.enable_mba = enable_mba
        if enable_mba:
            self.obj_model = ConditionalUnet1D(
                input_dim=obj_dim,
                local_cond_dim=None,
                global_cond_dim=global_cond_dim,
                diffusion_step_embed_dim=diffusion_step_embed_dim,
                down_dims=down_dims,
                kernel_size=kernel_size,
                n_groups=n_groups,
                cond_predict_scale=cond_predict_scale
            )
            self.fob=nn.Sequential(
                nn.Linear(obj_dim * horizon + global_cond_dim, global_cond_dim),
                nn.ReLU(),
                nn.Linear(global_cond_dim, global_cond_dim)
            )

        # create noise scheduler
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=100,
            beta_start=0.0001,
            beta_end=0.02,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            set_alpha_to_one=True,
            steps_offset=0,
            prediction_type="epsilon"
        )

        self.mask_generator = LowdimMaskGenerator(
            action_dim=action_dim,
            obs_dim=0,
            max_n_obs_steps=n_obs_steps,
            fix_obs_steps=True,
            action_visible=False
        )
        if enable_mba:
            self.obj_mask_generator = LowdimMaskGenerator(
                action_dim=obj_dim,
                obs_dim=0,
                max_n_obs_steps=n_obs_steps,
                fix_obs_steps=True,
                action_visible=False
            )
        self.horizon = horizon
        self.obs_feature_dim = obs_feature_dim
        self.action_dim = action_dim
        if enable_mba:
            self.obj_dim = obj_dim
        self.n_obs_steps = n_obs_steps
        self.base_global_cond_dim = global_cond_dim
        self.cond_extra_dim = cond_extra_dim
        self.rot_smooth_lambda = rot_smooth_lambda
        self.obj_pose_mode = obj_pose_mode
        self.pose_dims = 3 + ROT_DIM
        self.returns_absolute_pose = obj_pose_mode == "abs"
        if cond_extra_dim > 0:
            self.global_cond_proj = nn.Sequential(
                nn.Linear(global_cond_dim + cond_extra_dim, global_cond_dim),
                nn.ReLU(),
                nn.Linear(global_cond_dim, global_cond_dim)
            )
        else:
            self.global_cond_proj = nn.Identity()
        self.kwargs = kwargs

        if num_inference_steps is None:
            num_inference_steps = self.noise_scheduler.config.num_train_timesteps
        self.num_inference_steps = num_inference_steps

    def _expand_base_pose(self, base_pose: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Expand the base pose tensor to match the temporal dimension of the target tensor.
        """
        if base_pose.dim() == target.dim():
            return base_pose
        return base_pose.unsqueeze(1).expand(-1, target.shape[1], -1)

    def _absolute_to_delta(self, pose: torch.Tensor, base_pose: torch.Tensor) -> torch.Tensor:
        """
        Convert absolute pose sequence to delta representation relative to base_pose.
        """
        expanded_base = self._expand_base_pose(base_pose, pose)
        result = pose.clone()
        result[..., :3] = pose[..., :3] - expanded_base[..., :3]

        target_rot6 = pose[..., 3:3 + ROT_DIM].contiguous()
        base_rot6 = expanded_base[..., 3:3 + ROT_DIM].contiguous()
        target_rot_mat = rotation_6d_to_matrix(target_rot6.reshape(-1, ROT_DIM))
        base_rot_mat = rotation_6d_to_matrix(base_rot6.reshape(-1, ROT_DIM))
        delta_rot_mat = target_rot_mat @ base_rot_mat.transpose(-1, -2)
        delta_rot6 = matrix_to_rotation_6d(delta_rot_mat).reshape_as(target_rot6)
        result[..., 3:3 + ROT_DIM] = delta_rot6
        return result

    def _delta_to_absolute(self, delta_pose: torch.Tensor, base_pose: torch.Tensor) -> torch.Tensor:
        """
        Convert delta pose sequence back to absolute representation using base_pose.
        """
        expanded_base = self._expand_base_pose(base_pose, delta_pose)
        result = delta_pose.clone()
        result[..., :3] = delta_pose[..., :3] + expanded_base[..., :3]

        delta_rot6 = delta_pose[..., 3:3 + ROT_DIM].contiguous()
        base_rot6 = expanded_base[..., 3:3 + ROT_DIM].contiguous()
        delta_rot_mat = rotation_6d_to_matrix(delta_rot6.reshape(-1, ROT_DIM))
        base_rot_mat = rotation_6d_to_matrix(base_rot6.reshape(-1, ROT_DIM))
        abs_rot_mat = delta_rot_mat @ base_rot_mat
        abs_rot6 = matrix_to_rotation_6d(abs_rot_mat).reshape_as(delta_rot6)
        result[..., 3:3 + ROT_DIM] = abs_rot6
        self.returns_absolute_pose = True
        return result

    def _prepare_global_cond(self, readout: torch.Tensor, extra_cond: torch.Tensor = None) -> torch.Tensor:
        batch_size = readout.shape[0]
        global_cond = readout.reshape(batch_size, -1)
        if self.cond_extra_dim > 0:
            if extra_cond is None:
                extra = torch.zeros(
                    size=(batch_size, self.cond_extra_dim),
                    dtype=readout.dtype,
                    device=readout.device
                )
            else:
                extra = extra_cond.reshape(batch_size, -1).to(
                    dtype=readout.dtype,
                    device=readout.device
                )
                if extra.shape[1] != self.cond_extra_dim:
                    raise ValueError(
                        f"Expected extra conditioning dim {self.cond_extra_dim}, got {extra.shape[1]}"
                    )
            global_cond = torch.cat([global_cond, extra], dim=-1)
        global_cond = self.global_cond_proj(global_cond)
        return global_cond
    
    # ========= inference  ============
    def conditional_sample(self, 
            condition_data, condition_mask,
            local_cond=None, global_cond=None,
            generator=None,
            sample_obj=False,
            # keyword arguments to scheduler.step
            **kwargs
            ):
        if sample_obj:
            model = self.obj_model
        else:
            model = self.model
        scheduler = self.noise_scheduler

        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator)
    
        # set step values
        scheduler.set_timesteps(self.num_inference_steps)

        for t in scheduler.timesteps:
            # 1. apply conditioning
            trajectory[condition_mask] = condition_data[condition_mask]

            # 2. predict model output
            model_output = model(trajectory, t, 
                local_cond=local_cond, global_cond=global_cond)

            # 3. compute previous image: x_t -> x_t-1
            trajectory = scheduler.step(
                model_output, t, trajectory, 
                generator=generator,
                **kwargs
                ).prev_sample
        
        # finally make sure conditioning is enforced
        trajectory[condition_mask] = condition_data[condition_mask]        

        return trajectory

    def predict_action(self, readout, extra_cond=None) -> Dict[str, torch.Tensor]:
        B = readout.shape[0]
        T = self.horizon
        Da = self.action_dim
        Do = self.obs_feature_dim
        To = self.n_obs_steps

        # build input
        device = readout.device
        dtype = readout.dtype
        obs_features = readout
        assert obs_features.shape[0] == B * To
        
        # condition through global feature
        local_cond = None
        global_cond = None
        # reshape back to B, Do
        global_cond = self._prepare_global_cond(obs_features, extra_cond)

        if self.enable_mba:
            Da = self.obj_dim

            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

            # run sampling
            sample = self.conditional_sample(
                cond_data, 
                cond_mask,
                local_cond=local_cond,
                global_cond=global_cond,
                sample_obj=True,
                **self.kwargs)
            
            obj_pred = sample[...,:Da]
            if self.obj_pose_mode == "delta":
                if extra_cond is None:
                    raise ValueError("extra_cond (current object pose) is required for delta mode inference.")
                base_pose = extra_cond[:, :self.pose_dims]
                obj_pred = self._delta_to_absolute(obj_pred, base_pose)

            Da = self.action_dim

            obs_features = readout
            assert obs_features.shape[0] == B * To
            global_cond = self._prepare_global_cond(obs_features, extra_cond)
            obj_pred = obj_pred.reshape(B, -1)
            global_cond = torch.cat([global_cond, obj_pred], dim=-1)
            global_cond = self.fob(global_cond)

        # empty data for action
        cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

        # run sampling
        sample = self.conditional_sample(
            cond_data, 
            cond_mask,
            local_cond=local_cond,
            global_cond=global_cond,
            **self.kwargs)

        action_pred = sample[...,:Da]

        return action_pred

    def predict_obj(self, readout: torch.Tensor, extra_cond=None) -> torch.Tensor:
        if not self.enable_mba:
            raise RuntimeError("Object prediction requested but MBA is disabled.")
        B = readout.shape[0]
        T = self.horizon
        Da = self.obj_dim
        To = self.n_obs_steps

        device = readout.device
        dtype = readout.dtype
        obs_features = readout
        assert obs_features.shape[0] == B * To

        global_cond = self._prepare_global_cond(obs_features, extra_cond)
        cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
        cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)
        sample = self.conditional_sample(
            cond_data,
            cond_mask,
            local_cond=None,
            global_cond=global_cond,
            sample_obj=True,
            **self.kwargs
        )
        if self.obj_pose_mode == "delta":
            if extra_cond is None:
                raise ValueError("extra_cond (current object pose) is required for delta mode inference.")
            base_pose = extra_cond[:, :self.pose_dims]
            sample = self._delta_to_absolute(sample, base_pose)
        return sample

    # ========= training  ============
    def compute_obj_loss(self, readout, actions_obj, extra_cond=None):
        batch_size = readout.shape[0]
        # handle different ways of passing observation
        local_cond = None
        trajectory = actions_obj
        if self.obj_pose_mode == "delta":
            if extra_cond is None:
                raise ValueError("extra_cond (current object pose) is required for delta obj pose training.")
            base_pose = extra_cond[:, :self.pose_dims]
            trajectory = self._absolute_to_delta(actions_obj, base_pose)
        cond_data = trajectory
        assert readout.shape[0] == batch_size * self.n_obs_steps
        # reshape back to B, Do
        global_cond = self._prepare_global_cond(readout, extra_cond)

        # generate impainting mask
        condition_mask = self.obj_mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]
        
        # Predict the noise residual
        pred = self.obj_model(noisy_trajectory, timesteps, 
            local_cond=local_cond, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()

        if self.rot_smooth_lambda > 0:
            alphas_cumprod = self.noise_scheduler.alphas_cumprod.to(trajectory.device)
            alpha_t = alphas_cumprod[timesteps].view(-1, 1, 1)
            alpha_t = torch.clamp(alpha_t, min=1e-4, max=1.0)
            sqrt_alpha_t = torch.sqrt(alpha_t)
            sqrt_one_minus_alpha_t = torch.sqrt(1 - alpha_t)
            x0_pred = (noisy_trajectory - sqrt_one_minus_alpha_t * pred) / sqrt_alpha_t
            rot_pred = x0_pred[..., 3:3 + ROT_DIM]

            rot_pred = F.normalize(rot_pred, dim=-1) #numerically stable

            rot_delta = rot_pred[:, 1:, :] - rot_pred[:, :-1, :]
            smooth_loss = rot_delta.pow(2).mean()
            loss = loss + self.rot_smooth_lambda * smooth_loss
        return loss
        

    def compute_loss(self, readout, actions, extra_cond=None):
        if self.enable_mba:
            B = readout.shape[0]
            T = self.horizon
            Da = self.obj_dim
            Do = self.obs_feature_dim
            To = self.n_obs_steps

            # build input
            device = readout.device
            dtype = readout.dtype
            obs_features = readout
            assert obs_features.shape[0] == B * To
            
            # condition through global feature
            local_cond = None
            global_cond = self._prepare_global_cond(obs_features, extra_cond)
            # empty data for action
            cond_data = torch.zeros(size=(B, T, Da), device=device, dtype=dtype)
            cond_mask = torch.zeros_like(cond_data, dtype=torch.bool)

            # run sampling
            with torch.no_grad():
                sample = self.conditional_sample(
                    cond_data, 
                    cond_mask,
                    local_cond=local_cond,
                    global_cond=global_cond,
                    sample_obj=1,
                    **self.kwargs)
            obj_pred = sample[..., :Da]  # B, T, Da
            if self.obj_pose_mode == "delta":
                if extra_cond is None:
                    raise ValueError("extra_cond (current object pose) is required for delta obj pose training.")
                base_pose = extra_cond[:, :self.pose_dims]
                obj_pred = self._delta_to_absolute(obj_pred, base_pose)
        else:
            B = readout.shape[0]
            obj_pred = None

        batch_size = readout.shape[0]

        # handle different ways of passing observation
        local_cond = None
        trajectory = actions
        cond_data = trajectory
        assert readout.shape[0] == batch_size * self.n_obs_steps
        # reshape back to B, Do
        global_cond = self._prepare_global_cond(readout, extra_cond)
        if self.enable_mba:
            obj_pred = obj_pred.reshape(B, -1)
            global_cond = torch.cat([global_cond, obj_pred],dim=-1)
            global_cond = self.fob(global_cond)

        # generate impainting mask
        condition_mask = self.mask_generator(trajectory.shape)

        # Sample noise that we'll add to the images
        noise = torch.randn(trajectory.shape, device=trajectory.device)
        bsz = trajectory.shape[0]
        # Sample a random timestep for each image
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps, 
            (bsz,), device=trajectory.device
        ).long()
        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_trajectory = self.noise_scheduler.add_noise(
            trajectory, noise, timesteps)
        
        # compute loss mask
        loss_mask = ~condition_mask

        # apply conditioning
        noisy_trajectory[condition_mask] = cond_data[condition_mask]
        
        # Predict the noise residual
        pred = self.model(noisy_trajectory, timesteps, 
            local_cond=local_cond, global_cond=global_cond)

        pred_type = self.noise_scheduler.config.prediction_type 
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'sample':
            target = trajectory
        else:
            raise ValueError(f"Unsupported prediction type {pred_type}")

        loss = F.mse_loss(pred, target, reduction='none')
        loss = loss * loss_mask.type(loss.dtype)
        loss = reduce(loss, 'b ... -> b (...)', 'mean')
        loss = loss.mean()
        return loss
