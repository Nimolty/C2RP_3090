import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from functools import partial
from torch.distributions import VonMises
from torch.distributions.multivariate_normal import _batch_mahalanobis, _standard_normal, _batch_mv
from pyro.distributions import TorchDistribution, constraints
from pyro.distributions.util import broadcast_shape
from abc import ABCMeta, abstractmethod
from functools import partial
from pyro.distributions import MultivariateStudentT
from config import config
from utils.utils import AverageMeter
from utils.img import im_norm_255
import os
import utils.fancy_logger as logger
import time
from ops.rotation_conversions import matrix_to_quaternion,quaternion_to_matrix
from ops.pnp.camera import PerspectiveCamera
from ops.pnp.cost_fun import AdaptiveHuberPnPCost
from ops.pnp.levenberg_marquardt import LMSolver, RSLMSolver
from ops.pnp.epropnp import EProPnP6DoF

npoints = 10

##epropnp
class MonteCarloPoseLoss(nn.Module):

    def __init__(self, init_norm_factor=1.0, momentum=0.1):
        super(MonteCarloPoseLoss, self).__init__()
        self.register_buffer('norm_factor', torch.tensor(init_norm_factor, dtype=torch.float))
        self.momentum = momentum

    def forward(self, pose_sample_logweights, cost_target, norm_factor):
        """
        Args:
            pose_sample_logweights: Shape (mc_samples, num_obj)
            cost_target: Shape (num_obj, )
            norm_factor: Shape ()
        """
        if self.training:
            with torch.no_grad():
                self.norm_factor.mul_(
                    1 - self.momentum).add_(self.momentum * norm_factor)

        loss_tgt = cost_target
        loss_pred = torch.logsumexp(pose_sample_logweights, dim=0)  # (num_obj, )

        loss_pose = loss_tgt + loss_pred  # (num_obj, )
        loss_pose[torch.isnan(loss_pose)] = 0
        loss_pose = loss_pose.mean() / self.norm_factor

        return loss_pose.mean()
def skew(x):
    """
    Args:
        x (torch.Tensor): shape (*, 3)

    Returns:
        torch.Tensor: (*, 3, 3), skew symmetric matrices
    """
    mat = x.new_zeros(x.shape[:-1] + (3, 3))
    mat[..., [2, 0, 1], [1, 2, 0]] = x
    mat[..., [1, 2, 0], [2, 0, 1]] = -x
    return mat
def quaternion_to_rot_mat(quaternions):
    """
    Args:
        quaternions (torch.Tensor): (*, 4)

    Returns:
        torch.Tensor: (*, 3, 3)
    """
    if quaternions.requires_grad:
        w, i, j, k = torch.unbind(quaternions, -1)
        rot_mats = torch.stack((
            1 - 2 * (j * j + k * k),     2 * (i * j - k * w),     2 * (i * k + j * w),
                2 * (i * j + k * w), 1 - 2 * (i * i + k * k),     2 * (j * k - i * w),
                2 * (i * k - j * w),     2 * (j * k + i * w), 1 - 2 * (i * i + j * j)), dim=-1,
        ).reshape(quaternions.shape[:-1] + (3, 3))
    else:
        w, v = quaternions.split([1, 3], dim=-1)
        rot_mats = 2 * (w.unsqueeze(-1) * skew(v) + v.unsqueeze(-1) * v.unsqueeze(-2))
        diag = torch.diagonal(rot_mats, dim1=-2, dim2=-1)
        diag += w * w - (v.unsqueeze(-2) @ v.unsqueeze(-1)).squeeze(-1)
    return rot_mats
def yaw_to_rot_mat(yaw):
    """
    Args:
        yaw (torch.Tensor): (*)

    Returns:
        torch.Tensor: (*, 3, 3)
    """
    sin_yaw = torch.sin(yaw)
    cos_yaw = torch.cos(yaw)
    # [[ cos_yaw, 0, sin_yaw],
    #  [       0, 1,       0],
    #  [-sin_yaw, 0, cos_yaw]]
    rot_mats = yaw.new_zeros(yaw.shape + (3, 3))
    rot_mats[..., 0, 0] = cos_yaw
    rot_mats[..., 2, 2] = cos_yaw
    rot_mats[..., 0, 2] = sin_yaw
    rot_mats[..., 2, 0] = -sin_yaw
    rot_mats[..., 1, 1] = 1
    return rot_mats
def evaluate_pnp(x3d, x2d, w2d, pose, camera, cost_fun,
                 out_jacobian=False, out_residual=False, out_cost=False, **kwargs):
    """
    Args:
        x3d (torch.Tensor): Shape (*, n, 3)
        x2d (torch.Tensor): Shape (*, n, 2)
        w2d (torch.Tensor): Shape (*, n, 2)
        pose (torch.Tensor): Shape (*, 4 or 7)
        camera: Camera object of batch size (*, )
        cost_fun: PnPCost object of batch size (*, )
        out_jacobian (torch.Tensor | bool): When a tensor is passed, treated as the output tensor;
            when True, returns the Jacobian; when False, skip the computation and returns None
        out_residual (torch.Tensor | bool): When a tensor is passed, treated as the output tensor;
            when True, returns the residual; when False, skip the computation and returns None
        out_cost (torch.Tensor | bool): When a tensor is passed, treated as the output tensor;
            when True, returns the cost; when False, skip the computation and returns None

    Returns:
        Tuple:
            residual (torch.Tensor | None): Shape (*, n*2)
            cost (torch.Tensor | None): Shape (*, )
            jacobian (torch.Tensor | None): Shape (*, n*2, 4 or 6)
    """
    x2d_proj, jac_cam = camera.project(
        x3d, pose, out_jac=(
            out_jacobian.view(x2d.shape[:-1] + (2, out_jacobian.size(-1))
                              ) if isinstance(out_jacobian, torch.Tensor)
            else out_jacobian), **kwargs)
    residual, cost, jacobian = cost_fun.compute(
        x2d_proj, x2d, w2d, jac_cam=jac_cam,
        out_residual=out_residual,
        out_cost=out_cost,
        out_jacobian=out_jacobian)
    return residual, cost, jacobian
def pnp_normalize(x3d, pose=None, detach_transformation=True):
    """
    Args:
        x3d (torch.Tensor): Shape (*, n, 3)
        pose (torch.Tensor | None): Shape (*, 4)
        detach_transformation (bool)

    Returns:
        Tuple[torch.Tensor]:
            offset: Shape (*, 1, 3)
            x3d_norm: Shape (*, n, 3), normalized x3d
            pose_norm: Shape (*, ), transformed pose
    """
    offset = torch.mean(
        x3d.detach() if detach_transformation else x3d, dim=-2)  # (*, 3)
    x3d_norm = x3d - offset.unsqueeze(-2)
    if pose is not None:
        pose_norm = torch.empty_like(pose)
        pose_norm[..., 3:] = pose[..., 3:]
        pose_norm[..., :3] = pose[..., :3] + \
            ((yaw_to_rot_mat(pose[..., 3]) if pose.size(-1) == 4
              else quaternion_to_rot_mat(pose[..., 3:])) @ offset.unsqueeze(-1)).squeeze(-1)
    else:
        pose_norm = None
    return offset, x3d_norm, pose_norm
def pnp_denormalize(offset, pose_norm):
    pose = torch.empty_like(pose_norm)
    pose[..., 3:] = pose_norm[..., 3:]
    pose[..., :3] = pose_norm[..., :3] - \
        ((yaw_to_rot_mat(pose_norm[..., 3]) if pose_norm.size(-1) == 4
          else quaternion_to_rot_mat(pose_norm[..., 3:])) @ offset.unsqueeze(-1)).squeeze(-1)
    return pose
def cholesky_wrapper(mat, default_diag=None, force_cpu=True):
    device = mat.device
    if force_cpu:
        mat = mat.cpu()
    try:
        tril = torch.cholesky(mat, upper=False)
    except RuntimeError:
        n_dims = mat.size(-1)
        tril = []
        default_tril_single = torch.diag(mat.new_tensor(default_diag)) if default_diag is not None \
            else torch.eye(n_dims, dtype=mat.dtype, device=mat.device)
        for cov in mat.reshape(-1, n_dims, n_dims):
            try:
                tril.append(torch.cholesky(cov, upper=False))
            except RuntimeError:
                tril.append(default_tril_single)
        tril = torch.stack(tril, dim=0).reshape(mat.shape)
    return tril.to(device)
class AngularCentralGaussian(TorchDistribution):
    arg_constraints = {'scale_tril': constraints.lower_cholesky}
    has_rsample = True

    def __init__(self, scale_tril, validate_args=None, eps=1e-6):
        q = scale_tril.size(-1)
        assert q > 1
        assert scale_tril.shape[-2:] == (q, q)
        batch_shape = scale_tril.shape[:-2]
        event_shape = (q,)
        self.scale_tril = scale_tril.expand(batch_shape + (-1, -1))
        self._unbroadcasted_scale_tril = scale_tril
        self.q = q
        self.area = 2 * math.pi ** (0.5 * q) / math.gamma(0.5 * q)
        self.eps = eps
        super().__init__(batch_shape, event_shape, validate_args=validate_args)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        value = value.expand(
            broadcast_shape(value.shape[:-1], self._unbroadcasted_scale_tril.shape[:-2])
            + self.event_shape)
        M = _batch_mahalanobis(self._unbroadcasted_scale_tril, value)
        half_log_det = self._unbroadcasted_scale_tril.diagonal(dim1=-2, dim2=-1).log().sum(-1)
        return M.log() * (-self.q / 2) - half_log_det - math.log(self.area)

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        normal = _standard_normal(shape,
                                  dtype=self._unbroadcasted_scale_tril.dtype,
                                  device=self._unbroadcasted_scale_tril.device)
        gaussian_samples = _batch_mv(self._unbroadcasted_scale_tril, normal)
        gaussian_samples_norm = gaussian_samples.norm(dim=-1)
        samples = gaussian_samples / gaussian_samples_norm.unsqueeze(-1)
        samples[gaussian_samples_norm < self.eps] = samples.new_tensor(
            [1.] + [0. for _ in range(self.q - 1)])
        return samples
class VonMisesUniformMix(VonMises):

    def __init__(self, loc, concentration, uniform_mix=0.25, **kwargs):
        super(VonMisesUniformMix, self).__init__(loc, concentration, **kwargs)
        self.uniform_mix = uniform_mix

    @torch.no_grad()
    def sample(self, sample_shape=torch.Size()):
        assert len(sample_shape) == 1
        x = np.empty(tuple(self._extended_shape(sample_shape)), dtype=np.float32)
        uniform_samples = round(sample_shape[0] * self.uniform_mix)
        von_mises_samples = sample_shape[0] - uniform_samples
        x[:uniform_samples] = np.random.uniform(
            -math.pi, math.pi, size=tuple(self._extended_shape((uniform_samples,))))
        x[uniform_samples:] = np.random.vonmises(
            self.loc.cpu().numpy(), self.concentration.cpu().numpy(),
            size=tuple(self._extended_shape((von_mises_samples,))))
        return torch.from_numpy(x).to(self.loc.device)

    def log_prob(self, value):
        von_mises_log_prob = super(VonMisesUniformMix, self).log_prob(value) + np.log(1 - self.uniform_mix)
        log_prob = torch.logaddexp(
            von_mises_log_prob,
            torch.full_like(von_mises_log_prob, math.log(self.uniform_mix / (2 * math.pi))))
        return log_prob
class LMSolver(nn.Module):
    """
    Levenberg-Marquardt solver, with fixed number of iterations.

    - For 4DoF case, the pose is parameterized as [x, y, z, yaw], where yaw is the
    rotation around the Y-axis in radians.
    - For 6DoF case, the pose is parameterized as [x, y, z, w, i, j, k], where
    [w, i, j, k] is the unit quaternion.
    """
    def __init__(
            self,
            dof=4,
            num_iter=16,
            min_lm_diagonal=1e-6,
            max_lm_diagonal=1e32,
            min_relative_decrease=1e-3,
            initial_trust_region_radius=30.0,
            max_trust_region_radius=1e16,
            eps=1e-5,
            normalize=False,
            init_solver=None):
        super(LMSolver, self).__init__()
        self.dof = dof
        self.num_iter = num_iter
        self.min_lm_diagonal = min_lm_diagonal
        self.max_lm_diagonal = max_lm_diagonal
        self.min_relative_decrease = min_relative_decrease
        self.initial_trust_region_radius = initial_trust_region_radius
        self.max_trust_region_radius = max_trust_region_radius
        self.eps = eps
        self.normalize = normalize
        self.init_solver = init_solver

    def forward(self, x3d, x2d, w2d, camera, cost_fun, with_pose_opt_plus=False,
                pose_init=None, normalize_override=None, **kwargs):
        if isinstance(normalize_override, bool):
            normalize = normalize_override
        else:
            normalize = self.normalize
        if normalize:
            transform, x3d, pose_init = pnp_normalize(x3d, pose_init, detach_transformation=True)

        pose_opt, pose_cov, cost = self.solve(
            x3d, x2d, w2d, camera, cost_fun, pose_init=pose_init, **kwargs)
        if with_pose_opt_plus:
            step = self.gn_step(x3d, x2d, w2d, pose_opt, camera, cost_fun)
            pose_opt_plus = self.pose_add(pose_opt, step, camera)
        else:
            pose_opt_plus = None

        if normalize:
            pose_opt = pnp_denormalize(transform, pose_opt)
            if pose_cov is not None:
                raise NotImplementedError('Normalized covariance unsupported')
            if pose_opt_plus is not None:
                pose_opt_plus = pnp_denormalize(transform, pose_opt_plus)
        return pose_opt, pose_cov, cost, pose_opt_plus

    def solve(self, x3d, x2d, w2d, camera, cost_fun, pose_init=None, cost_init=None,
              with_pose_cov=False, with_cost=False, force_init_solve=False, fast_mode=False):
        """
        Args:
            x3d (Tensor): Shape (num_obj, num_pts, 3)
            x2d (Tensor): Shape (num_obj, num_pts, 2)
            w2d (Tensor): Shape (num_obj, num_pts, 2)
            camera: Camera object of batch size (num_obj, )
            cost_fun: PnPCost object of batch size (num_obj, )
            pose_init (None | Tensor): Shape (num_obj, 4 or 7) in [x, y, z, yaw], optional
            cost_init (None | Tensor): Shape (num_obj, ), PnP cost of pose_init, optional
            with_pose_cov (bool): Whether to compute the covariance of pose_opt
            with_cost (bool): Whether to compute the cost of pose_opt
            force_init_solve (bool): Whether to force using the initialization solver when
                pose_init is not None
            fast_mode (bool): Fall back to Gauss-Newton for fast inference

        Returns:
            tuple:
                pose_opt (Tensor): Shape (num_obj, 4 or 7)
                pose_cov (Tensor | None): Shape (num_obj, 4, 4) or (num_obj, 6, 6), covariance
                    of local pose parameterization
                cost (Tensor | None): Shape (num_obj, )
        """
        with torch.no_grad():
            num_obj, num_pts, _ = x2d.size()
            tensor_kwargs = dict(dtype=x2d.dtype, device=x2d.device)

            if num_obj > 0:
                # evaluate_fun(pose, out_jacobian=None, out_residual=None, out_cost=None)
                evaluate_fun = partial(
                    evaluate_pnp,
                    x3d=x3d, x2d=x2d, w2d=w2d, camera=camera, cost_fun=cost_fun,
                    clip_jac=not fast_mode)

                if pose_init is None or force_init_solve:
                    assert self.init_solver is not None
                    if pose_init is None:
                        pose_init_solve, _, _ = self.init_solver.solve(
                            x3d, x2d, w2d, camera, cost_fun, fast_mode=fast_mode)
                        pose_opt = pose_init_solve
                    else:
                        if cost_init is None:
                            cost_init = evaluate_fun(pose=pose_init, out_cost=True)[1]
                        pose_init_solve, _, cost_init_solve = self.init_solver.solve(
                            x3d, x2d, w2d, camera, cost_fun, with_cost=True, fast_mode=fast_mode)
                        # print('ghjh',pose_init_solve.shape,pose_init.shape)
                        use_init = cost_init < cost_init_solve

                        pose_init_solve[use_init] = pose_init[use_init]
                        pose_opt = pose_init_solve
                else:
                    pose_opt = pose_init.clone()

                jac = torch.empty((num_obj, num_pts * 2, self.dof), **tensor_kwargs)
                residual = torch.empty((num_obj, num_pts * 2), **tensor_kwargs)
                cost = torch.empty((num_obj,), **tensor_kwargs)

                if fast_mode:  # disable trust region
                    for i in range(self.num_iter):
                        evaluate_fun(pose=pose_opt, out_jacobian=jac, out_residual=residual, out_cost=cost)
                        jac_t = jac.transpose(-1, -2)  # (num_obj, 4 or 6, num_pts * 2)
                        jtj = jac_t @ jac  # (num_obj, 4, 4) or (num_obj, 6, 6)
                        diagonal = torch.diagonal(jtj, dim1=-2, dim2=-1)  # (num_obj, 4 or 6)
                        diagonal += self.eps  # add to jtj
                        # (num_obj, 4 or 6, 1) = (num_obj, 4 or 6, num_pts * 2) @ (num_obj, num_pts * 2, 1)
                        gradient = jac_t @ residual.unsqueeze(-1)
                        if self.dof == 4:
                            pose_opt -= solve_wrapper(gradient, jtj).squeeze(-1)
                        else:
                            step = -solve_wrapper(gradient, jtj).squeeze(-1)
                            pose_opt[..., :3] += step[..., :3]
                            pose_opt[..., 3:] = F.normalize(pose_opt[..., 3:] + (
                                    camera.get_quaternion_transfrom_mat(pose_opt[..., 3:]) @ step[..., 3:, None]
                                ).squeeze(-1), dim=-1)
                else:
                    evaluate_fun(pose=pose_opt, out_jacobian=jac, out_residual=residual, out_cost=cost)
                    jac_new = torch.empty_like(jac)
                    residual_new = torch.empty_like(residual)
                    cost_new = torch.empty_like(cost)
                    radius = x2d.new_full((num_obj,), self.initial_trust_region_radius)
                    decrease_factor = x2d.new_full((num_obj,), 2.0)
                    step_is_successful = x2d.new_zeros((num_obj,), dtype=torch.bool)
                    i = 0
                    while i < self.num_iter:
                        self._lm_iter(
                            pose_opt,
                            jac, residual, cost,
                            jac_new, residual_new, cost_new,
                            step_is_successful, radius, decrease_factor,
                            evaluate_fun, camera)
                        i += 1
                    if with_pose_cov:
                        jac[step_is_successful] = jac_new[step_is_successful]
                        jtj = jac.transpose(-1, -2) @ jac
                        diagonal = torch.diagonal(jtj, dim1=-2, dim2=-1)  # (num_obj, 4 or 6)
                        diagonal += self.eps  # add to jtj
                    if with_cost:
                        cost[step_is_successful] = cost_new[step_is_successful]

                if with_pose_cov:
                    pose_cov = torch.inverse(jtj)
                else:
                    pose_cov = None
                if not with_cost:
                    cost = None

            else:
                pose_opt = torch.empty((0, 4 if self.dof == 4 else 7), **tensor_kwargs)
                pose_cov = torch.empty((0, self.dof, self.dof), **tensor_kwargs) if with_pose_cov else None
                cost = torch.empty((0, ), **tensor_kwargs) if with_cost else None

            return pose_opt, pose_cov, cost

    def _lm_iter(
            self,
            pose_opt,
            jac, residual, cost,
            jac_new, residual_new, cost_new,
            step_is_successful, radius, decrease_factor,
            evaluate_fun, camera):
        jac[step_is_successful] = jac_new[step_is_successful]
        residual[step_is_successful] = residual_new[step_is_successful]
        cost[step_is_successful] = cost_new[step_is_successful]

        # compute step
        residual_ = residual.unsqueeze(-1)
        jac_t = jac.transpose(-1, -2)  # (num_obj, 4 or 6, num_pts * 2)
        jtj = jac_t @ jac  # (num_obj, 4, 4) or (num_obj, 6, 6)

        jtj_lm = jtj.clone()
        diagonal = torch.diagonal(jtj_lm, dim1=-2, dim2=-1)  # (num_obj, 4 or 6)
        diagonal += diagonal.clamp(min=self.min_lm_diagonal, max=self.max_lm_diagonal
                                   ) / radius[:, None] + self.eps  # add to jtj_lm

        # (num_obj, 4 or 6, 1) = (num_obj, 4 or 6, num_pts * 2) @ (num_obj, num_pts * 2, 1)
        gradient = jac_t @ residual_
        # (num_obj, 4 or 6, 1)
        step_ = -solve_wrapper(gradient, jtj_lm)

        # evaluate step quality
        pose_new = self.pose_add(pose_opt, step_.squeeze(-1), camera)
        evaluate_fun(pose=pose_new,
                     out_jacobian=jac_new,
                     out_residual=residual_new,
                     out_cost=cost_new)

        model_cost_change = -(step_.transpose(-1, -2) @ ((jtj @ step_) / 2 + gradient)).flatten()

        relative_decrease = (cost - cost_new) / model_cost_change
        torch.bitwise_and(relative_decrease >= self.min_relative_decrease, model_cost_change > 0.0,
                          out=step_is_successful)

        # step accepted
        pose_opt[step_is_successful] = pose_new[step_is_successful]
        radius[step_is_successful] /= (
                1.0 - (2.0 * relative_decrease[step_is_successful] - 1.0) ** 3).clamp(min=1.0 / 3.0)
        radius.clamp_(max=self.max_trust_region_radius, min=self.eps)
        decrease_factor.masked_fill_(step_is_successful, 2.0)

        # step rejected
        radius[~step_is_successful] /= decrease_factor[~step_is_successful]
        decrease_factor[~step_is_successful] *= 2.0
        return

    def gn_step(self, x3d, x2d, w2d, pose, camera, cost_fun):
        residual, _, jac = evaluate_pnp(
            x3d, x2d, w2d, pose, camera, cost_fun,
            out_jacobian=True, out_residual=True)
        jac_t = jac.transpose(-1, -2)  # (num_obj, 4 or 6, num_pts * 2)
        jtj = jac_t @ jac  # (num_obj, 4, 4) or (num_obj, 6, 6)
        jtj = jtj + torch.eye(self.dof, device=jtj.device, dtype=jtj.dtype) * self.eps
        # (num_obj, 4 or 6, 1) = (num_obj, 4 or 6, num_pts * 2) @ (num_obj, num_pts * 2, 1)
        gradient = jac_t @ residual.unsqueeze(-1)
        step = -solve_wrapper(gradient, jtj).squeeze(-1)
        return step

    def pose_add(self, pose_opt, step, camera):
        if self.dof == 4:
            pose_new = pose_opt + step
        else:
            pose_new = torch.cat(
                (pose_opt[..., :3] + step[..., :3],
                 F.normalize(pose_opt[..., 3:] + (
                         camera.get_quaternion_transfrom_mat(pose_opt[..., 3:]) @ step[..., 3:, None]
                     ).squeeze(-1), dim=-1)),
                dim=-1)
        return pose_new
class RSLMSolver(LMSolver):
    """
    Random Sample Levenberg-Marquardt solver, a generalization of RANSAC.
    Used for initialization in ambiguous problems.
    """
    def __init__(
            self,
            num_points=16,
            num_proposals=64,
            num_iter=3,
            **kwargs):
        super(RSLMSolver, self).__init__(num_iter=num_iter, **kwargs)
        self.num_points = num_points
        self.num_proposals = num_proposals

    def center_based_init(self, x2d, x3d, camera, eps=1e-6):
        # print(x2d.shape)
        # print(F.pad(x2d, [0, 1], mode='constant', value=1.).transpose(-1, -2).shape)
        x2dc = solve_wrapper(F.pad(x2d, [0, 1], mode='constant', value=1.).transpose(-1, -2),
                             camera.cam_mats).transpose(-1, -2)
        x2dc = x2dc[..., :2] / x2dc[..., 2:].clamp(min=eps)
        x2dc_std, x2dc_mean = torch.std_mean(x2dc, dim=-2)
        x3d_std = torch.std(x3d, dim=-2)
        if self.dof == 4:
            t_vec = F.pad(
                x2dc_mean, [0, 1], mode='constant', value=1.
            ) * (x3d_std[..., 1] / x2dc_std[..., 1].clamp(min=eps)).unsqueeze(-1)
        else:
            t_vec = F.pad(
                x2dc_mean, [0, 1], mode='constant', value=1.
            ) * (math.sqrt(2 / 3) * x3d_std.norm(dim=-1) / x2dc_std.norm(dim=-1).clamp(min=eps)
                 ).unsqueeze(-1)
        return t_vec

    def solve(self, x3d, x2d, w2d, camera, cost_fun, **kwargs):
        with torch.no_grad():
            bs, pn, _ = x2d.size()
            # print(bs,pn,_)
            if bs > 0:
                mean_weight = w2d.mean(dim=-1).reshape(1, bs, pn).expand(self.num_proposals, -1, -1)
                # print(mean_weight.shape,self.num_points)
                inds = torch.multinomial(
                    mean_weight.reshape(-1,pn), self.num_points
                ).reshape(self.num_proposals, bs, self.num_points)
                bs_inds = torch.arange(bs, device=inds.device)
                inds += (bs_inds * pn)[:, None]

                x2d_samples = x2d.reshape(-1, 2)[inds]  # (num_proposals, bs, num_points, 2)
                x3d_samples = x3d.reshape(-1, 3)[inds]  # (num_proposals, bs, num_points, 3)
                w2d_samples = w2d.reshape(-1, 2)[inds]  # (num_proposals, bs, num_points, 3)

                pose_init = x2d.new_empty((self.num_proposals, bs, 4 if self.dof == 4 else 7))
                pose_init[..., :3] = self.center_based_init(x2d, x3d, camera)
                if self.dof == 4:
                    # pass
                    pose_init[..., 3] = torch.rand(
                        (self.num_proposals, bs), dtype=x2d.dtype, device=x2d.device) * (2 * math.pi)
                else:
                    pose_init[..., 3:] = torch.randn(
                        (self.num_proposals, bs, 4), dtype=x2d.dtype, device=x2d.device)
                    q_norm = pose_init[..., 3:].norm(dim=-1)
                    pose_init[..., 3:] /= q_norm.unsqueeze(-1)
                    pose_init.view(-1, 7)[(q_norm < self.eps).flatten(), 3:] = x2d.new_tensor([1, 0, 0, 0])

                camera_expand = camera.shallow_copy()
                camera_expand.repeat_(self.num_proposals)
                cost_fun_expand = cost_fun.shallow_copy()
                cost_fun_expand.repeat_(self.num_proposals)

                pose, _, _ = super(RSLMSolver, self).solve(
                    x3d_samples.reshape(self.num_proposals * bs, self.num_points, 3),
                    x2d_samples.reshape(self.num_proposals * bs, self.num_points, 2),
                    w2d_samples.reshape(self.num_proposals * bs, self.num_points, 2),
                    camera_expand,
                    cost_fun_expand,
                    pose_init=pose_init.reshape(self.num_proposals * bs, pose_init.size(-1)),
                    **kwargs)

                pose = pose.reshape(self.num_proposals, bs, pose.size(-1))

                cost = evaluate_pnp(x3d, x2d, w2d, pose, camera, cost_fun, out_cost=True)[1]

                min_cost, min_cost_ind = cost.min(dim=0)
                pose = pose[min_cost_ind, torch.arange(bs, device=pose.device)]

            else:
                pose = x2d.new_empty((0, 4 if self.dof == 4 else 7))
                min_cost = x2d.new_empty((0, ))

            return pose, None, min_cost
def solve_wrapper(A,b):
    if A.numel() > 0:
        return torch.linalg.solve(b, A)[0]
    else:
        return b + A.reshape_as(b)
class EProPnPBase(torch.nn.Module, metaclass=ABCMeta):
    """
    End-to-End Probabilistic Perspective-n-Points.

    Args:
        mc_samples (int): Number of total Monte Carlo samples
        num_iter (int): Number of AMIS iterations
        normalize (bool)
        eps (float)
        solver (dict): PnP solver
    """
    def __init__(
            self,
            mc_samples=512,
            num_iter=4,
            normalize=False,
            eps=1e-5,
            solver=None):
        super(EProPnPBase, self).__init__()
        assert num_iter > 0
        assert mc_samples % num_iter == 0
        self.mc_samples = mc_samples
        self.num_iter = num_iter
        self.iter_samples = self.mc_samples // self.num_iter
        self.eps = eps
        self.normalize = normalize
        self.solver = solver

    @abstractmethod
    def allocate_buffer(self, *args, **kwargs):
        pass

    @abstractmethod
    def initial_fit(self, *args, **kwargs):
        pass

    @abstractmethod
    def gen_new_distr(self, *args, **kwargs):
        pass

    @abstractmethod
    def gen_old_distr(self, *args, **kwargs):
        pass

    @abstractmethod
    def estimate_params(self, *args, **kwargs):
        pass

    def forward(self, *args, **kwargs):
        return self.solver(*args, **kwargs)

    def monte_carlo_forward(self, x3d, x2d, w2d, camera, cost_fun,
                            pose_init=None, force_init_solve=True, **kwargs):
        """
        Monte Carlo PnP forward. Returns weighted pose samples drawn from the probability
        distribution of pose defined by the correspondences {x_{3D}, x_{2D}, w_{2D}}.

        Args:
            x3d (Tensor): Shape (num_obj, num_points, 3)
            x2d (Tensor): Shape (num_obj, num_points, 2)
            w2d (Tensor): Shape (num_obj, num_points, 2)
            camera: Camera object of batch size (num_obj, )
            cost_fun: PnPCost object of batch size (num_obj, )
            pose_init (Tensor | None): Shape (num_obj, 4 or 7), optional. The target pose
                (y_{gt}) can be passed for training with Monte Carlo pose loss
            force_init_solve (bool): Whether to force using the initialization solver when
                pose_init is not None

        Returns:
            Tuple:
                pose_opt (Tensor): Shape (num_obj, 4 or 7), PnP solution y*
                cost (Tensor | None): Shape (num_obj, ), is not None when with_cost=True
                pose_opt_plus (Tensor | None): Shape (num_obj, 4 or 7), y* + Δy, used in derivative
                    regularization loss, is not None when with_pose_opt_plus=True, can be backpropagated
                pose_samples (Tensor): Shape (mc_samples, num_obj, 4 or 7)
                pose_sample_logweights (Tensor): Shape (mc_samples, num_obj), can be backpropagated
                cost_init (Tensor | None): Shape (num_obj, ), is None when pose_init is None, can be
                    backpropagated
        """
        if self.normalize:
            transform, x3d, pose_init = pnp_normalize(x3d, pose_init, detach_transformation=True)

        assert x3d.dim() == x2d.dim() == w2d.dim() == 3
        num_obj = x3d.size(0)

        evaluate_fun = partial(
            evaluate_pnp,
            x3d=x3d, x2d=x2d, w2d=w2d, camera=camera, cost_fun=cost_fun, out_cost=True)
        cost_init = evaluate_fun(pose=pose_init)[1] if pose_init is not None else None

        pose_opt, pose_cov, cost, pose_opt_plus = self.solver(
            x3d, x2d, w2d, camera, cost_fun,
            pose_init=pose_init, cost_init=cost_init,
            with_pose_cov=True, force_init_solve=force_init_solve,
            normalize_override=False, **kwargs)

        if num_obj > 0:
            pose_samples = x3d.new_empty((self.num_iter, self.iter_samples) + pose_opt.size())
            logprobs = x3d.new_empty((self.num_iter, self.num_iter, self.iter_samples, num_obj))
            cost_pred = x3d.new_empty((self.num_iter, self.iter_samples, num_obj))

            distr_params = self.allocate_buffer(num_obj, dtype=x3d.dtype, device=x3d.device)

            with torch.no_grad():
                self.initial_fit(pose_opt, pose_cov, camera, *distr_params)

            for i in range(self.num_iter):
                # ===== step 1: generate samples =====
                new_trans_distr, new_rot_distr = self.gen_new_distr(i, *distr_params)
                # (iter_sample, num_obj, 3)
                pose_samples[i, :, :, :3] = new_trans_distr.sample((self.iter_samples, ))
                # (iter_sample, num_obj, 1 or 4)
                pose_samples[i, :, :, 3:] = new_rot_distr.sample((self.iter_samples, ))

                # ===== step 2: evaluate integrand =====
                cost_pred[i] = evaluate_fun(pose=pose_samples[i])[1]

                # ===== step 3: evaluate proposal mixture logprobs =====
                # (i + 1, iter_sample, num_obj)
                # all samples (i + 1, iter_sample, num_obj) on new distr (num_obj, )
                logprobs[i, :i + 1] = new_trans_distr.log_prob(pose_samples[:i + 1, :, :, :3]) \
                                      + new_rot_distr.log_prob(pose_samples[:i + 1, :, :, 3:]).flatten(2)
                if i > 0:
                    old_trans_distr, old_rot_distr = self.gen_old_distr(i, *distr_params)
                    # (i, iter_sample, num_obj)
                    # new samples (iter_sample, num_obj) on old distr (i, 1, num_obj)
                    logprobs[:i, i] = old_trans_distr.log_prob(pose_samples[i, :, :, :3]) \
                                      + old_rot_distr.log_prob(pose_samples[i, :, :, 3:]).flatten(2)
                # (i + 1, i + 1, iter_sample, num_obj) -> (i + 1, iter_sample, num_obj)
                mix_logprobs = torch.logsumexp(logprobs[:i + 1, :i + 1], dim=0) - math.log(i + 1)

                # ===== step 4: get sample weights =====
                # (i + 1, iter_sample, num_obj)
                pose_sample_logweights = -cost_pred[:i + 1] - mix_logprobs

                # ===== step 5: estimate new proposal =====
                if i == self.num_iter - 1:
                    break  # break at last iter
                with torch.no_grad():
                    self.estimate_params(
                        i,
                        pose_samples[:i + 1].reshape(((i + 1) * self.iter_samples, ) + pose_opt.size()),
                        pose_sample_weights.reshape((i + 1) * self.iter_samples, num_obj),
                        *distr_params)

            pose_samples = pose_samples.reshape((self.mc_samples, ) + pose_opt.size())
            pose_sample_logweights = pose_sample_logweights.reshape(self.mc_samples, num_obj)

        else:
            pose_samples = x2d.new_zeros((self.mc_samples, ) + pose_opt.size())
            pose_sample_logweights = x3d.reshape(self.mc_samples, 0) \
                                     + x2d.reshape(self.mc_samples, 0) \
                                     + w2d.reshape(self.mc_samples, 0)

        if self.normalize:
            pose_opt = pnp_denormalize(transform, pose_opt)
            pose_samples = pnp_denormalize(transform, pose_samples)
            if pose_opt_plus is not None:
                pose_opt_plus = pnp_denormalize(transform, pose_opt_plus)

        return pose_opt, cost, pose_opt_plus, pose_samples, pose_sample_logweights, cost_init
class EProPnPeasy(EProPnPBase):
    """
    End-to-End Probabilistic Perspective-n-Points for 6DoF pose estimation.
    The pose is parameterized as [x, y, z, w, i, j, k], where [w, i, j, k]
    is the unit quaternion.
    Adopted proposal distributions:
        position: multivariate t-distribution, degrees of freedom = 3
        orientation: angular central Gaussian distribution
    """

    def __init__(self,
                 *args,
                 acg_mle_iter=3,
                 acg_dispersion=0.001,
                 **kwargs):
        super(EProPnPeasy, self).__init__(*args, **kwargs)
        self.acg_mle_iter = acg_mle_iter
        self.acg_dispersion = acg_dispersion

    def allocate_buffer(self, num_obj, dtype=torch.float32, device=None):
        trans_mode = torch.empty((self.num_iter, num_obj, 3), dtype=dtype, device=device)
        trans_cov_tril = torch.empty((self.num_iter, num_obj, 3, 3), dtype=dtype, device=device)
        rot_cov_tril = torch.empty((self.num_iter, num_obj, 4, 4), dtype=dtype, device=device)
        return trans_mode, trans_cov_tril, rot_cov_tril

    def initial_fit(self,
                    pose_opt, pose_cov, camera,
                    trans_mode, trans_cov_tril,
                    rot_cov_tril):
        trans_mode[0], rot_mode = pose_opt.split([3, 4], dim=-1)
        trans_cov_tril[0] = cholesky_wrapper(pose_cov[:, :3, :3])

        eye_4 = torch.eye(4, dtype=pose_opt.dtype, device=pose_opt.device)
        transform_mat = camera.get_quaternion_transfrom_mat(rot_mode)
        rot_cov = (transform_mat @ pose_cov[:, 3:, 3:].inverse() @ transform_mat.transpose(-1, -2)
                   + eye_4).inverse()
        rot_cov.div_(rot_cov.diagonal(
            offset=0, dim1=-1, dim2=-2).sum(-1)[..., None, None])
        rot_cov_tril[0] = cholesky_wrapper(
            rot_cov + rot_cov.det()[:, None, None] ** 0.25 * (self.acg_dispersion * eye_4))

    @staticmethod
    def gen_new_distr(iter_id, trans_mode, trans_cov_tril, rot_cov_tril):
        new_trans_distr = MultivariateStudentT(3, trans_mode[iter_id], trans_cov_tril[iter_id])
        new_rot_distr = AngularCentralGaussian(rot_cov_tril[iter_id])
        return new_trans_distr, new_rot_distr

    @staticmethod
    def gen_old_distr(iter_id, trans_mode, trans_cov_tril, rot_cov_tril):
        mix_trans_distr = MultivariateStudentT(
            3, trans_mode[:iter_id, None], trans_cov_tril[:iter_id, None])
        mix_rot_distr = AngularCentralGaussian(rot_cov_tril[:iter_id, None])
        return mix_trans_distr, mix_rot_distr

    def estimate_params(self, iter_id, pose_samples, pose_sample_logweights,
                        trans_mode, trans_cov_tril, rot_cov_tril):
        sample_weights_norm = torch.softmax(pose_sample_logweights, dim=0)
        # translation var mean
        # (cum_sample, num_obj, 3) -> (num_obj, 3)
        trans_mode[iter_id + 1] = (sample_weights_norm[..., None] * pose_samples[..., :3]).sum(dim=0)
        trans_dev = pose_samples[..., :3] - trans_mode[iter_id + 1]  # (cum_sample, num_obj, 3)
        # (cum_sample, num_obj, 1, 1) * (cum_sample, num_obj, 3, 1)
        # * (cum_sample, num_obj, 1, 3) -> (num_obj, 3, 3)
        trans_cov = (sample_weights_norm[..., None, None] * trans_dev.unsqueeze(-1)
                     * trans_dev.unsqueeze(-2)).sum(dim=0)
        trans_cov_tril[iter_id + 1] = cholesky_wrapper(trans_cov)
        # rotation estimation
        eye_4 = torch.eye(4, dtype=pose_samples.dtype, device=pose_samples.device)
        rot = pose_samples[..., 3:]  # (cum_sample, num_obj, 4)
        r_r_t = rot[:, :, :, None] * rot[:, :, None, :]  # (cum_sample, num_obj, 4, 4)
        rot_cov = eye_4.expand(pose_samples.size(1), 4, 4).clone()
        for _ in range(self.acg_mle_iter):
            # (cum_sample, num_obj, 1, 1)
            M = rot[:, :, None, :] @ rot_cov.inverse() @ rot[:, :, :, None]
            invM_weighted = sample_weights_norm[..., None, None] / M.clamp(min=self.eps)
            invM_weighted_norm = invM_weighted / invM_weighted.sum(dim=0)
            # (num_obj, 4, 4) trace equals 1
            rot_cov = (invM_weighted_norm * r_r_t).sum(dim=0) + eye_4 * self.eps
        rot_cov_tril[iter_id + 1] = cholesky_wrapper(
            rot_cov + rot_cov.det()[:, None, None] ** 0.25 * (self.acg_dispersion * eye_4))

camera_intrinsic = np.asarray([[320, 0, 320], [0, 320, 240], [0, 0, 1]], dtype = np.float64)

epropnp = EProPnPeasy(
        mc_samples=512,
        num_iter=4,
        solver=LMSolver(
            dof=6,
            num_iter=4,
            init_solver=RSLMSolver(
                dof=6,
                num_points=6,
                num_proposals=128,
                num_iter=4)))
def train(epoch,x2d,x3d,rmat,tval,model):
    model = 0
    # model.train(
    tval = torch.tensor(tval.reshape(3))
    # print(tval.shape)
    Loss = AverageMeter()
    Loss_rot = AverageMeter()
    Loss_trans = AverageMeter()
    Loss_mc = AverageMeter()
    Loss_t = AverageMeter()
    Loss_r = AverageMeter()
    rot_quat = matrix_to_quaternion(torch.tensor(rmat))
    pose_gt = torch.tensor(np.ones((batch_size,npoints))).double()
    x2d = torch.tensor(x2d)
    x3d = torch.tensor(x3d)
    w2d = torch.tensor(np.ones((batch_size,npoints, 2)) / x2d.shape[1])
    bs = x2d.shape[0]
    # print(x2d.shape, x3d.shape, w2d.shape,pose_gt.shape)
    camera = PerspectiveCamera(
        cam_mats=torch.tensor(camera_intrinsic.reshape((1, 3, 3))),
        z_min=0.01,
        lb=torch.zeros((bs,2)),
        ub=torch.ones((bs,2)))
    ############################################################
    cost_fun = AdaptiveHuberPnPCost(
        relative_delta=0.1)
    cost_fun.set_param(x2d, w2d)
    _, _, pose_opt_plus, _, pose_sample_logweights, cost_tgt = epropnp.monte_carlo_forward(
        x3d, x2d, w2d, camera, cost_fun,
        pose_init=pose_gt, force_init_solve=True, with_pose_opt_plus=True)
    loss_mc = 0
    # loss_mc = model.monte_carlo_pose_loss(
    #     pose_sample_logweights, cost_tgt, bs)
    ############################################################
    # print(pose_opt_plus.shape,pose_gt.shape)
    loss_t = (pose_opt_plus[:, :3] - pose_gt[:, :3]).norm(dim=-1)
    beta = 0.05
    loss_t = torch.where(loss_t < beta, 0.5 * loss_t.square() / beta,
                         loss_t - 0.5 * beta)
    loss_t = loss_t.mean()

    dot_quat = (pose_opt_plus[:, None, 3:] @ pose_gt[:, 3:, None]).squeeze(-1).squeeze(-1)
    loss_r = (1 - dot_quat.square()) * 2
    loss_r = loss_r.mean()

    # loss_rot = torch.nn.MSELoss(
    #     loss_msk_var[:, :3] * noc, loss_msk_var[:, :3] * target_var[:, :3])
    loss_rot = 0
    loss_trans = 0
    loss = 1 * loss_rot + 1 * loss_trans \
           + 0.02 * loss_mc + 0. * loss_t \
           + 0. * loss_r

    Loss.update(loss.item() if loss != 0 else 0, bs)
    Loss_rot.update(loss_rot.item() if loss_rot != 0 else 0, bs)
    Loss_trans.update(loss_trans.item() if loss_trans != 0 else 0, bs)
    Loss_mc.update(loss_mc.item() if loss_mc != 0 else 0, bs)
    Loss_t.update(loss_t.item() if loss_t != 0 else 0,bs)
    Loss_r.update(loss_r.item() if loss_r != 0 else 0, bs)

def test(x2d,x3d,rmat,tval):

    # print(tval)
    rot_quat = torch.tensor([list(matrix_to_quaternion(x)) for x in rmat])
    print(rot_quat.shape,tval.shape)

    pose_gt = torch.cat((tval,rot_quat),dim = -1).double()
    x2d = torch.tensor(x2d)
    x3d = torch.tensor(x3d)
    w2d = torch.tensor(np.ones((batch_size, npoints, 2)) / x2d.shape[1])

    camera = PerspectiveCamera(
        cam_mats=torch.tensor(camera_intrinsic.reshape((1, 3, 3))),
        z_min=0.01,
        lb=torch.ones((batch_size,2))*-100,
        ub=torch.ones((batch_size,2))*100)
    ############################################################
    cost_fun = AdaptiveHuberPnPCost(
        relative_delta=0.1)
    cost_fun.set_param(x2d, w2d)
    pot, _, pose_opt_plus, _, pose_sample_logweights, cost_tgt = epropnp.monte_carlo_forward(
        x3d, x2d, w2d, camera, cost_fun,
        pose_init=pose_gt, force_init_solve=True, with_pose_opt_plus=True)

    # print(pose_opt_plus.shape)
    return pot
#
#
# def test(x2d,x3d,rmat,tval):
#
#     # print(tval)
#     rot_quat = torch.tensor([list(matrix_to_quaternion(x)) for x in rmat])
#     print(rot_quat.shape,tval.shape)
#
#     pose_gt = torch.cat((tval,rot_quat),dim = -1).double()
#     x2d = torch.tensor(x2d)
#     x3d = torch.tensor(x3d)
#     w2d = torch.tensor(np.ones((batch_size, npoints, 2)) / x2d.shape[1])
#
#     camera = PerspectiveCamera(
#         cam_mats=torch.tensor(camera_intrinsic.reshape((1, 3, 3))),
#         z_min=0.01,
#         lb=torch.ones((batch_size,2))*-100,
#         ub=torch.ones((batch_size,2))*100)
#     ############################################################
#     cost_fun = AdaptiveHuberPnPCost(
#         relative_delta=0.1)
#     cost_fun.set_param(x2d, w2d)
#     pot, _, pose_opt_plus, _, pose_sample_logweights, cost_tgt = epropnp.monte_carlo_forward(
#         x3d, x2d, w2d, camera, cost_fun,
#         pose_init=pose_gt, force_init_solve=True, with_pose_opt_plus=True)
#
#     # print(pose_opt_plus.shape)
#     return pot
#
# class Model(nn.Module):
#
#     def __init__(
#             self,
#             num_points=10,  # number of 2D-3D pairs
#             mlp_layers=[1024],  # a single hidden layer
#             epropnp=EProPnP6DoF(
#                 mc_samples=128,
#                 num_iter=4,
#                 solver=LMSolver(
#                     dof=6,
#                     num_iter=6,
#                     init_solver=RSLMSolver(
#                         dof=6,
#                         num_points=8,
#                         num_proposals=128,
#                         num_iter=6))),
#             camera=PerspectiveCamera(),
#             cost_fun=AdaptiveHuberPnPCost(
#                 relative_delta=0.5)):
#         super().__init__()
#         self.num_points = num_points
#         mlp_layers = [2] + mlp_layers
#         print(mlp_layers)
#         mlp = []
#         for i in range(len(mlp_layers) - 1):
#             mlp.append(nn.Linear(mlp_layers[i], mlp_layers[i + 1]))
#             mlp.append(nn.LeakyReLU())
#         mlp.append(nn.Linear(mlp_layers[-1], num_points * (2)))
#         self.mlp = nn.Sequential(*mlp)
#         # print(self.mlp)
#         # Here we use static weight_scale because the data noise is homoscedastic
#         self.log_weight_scale = nn.Parameter(torch.zeros(2))
#         self.epropnp = epropnp
#         self.camera = camera
#         self.cost_fun = cost_fun
#
#     def forward_correspondence(self, in_pose):
#         w2d = self.mlp(in_pose).reshape(-1, self.num_points, 2)
#         # print(w2d)
#         w2d = (w2d.log_softmax(dim=-2) + self.log_weight_scale).exp()
#         # print(w2d.shape)
#         # equivalant to:
#         #     w2d = w2d.softmax(dim=-2) * self.log_weight_scale.exp()
#         return w2d
#
#     def forward_train(self,x2d,x3d,cam_mats, out_pose):
#         w2d = self.forward_correspondence(torch.randn([32, 2], device=device))
#         self.camera.set_param(cam_mats)
#
#         self.cost_fun.set_param(x2d.detach(), w2d)  # compute dynamic delta
#         pose_opt, cost, pose_opt_plus, pose_samples, pose_sample_logweights, cost_tgt = self.epropnp.monte_carlo_forward(
#             x3d,
#             x2d,
#             w2d,
#             self.camera,
#             self.cost_fun,
#             pose_init=out_pose,
#             force_init_solve=True,
#             with_pose_opt_plus=True)  # True for derivative regularization loss
#         norm_factor = model.log_weight_scale.detach().exp().mean()
#         return pose_opt, cost, pose_opt_plus, pose_samples, pose_sample_logweights, cost_tgt, norm_factor
#
#     def forward_test(self,x2d,x3d,w2d,cam_mats, fast_mode=False):
#
#         self.camera.set_param(cam_mats)
#         self.cost_fun.set_param(x2d.detach(), w2d)
#         # returns a mode of the distribution
#         pose_opt, _, _, _ = self.epropnp(
#             x3d, x2d, w2d, self.camera, self.cost_fun,
#             fast_mode=fast_mode)  # fast_mode=True activates Gauss-Newton solver (no trust region)
#         return pose_opt
#         # or returns weighted samples drawn from the distribution (slower):
#         #     _, _, _, pose_samples, pose_sample_logweights, _ = self.epropnp.monte_carlo_forward(
#         #         x3d, x2d, w2d, self.camera, self.cost_fun, fast_mode=fast_mode)
#         #     pose_sample_weights = pose_sample_logweights.softmax(dim=0)
#         #     return pose_samples, pose_sample_weights
#
#
# class MonteCarloPoseLoss(nn.Module):
#
#     def __init__(self, init_norm_factor=1.0, momentum=0.1):
#         super(MonteCarloPoseLoss, self).__init__()
#         self.register_buffer('norm_factor', torch.tensor(init_norm_factor, dtype=torch.float))
#         self.momentum = momentum
#
#     def forward(self, pose_sample_logweights, cost_target, norm_factor):
#         """
#         Args:
#             pose_sample_logweights: Shape (mc_samples, num_obj)
#             cost_target: Shape (num_obj, )
#             norm_factor: Shape ()
#         """
#         if self.training:
#             with torch.no_grad():
#                 self.norm_factor.mul_(
#                     1 - self.momentum).add_(self.momentum * norm_factor)
#
#         loss_tgt = cost_target
#         loss_pred = torch.logsumexp(pose_sample_logweights, dim=0)  # (num_obj, )
#
#         loss_pose = loss_tgt + loss_pred  # (num_obj, )
#         loss_pose[torch.isnan(loss_pose)] = 0
#         loss_pose = loss_pose.mean() / self.norm_factor
#
#         return loss_pose.mean()
#
# datasize = 2500
# data = np.loadtxt('data.csv')
# _x2d = torch.tensor(data[:,0:2])
# _x3d = torch.tensor(data[:,2:])
# # print(_x2d.shape,_x3d.shape)
# pose = np.loadtxt('gt_pose.csv')
# pose = pose.reshape(-1,3,4)
# _rmat = pose[:,:,0:3]
# # print(_rmat[3].shape)
# _rqua = torch.tensor([np.asarray(matrix_to_quaternion(torch.tensor(x).unsqueeze(0))) for x in _rmat]).transpose(-1,-2)
# _tval = torch.tensor(pose[:,:,3:])
# # print(_rqua.shape,_tval.shape)
# out_pose = torch.cat((_rqua,_tval),dim = -2).squeeze()
# # print(out_pose)
# # poset = np.zeros((1,npoints))
# #
# # batch_size = 1
# # for i in range(datasize):
# #     x2d = torch.tensor(_x2d[i*npoints*batch_size:i*npoints*batch_size+npoints*batch_size]).reshape(batch_size,npoints,2)
# #     x3d = torch.tensor(_x3d[i * npoints*batch_size:i * npoints*batch_size + npoints*batch_size]).reshape(batch_size,npoints,3)
# #     # print(x2d)
# #     rmat = torch.tensor(_rmat[i*batch_size:i*batch_size+batch_size]).reshape(batch_size,3,3)
# #     tval = torch.tensor(_tval[i*batch_size:i*batch_size+batch_size]).reshape(batch_size,3)
# #     p = test(x2d,x3d,rmat,tval)
# #     print(p.shape)
# #     pose_t = p[:,0:3]
# #     pose_q = p[:,3:]
# #     # r = matrix_to_quaternion(rmat)
# #     mat = quaternion_to_rot_mat(pose_q)
# #     print("rot",rmat-mat)
# #     print("tra",tval-pose_t)
# #     print("\n")
#
# # print(pose_t.shape)
# # np.savetxt('testpose.csv',poset)
# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.utils.data as Data
#
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# n_data = 7000
# batch_size = 32
# n_epoch = 10
# noise = 0.01
#
# # out_pose = torch.randn([n_data, 7], device=device) * noise
# # out_pose[:, 3:] = F.normalize(out_pose[:, 3:], dim=-1)  # normalize to unit quaternion
#
# # cam_mats = torch.tensor(camera_intrinsic)
# print(out_pose.shape,_x2d.shape,_x3d.shape)
# # dataset = Data.TensorDataset(_x2d,_x3d,out_pose)
# # loader = Data.DataLoader(
# #     dataset=dataset,
# #     batch_size=batch_size,
# #     shuffle=True)
#
# # setup model
# model = Model().to(device)
# mc_loss_fun = MonteCarloPoseLoss().to(device)
#
# optimizer = torch.optim.Adam([
#                 {'params': model.mlp.parameters()},
#                 {'params': model.log_weight_scale, 'lr': 1e-2}
#             ], lr=1e-4)
# #
# # for i,(x2d,x3d,gt_pose) in enumerate(loader):
# #     print(i,(x2d.shape,x3d.shape,gt_pose.shape))
#
# for epoch_id in range(n_epoch):
#     # for iter_id, (x2d,x3d, gt_pose) in enumerate(loader):  # for each training step
#     for i in range(250):
#         # print(w2d)
#         x2d = torch.tensor(_x2d[i*npoints*batch_size:i*npoints*batch_size+npoints*batch_size]).reshape(batch_size,npoints,2)
#         x3d = torch.tensor(_x3d[i * npoints*batch_size:i * npoints*batch_size + npoints*batch_size]).reshape(batch_size,npoints,3)
#         rmat = torch.tensor(_rmat[i*batch_size:i*batch_size+batch_size])
#         tval = torch.tensor(_tval[i*batch_size:i*batch_size+batch_size]).squeeze()
#         rot_quat = torch.tensor([list(matrix_to_quaternion(x)) for x in rmat])
#         gt_pose = torch.cat((tval,rot_quat),dim = -1).double()
#
#
#         batch_cam_mats = torch.tensor(camera_intrinsic).expand(batch_size, -1, -1)
#         _, _, pose_opt_plus, _, pose_sample_logweights, cost_tgt, norm_factor = model.forward_train(
#             x2d,x3d,
#             batch_cam_mats,
#             gt_pose)
#         # print(model.w2d)
#         distCoeffs = np.asarray([0, 0, 0, 0, 0], dtype=np.float64)
#         # _, rrval, ttval = cv2.solvePnP(np.asarray(x3d), np.asarray(x2d), camera_intrinsic, distCoeffs)
#         # t_pose = torch.cat((torch.tensor(ttval), torch.tensor(rrval)))
#         for p in range(len(pose_opt_plus)):
#             print("test:",pose_opt_plus[p]-gt_pose[p])
#             _, rrval, ttval = cv2.solvePnP(np.asarray(x3d[p]), np.asarray(x2d[p]), camera_intrinsic, distCoeffs)
#             rmat_val,_ = cv2.Rodrigues(rrval)
#             # print(rmat_val.shape)
#             rqval = matrix_to_quaternion(torch.tensor(rmat_val)).unsqueeze(1)
#             # print(ttval.shape,rqval.shape)
#             t_pose = torch.cat((torch.tensor(ttval), torch.tensor(rqval))).reshape(1,7)
#             # print(ttval.shape,rrval.shape,t_pose.shape,gt_pose.shape)
#             print("gt:",sum(t_pose-gt_pose[p]))
#         # print(cost_tgt.shape)
#         # # monte carlo pose loss
#         loss_mc = mc_loss_fun(
#             pose_sample_logweights,
#             cost_tgt,
#             norm_factor)
#
#         # derivative regularization
#         dist_t = (pose_opt_plus[:, :3] - gt_pose[:, :3]).norm(dim=-1)
#         beta = 1.0
#         loss_t = torch.where(dist_t < beta, 0.5 * dist_t.square() / beta,
#                              dist_t - 0.5 * beta)
#         loss_t = loss_t.mean()
#
#         dot_quat = (pose_opt_plus[:, None, 3:] @ gt_pose[:, 3:, None]).squeeze(-1).squeeze(-1)
#         loss_r = (1 - dot_quat.square()) * 2
#         loss_r = loss_r.mean()
#
#         loss = 1 * loss_mc + 0.1 * loss_t + 0.1 * loss_r
#         loss.requires_grad_(True)
#         optimizer.zero_grad()
#         loss.backward()
#
#         # grad_norm = []
#         # for p in model.parameters():
#         #     if (p.grad is None) or (not p.requires_grad):
#         #         continue
#         #     else:
#         #         grad_norm.append(torch.norm(p.grad.detach()))
#         # grad_norm = torch.norm(torch.stack(grad_norm))
#
#         optimizer.step()
#
#         print(
#             'Epoch {}: {}/{} - loss_mc={:.4f}, loss_t={:.4f}, loss_r={:.4f}, loss={:.4f}, norm_factor={:.4f}'.format(
#                 epoch_id + 1, i*npoints*batch_size , datasize*batch_size  , loss_mc, loss_t, loss_r, loss, norm_factor))