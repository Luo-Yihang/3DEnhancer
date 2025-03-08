import torch
from kornia.core import Tensor, concatenate

import torch
import math
import numpy as np
from torch import nn
from kiui.cam import orbit_camera


# gaussian splatting utils.graphics_utils
def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))


# gaussian splatting scene.camera
class Camera(nn.Module):
    def __init__(self, R, T, FoVx, FoVy,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0
                 ):
        super(Camera, self).__init__()

        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1)
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1)
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]


# gaussian splatting utils.camera_utils
def loadCam(c2w, fovx, image_height=512, image_width=512):
    # load_camera
    w2c = np.linalg.inv(c2w)

    R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
    T = w2c[:3, 3]

    fovy = focal2fov(fov2focal(fovx, image_width), image_height)
    FovY = fovy
    FovX = fovx

    return Camera(R=R, T=T, 
                  FoVx=FovX, FoVy=FovY)


# epipolar calculation related
@torch.no_grad()
def fundamental_from_projections(P1: Tensor, P2: Tensor) -> Tensor:
    r"""Get the Fundamental matrix from Projection matrices.

    Args:
        P1: The projection matrix from first camera with shape :math:`(*, 3, 4)`.
        P2: The projection matrix from second camera with shape :math:`(*, 3, 4)`.

    Returns:
         The fundamental matrix with shape :math:`(*, 3, 3)`.
    """
    if not (len(P1.shape) >= 2 and P1.shape[-2:] == (3, 4)):
        raise AssertionError(P1.shape)
    if not (len(P2.shape) >= 2 and P2.shape[-2:] == (3, 4)):
        raise AssertionError(P2.shape)
    if P1.shape[:-2] != P2.shape[:-2]:
        raise AssertionError

    def vstack(x: Tensor, y: Tensor) -> Tensor:
        return concatenate([x, y], dim=-2)

    X1 = P1[..., 1:, :]
    X2 = vstack(P1[..., 2:3, :], P1[..., 0:1, :])
    X3 = P1[..., :2, :]

    Y1 = P2[..., 1:, :]
    Y2 = vstack(P2[..., 2:3, :], P2[..., 0:1, :])
    Y3 = P2[..., :2, :]

    X1Y1, X2Y1, X3Y1 = vstack(X1, Y1), vstack(X2, Y1), vstack(X3, Y1)
    X1Y2, X2Y2, X3Y2 = vstack(X1, Y2), vstack(X2, Y2), vstack(X3, Y2)
    X1Y3, X2Y3, X3Y3 = vstack(X1, Y3), vstack(X2, Y3), vstack(X3, Y3)

    F_vec = torch.cat(
        [
            X1Y1.det().reshape(-1, 1),
            X2Y1.det().reshape(-1, 1),
            X3Y1.det().reshape(-1, 1),
            X1Y2.det().reshape(-1, 1),
            X2Y2.det().reshape(-1, 1),
            X3Y2.det().reshape(-1, 1),
            X1Y3.det().reshape(-1, 1),
            X2Y3.det().reshape(-1, 1),
            X3Y3.det().reshape(-1, 1),
        ],
        dim=1,
    )

    return F_vec.view(*P1.shape[:-2], 3, 3)


def get_fundamental_matrix_with_H(cam1, cam2, current_H, current_W):
    NDC_2_pixel = torch.tensor([[current_W / 2, 0, current_W / 2], [0, current_H / 2, current_H / 2], [0, 0, 1]])
    # NDC_2_pixel_inversed = torch.inverse(NDC_2_pixel)
    NDC_2_pixel = NDC_2_pixel.float()
    cam_1_tranformation = cam1.full_proj_transform[:, [0,1,3]].T.float()
    cam_2_tranformation = cam2.full_proj_transform[:, [0,1,3]].T.float()
    cam_1_pixel = NDC_2_pixel@cam_1_tranformation
    cam_2_pixel = NDC_2_pixel@cam_2_tranformation

    # print(NDC_2_pixel.dtype, cam_1_tranformation.dtype, cam_2_tranformation.dtype, cam_1_pixel.dtype, cam_2_pixel.dtype)

    cam_1_pixel = cam_1_pixel.float()
    cam_2_pixel = cam_2_pixel.float()
    # print("cam_1", cam_1_pixel.dtype, cam_1_pixel.shape)
    # print("cam_2", cam_2_pixel.dtype, cam_2_pixel.shape)
    # print(NDC_2_pixel@cam_1_tranformation, NDC_2_pixel@cam_2_tranformation)
    return fundamental_from_projections(cam_1_pixel, cam_2_pixel)


def point_to_line_dist(points, lines):
    """
    Calculate the distance from points to lines in 2D.
    points: Nx3
    lines: Mx3

    return distance: NxM
    """
    numerator = torch.abs(lines @ points.T)
    denominator = torch.linalg.norm(lines[:,:2], dim=1, keepdim=True)
    return numerator / denominator


def compute_epipolar_constrains(cam1, cam2, current_H=64, current_W=64):
    n_frames = 1
    # sequence_length = current_W * current_H
    fundamental_matrix_1 = []
    
    fundamental_matrix_1.append(get_fundamental_matrix_with_H(cam1, cam2, current_H, current_W))
    fundamental_matrix_1 = torch.stack(fundamental_matrix_1, dim=0)

    x = torch.arange(current_W)
    y = torch.arange(current_H)
    x, y = torch.meshgrid(x, y, indexing='xy')
    x = x.reshape(-1)
    y = y.reshape(-1)
    heto_cam2 = torch.stack([x, y, torch.ones(size=(len(x),))], dim=1).view(-1, 3)
    heto_cam1 = torch.stack([x, y, torch.ones(size=(len(x),))], dim=1).view(-1, 3)
    # epipolar_line: n_frames X seq_len,  3
    line1 = (heto_cam2.unsqueeze(0).repeat(n_frames, 1, 1) @ fundamental_matrix_1).view(-1, 3)
    
    distance1 = point_to_line_dist(heto_cam1, line1)

    idx1_epipolar = distance1 > 1 # sequence_length x sequence_lengths

    return idx1_epipolar


def compute_camera_distance(cams, key_cams):
    cam_centers = [cam.camera_center for cam in cams]
    key_cam_centers = [cam.camera_center for cam in key_cams] 
    cam_centers = torch.stack(cam_centers)
    key_cam_centers = torch.stack(key_cam_centers)
    cam_distance = torch.cdist(cam_centers, key_cam_centers)

    return cam_distance


def get_intri(target_im=None, h=None, w=None, normalize=False):
    if target_im is None:
        assert (h is not None and w is not None)
    else:
        h, w = target_im.shape[:2]

    fx = fy = 1422.222
    res_raw = 1024
    f_x = f_y = fx * h / res_raw
    K = np.array([f_x, 0, w / 2, 0, f_y, h / 2, 0, 0, 1]).reshape(3, 3)
    if normalize:  # center is [0.5, 0.5], eg3d renderer tradition
        K[:2] /= h
    return K


def normalize_camera(c, c_frame0):
    B = c.shape[0]
    camera_poses = c[:, :16].reshape(B, 4, 4)  # 3x4
    canonical_camera_poses = c_frame0[:, :16].reshape(1, 4, 4)
    inverse_canonical_pose = np.linalg.inv(canonical_camera_poses)
    inverse_canonical_pose = np.repeat(inverse_canonical_pose, B, 0)

    cam_radius = np.linalg.norm(
        c_frame0[:, :16].reshape(1, 4, 4)[:, :3, 3],
        axis=-1,
        keepdims=False)  # since g-buffer adopts dynamic radius here.

    frame1_fixed_pos = np.repeat(np.eye(4)[None], 1, axis=0)
    frame1_fixed_pos[:, 2, -1] = -cam_radius

    transform = frame1_fixed_pos @ inverse_canonical_pose

    new_camera_poses = np.repeat(
        transform, 1, axis=0
    ) @ camera_poses  # [v, 4, 4]. np.repeat() is th.repeat_interleave()

    c = np.concatenate([new_camera_poses.reshape(B, 16), c[:, 16:]],
                        axis=-1)

    return c


def gen_rays(c2w, intrinsics, h, w):
    # Generate rays
    yy, xx = torch.meshgrid(
        torch.arange(h, dtype=torch.float32) + 0.5,
        torch.arange(w, dtype=torch.float32) + 0.5,
        indexing='ij')

    # normalize to 0-1 pixel range
    yy = yy / h
    xx = xx / w

    cx, cy, fx, fy = intrinsics[2], intrinsics[
        5], intrinsics[0], intrinsics[4]

    xx = (xx - cx) / fx
    yy = (yy - cy) / fy
    zz = torch.ones_like(xx)
    dirs = torch.stack((xx, yy, zz), dim=-1)  # OpenCV convention
    dirs /= torch.norm(dirs, dim=-1, keepdim=True)
    dirs = dirs.reshape(-1, 3, 1)
    del xx, yy, zz

    dirs = (c2w[None, :3, :3] @ dirs)[..., 0]

    origins = c2w[None, :3, 3].expand(h * w, -1).contiguous()
    origins = origins.view(h, w, 3)
    dirs = dirs.view(h, w, 3)

    return origins, dirs


def get_c2ws(elevations, amuziths, camera_radius=1.5):
    c2ws = np.stack([
        orbit_camera(elevation, amuzith, radius=camera_radius) for elevation, amuzith in zip(elevations, amuziths)
    ], axis=0)

    # change kiui opengl camera system to our camera system
    c2ws[:, :3, 1:3] *= -1
    c2ws[:, [0, 1, 2], :] = c2ws[:, [2, 0, 1], :]
    c2ws = c2ws.reshape(-1, 16)

    return c2ws


def get_camera_poses(c2ws, fov, h, w, intrinsics=None):
    if intrinsics is None:
        intrinsics = get_intri(h=64, w=64, normalize=True).reshape(9)

    c2ws = normalize_camera(c2ws, c2ws[0:1])

    rays_pluckers = []
    c2ws = c2ws.reshape((-1, 4, 4))
    c2ws = torch.from_numpy(c2ws).float()

    gs_cams = []
    for i, c2w in enumerate(c2ws):
        gs_cams.append(loadCam(c2w.numpy(), fov, h, w))
        rays_o, rays_d = gen_rays(c2w, intrinsics, h, w)
        rays_plucker = torch.cat([torch.cross(rays_o, rays_d, dim=-1), rays_d],
                                    dim=-1)  # [h, w, 6]
        rays_pluckers.append(rays_plucker.permute(2, 0, 1)) # [6, h, w]

    n_views = len(gs_cams)
    epipolar_constrains = []
    cam_distances = []
    for i in range(n_views):
        cur_epipolar_constrains = []
        kv_idxs = [(i-1)%n_views, (i+1)%n_views]
        for kv_idx in kv_idxs:
            # False means that the position is on the epipolar line
            cam_epipolar_constrain = compute_epipolar_constrains(gs_cams[kv_idx], gs_cams[i], current_H=h//16, current_W=w//16)
            cur_epipolar_constrains.append(cam_epipolar_constrain)

        cam_distances.append(compute_camera_distance([gs_cams[i]], [gs_cams[kv_idxs[0]], gs_cams[kv_idxs[1]]])) # 1, 2
        epipolar_constrains.append(torch.stack(cur_epipolar_constrains, dim=0))

    rays_pluckers = torch.stack(rays_pluckers) # [v, 6, h, w]
    cam_distances = torch.cat(cam_distances, dim=0) # [v, 2]
    epipolar_constrains = torch.stack(epipolar_constrains, dim=0) # [v, 2, 1024, 1024]

    return rays_pluckers, epipolar_constrains, cam_distances