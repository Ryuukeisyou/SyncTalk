import os
import numpy as np
from scipy.io import loadmat

cur_dir = os.path.dirname(__file__)
mat_path = os.path.join(cur_dir, "3DMM/01_MorphableModel.mat")
topology_info_path = os.path.join(cur_dir, "3DMM/topology_info.npy")

original_BFM = loadmat(mat_path)
sub_inds = np.load(topology_info_path, allow_pickle=True).item()["sub_inds"]

shapePC = original_BFM["shapePC"]
shapeEV = original_BFM["shapeEV"]
shapeMU = original_BFM["shapeMU"]
texPC = original_BFM["texPC"]
texEV = original_BFM["texEV"]
texMU = original_BFM["texMU"]

b_shape = shapePC.reshape(-1, 199).transpose(1, 0).reshape(199, -1, 3)
mu_shape = shapeMU.reshape(-1, 3)

b_tex = texPC.reshape(-1, 199).transpose(1, 0).reshape(199, -1, 3)
mu_tex = texMU.reshape(-1, 3)

b_shape = b_shape[:, sub_inds, :].reshape(199, -1)
mu_shape = mu_shape[sub_inds, :].reshape(-1)
b_tex = b_tex[:, sub_inds, :].reshape(199, -1)
mu_tex = mu_tex[sub_inds, :].reshape(-1)

exp_info_path = os.path.join(cur_dir, "3DMM/exp_info.npy")
_3dmm_info_path = os.path.join(cur_dir, "3DMM/3DMM_info.npy")

exp_info = np.load(exp_info_path, allow_pickle=True).item()
np.save(
    _3dmm_info_path,
    {
        "mu_shape": mu_shape,
        "b_shape": b_shape,
        "sig_shape": shapeEV.reshape(-1),
        "mu_exp": exp_info["mu_exp"],
        "b_exp": exp_info["base_exp"],
        "sig_exp": exp_info["sig_exp"],
        "mu_tex": mu_tex,
        "b_tex": b_tex,
        "sig_tex": texEV.reshape(-1),
    },
)
