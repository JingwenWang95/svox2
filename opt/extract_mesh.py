import os
import numpy as np
from skimage import measure
import trimesh

data_dir = "../logs/lego/1"
data = np.load(os.path.join(data_dir, 'ckpt.npz'))


def get_scene_mesh(data, threshold=150.):
    indices_flat = data['links'].reshape(-1)
    density = data['density_data'][data['links'].reshape(-1)]
    density[indices_flat < 0] = 0.
    density = density.reshape(data['links'].shape)

    vertices, faces, normals, _ = measure.marching_cubes(density,
                                                         threshold,
                                                         spacing=(2. / data['links'].shape[0],
                                                                  2. / data['links'].shape[1],
                                                                  2. / data['links'].shape[2]),
                                                         allow_degenerate=False)
    vertices = np.array(vertices) + np.array([-1, -1, -1])
    normals = np.array(normals)

    return trimesh.Trimesh(vertices, faces, vertex_normals=normals)


mesh = get_scene_mesh(data)
mesh.export(os.path.join(data_dir, 'mesh.ply'))
mesh.show()