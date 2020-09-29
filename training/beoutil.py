import hdf5storage
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import torch
from skimage import measure

BASIS_FILE_PATH = 'data/knives/knife_basis.mat'
VOXEL_OBJECT_FILE_PATH = 'data/knives/knives_voxel/knife_3.mat'

# PyTorch utils
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def voxel_object_to_vector(voxel_object):
    return np.squeeze(np.reshape(voxel_object, (voxel_object.size, 1, 1),  order="F"))


def vector_to_voxel_object(object_vector, shape):
    return np.reshape(object_vector, shape, order="F")

def visualize_reconstruction(reconstructed_object, cuttoff_value=0.4):
    # threshold at cutoff_value
    reconstructed_object_thresholded = np.copy(reconstructed_object)
    reconstructed_object_thresholded[reconstructed_object_thresholded < cuttoff_value] = 0
    reconstructed_object_thresholded[reconstructed_object_thresholded > 0] = 1

    fig = plt.figure()

    ax = fig.add_subplot(1, 1, 1, projection='3d')
    ax.voxels(reconstructed_object_thresholded, edgecolor='k', facecolors='b')
    ax.grid(False)
    ax.view_init(elev=0, azim=45)
    #plt.axis('off')
    #fig.axes.get_xaxis().set_visible(False)
    #fig.axes.get_yaxis().set_visible(False)

    #plt.savefig('pict.png', bbox_inches='tight', pad_inches = 0)
    return plt

def visualize_reconstruction_marching_cubes(reconstructed_object, cuttoff_value=0.4):
    # threshold at cutoff_value
    reconstructed_object_thresholded = np.copy(reconstructed_object)
    reconstructed_object_thresholded[reconstructed_object_thresholded < cuttoff_value] = 0
    reconstructed_object_thresholded[reconstructed_object_thresholded > 0] = 1

    verts, faces, normals, values = measure.marching_cubes_lewiner(reconstructed_object_thresholded, 0, spacing=(0.1, 0.1, 0.1))

    fig = plt.figure()

    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(verts[:, 0], verts[:,1], faces, verts[:, 2],
                cmap='Spectral', lw=1)
    return plt

def visualize_object_and_reconstruction(input_object, reconstructed_object, cuttoff_value=0.4):
    # threshold at cutoff_value
    reconstructed_object_thresholded = np.copy(reconstructed_object)
    reconstructed_object_thresholded[reconstructed_object_thresholded < cuttoff_value] = 0
    reconstructed_object_thresholded[reconstructed_object_thresholded > 0] = 1

    fig = plt.figure()

    ax = fig.add_subplot(1, 2, 1, projection='3d')
    ax.voxels(input_object, edgecolor='k', facecolors='b')
    ax.grid(False)
    ax.view_init(elev=0, azim=45)
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.voxels(reconstructed_object_thresholded, edgecolor='k', facecolors='r')
    ax.grid(False)
    ax.view_init(elev=0, azim=45)

    return plt
def project_object_with_mat(mat, voxel_file):
    # load voxelized object into memory
    voxel_model_contents = hdf5storage.loadmat(voxel_file)
    voxel_object = voxel_model_contents['instance']

    # stack voxel object into vector
    vector_object = Tensor(voxel_object_to_vector(voxel_object).astype(np.float))

    # project voxel object onto basis (subspace)
    projected_object = torch.matmul(vector_object, mat)
    return projected_object.cpu().numpy()

def project_object(basis_file, voxel_file):
    contents = hdf5storage.loadmat(basis_file)
    basis = Tensor(contents['basis_300'])

    # load voxelized object into memory
    voxel_model_contents = hdf5storage.loadmat(voxel_file)
    voxel_object = voxel_model_contents['instance']

    # stack voxel object into vector
    vector_object = Tensor(voxel_object_to_vector(voxel_object).astype(np.float))

    # project voxel object onto basis (subspace)
    projected_object = torch.matmul(vector_object, basis)
    return projected_object.cpu().numpy()

def reconstruct_vector_with_mat(mat, vector):
    # back-project point in subspace into voxel space
    reconstructed_object_vector = torch.matmul(vector, mat)

    # reshape into voxel object
    reconstructed_object = vector_to_voxel_object(reconstructed_object_vector.cpu().numpy(), (64, 64, 64))
    return reconstructed_object

def reconstruct_vector(basis_file, vector):
    contents = hdf5storage.loadmat(basis_file)
    basis = contents['basis_300']

    # back-project point in subspace into voxel space
    reconstructed_object_vector = np.dot(vector, np.transpose(basis))

    # reshape into voxel object
    reconstructed_object = vector_to_voxel_object(reconstructed_object_vector, (64, 64, 64))
    return reconstructed_object

def reconstruct_object(basis_file, voxel_file):
    contents = hdf5storage.loadmat(basis_file)

    basis = contents['basis_300']

    # load voxelized object into memory
    voxel_model_contents = hdf5storage.loadmat(voxel_file)
    voxel_object = voxel_model_contents['instance']

    # stack voxel object into vector
    vector_object = voxel_object_to_vector(voxel_object)

    # project voxel object onto basis (subspace)
    projected_object = np.dot(vector_object, basis).astype(np.float)

    # back-project point in subspace into voxel space
    reconstructed_object_vector = np.dot(projected_object, np.transpose(basis))

    # reshape into voxel object
    reconstructed_object = vector_to_voxel_object(reconstructed_object_vector, voxel_object.shape)
    return reconstructed_object

def reconstruct_object_with_mat(mat, object):
    vector_object = Tensor(voxel_object_to_vector(object).astype(np.float))

    # project voxel object onto basis (subspace)
    reconstructed_object = torch.matmul(torch.matmul(vector_object, mat), mat.t())
    return vector_to_voxel_object(reconstructed_object.cpu().numpy(), (64, 64, 64))
