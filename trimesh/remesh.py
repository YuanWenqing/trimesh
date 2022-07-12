"""
remesh.py
-------------

Deal with re- triangulation of existing meshes.
"""

import numpy as np

from . import util
from . import grouping
from .geometry import faces_to_edges


def subdivide(vertices,
              faces,
              face_index=None,
              vertex_attributes=None,
              return_index=False):
    """
    Subdivide a mesh into smaller triangles.

    Note that if `face_index` is passed, only those
    faces will be subdivided and their neighbors won't
    be modified making the mesh no longer "watertight."

    Parameters
    ------------
    vertices : (n, 3) float
      Vertices in space
    faces : (m, 3) int
      Indexes of vertices which make up triangular faces
    face_index : faces to subdivide.
      if None: all faces of mesh will be subdivided
      if (n,) int array of indices: only specified faces
    vertex_attributes : dict
      Contains (n, d) attribute data
    return_index : bool
      If True, return index of original face for new faces

    Returns
    ----------
    new_vertices : (q, 3) float
      Vertices in space
    new_faces : (p, 3) int
      Remeshed faces
    index_dict : dict
      Only returned if `return_index`, {index of
      original face : index of new faces}.
    """
    if face_index is None:
        face_index = np.arange(len(faces))
    else:
        face_index = np.asanyarray(face_index)

    # the (c, 3) int array of vertex indices
    faces_subset = faces[face_index]  # (F,3)

    # find max edge of each face
    face_edges = faces_subset[:, [0, 1, 1, 2, 2, 0]].reshape((-1, 3, 2))  # (F,3,2)
    face_edges_length = ((np.diff(vertices[face_edges], axis=2) ** 2).sum(axis=3) ** 0.5).reshape((-1, 3))  # (F,3)
    face_edges_argmax = np.argmax(face_edges_length, axis=1)  # (F,)
    face_max_edge = face_edges[np.arange(len(face_edges_argmax)), face_edges_argmax]  # (F,2)

    # subdivide max_edge
    mid = vertices[face_max_edge].mean(axis=1)
    mid_idx = np.arange(len(mid)) + len(vertices)

    # find another vertex of triangle out of max edge
    vertex_in_edge = np.zeros_like(faces_subset, dtype=bool)
    for i in range(faces_subset.shape[1]):
        vertex_in_edge[:, i] = np.logical_or(faces_subset[:, i] == face_max_edge[:, 0],
                                             faces_subset[:, i] == face_max_edge[:, 1])
    another_vertices = faces_subset[np.logical_not(vertex_in_edge)]

    # the new faces_subset with correct winding
    f = np.column_stack([another_vertices,
                         face_max_edge[:, 0],
                         mid_idx,

                         mid_idx,
                         face_max_edge[:, 1],
                         another_vertices,
                         ]).reshape((-1, 3))  # (2F,3)
    # add new faces_subset per old face
    new_faces = np.vstack((faces, f[1::2]))
    # replace the old face with a smaller face
    new_faces[face_index] = f[0::2]

    new_vertices = np.vstack((vertices, mid))

    if vertex_attributes is not None:
        new_attributes = {}
        for key, values in vertex_attributes.items():
            attr_tris = values[faces_subset]
            attr_mid = attr_tris[face_max_edge].mean(axis=1)
            new_attributes[key] = np.vstack((
                values, attr_mid))
        return new_vertices, new_faces, new_attributes

    if return_index:
        index_dict = {f: [f, len(faces) + fidx]
                      for fidx, f in enumerate(face_index)}
        return new_vertices, new_faces, index_dict

    return new_vertices, new_faces


def subdivide_to_size(vertices,
                      faces,
                      max_edge,
                      max_iter=10,
                      return_index=False):
    """
    Subdivide a mesh until every edge is shorter than a
    specified length.

    Will return a triangle soup, not a nicely structured mesh.

    Parameters
    ------------
    vertices : (n, 3) float
      Vertices in space
    faces : (m, 3) int
      Indices of vertices which make up triangles
    max_edge : float
      Maximum length of any edge in the result
    max_iter : int
      The maximum number of times to run subdivision
    return_index : bool
      If True, return index of original face for new faces

    Returns
    ------------
    vertices : (j, 3) float
      Vertices in space
    faces : (q, 3) int
      Indices of vertices
    index : (q, 3) int
      Only returned if `return_index`, index of
      original face for each new face.
    """
    # store completed
    done_face = []
    done_vert = []
    done_idx = []

    # copy inputs and make sure dtype is correct
    current_faces = np.array(
        faces, dtype=np.int64, copy=True)
    current_vertices = np.array(
        vertices, dtype=np.float64, copy=True)
    current_index = np.arange(len(faces))

    # loop through iteration cap
    for i in range(max_iter + 1):
        # compute the length of every triangle edge
        edge_length = (np.diff(
            current_vertices[current_faces[:, [0, 1, 2, 0]], :3],
            axis=1) ** 2).sum(axis=2) ** 0.5
        # check edge length against maximum
        too_long = (edge_length > max_edge).any(axis=1)
        # faces that are OK
        face_ok = ~too_long

        # clean up the faces a little bit so we don't
        # store a ton of unused vertices
        unique, inverse = grouping.unique_bincount(
            current_faces[face_ok].flatten(),
            return_inverse=True)

        # store vertices and faces meeting criteria
        done_vert.append(current_vertices[unique])
        done_face.append(inverse.reshape((-1, 3)))
        done_idx.append(current_index[face_ok])

        # met our goals so exit
        if not too_long.any():
            break

        current_index = np.tile(current_index[too_long], (2, 1)).T.ravel()
        # run subdivision again
        (current_vertices,
         current_faces) = subdivide(current_vertices,
                                    current_faces[too_long])

    if i >= max_iter:
        # max_iter is too small to generate short-enough edges
        raise ValueError('edges too long at exit! try increasing max_iter')

    # stack sequence into nice (n, 3) arrays
    final_vertices, final_faces = util.append_faces(
        done_vert, done_face)

    if return_index:
        final_index = np.concatenate(done_idx)
        assert len(final_index) == len(final_faces)
        return final_vertices, final_faces, final_index

    return final_vertices, final_faces
