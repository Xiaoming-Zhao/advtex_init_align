import numpy as np
import numpy.linalg


class RedwoodCameraPose:
    # http://redwood-data.org/indoor/fileformat.html
    def __init__(self, meta, mat):
        self.metadata = meta
        self.pose = mat

    def __str__(self):
        return (
            "Metadata : "
            + " ".join(map(str, self.metadata))
            + "\n"
            + "Pose : "
            + "\n"
            + np.array_str(self.pose)
        )


def read_redwood_camera_trajectory(filename):
    traj = []
    with open(filename, "r") as f:
        metastr = f.readline()
        while metastr:
            metadata = map(int, metastr.split())
            mat = np.zeros(shape=(4, 4))
            for i in range(4):
                matstr = f.readline()
                mat[i, :] = np.fromstring(matstr, dtype=float, sep=" \t")
            traj.append(RedwoodCameraPose(metadata, mat))
            metastr = f.readline()
    return traj


def write_redoowd_camera_trajectory(traj, filename):
    with open(filename, "w") as f:
        for x in traj:
            p = x.pose.tolist()
            f.write(" ".join(map(str, x.metadata)) + "\n")
            f.write("\n".join(" ".join(map("{0:.12f}".format, p[i])) for i in range(4)))
            f.write("\n")


def proj_mat_from_K(K, height, width):

    # NOTE: originally, projection matrix maps to NDC with range [0, 1]
    # to align with our CPP implementation, we modify it to make points mapped to NDC with range [-1, 1].
    # Specially, assume original projection matrix is the following:
    # M1 = [[fu, 0, u],
    #       [0, fv, v],
    #       [0, 0,  1]]
    # where fu, fv are focal lengths and (u, v) marks the principal point.
    # Now we change the projection matrix to:
    # M2 = [[2fu, 0,   2u - 1],
    #       [0,   2fv, 2v - 1],
    #       [0,   0,   1]]
    #
    # The validity can be verified as following:
    # a) left end value:
    # assume point p0 = (h0, w0, 1)^T is mapped to (0, 0, 1), namely:
    # M1 * p0 = (0, 0, 1)^T
    # ==> h0 = -u / fu, w0 = -v / fv
    # ==> M2 * p0 = (-1, -1, 1)
    #
    # b) right end value:
    # assume point p1 = (h1, w1, 1)^T is mapped to (1, 1, 1), namely:
    # M1 * p1 = (1, 1, 1)^T
    # ==> h1 = (1 - u) / fu, w0 = (1 - v) / fv
    # ==> M2 * p1 = (1, 1, 1)

    proj_mat = np.eye(4)

    proj_mat[0, 0] = 2 * K[0, 0] / width
    proj_mat[1, 1] = 2 * K[1, 1] / height
    proj_mat[0, 2] = 2 * K[0, 2] / width - 1
    proj_mat[1, 2] = 2 * K[1, 2] / height - 1
    proj_mat[2, 2] = 1
    proj_mat[3, 3] = 0
    proj_mat[3, 2] = 1

    return proj_mat
