import numpy as np

dim_x = 7
dim_z = 4
num_states = 20


class VectorizedKalmanFilter(object):
    """
        Implements a vectorized Kalman Filter in np
    """

    def __init__(self, dim_x, dim_z, num_states):
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.num_states = num_states
        self.x = np.zeros((num_states, dim_x)) # state
        self.P = np.expand_dims(np.eye(dim_x),0).repeat(num_states,0)# uncertainty covariance
        self.Q = np.eye(dim_x)  # process uncertainty
        self.F = np.eye(dim_x)  # state transition matrix
        self.H = np.zeros((dim_z, dim_x))# Measurement function
        self.R = np.eye(dim_z)
        self.z = np.zeros((num_states, dim_z)) # measurement
        self._I = np.eye(dim_x)

    def predict(self):

        self.x = self.x @ self.F.T

        self.P = self.F @ (self.P @ self.F.T) + self.Q


    def update(self, z):

        self.z = z

        y = self.z - self.x.dot(self.H.T)

        PHT = self.P.dot(self.H.T)

        S = np.matmul(self.H, PHT) + self.R

        SI = np.linalg.inv(S)

        K = np.matmul(PHT, SI)
        # TODO make numpy
        Ky = np.matmul(np.expand_dims(y,1),K.transpose(0,2,1) ).squeeze()
        self.x = self.x + Ky

        I_KH = self._I - K@self.H

        self.P =((I_KH @ self.P) @ I_KH.transpose(0,2,1)) + (K @self.R) @ K.transpose(0,2,1)







