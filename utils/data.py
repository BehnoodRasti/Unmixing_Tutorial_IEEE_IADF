import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio

from . import EPS


class HSI:
    def __init__(
        self,
        name: str,
        data_dir: str = "./Unmixing_Tutorial_IEEE_IADF/datasets/",
    ):
        path = os.path.join(data_dir, name)
        assert os.path.isfile(path)
        self.shortname = name.strip(".mat")

        data = sio.loadmat(path)
        print(f"Matlab data keys: {data.keys()}")

        for key in filter(
            lambda k: not k.startswith("__"),
            data.keys(),
        ):
            self.__setattr__(key, data[key])

        # Data format check
        self.H = self.H.item()
        self.W = self.W.item()
        self.L = self.L.item()
        self.p = self.p.item()

        self.N = self.H * self.W

        assert self.E.shape == (self.L, self.p)
        assert self.Y.shape == (self.L, self.N)

        # Data normalization
        self.Y = (self.Y - self.Y.min()) / (self.Y.max() - self.Y.min())

        try:
            assert len(self.labels) == self.p
            # Curate labels from MATLAB string formatting
            tmp_labels = list(self.labels)
            self.labels = [s.strip(" ") for s in tmp_labels]

        except AssertionError:
            # Create pseudo labels
            self.labels = [f"#{ii}" for ii in range(self.p)]

        assert self.A.shape == (self.p, self.N)
        # Abundance Sum to One Constraint (ASC)
        assert np.allclose(self.A.sum(0), np.ones(self.N))
        # Abundance Non-negative Constraint (ANC)
        assert np.all(self.A >= -EPS)
        # Endmembers Non-negative Constraint
        self.E = np.maximum(self.E, 0)
        assert np.all(self.E >= -EPS)

    def __repr__(self):
        msg = f"HSI => {self.shortname}\n"
        msg += "---------------------\n"
        msg += f"{self.L} bands,\n"
        msg += f"{self.H} lines, {self.W} samples, ({self.N} pixels),\n"
        msg += f"{self.p} endmembers ({self.labels})\n"
        msg += f"GlobalMinValue: {self.Y.min()}, GlobalMaxValue: {self.Y.max()}\n"
        return msg

    def __call__(self):
        Y = np.copy(self.Y)
        E = np.copy(self.E)
        A = np.copy(self.A)
        return (Y, E, A)

    def plot_endmembers(
        self,
        E0=None,
    ):
        """
        Display endmembers spectrum signature
        """
        # Plot attributes
        title = f"{self.shortname}"
        ylabel = "Reflectance"
        xlabel = "# Bands"
        if E0 is None:
            E = np.copy(self.E)
            title += " GT Endmembers\n"
            linestyle = "-"
        else:
            assert self.E.shape == E0.shape
            E = np.copy(E0)
            title += " Estimated Endmembers\n"
            linestyle = "--"
        # Figure
        plt.figure(figsize=(6, 6))
        for pp in range(self.p):
            data = E[:, pp]
            plt.plot(data, label=self.labels[pp], linestyle=linestyle)
        plt.title(title)
        plt.legend(frameon=True)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.show()

    def plot_abundances(
        self,
        A0=None,
        transpose=False,
    ):
        """
        Display abundances maps
        """
        nrows, ncols = (1, self.p)
        title = f"{self.shortname}"
        if A0 is None:
            A = np.copy(self.A)
            title += " GT Abundances"
        else:
            assert self.A.shape == A0.shape
            A = np.copy(A0)
            title += " Estimated Abundances"
        A = A.reshape(self.p, self.H, self.W)

        if transpose:
            A = A.transpose(0, 2, 1)

        fig, ax = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(12, 4 * nrows),
        )
        kk = 0
        for ii in range(nrows):
            for jj in range(ncols):
                if nrows == 1:
                    curr_ax = ax[jj]
                else:
                    curr_ax = ax[ii, jj]
                mappable = curr_ax.imshow(
                    A[kk, :, :],
                    vmin=0.0,
                    vmax=1.0,
                )
                curr_ax.set_title(f"{self.labels[kk]}")
                curr_ax.axis("off")
                fig.colorbar(
                    mappable,
                    ax=curr_ax,
                    shrink=0.5,
                )
                kk += 1

                if kk == self.p:
                    break

        plt.suptitle(title)
        plt.show()
