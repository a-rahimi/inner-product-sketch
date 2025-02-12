from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class Teacher(nn.Linear):
    def __init__(self, num_rows: int, num_cols: int, rank: int, nnz: int = 50):
        U = torch.randn(num_rows, rank)
        V = torch.randn(rank, num_cols)

        sparse = torch.zeros(num_rows, num_cols)
        i = torch.randint(0, num_rows, (nnz,))
        j = torch.randint(0, num_cols, (nnz,))
        sparse[i, j] = 10 * torch.randn(nnz)

        super().__init__(in_features=num_cols, out_features=num_rows, bias=False)
        self.U = nn.Parameter(U, requires_grad=False)
        self.V = nn.Parameter(V, requires_grad=False)
        self.sparse = nn.Parameter(sparse, requires_grad=False)
        self.weight = nn.Parameter(U @ V + sparse, requires_grad=False)

    def shape(self) -> torch.Size:
        return self.weight.shape


class Trainable(nn.Module):
    def penalty(self) -> torch.Tensor:
        return 0

    def prox(self, lr: float):
        pass

    def low_rank_tensor(self):
        raise NotImplementedError()

    def full_tensor(self):
        return self.low_rank_tensor() + self.sparse.weight

    def train(
        self,
        teacher: Teacher,
        num_iterations=40_000,
        batch_size=2,
        learning_rate=0.01,
        ax: Optional[plt.Axes] = None,
    ) -> "Trainable":
        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
        losses = torch.empty(num_iterations, dtype=float)
        for it in range(num_iterations):
            with torch.no_grad():
                x = torch.randn((batch_size, teacher.shape()[1]))
                y = teacher(x)

            optimizer.zero_grad()

            loss = ((self(x) - y) ** 2).mean()

            (loss + self.penalty()).backward()
            optimizer.step()

            with torch.no_grad():
                self.prox(learning_rate / (it + 1))

            losses[it] = loss.item()

        if ax:
            ax.semilogy(losses)
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Loss")
            print("Final loss:", losses[-1000].mean().item())

        return self


class SparseRecovery_UVFactorization(Trainable):
    def __init__(
        self,
        num_rows: int,
        num_cols: int,
        rank: int,
        scale_uv_norm: float = 1.0,
        scale_sparse_norm: float = 1.0,
    ):
        super().__init__()
        self.U = nn.Linear(rank, num_rows, bias=False)
        self.V = nn.Linear(num_cols, rank, bias=False)
        self.sparse = nn.Linear(num_cols, num_rows, bias=False)
        self.scale_uv_norm = scale_uv_norm
        self.scale_sparse_norm = scale_sparse_norm

    def forward(self, x):
        return self.U(self.V(x)) + self.sparse(x)

    def low_rank_tensor(self):
        return self.U.weight @ self.V.weight

    def penalty(self):
        return (
            self.scale_uv_norm * (self.U.weight**2).mean()
            + self.scale_uv_norm * (self.V.weight**2).mean()
            + self.scale_sparse_norm * self.sparse.weight.abs().mean()
        )


class SparseRecovery_NuclearNorm(Trainable):
    def __init__(
        self,
        num_rows: int,
        num_cols: int,
        scale_low_rank_norm: float = 1.0,
        scale_sparse_norm: float = 1.0,
    ):
        super().__init__()
        self.low_rank = nn.Linear(num_cols, num_rows, bias=False)
        self.sparse = nn.Linear(num_cols, num_rows, bias=False)
        self.scale_low_rank_norm = scale_low_rank_norm
        self.scale_sparse_norm = scale_sparse_norm

    def forward(self, x):
        return self.low_rank(x) + self.sparse(x)

    def low_rank_tensor(self):
        return self.low_rank.weight

    def penalty(self):
        return (
            self.scale_low_rank_norm
            * torch.linalg.svdvals(self.low_rank.weight).sqrt().mean()
            + self.scale_sparse_norm * self.sparse.weight.abs().mean()
        )


def prox_l1(x: torch.Tensor, lr: float):
    # Update the tensor x in place with
    #     argmin_y ||y||_1 + 1/2/lr ||y - x||_2^2
    # See section 7.1.1 of
    #   https://web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf
    i_large = x.abs() > lr
    x[i_large] -= lr * x[i_large].sign()
    x[~i_large] = 0


def prox_nuclear_norm(X: torch.Tensor, lr: float):
    # Update the matrix X in place with
    #     argmin_Y ||Y||_* + 1/2/lr ||Y - X||_2^2
    # See section 6.7.3 of
    #   https://web.stanford.edu/~boyd/papers/pdf/prox_algs.pdf
    if X.dim() != 2:
        raise ValueError("X must be a 2D tensor")

    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    prox_l1(S, lr)
    X.copy_(U @ (S[:, None] * Vh))


class SparseRecovery_Proximal(SparseRecovery_NuclearNorm):
    def penalty(self):
        return 0

    def prox(self, lr: float):
        prox_nuclear_norm(self.low_rank.weight, self.scale_low_rank_norm * lr)
        prox_l1(self.sparse.weight, self.scale_sparse_norm * lr)


def plot_spectra(model: Trainable, teacher: Teacher):
    plt.plot(
        torch.linalg.svdvals(teacher.weight).detach(),
        lw=4,
        alpha=0.5,
        color="green",
        label="Spectrum of Teacher",
    )
    plt.plot(
        torch.linalg.svdvals(teacher.U @ teacher.V),
        lw=3,
        alpha=0.5,
        color="red",
        label="Spectrum of teacher low rank",
    )

    plt.plot(
        torch.linalg.svdvals(model.full_tensor().detach()).detach(),
        color="green",
        marker="o",
        label="Spectrum of student",
    )
    plt.plot(
        torch.linalg.svdvals(model.low_rank_tensor().detach()),
        color="red",
        label="Spectrum of student low rank",
    )
    plt.legend(loc="upper right")


def show_matrices(model: Trainable, teacher: Teacher):
    _, axs = plt.subplots(2, 2, figsize=(10, 6))

    axs[0, 0].imshow(teacher.weight.detach())
    axs[0, 0].set_title("Teacher full")

    axs[0, 1].imshow(teacher.sparse.detach())
    axs[0, 1].set_title("Teacher sparse")
    plt.colorbar(axs[0, 1].get_images()[0], ax=axs[0, 1])

    axs[1, 0].imshow(model.full_tensor().detach())
    axs[1, 0].set_title("Student full")

    axs[1, 1].imshow(model.sparse.weight.detach())
    axs[1, 1].set_title("Student sparse ")
    plt.colorbar(axs[1, 1].get_images()[0], ax=axs[1, 1])
