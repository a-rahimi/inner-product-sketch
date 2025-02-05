import torch


class MatrixProductSketch:
    def __init__(self, dense_matrix: torch.Tensor, num_sketches: int):
        if dense_matrix.dim() != 2:
            raise ValueError("Dense matrix must be a 2D tensor")

        if False and not torch.allclose(
            dense_matrix.sum(axis=1), torch.tensor(0.0, dtype=dense_matrix.dtype)
        ):
            # The algorithm works better when the norm of the entries of each row
            # has very high entropy. One easy way to do that is to subtract the mean
            # from each row. This mean can be added back later after the dot
            # product is computed.
            #
            # Concretely, suppose M is the dense matrix, and let m be a vector
            # whose entries are the means of the rows of M. Let M~ = M - m 1' be
            # the matrix where we've subtracted m[r] from row r of M (1 is the
            # column vector of all 1's). Then  for any vector v,
            #
            #    M v = (M~ + m 1') v = M~ v + (m 1'v)
            #
            # So we compute M~ v quickly using this class, then add the vector m
            # scaled by the scalar (1'v) to the result to recover M v.
            raise Warning(
                "The entries in each row of the dense matrix must sum to zero"
            )

        self.num_rows, self.num_columns = dense_matrix.shape
        self.num_sketches = num_sketches

        self.row_scales = torch.empty(self.num_rows, dtype=dense_matrix.dtype)

        sampled_rows = torch.empty((self.num_rows, num_sketches), dtype=torch.int64)
        sampled_columns = torch.empty_like(sampled_rows)
        sample_is_positive = torch.empty_like(sampled_rows, dtype=torch.bool)

        for irow, matrix_row in enumerate(dense_matrix):
            columns = torch.multinomial(
                matrix_row.abs(), num_sketches, replacement=False
            )

            sampled_rows[irow] = irow
            sampled_columns[irow] = columns
            sample_is_positive[irow] = matrix_row[columns] > 0

            self.row_scales[irow] = matrix_row.abs().sum() / num_sketches

        self.sampled_rows_positive = sampled_rows[sample_is_positive]
        self.sampled_columns_positive = sampled_columns[sample_is_positive]
        self.sampled_rows_negative = sampled_rows[~sample_is_positive]
        self.sampled_columns_negative = sampled_columns[~sample_is_positive]

        # TODO: sort the sampled arrays by column to reduce cache misses later.

    def matrix_vector_product(self, dest: torch.Tensor, vector: torch.Tensor) -> None:
        if vector.shape != (self.num_columns,):
            raise ValueError(f"Vector must be a 1D tensor of size {self.num_columns}")

        if dest.shape != (self.num_rows,):
            raise ValueError(f"Destination must be a 1D tensor of size {self.num_rows}")

        dest.scatter_add_(
            0, self.sampled_rows_positive, vector[self.sampled_columns_positive]
        )
        dest.scatter_add_(
            0, self.sampled_rows_negative, -vector[self.sampled_columns_negative]
        )

        dest *= self.row_scales

    def __matmul__(self, other: torch.Tensor) -> torch.Tensor:
        dest = torch.zeros(self.num_rows, dtype=other.dtype)
        self.matrix_vector_product(dest, other)
        return dest
