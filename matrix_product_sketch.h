#include <cassert>
#include <cmath>
#include <concepts>
#include <random>
#include <type_traits>
#include <vector>

// TODO: replace T with FloatType and use a concept to enforce it.

template <std::floating_point FloatType>
struct MatrixProductSketch
{
  size_t num_rows;
  size_t num_cols;
  size_t num_sketches;

  struct Sample
  {
    size_t row, col;
  };

  std::vector<FloatType> row_scales;
  std::vector<Sample> sampled_indices_to_add;
  std::vector<Sample> sampled_indices_to_subtract;

  MatrixProductSketch(const FloatType *dense_matrix, size_t num_rows, size_t num_cols,
                      size_t num_sketches)
      : num_rows(num_rows), num_cols(num_cols), num_sketches(num_sketches), row_scales(num_rows)
  {
    // TODO: Take the random number generated as an argument.
    std::random_device rd;
    std::mt19937 gen(rd());

    for (size_t row = 0; row < num_rows; row++)
    {
      const FloatType *matrix_row = &dense_matrix[row * num_cols];

      std::vector<FloatType> scaled_probabilities(num_cols);
      for (size_t col = 0; col < num_cols; col++)
      {
        auto abs = std::fabs(matrix_row[col]);
        scaled_probabilities[col] = abs;
        row_scales[row] += abs;
      }
      row_scales[row] /= num_sketches;

      std::discrete_distribution<std::size_t> column_sampler(
          scaled_probabilities.begin(), scaled_probabilities.end());

      for (size_t sketch = 0; sketch < num_sketches; sketch++)
      {
        std::size_t col = column_sampler(gen);
        (matrix_row[col] < 0 ? sampled_indices_to_add : sampled_indices_to_subtract).push_back(Sample{row, col});
      }
    }

    // TODO Reorder the sample by column to reduce cache misses later.
  }

  void approximate_matmul(std::vector<FloatType> &dest, const std::vector<FloatType> &vec) const
  {
    assert(dest.size() == num_rows);
    assert(vec.size() == num_cols);

    for (const Sample &sample : sampled_indices_to_add)
      dest[sample.row] += vec[sample.col];

    for (const Sample &sample : sampled_indices_to_subtract)
      dest[sample.row] -= vec[sample.col];

    for (size_t row = 0; row < num_rows; row++)
      dest[row] *= row_scales[row];
  }
};