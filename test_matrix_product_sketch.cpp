#include "matrix_product_sketch.h"
#include <gtest/gtest.h>
#include <vector>
#include <numeric>

TEST(MatrixProductSketchTest, BasicTest)
{
    // Example usage and test
    float dense_matrix[] = {
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f};
    int num_rows = 3;
    int num_cols = 3;
    int num_sketches = 2;

    MatrixProductSketch<float> sketch(dense_matrix, num_rows, num_cols, num_sketches);

    std::vector<float> x = {1.0f, 1.0f, 1.0f};
    std::vector<float> dest(num_rows, 0.0f);
    sketch.approximate_matmul(dest, x);

    // Basic check - you'll want to refine these based on the expected behavior
    ASSERT_EQ(dest.size(), num_rows);
    // Since this is a randomized sketch, the exact result will vary.  We can check for reasonableness
    for (float val : dest)
    {
        ASSERT_GT(val, 0.0f); // Should be positive in this case
    }
}

TEST(MatrixProductSketchTest, ZeroMatrixTest)
{
    float dense_matrix[] = {
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f};
    int num_rows = 3;
    int num_cols = 3;
    int num_sketches = 2;

    MatrixProductSketch<float> sketch(dense_matrix, num_rows, num_cols, num_sketches);

    std::vector<float> x = {1.0f, 1.0f, 1.0f};
    std::vector<float> dest(num_rows, 0.0f);
    sketch.approximate_matmul(dest, x);

    for (float val : dest)
    {
        ASSERT_EQ(val, 0.0f); // Should be zero
    }
}

TEST(MatrixProductSketchTest, SingleValueRow)
{
    float dense_matrix[] = {
        5.0f, 0.0f, 0.0f,
        0.0f, 6.0f, 0.0f,
        0.0f, 0.0f, 7.0f};
    int num_rows = 3;
    int num_cols = 3;
    int num_sketches = 2;

    MatrixProductSketch<float> sketch(dense_matrix, num_rows, num_cols, num_sketches);

    std::vector<float> x = {1.0f, 1.0f, 1.0f};
    std::vector<float> dest(num_rows, 0.0f);
    sketch.approximate_matmul(dest, x);

    ASSERT_EQ(dest[0], 5.0f * 2.0f); // Should be 5 * 2 (two sketches)
    ASSERT_EQ(dest[1], 6.0f * 2.0f); // Should be 6 * 2
    ASSERT_EQ(dest[2], 7.0f * 2.0f); // Should be 7 * 2
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}