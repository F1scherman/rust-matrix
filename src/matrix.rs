/// Brayden Jonsson, 2023
/// https://github.com/BraydenJonsson/rust-matrix
///
/// Contains a struct and methods for representing a mathematical matrix
use num_traits;
use gen_ops::gen_ops;
use std::cmp;
use std::ops;
use trait_set::trait_set;

trait_set! {
    pub trait MatrixCompatible = num_traits::NumAssign
    + num_traits::sign::Signed
    + Copy;
}

/// Represents a mathematical matrix, zero-indexed
#[derive(Debug)]
pub struct Matrix<T>
    where
        T: MatrixCompatible,
{
    matrix: Vec<Vec<T>>,
    rows: usize,
    columns: usize,
}

impl<T> Matrix<T>
    where
        T: MatrixCompatible,
{
    // -----CONSTRUCTORS-----

    /// Creates a new zero matrix with the given size parameters
    pub fn new(rows: usize, columns: usize) -> Self {
        let matrix: Vec<Vec<T>> = vec![vec![T::zero(); columns]; rows];

        Self {
            matrix,
            rows,
            columns,
        }
    }

    /// Creates a new square zero matrix with the given size parameters
    pub fn square_matrix(size: usize) -> Self {
        Self::new(size, size)
    }

    /// Creates a new matrix from the given 2D vector array. The array must have consistent rectangular sizing
    pub fn from_vector(vector: Vec<Vec<T>>) -> Self {
        let rows: usize = vector.capacity();
        let columns: usize = vector[0].capacity();

        for row in vector {
            if columns != row.capacity() {
                panic!("This matrix doesn't have equal column sizes!")
            }
        }

        let matrix: Vec<Vec<T>> = vector.clone();

        Self {
            matrix,
            rows,
            columns,
        }
    }

    /// Creates a new identity matrix with the given size
    pub fn identity_matrix(size: usize) -> Self {
        let mut matrix: Self = Self::square_matrix(size);

        for i in 0..matrix.rows {
            matrix.set_value(i, i, T::one());
        }

        matrix
    }

    /// Constructs a new square matrix from the given list of numbers, listed left-to-right, up-to-down.
    /// The length of the list must be a perfect square.
    pub fn square_matrix_from_list(list_of_numbers: &[T]) -> Self {
        let list_length: f64 = list_of_numbers.len() as f64;
        if f64::sqrt(list_length).fract() != 0.0 {
            panic!("This list size is not a perfect square!");
        }

        let matrix_size: usize = f64::sqrt(list_length) as usize;
        Self::matrix_from_list(list_of_numbers, matrix_size, matrix_size)
    }

    /// Constructs a new matrix from the given list of numbers, listed left-to-right, up-to-down.
    /// The length of the list must be match the dimensions
    pub fn matrix_from_list(list_of_numbers: &[T], rows: usize, columns: usize) -> Self {
        if list_of_numbers.len() != rows * columns {
            panic!("This list size does not match the dimensions!");
        }

        let mut matrix: Self = Self::new(rows, columns);
        let mut list_index: usize = 0;

        for row_index in 0..rows {
            for column_index in 0..columns {
                matrix.set_value(row_index, column_index, list_of_numbers[list_index]);
                list_index += 1;
            }
        }

        matrix
    }

    // -----PRIVATE HELPERS-----

    /// Calculates the inner product of two input Vec<T> objects
    fn inner_product(a: &Vec<T>, b: &Vec<T>) -> T {
        if a.len() != b.len() {
            panic!("These vectors are of different sizes!");
        }

        let mut output: T = T::zero();

        for i in 0..a.len() {
            output += a[i] * b[i];
        }

        output
    }

    /// Partitions the matrix such that a new matrix is created where the rows/columns of the new matrix are defined by being within the parameters bounds (ending is exclusive)
    ///
    /// ie. Partitioning a matrix "example_matrix" with parameters "example_matrix.partition(0, example_matrix.rows, 0, example_matrix.columns)" will return a matrix equivalent to example_matrix.
    fn partition(
        &self,
        starting_row: usize,
        ending_row: usize,
        starting_column: usize,
        ending_column: usize,
    ) -> Self {
        let mut new_matrix: Self =
            Self::new(ending_row - starting_row, ending_column - starting_column);

        for row in starting_row..ending_row {
            for column in starting_column..ending_column {
                new_matrix.set_value(
                    row - starting_row,
                    column - starting_column,
                    self[row][column],
                );
            }
        }

        new_matrix
    }

    /// Combines the self matrix and the input matrix such that both are side-by-side, with the input matrix (rhs) on the right.
    fn combine(&self, rhs: &Self) -> Self {
        if self.rows != rhs.rows {
            panic!("These two matrices must have the same number of rows!");
        }
        let mut new_matrix: Self = Self::new(self.rows, self.columns + rhs.columns);

        for row in 0..self.rows {
            for column in 0..self.columns {
                new_matrix.set_value(row, column, self[row][column]);
            }
            for column in 0..rhs.columns {
                new_matrix.set_value(row, column + self.columns, rhs[row][column]);
            }
        }

        new_matrix
    }

    /// Returns the x input vector of a solved matrix
    fn get_x_vector(solved_matrix: Matrix<T>) -> Vec<T> {
        let last_column_index: usize = solved_matrix.columns - 1;
        let zero: T = T::zero();
        let one: T = T::one();

        let mut x_vector: Vec<T> = Vec::with_capacity(last_column_index);
        let mut current_row_index: usize = 0;

        for column_index in 0..last_column_index {
            if solved_matrix[current_row_index][column_index] == one {
                x_vector.push(solved_matrix[current_row_index][last_column_index]);
                current_row_index += 1;
            } else {
                x_vector.push(zero);
            }
        }

        x_vector
    }

    // -----PUBLIC METHODS-----

    /// Gets the value of the matrix at the given indices (0 indexed). Functionally equivalent to Matrix\[row\]\[column\]
    pub fn get_value(&self, row: usize, column: usize) -> T {
        self.matrix[row][column]
    }

    /// Sets the value of the matrix at the given indices (0 indexed)
    pub fn set_value(&mut self, row: usize, column: usize, value: T) {
        self.matrix[row][column] = value;
    }

    /// Calculates the reduced echelon form and determinant of this matrix (determinant is an error if the matrix is non-square)
    pub fn reduced_echelon_and_det(&self) -> (Self, Result<T, &'static str>) {
        let mut operating_matrix: Vec<Vec<T>> = self.clone().matrix;

        let mut current_pivot_row: usize = 0;
        let mut current_pivot_column: usize = 0;
        let mut factor: T;
        let mut determinant: T = T::one();

        let negative_one: T = T::one().neg();
        let zero: T = T::zero();

        #[allow(clippy::mut_range_bound)]
        while self.rows - current_pivot_row > 0 && self.columns - current_pivot_column > 0 {
            let mut changed: bool = false;

            // Find the next pivot
            for column in current_pivot_column..self.columns {
                for row in current_pivot_row..self.rows {
                    if operating_matrix[row][column] != zero {
                        // Row swap if necessary
                        if current_pivot_row != row {
                            operating_matrix.swap(row, current_pivot_row);
                            determinant *= negative_one;
                        }
                        // Update the column
                        current_pivot_column = column;
                        changed = true;
                        break;
                    }
                }
                if changed {
                    break;
                }
            }

            // If the pivot didn't change, then we have exhausted all pivots
            if !changed {
                break;
            }

            // Set the pivot to 1
            factor = operating_matrix[current_pivot_row][current_pivot_column];
            for column in current_pivot_column..self.columns {
                operating_matrix[current_pivot_row][column] /= factor;
            }
            determinant *= factor;

            // Reduce down all rows above and underneath
            for row in 0..self.rows {
                if operating_matrix[row][current_pivot_column] == zero || row == current_pivot_row {
                    continue;
                }
                factor = operating_matrix[row][current_pivot_column];
                for column in current_pivot_column..self.columns {
                    let subtraction_factor: T =
                        operating_matrix[current_pivot_row][column] * factor;
                    operating_matrix[row][column] -= subtraction_factor;
                }
            }

            // Force the pivot to update
            current_pivot_row += 1;
            current_pivot_column += 1;
        }

        let det_output: Result<T, &'static str>;
        // Checks if this matrix is square and has so has a determinant, then checks that this matrix is equal to In
        if self.rows != self.columns {
            det_output = Err("The matrix was not square");
        } else {
            for (i, row) in operating_matrix.iter().enumerate() {
                if row[i] == zero {
                    determinant = zero;
                    break;
                }
            }
            det_output = Ok(determinant);
        }

        (Self::from_vector(operating_matrix), det_output)
    }

    /// Calculates and returns the reduced echelon form of this matrix
    pub fn reduced_echelon_form(&self) -> Self {
        self.reduced_echelon_and_det().0
    }

    /// Calculates and returns the determinant if this matrix is square
    pub fn determinant(&self) -> T {
        if self.rows != self.columns {
            panic!("This matrix is not square!");
        }
        self.reduced_echelon_and_det().1.unwrap()
    }

    /// Calculates and returns the inverse of this matrix, if this matrix is invertible
    pub fn inverse(&self) -> Result<Self, &'static str> {
        if self.rows != self.columns {
            panic!("This matrix is not square!");
        }

        let identity_matrix: Self = Self::identity_matrix(self.rows);

        let reduced_matrix: Self = self.combine(&identity_matrix).reduced_echelon_form();

        if reduced_matrix.partition(0, self.rows, 0, self.columns) != identity_matrix {
            return Err("Matrix is not invertible");
        }

        let inverse_matrix: Self =
            reduced_matrix.partition(0, self.rows, self.columns, reduced_matrix.columns);
        Ok(inverse_matrix)
    }

    /// Returns a transpose of this matrix
    pub fn transpose(&self) -> Self {
        let mut transpose_matrix: Self = Self::new(self.columns, self.rows);

        for row in 0..self.rows {
            for column in 0..self.columns {
                transpose_matrix.set_value(column, row, self[row][column]);
            }
        }

        transpose_matrix
    }

    /// Returns a least squares solution of Ax = b. Uses the ATAx = ATb method.
    pub fn least_squares_solution(&self, b: &[T]) -> Result<Vec<T>, &'static str> {
        if b.len() != self.rows {
            panic!("Your b vector is not the correct length!");
        }

        let b_matrix: Self = Self::matrix_from_list(&b, b.len(), 1);

        let a_transpose_a_matrix: Self = self.transpose() * self.clone();
        let a_transpose_b_matrix: Self = self.transpose() * b_matrix;

        let solved_matrix: Self = a_transpose_a_matrix
            .combine(&a_transpose_b_matrix)
            .reduced_echelon_form();

        let last_column_index: usize = solved_matrix.columns - 1;
        let zero: T = T::zero();
        for row_index in 0..solved_matrix.rows {
            if solved_matrix[row_index][last_column_index] == zero {
                continue;
            }

            let mut check_passed: bool = false;
            for column_index in 0..last_column_index {
                if solved_matrix[row_index][column_index] != zero {
                    check_passed = true;
                    break;
                }
            }

            if !check_passed {
                return Err("The system was inconsistent and there is no solution for b. (In this case, these means an arithmetic problem, probably due to floating point inaccuracy).");
            }
        }

        Ok(Self::get_x_vector(solved_matrix))
    }

    /// Returns a solution to the given Ax = b equation, or an error if a solution does not exist
    pub fn solve(&self, b: &[T]) -> Result<Vec<T>, &'static str> {
        if b.len() != self.rows {
            panic!("Your b vector is not the correct length!");
        }

        let b_matrix: Self = Self::matrix_from_list(&b, b.len(), 1);

        let solved_matrix: Self = self.combine(&b_matrix).reduced_echelon_form();

        let last_column_index: usize = solved_matrix.columns - 1;
        let zero: T = T::zero();
        for row_index in 0..solved_matrix.rows {
            if solved_matrix[row_index][last_column_index] == zero {
                continue;
            }

            let mut check_passed: bool = false;
            for column_index in 0..last_column_index {
                if solved_matrix[row_index][column_index] != zero {
                    check_passed = true;
                    break;
                }
            }

            if !check_passed {
                return Err("The system was inconsistent and there is no solution for b.");
            }
        }

        Ok(Self::get_x_vector(solved_matrix))
    }

    /// Returns true if these two matrices are equal, within the given delta
    pub fn equals(&self, other: &Self, delta: T) -> bool {
        if self.rows != other.rows || self.columns != other.columns {
            return false;
        }

        for row in 0..self.rows {
            for column in 0..self.columns {
                let difference: T =
                    num_traits::sign::abs_sub(self[row][column], other[row][column]);
                // is_positive() should exclude zero, but in my testing it doesn't
                if (difference - delta).is_positive() && !(difference - delta).is_zero() {
                    return false;
                }
            }
        }

        true
    }
}

impl<T> Clone for Matrix<T>
    where
        T: MatrixCompatible,
{
    /// Safely clones this matrix
    fn clone(&self) -> Self {
        let mut matrix: Vec<Vec<T>> = Vec::with_capacity(self.rows);

        for i in 0..self.rows {
            matrix.push(self.matrix[i].clone());
        }

        Self {
            matrix,
            rows: self.rows,
            columns: self.columns,
        }
    }
}

gen_ops!(
    <T>;
    types Matrix<T>, Matrix<T> => Matrix<T>;

    /// Adds two matrices
    for + call |a: &Matrix<T>, b: &Matrix<T>| {
        if a.rows != b.rows || a.columns != b.columns {
            panic!("Matrix size mismatch!");
        }

        let mut output: Self = Self::new(a.rows, a.columns);

        for row_index in 0..a.rows {
            for column_index in 0..a.columns {
                let value: T = a[row_index][column_index] + b[row_index][column_index];
                output.set_value(row_index, column_index, value);
            }
        }

        output
    };

    /// Subtracts two matrices. Multiplies the right matrix by -1, adds those matrices.
    for - call |a: &Matrix<T>, b: &Matrix<T>| {
        let negative_rhs: Matrix<T> = b.clone() * T::one().neg();
        a.clone() + negative_rhs
    };

    /// Multiplies two matrices
    for * call |a: &Matrix<T>, b: &Matrix<T>| {
        if a.columns != b.rows {
            panic!("Left hand columns must equal right hand rows!");
        }

        let common_size: usize = a.columns;

        let mut output: Matrix<T> = Matrix::new(a.rows, b.columns);

        for output_row in 0..a.rows {
            for output_column in 0..b.columns {
                let mut a_row: Vec<T> = Vec::with_capacity(common_size);
                for i in 0..common_size {
                    a_row.push(a[output_row][i]);
                }

                let mut b_row: Vec<T> = Vec::with_capacity(common_size);
                for i in 0..common_size {
                    b_row.push(b[i][output_column]);
                }

                output.set_value(output_row, output_column, Matrix::inner_product(&a_row, &b_row));
            }
        }

        output
    };

    where T: MatrixCompatible
);

gen_ops!(
    <T>;
    types Matrix<T>, Matrix<T>;

    /// Add-assigns a matrix
    for += call |a: &mut Matrix<T>, b: &Matrix<T>| {*a = a.clone() + b.clone()};

    /// Sub-assigns a matrix
    for -= call |a: &mut Matrix<T>, b: &Matrix<T>| {*a = a.clone() - b.clone()};

    /// Mul-assigns a matrix
    for *= call |a: &mut Matrix<T>, b: &Matrix<T>| {*a = a.clone() * b.clone()};

    where T: MatrixCompatible
);

gen_ops!(
    <T>;
    types Matrix<T>, T => Matrix<T>;

    for * call |a: &Matrix<T>, scalar: &T| {
        let mut output: Matrix<T> = Matrix::new(a.rows, a.columns);

        for row_index in 0..a.rows {
            for column_index in 0..a.columns {
                let value: T = a[row_index][column_index] * *scalar;
                output.set_value(row_index, column_index, value);
            }
        }

        output
    };

    where T: MatrixCompatible
);

gen_ops!(
    <T>;
    types Matrix<T>, T;

    /// Scale-assigns a matrix
    for *= call |a: &mut Matrix<T>, b: &T| {*a = a.clone() * *b};

    where T: MatrixCompatible
);

impl<T> cmp::PartialEq for Matrix<T>
    where
        T: MatrixCompatible,
{
    fn eq(&self, other: &Self) -> bool {
        self.equals(other, T::zero())
    }
}

impl<T> ops::Index<usize> for Matrix<T>
    where
        T: MatrixCompatible,
{
    type Output = Vec<T>;

    /// Grabs the indicated row of the matrix. Can then index that row to get a value, ie Matrix\[row\]\[column\]
    fn index(&self, index: usize) -> &Self::Output {
        return self.matrix[index].as_ref();
    }
}
