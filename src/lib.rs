/// Brayden Jonsson, 2023
/// https://github.com/BraydenJonsson/rust-matrix
///
/// Contains tests for the matrix library
mod matrix;

#[cfg(test)]
mod tests {
    use crate::matrix::Matrix;

    const STANDARD_MATRIX_A: &'static [f64] = &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    const STANDARD_MATRIX_B: &'static [f64] = &[5.7, 1.2, 0.0, 4.9, -7.1, -2.1, 77.1, 0.0, 9.1];

    #[test]
    fn square_addition() {
        let a: Matrix = Matrix::square_matrix_from_list(&STANDARD_MATRIX_A.to_vec());
        let b: Matrix = Matrix::square_matrix_from_list(&STANDARD_MATRIX_B.to_vec());

        let mut solution_list: Vec<f64> = Vec::with_capacity(9);
        for index in 0..9 {
            solution_list.push(STANDARD_MATRIX_A[index] + STANDARD_MATRIX_B[index]);
        }
        let solution_matrix: Matrix = Matrix::square_matrix_from_list(&solution_list);
        assert_eq!(a + b, solution_matrix);
    }

    #[test]
    fn square_subtraction() {
        let a: Matrix = Matrix::square_matrix_from_list(&STANDARD_MATRIX_A.to_vec());
        let b: Matrix = Matrix::square_matrix_from_list(&STANDARD_MATRIX_B.to_vec());

        let mut solution_list: Vec<f64> = Vec::with_capacity(9);
        for index in 0..9 {
            solution_list.push(STANDARD_MATRIX_A[index] - STANDARD_MATRIX_B[index]);
        }
        let solution_matrix: Matrix = Matrix::square_matrix_from_list(&solution_list);
        assert_eq!(a - b, solution_matrix);
    }

    #[test]
    fn square_scalar() {
        let a: Matrix = Matrix::square_matrix_from_list(&STANDARD_MATRIX_A.to_vec());

        let mut solution_list: Vec<f64> = Vec::with_capacity(9);
        for index in 0..9 {
            solution_list.push(STANDARD_MATRIX_A[index] * 3.7);
        }
        let solution_matrix: Matrix = Matrix::square_matrix_from_list(&solution_list);
        assert_eq!(3.7 * a, solution_matrix);
    }
}
