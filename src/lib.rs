/// Brayden Jonsson, 2023
/// https://github.com/BraydenJonsson/rust-matrix
///
/// Contains tests for the matrix library
mod matrix;

#[cfg(test)]
mod tests {
    use crate::matrix::Matrix;

    const COMPARISON_TOLERANCE: f64 = 0.000000001;
    const STANDARD_MATRIX_A: &[f64] = &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    const STANDARD_MATRIX_B: &[f64] = &[5.7, 1.2, 0.0, 4.9, -7.1, -2.1, 77.1, 0.0, 9.1];
    // These matrices were calculated using reshish.com
    const STANDARD_MATRIX_MULTIPLICATION_SOLUTION: &[f64] =
        &[246.8, -13.0, 23.1, 509.9, -30.7, 44.1, 773.0, -48.4, 65.1];
    const STANDARD_MATRIX_REVERSE_MULTIPLICATION_SOLUTION: &[f64] =
        &[10.5, 17.4, 24.3, -38.2, -42.5, -46.8, 140.8, 227.0, 313.2];
    const STANDARD_MATRIX_B_INVERSE_SOLUTION: &[f64] = &[
        0.10487325447955369214,
        0.017725057095135835292,
        0.0040903977911851927602,
        0.33518537455545329559,
        -0.084194021201895217641,
        -0.019429389508129665611,
        -0.88854152971138347965,
        -0.15017603319065636276,
        0.075234102230727652557,
    ];

    const STANDARD_MATRIX_A_DETERMINANT: f64 = 0.0;
    const STANDARD_MATRIX_B_DETERMINANT: f64 = -616.077;

    const STANDARD_MATRIX_A_REF: &[f64] = &[1.0, 0.0, -1.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0];

    const STANDARD_MATRIX_B_REF: &[f64] = &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

    #[test]
    fn square_addition() {
        let a: Matrix = Matrix::square_matrix_from_list(&STANDARD_MATRIX_A.to_vec());
        let b: Matrix = Matrix::square_matrix_from_list(&STANDARD_MATRIX_B.to_vec());

        let mut solution_list: Vec<f64> = Vec::with_capacity(9);
        for index in 0..9 {
            solution_list.push(STANDARD_MATRIX_A[index] + STANDARD_MATRIX_B[index]);
        }
        let solution_matrix: Matrix = Matrix::square_matrix_from_list(&solution_list);
        assert!(solution_matrix.equals(&(a + b), COMPARISON_TOLERANCE));
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
        assert!(solution_matrix.equals(&(a - b), COMPARISON_TOLERANCE));
    }

    #[test]
    fn square_scalar() {
        let a: Matrix = Matrix::square_matrix_from_list(&STANDARD_MATRIX_A.to_vec());

        let mut solution_list: Vec<f64> = Vec::with_capacity(9);
        for index in 0..9 {
            solution_list.push(STANDARD_MATRIX_A[index] * 3.7);
        }
        let solution_matrix: Matrix = Matrix::square_matrix_from_list(&solution_list);
        assert!(solution_matrix.equals(&(3.7 * a), COMPARISON_TOLERANCE));
    }

    #[test]
    fn square_multiplication() {
        let a: Matrix = Matrix::square_matrix_from_list(&STANDARD_MATRIX_A.to_vec());
        let b: Matrix = Matrix::square_matrix_from_list(&STANDARD_MATRIX_B.to_vec());

        let solution_matrix: Matrix =
            Matrix::square_matrix_from_list(&STANDARD_MATRIX_MULTIPLICATION_SOLUTION.to_vec());
        assert!(solution_matrix.equals(&(a * b), COMPARISON_TOLERANCE));
    }

    #[test]
    fn square_reverse_multiplication() {
        let a: Matrix = Matrix::square_matrix_from_list(&STANDARD_MATRIX_A.to_vec());
        let b: Matrix = Matrix::square_matrix_from_list(&STANDARD_MATRIX_B.to_vec());

        let solution_matrix: Matrix = Matrix::square_matrix_from_list(
            &STANDARD_MATRIX_REVERSE_MULTIPLICATION_SOLUTION.to_vec(),
        );
        assert!(solution_matrix.equals(&(b * a), COMPARISON_TOLERANCE));
    }

    #[test]
    fn a_inverse() {
        let a: Matrix = Matrix::square_matrix_from_list(&STANDARD_MATRIX_A.to_vec());

        assert_eq!(a.inverse().unwrap_err(), "Matrix is not invertible");
    }

    #[test]
    fn b_inverse() {
        let b: Matrix = Matrix::square_matrix_from_list(&STANDARD_MATRIX_B.to_vec());

        let solution_matrix: Matrix =
            Matrix::square_matrix_from_list(&STANDARD_MATRIX_B_INVERSE_SOLUTION.to_vec());
        assert!(solution_matrix.equals(&(b.inverse()).unwrap(), COMPARISON_TOLERANCE));
    }

    #[test]
    fn a_determinant() {
        let a: Matrix = Matrix::square_matrix_from_list(&STANDARD_MATRIX_A.to_vec());

        let determinant: f64 = a.determinant();

        assert!((determinant - STANDARD_MATRIX_A_DETERMINANT).abs() < COMPARISON_TOLERANCE);
    }

    #[test]
    fn b_determinant() {
        let b: Matrix = Matrix::square_matrix_from_list(&STANDARD_MATRIX_B.to_vec());

        let determinant: f64 = b.determinant();

        assert!((determinant - STANDARD_MATRIX_B_DETERMINANT).abs() < COMPARISON_TOLERANCE);
    }

    #[test]
    fn a_ref() {
        let a: Matrix = Matrix::square_matrix_from_list(&STANDARD_MATRIX_A.to_vec());

        let solution_matrix: Matrix =
            Matrix::square_matrix_from_list(&STANDARD_MATRIX_A_REF.to_vec());
        assert!(solution_matrix.equals(&(a.reduced_echelon_form()), COMPARISON_TOLERANCE));
    }

    #[test]
    fn b_ref() {
        let b: Matrix = Matrix::square_matrix_from_list(&STANDARD_MATRIX_B.to_vec());

        let solution_matrix: Matrix =
            Matrix::square_matrix_from_list(&STANDARD_MATRIX_B_REF.to_vec());
        assert!(solution_matrix.equals(&(b.reduced_echelon_form()), COMPARISON_TOLERANCE));
    }
}
