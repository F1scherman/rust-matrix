/// Brayden Jonsson, 2023
/// https://github.com/BraydenJonsson/rust-matrix
///
/// Contains tests for the matrix library
mod matrix;

#[cfg(test)]
mod f64tests {
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

    const STANDARD_MATRIX_A_TRANSPOSE: &[f64] = &[1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0];

    const STANDARD_MATRIX_B_TRANSPOSE: &[f64] = &[5.7, 4.9, 77.1, 1.2, -7.1, 0.0, 0.0, -2.1, 9.1];

    const B_VECTOR: &[f64] = &[3.9, 7.2, -1.0];

    const WRONG_LENGTH_B_VECTOR: &[f64] = &[-2.0, 1.0, 9.8, -0.1];

    const STANDARD_MATRIX_B_SOLUTION: &[f64] = &[
        0.53253570576405222074,
        0.72045539762075195146,
        -4.6218135070778490354,
    ];

    #[test]
    fn square_addition() {
        let a: Matrix<f64> = Matrix::square_matrix_from_list(&STANDARD_MATRIX_A.to_vec());
        let b: Matrix<f64> = Matrix::square_matrix_from_list(&STANDARD_MATRIX_B.to_vec());

        let mut solution_list: Vec<f64> = Vec::with_capacity(9);
        for index in 0..9 {
            solution_list.push(STANDARD_MATRIX_A[index] + STANDARD_MATRIX_B[index]);
        }
        let solution_matrix: Matrix<f64> = Matrix::square_matrix_from_list(&solution_list);
        assert!(solution_matrix.equals(&(a + b), COMPARISON_TOLERANCE));
    }

    #[test]
    fn square_subtraction() {
        let a: Matrix<f64> = Matrix::square_matrix_from_list(&STANDARD_MATRIX_A.to_vec());
        let b: Matrix<f64> = Matrix::square_matrix_from_list(&STANDARD_MATRIX_B.to_vec());

        let mut solution_list: Vec<f64> = Vec::with_capacity(9);
        for index in 0..9 {
            solution_list.push(STANDARD_MATRIX_A[index] - STANDARD_MATRIX_B[index]);
        }
        let solution_matrix: Matrix<f64> = Matrix::square_matrix_from_list(&solution_list);
        assert!(solution_matrix.equals(&(a - b), COMPARISON_TOLERANCE));
    }

    #[test]
    fn square_scalar() {
        let a: Matrix<f64> = Matrix::square_matrix_from_list(&STANDARD_MATRIX_A.to_vec());

        let mut solution_list: Vec<f64> = Vec::with_capacity(9);
        for index in 0..9 {
            solution_list.push(STANDARD_MATRIX_A[index] * 3.7);
        }
        let solution_matrix: Matrix<f64> = Matrix::square_matrix_from_list(&solution_list);
        assert!(solution_matrix.equals(&(a * 3.7), COMPARISON_TOLERANCE));
    }

    #[test]
    fn square_multiplication() {
        let a: Matrix<f64> = Matrix::square_matrix_from_list(&STANDARD_MATRIX_A.to_vec());
        let b: Matrix<f64> = Matrix::square_matrix_from_list(&STANDARD_MATRIX_B.to_vec());

        let solution_matrix: Matrix<f64> =
            Matrix::square_matrix_from_list(&STANDARD_MATRIX_MULTIPLICATION_SOLUTION.to_vec());
        assert!(solution_matrix.equals(&(a * b), COMPARISON_TOLERANCE));
    }

    #[test]
    fn square_reverse_multiplication() {
        let a: Matrix<f64> = Matrix::square_matrix_from_list(&STANDARD_MATRIX_A.to_vec());
        let b: Matrix<f64> = Matrix::square_matrix_from_list(&STANDARD_MATRIX_B.to_vec());

        let solution_matrix: Matrix<f64> = Matrix::square_matrix_from_list(
            &STANDARD_MATRIX_REVERSE_MULTIPLICATION_SOLUTION.to_vec(),
        );
        assert!(solution_matrix.equals(&(b * a), COMPARISON_TOLERANCE));
    }

    #[test]
    fn a_inverse() {
        let a: Matrix<f64> = Matrix::square_matrix_from_list(&STANDARD_MATRIX_A.to_vec());

        assert_eq!(a.inverse().unwrap_err(), "Matrix is not invertible");
    }

    #[test]
    fn b_inverse() {
        let b: Matrix<f64> = Matrix::square_matrix_from_list(&STANDARD_MATRIX_B.to_vec());

        let solution_matrix: Matrix<f64> =
            Matrix::square_matrix_from_list(&STANDARD_MATRIX_B_INVERSE_SOLUTION.to_vec());
        assert!(solution_matrix.equals(&(b.inverse()).unwrap(), COMPARISON_TOLERANCE));
    }

    #[test]
    fn a_determinant() {
        let a: Matrix<f64> = Matrix::square_matrix_from_list(&STANDARD_MATRIX_A.to_vec());

        let determinant: f64 = a.determinant();

        assert!((determinant - STANDARD_MATRIX_A_DETERMINANT).abs() < COMPARISON_TOLERANCE);
    }

    #[test]
    fn b_determinant() {
        let b: Matrix<f64> = Matrix::square_matrix_from_list(&STANDARD_MATRIX_B.to_vec());

        let determinant: f64 = b.determinant();

        assert!((determinant - STANDARD_MATRIX_B_DETERMINANT).abs() < COMPARISON_TOLERANCE);
    }

    #[test]
    fn a_ref() {
        let a: Matrix<f64> = Matrix::square_matrix_from_list(&STANDARD_MATRIX_A.to_vec());

        let solution_matrix: Matrix<f64> =
            Matrix::square_matrix_from_list(&STANDARD_MATRIX_A_REF.to_vec());
        assert!(solution_matrix.equals(&(a.reduced_echelon_form()), COMPARISON_TOLERANCE));
    }

    #[test]
    fn b_ref() {
        let b: Matrix<f64> = Matrix::square_matrix_from_list(&STANDARD_MATRIX_B.to_vec());

        let solution_matrix: Matrix<f64> =
            Matrix::square_matrix_from_list(&STANDARD_MATRIX_B_REF.to_vec());
        assert!(solution_matrix.equals(&(b.reduced_echelon_form()), COMPARISON_TOLERANCE));
    }

    #[test]
    fn a_transpose() {
        let a: Matrix<f64> = Matrix::square_matrix_from_list(&STANDARD_MATRIX_A.to_vec());

        let solution_matrix: Matrix<f64> =
            Matrix::square_matrix_from_list(&STANDARD_MATRIX_A_TRANSPOSE.to_vec());
        assert!(solution_matrix.equals(&(a.transpose()), COMPARISON_TOLERANCE));
    }

    #[test]
    fn b_transpose() {
        let b: Matrix<f64> = Matrix::square_matrix_from_list(&STANDARD_MATRIX_B.to_vec());

        let solution_matrix: Matrix<f64> =
            Matrix::square_matrix_from_list(&STANDARD_MATRIX_B_TRANSPOSE.to_vec());
        assert!(solution_matrix.equals(&(b.transpose()), COMPARISON_TOLERANCE));
    }

    #[test]
    #[should_panic]
    fn wrong_length_b_vector() {
        let a: Matrix<f64> = Matrix::square_matrix_from_list(&STANDARD_MATRIX_A.to_vec());

        a.solve(WRONG_LENGTH_B_VECTOR.to_vec());
    }

    #[test]
    fn solve_a() {
        let a: Matrix<f64> = Matrix::square_matrix_from_list(&STANDARD_MATRIX_A.to_vec());

        assert_eq!(
            a.solve(B_VECTOR.to_vec()).unwrap_err(),
            "The system was inconsistent and there is no solution for b."
        );
    }

    #[test]
    fn solve_b() {
        let b: Matrix<f64> = Matrix::square_matrix_from_list(&STANDARD_MATRIX_B.to_vec());

        let solution_vector: Vec<f64> = STANDARD_MATRIX_B_SOLUTION.to_vec();

        let b_solution: Vec<f64> = b.solve(B_VECTOR.to_vec()).unwrap();

        for i in 0..solution_vector.len() {
            assert!((solution_vector[i] - b_solution[i]).abs() < COMPARISON_TOLERANCE);
        }
    }

    #[test]
    fn least_squares_a() {
        let a: Matrix<f64> = Matrix::square_matrix_from_list(&STANDARD_MATRIX_A.to_vec());

        assert_eq!(
            a.least_squares_solution(B_VECTOR.to_vec()).unwrap_err(),
            "The system was inconsistent and there is no solution for b. (In this case, these means an arithmetic problem, probably due to floating point inaccuracy)."
        );
    }

    #[test]
    fn least_squares_b() {
        let b: Matrix<f64> = Matrix::square_matrix_from_list(&STANDARD_MATRIX_B.to_vec());

        let solution_vector: Vec<f64> = STANDARD_MATRIX_B_SOLUTION.to_vec();

        let b_solution: Vec<f64> = b.least_squares_solution(B_VECTOR.to_vec()).unwrap();

        for i in 0..solution_vector.len() {
            assert!((solution_vector[i] - b_solution[i]).abs() < COMPARISON_TOLERANCE);
        }
    }
}

// Runs the exact same tests* as f64tests, but with f32 instead
// least_squares_a is different because it can actually solve the system consistently
mod f32tests {
    use crate::matrix::Matrix;

    const COMPARISON_TOLERANCE: f32 = 0.001;
    const STANDARD_MATRIX_A: &[f32] = &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    const STANDARD_MATRIX_B: &[f32] = &[5.7, 1.2, 0.0, 4.9, -7.1, -2.1, 77.1, 0.0, 9.1];
    // These matrices were calculated using reshish.com
    const STANDARD_MATRIX_MULTIPLICATION_SOLUTION: &[f32] =
        &[246.8, -13.0, 23.1, 509.9, -30.7, 44.1, 773.0, -48.4, 65.1];
    const STANDARD_MATRIX_REVERSE_MULTIPLICATION_SOLUTION: &[f32] =
        &[10.5, 17.4, 24.3, -38.2, -42.5, -46.8, 140.8, 227.0, 313.2];
    const STANDARD_MATRIX_B_INVERSE_SOLUTION: &[f32] = &[
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

    const STANDARD_MATRIX_A_DETERMINANT: f32 = 0.0;
    const STANDARD_MATRIX_B_DETERMINANT: f32 = -616.077;

    const STANDARD_MATRIX_A_REF: &[f32] = &[1.0, 0.0, -1.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0];

    const STANDARD_MATRIX_B_REF: &[f32] = &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];

    const STANDARD_MATRIX_A_TRANSPOSE: &[f32] = &[1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0];

    const STANDARD_MATRIX_B_TRANSPOSE: &[f32] = &[5.7, 4.9, 77.1, 1.2, -7.1, 0.0, 0.0, -2.1, 9.1];

    const B_VECTOR: &[f32] = &[3.9, 7.2, -1.0];

    const WRONG_LENGTH_B_VECTOR: &[f32] = &[-2.0, 1.0, 9.8, -0.1];

    const STANDARD_MATRIX_B_SOLUTION: &[f32] = &[
        0.53253570576405222074,
        0.72045539762075195146,
        -4.6218135070778490354,
    ];

    const LEAST_SQUARES_A_SOLUTION: &[f32] = &[-7.45, 6.6333333333333332766, 0.0];

    #[test]
    fn square_addition() {
        let a: Matrix<f32> = Matrix::square_matrix_from_list(&STANDARD_MATRIX_A.to_vec());
        let b: Matrix<f32> = Matrix::square_matrix_from_list(&STANDARD_MATRIX_B.to_vec());

        let mut solution_list: Vec<f32> = Vec::with_capacity(9);
        for index in 0..9 {
            solution_list.push(STANDARD_MATRIX_A[index] + STANDARD_MATRIX_B[index]);
        }
        let solution_matrix: Matrix<f32> = Matrix::square_matrix_from_list(&solution_list);
        assert!(solution_matrix.equals(&(a + b), COMPARISON_TOLERANCE));
    }

    #[test]
    fn square_subtraction() {
        let a: Matrix<f32> = Matrix::square_matrix_from_list(&STANDARD_MATRIX_A.to_vec());
        let b: Matrix<f32> = Matrix::square_matrix_from_list(&STANDARD_MATRIX_B.to_vec());

        let mut solution_list: Vec<f32> = Vec::with_capacity(9);
        for index in 0..9 {
            solution_list.push(STANDARD_MATRIX_A[index] - STANDARD_MATRIX_B[index]);
        }
        let solution_matrix: Matrix<f32> = Matrix::square_matrix_from_list(&solution_list);
        assert!(solution_matrix.equals(&(a - b), COMPARISON_TOLERANCE));
    }

    #[test]
    fn square_scalar() {
        let a: Matrix<f32> = Matrix::square_matrix_from_list(&STANDARD_MATRIX_A.to_vec());

        let mut solution_list: Vec<f32> = Vec::with_capacity(9);
        for index in 0..9 {
            solution_list.push(STANDARD_MATRIX_A[index] * 3.7);
        }
        let solution_matrix: Matrix<f32> = Matrix::square_matrix_from_list(&solution_list);
        assert!(solution_matrix.equals(&(a * 3.7), COMPARISON_TOLERANCE));
    }

    #[test]
    fn square_multiplication() {
        let a: Matrix<f32> = Matrix::square_matrix_from_list(&STANDARD_MATRIX_A.to_vec());
        let b: Matrix<f32> = Matrix::square_matrix_from_list(&STANDARD_MATRIX_B.to_vec());

        let solution_matrix: Matrix<f32> =
            Matrix::square_matrix_from_list(&STANDARD_MATRIX_MULTIPLICATION_SOLUTION.to_vec());
        assert!(solution_matrix.equals(&(a * b), COMPARISON_TOLERANCE));
    }

    #[test]
    fn square_reverse_multiplication() {
        let a: Matrix<f32> = Matrix::square_matrix_from_list(&STANDARD_MATRIX_A.to_vec());
        let b: Matrix<f32> = Matrix::square_matrix_from_list(&STANDARD_MATRIX_B.to_vec());

        let solution_matrix: Matrix<f32> = Matrix::square_matrix_from_list(
            &STANDARD_MATRIX_REVERSE_MULTIPLICATION_SOLUTION.to_vec(),
        );
        assert!(solution_matrix.equals(&(b * a), COMPARISON_TOLERANCE));
    }

    #[test]
    fn a_inverse() {
        let a: Matrix<f32> = Matrix::square_matrix_from_list(&STANDARD_MATRIX_A.to_vec());

        assert_eq!(a.inverse().unwrap_err(), "Matrix is not invertible");
    }

    #[test]
    fn b_inverse() {
        let b: Matrix<f32> = Matrix::square_matrix_from_list(&STANDARD_MATRIX_B.to_vec());

        let solution_matrix: Matrix<f32> =
            Matrix::square_matrix_from_list(&STANDARD_MATRIX_B_INVERSE_SOLUTION.to_vec());
        assert!(solution_matrix.equals(&(b.inverse()).unwrap(), COMPARISON_TOLERANCE));
    }

    #[test]
    fn a_determinant() {
        let a: Matrix<f32> = Matrix::square_matrix_from_list(&STANDARD_MATRIX_A.to_vec());

        let determinant: f32 = a.determinant();

        assert!((determinant - STANDARD_MATRIX_A_DETERMINANT).abs() < COMPARISON_TOLERANCE);
    }

    #[test]
    fn b_determinant() {
        let b: Matrix<f32> = Matrix::square_matrix_from_list(&STANDARD_MATRIX_B.to_vec());

        let determinant: f32 = b.determinant();

        assert!((determinant - STANDARD_MATRIX_B_DETERMINANT).abs() < COMPARISON_TOLERANCE);
    }

    #[test]
    fn a_ref() {
        let a: Matrix<f32> = Matrix::square_matrix_from_list(&STANDARD_MATRIX_A.to_vec());

        let solution_matrix: Matrix<f32> =
            Matrix::square_matrix_from_list(&STANDARD_MATRIX_A_REF.to_vec());
        assert!(solution_matrix.equals(&(a.reduced_echelon_form()), COMPARISON_TOLERANCE));
    }

    #[test]
    fn b_ref() {
        let b: Matrix<f32> = Matrix::square_matrix_from_list(&STANDARD_MATRIX_B.to_vec());

        let solution_matrix: Matrix<f32> =
            Matrix::square_matrix_from_list(&STANDARD_MATRIX_B_REF.to_vec());
        assert!(solution_matrix.equals(&(b.reduced_echelon_form()), COMPARISON_TOLERANCE));
    }

    #[test]
    fn a_transpose() {
        let a: Matrix<f32> = Matrix::square_matrix_from_list(&STANDARD_MATRIX_A.to_vec());

        let solution_matrix: Matrix<f32> =
            Matrix::square_matrix_from_list(&STANDARD_MATRIX_A_TRANSPOSE.to_vec());
        assert!(solution_matrix.equals(&(a.transpose()), COMPARISON_TOLERANCE));
    }

    #[test]
    fn b_transpose() {
        let b: Matrix<f32> = Matrix::square_matrix_from_list(&STANDARD_MATRIX_B.to_vec());

        let solution_matrix: Matrix<f32> =
            Matrix::square_matrix_from_list(&STANDARD_MATRIX_B_TRANSPOSE.to_vec());
        assert!(solution_matrix.equals(&(b.transpose()), COMPARISON_TOLERANCE));
    }

    #[test]
    #[should_panic]
    fn wrong_length_b_vector() {
        let a: Matrix<f32> = Matrix::square_matrix_from_list(&STANDARD_MATRIX_A.to_vec());

        a.solve(WRONG_LENGTH_B_VECTOR.to_vec());
    }

    #[test]
    fn solve_a() {
        let a: Matrix<f32> = Matrix::square_matrix_from_list(&STANDARD_MATRIX_A.to_vec());

        assert_eq!(
            a.solve(B_VECTOR.to_vec()).unwrap_err(),
            "The system was inconsistent and there is no solution for b."
        );
    }

    #[test]
    fn solve_b() {
        let b: Matrix<f32> = Matrix::square_matrix_from_list(&STANDARD_MATRIX_B.to_vec());

        let solution_vector: Vec<f32> = STANDARD_MATRIX_B_SOLUTION.to_vec();

        let b_solution: Vec<f32> = b.solve(B_VECTOR.to_vec()).unwrap();

        for i in 0..solution_vector.len() {
            assert!((solution_vector[i] - b_solution[i]).abs() < COMPARISON_TOLERANCE);
        }
    }

    #[test]
    fn least_squares_a() {
        let a: Matrix<f32> = Matrix::square_matrix_from_list(&STANDARD_MATRIX_A.to_vec());

        let solution_vector: Vec<f32> = LEAST_SQUARES_A_SOLUTION.to_vec();

        let a_solution: Vec<f32> = a.least_squares_solution(B_VECTOR.to_vec()).unwrap();

        for i in 0..solution_vector.len() {
            assert!((solution_vector[i] - a_solution[i]).abs() < COMPARISON_TOLERANCE);
        }
    }

    #[test]
    fn least_squares_b() {
        let b: Matrix<f32> = Matrix::square_matrix_from_list(&STANDARD_MATRIX_B.to_vec());

        let solution_vector: Vec<f32> = STANDARD_MATRIX_B_SOLUTION.to_vec();

        let b_solution: Vec<f32> = b.least_squares_solution(B_VECTOR.to_vec()).unwrap();

        for i in 0..solution_vector.len() {
            assert!((solution_vector[i] - b_solution[i]).abs() < COMPARISON_TOLERANCE);
        }
    }
}
