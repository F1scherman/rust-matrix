/// Brayden Jonsson, 2023
/// https://github.com/BraydenJonsson/rust-matrix
///

/// Contains tests for the matrix library
/// All solution matrices were calculated with reshish.com
mod matrix;
mod complex;

#[cfg(test)]
#[allow(unused)]
mod f64tests {
    use crate::matrix::Matrix;

    const COMPARISON_TOLERANCE: f64 = 0.000000001;

    #[test]
    fn square_addition() {
        let a_array = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let b_array = [5.7, 1.2, 0.0, 4.9, -7.1, -2.1, 77.1, 0.0, 9.1];

        let a = Matrix::square_matrix_from_list(&a_array);
        let b = Matrix::square_matrix_from_list(&b_array);

        let mut solution_list = Vec::with_capacity(9);
        for index in 0..9 {
            solution_list.push(a_array[index] + b_array[index]);
        }
        let solution_matrix = Matrix::square_matrix_from_list(&solution_list);
        assert!(solution_matrix.equals(&(a + b), COMPARISON_TOLERANCE));
    }

    #[test]
    fn square_subtraction() {
        let a_array = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let b_array = [5.7, 1.2, 0.0, 4.9, -7.1, -2.1, 77.1, 0.0, 9.1];

        let a = Matrix::square_matrix_from_list(&a_array);
        let b = Matrix::square_matrix_from_list(&b_array);

        let mut solution_list = Vec::with_capacity(9);
        for index in 0..9 {
            solution_list.push(a_array[index] - b_array[index]);
        }
        let solution_matrix = Matrix::square_matrix_from_list(&solution_list);
        assert!(solution_matrix.equals(&(a - b), COMPARISON_TOLERANCE));
    }

    #[test]
    fn square_scalar() {
        let matrix_array = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

        let matrix = Matrix::square_matrix_from_list(&matrix_array.to_vec());

        let mut solution_list = Vec::with_capacity(9);
        for index in 0..9 {
            solution_list.push(matrix_array[index] * 3.7);
        }
        let solution_matrix = Matrix::square_matrix_from_list(&solution_list);
        assert!(solution_matrix.equals(&(matrix * 3.7), COMPARISON_TOLERANCE));
    }

    #[test]
    fn square_multiplication() {
        let a_array = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let b_array = [5.7, 1.2, 0.0, 4.9, -7.1, -2.1, 77.1, 0.0, 9.1];

        let a = Matrix::square_matrix_from_list(&a_array);
        let b = Matrix::square_matrix_from_list(&b_array);
      
        let solution_array = [246.8, -13.0, 23.1, 509.9, -30.7, 44.1, 773.0, -48.4, 65.1];
        let solution = Matrix::square_matrix_from_list(&solution_array);
        assert!(solution.equals(&(a * b), COMPARISON_TOLERANCE));
    }

    #[test]
    fn square_reverse_multiplication() {
        let a_array = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let b_array = [5.7, 1.2, 0.0, 4.9, -7.1, -2.1, 77.1, 0.0, 9.1];

        let a = Matrix::square_matrix_from_list(&a_array);
        let b = Matrix::square_matrix_from_list(&b_array);

        let solution_array = [10.5, 17.4, 24.3, -38.2, -42.5, -46.8, 140.8, 227.0, 313.2];
        let solution = Matrix::square_matrix_from_list(&solution_array);
        assert!(solution.equals(&(b * a), COMPARISON_TOLERANCE));
    }

    #[test]
    fn inverse_1() {
        let matrix_array = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

        let matrix = Matrix::square_matrix_from_list(&matrix_array);

        assert_eq!(matrix.inverse().unwrap_err(), "Matrix is not invertible");
    }

    #[test]
    fn inverse_2() {
        let matrix_array = [5.7, 1.2, 0.0, 4.9, -7.1, -2.1, 77.1, 0.0, 9.1];
        let matrix = Matrix::square_matrix_from_list(&matrix_array);

        let solution_array = [
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
        let solution = Matrix::square_matrix_from_list(&solution_array);
        assert!(solution.equals(&(matrix.inverse()).unwrap(), COMPARISON_TOLERANCE));
    }

    #[test]
    fn determinant_1() {
        let matrix_array = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let matrix = Matrix::square_matrix_from_list(&matrix_array);

        let determinant = matrix.determinant();
        assert!((determinant - 0.0 as f64).abs() < COMPARISON_TOLERANCE);
    }

    #[test]
    fn determinant_2() {
        let matrix_array = [5.7, 1.2, 0.0, 4.9, -7.1, -2.1, 77.1, 0.0, 9.1];
        let matrix = Matrix::square_matrix_from_list(&matrix_array);

        let determinant = matrix.determinant();
        assert!((determinant - -616.077 as f64).abs() < COMPARISON_TOLERANCE);
    }

    #[test]
    fn ref_1() {
        let matrix_array = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let matrix = Matrix::square_matrix_from_list(&matrix_array);

        let solution_array = [1.0, 0.0, -1.0, 0.0, 1.0, 2.0, 0.0, 0.0, 0.0];
        let solution_matrix = Matrix::square_matrix_from_list(&solution_array);
        assert!(solution_matrix.equals(&(matrix.reduced_echelon_form()), COMPARISON_TOLERANCE));
    }

    #[test]
    fn ref_2() {
        let matrix_array = [5.7, 1.2, 0.0, 4.9, -7.1, -2.1, 77.1, 0.0, 9.1];
        let matrix = Matrix::square_matrix_from_list(&matrix_array);

        let solution_array = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let solution = Matrix::square_matrix_from_list(&solution_array);
        assert!(solution.equals(&(matrix.reduced_echelon_form()), COMPARISON_TOLERANCE));
    }

    #[test]
    fn transpose_1() {
        let matrix_array = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let matrix = Matrix::square_matrix_from_list(&matrix_array);


        let solution_array = [1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0];
        let solution_matrix = Matrix::square_matrix_from_list(&solution_array);
        assert!(solution_matrix.equals(&(matrix.transpose()), COMPARISON_TOLERANCE));
    }

    #[test]
    fn transpose_2() {
        let matrix_array = [5.7, 1.2, 0.0, 4.9, -7.1, -2.1, 77.1, 0.0, 9.1];
        let matrix = Matrix::square_matrix_from_list(&matrix_array);

        let solution_array = [5.7, 4.9, 77.1, 1.2, -7.1, 0.0, 0.0, -2.1, 9.1];
        let solution = Matrix::square_matrix_from_list(&solution_array);
        assert!(solution.equals(&(matrix.transpose()), COMPARISON_TOLERANCE));
    }

    #[test]
    #[should_panic]
    fn wrong_length_b_vector() {
        let matrix_array = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let matrix = Matrix::square_matrix_from_list(&matrix_array.to_vec());

        matrix.solve(&[-2.0, -0.1]);
    }

    #[test]
    fn solve_1() {
        let matrix_array = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

        let matrix = Matrix::square_matrix_from_list(&matrix_array.to_vec());

        let b_vector = [3.9, 7.2, -1.0];
        assert_eq!(
            matrix.solve(&b_vector).unwrap_err(),
            "The system was inconsistent and there is no solution for b."
        );
    }

    #[test]
    fn solve_2() {
        let matrix_array = [5.7, 1.2, 0.0, 4.9, -7.1, -2.1, 77.1, 0.0, 9.1];
        let matrix = Matrix::square_matrix_from_list(&matrix_array.to_vec());

        let expected_solution = [
            0.53253570576405222074,
            0.72045539762075195146,
            -4.6218135070778490354,
        ];

        let b_vector = [3.9, 7.2, -1.0];
        let actual_solution = matrix.solve(&b_vector).unwrap();

        for i in 0..expected_solution.len() {
            assert!((expected_solution[i] - actual_solution[i] as f64).abs() < COMPARISON_TOLERANCE);
        }
    }

    #[test]
    fn least_squares_1() {
        let matrix_array = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

        let matrix = Matrix::square_matrix_from_list(&matrix_array.to_vec());
        let b_vector = [3.9, 7.2, -1.0];
        assert_eq!(
            matrix.least_squares_solution(&b_vector).unwrap_err(),
            "The system was inconsistent and there is no solution for b. (In this case, these means an arithmetic problem, probably due to floating point inaccuracy)."
        );
    }

    #[test]
    fn least_squares_2() {
        let matrix_array = [5.7, 1.2, 0.0, 4.9, -7.1, -2.1, 77.1, 0.0, 9.1];
        let matrix = Matrix::square_matrix_from_list(&matrix_array);

        let expected_solution = [
            0.53253570576405222074,
            0.72045539762075195146,
            -4.6218135070778490354,
        ];

        let b_vector = [3.9, 7.2, -1.0];
        let actual_solution = matrix.least_squares_solution(&b_vector).unwrap();

        for i in 0..expected_solution.len() {
            assert!((expected_solution[i] - actual_solution[i] as f64).abs() < COMPARISON_TOLERANCE);
        }
    }
}

#[cfg(test)]
#[allow(unused)]
mod complex_tests {
    use crate::complex::ComplexNumber;

    #[test]
    fn create_complex_number() {
        let mut z: ComplexNumber<f64> = ComplexNumber {
            real: 5.6,
            imaginary: 2.5,
        };
        assert_eq!(z.real, 5.6);
        assert_eq!(z.imaginary, 2.5);
        z += ComplexNumber {
            real: 5.6,
            imaginary: 2.5,
        };
        assert_eq!(z.real, 11.2);
    }

    #[test]
    fn complex_number_arithmetic() {
        let z1: ComplexNumber<f64> = ComplexNumber {
            real: 5.6,
            imaginary: 2.5,
        };
        let z2: ComplexNumber<f64> = ComplexNumber {
            real: -12.1,
            imaginary: 4.8,
        };

        let addition: ComplexNumber<f64> = z1 + z2;
        let subtraction: ComplexNumber<f64> = z1 - z2;
        let multiplication: ComplexNumber<f64> = z1 * z2;
        let division: ComplexNumber<f64> = z1 / z2;

        assert_eq!(addition.real, -6.5);
        assert_eq!(addition.imaginary, 7.3);

        assert_eq!(subtraction.real, 17.7);
        assert_eq!(subtraction.imaginary, -2.3);

        assert_eq!(multiplication.real, -79.759999999999991);
        assert_eq!(multiplication.imaginary, -3.3700000000000010);

        assert_eq!(division.real, -0.32906462083210386);
        assert_eq!(division.imaginary, -0.33714960165240482);
    }

    #[test]
    fn complex_number_parsing() {
        let z:ComplexNumber<f64> = "5.9 + 6.4i".parse().unwrap();

        assert_eq!(z.real, 5.9);
        assert_eq!(z.imaginary, 6.4);
    }

    #[test]
    fn complex_number_magnitude() {
        let z:ComplexNumber<f64> = "5.9 + 0i".parse().unwrap();

        assert_eq!(z.magnitude(), 5.9);
    }
}
