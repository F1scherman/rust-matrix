/// Brayden Jonsson, 2023
/// https://github.com/BraydenJonsson/rust-matrix
/// 
/// Contains tests for the matrix library
mod matrix;

#[cfg(test)]
mod tests {
    use crate::matrix::Matrix;

    #[test]
    fn it_works() {
        let a : Matrix = Matrix::new(4, 3);
        assert_eq!(a, Matrix::new(4, 3));
    }
}
