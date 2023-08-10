/// Brayden Jonsson, 2023
/// https://github.com/BraydenJonsson/rust-matrix
///
/// Contains a struct and methods for representing a complex number

trait_set! {
    pub trait ComplexCompatible = num_traits::NumAssign
    + num_traits::sign::Signed
    + Copy;
}

/// Represents a complex number
#[derive(Debug)]
pub struct complex_number<T>
where 
T: ComplexCompatible, {
    real : T,
    imaginary : T
}