/// Brayden Jonsson, 2023
/// https://github.com/BraydenJonsson/rust-matrix
///
/// Contains a struct and methods for representing a complex number

#[macro_use] extern crate gen_ops;
use std::ops;

trait_set! {
    pub trait ComplexCompatible = num_traits::NumAssign
    + num_traits::sign::Signed
    + Copy;
}

/// Represents a complex number
#[derive(Debug)]
pub struct ComplexNumber<T>
where
    T: ComplexCompatible,
{
    real: T,
    imaginary: T,
}

impl<T> ComplexNumber<T>
where T: ComplexCompatible, {

    pub fn conjugate (&self) -> Self {
        Self {
            real,
            imaginary.neg(),
        }
    }
}

gen_ops!(
    <T>;
    types ComplexNumber<T>, ComplexNumber<T> => ComplexNumber<T>;

    /// Add two complex numbers
    /// 
    /// Returns a complex number
    for + call |a: &ComplexNumber<T>, b: &ComplexNumber<T>| ComplexNumber(a.real + b.real, a.imaginary + b.imaginary);

    /// Subtract two complex numbers
    /// 
    /// Returns a complex number
    for - call |a: &ComplexNumber<T>, b: &ComplexNumber<T>| ComplexNumber(a.real - b.real, a.imaginary - b.imaginary);

    /// Multiply two complex numbers
    /// 
    /// Returns a complex number
    for * call |a: &ComplexNumber<T>, b: &ComplexNumber<T>| ComplexNumber(a.real * b.real - a.imaginary * b.imaginary, a.real * b.imaginary + a.imaginary * b.real);

    /// Divide two complex numbers
    /// 
    /// Returns a complex number
    for / call |a: &ComplexNumber<T>, b: &ComplexNumber<T>| (a * b.conjugate()) / (b.real * b.real + b.imaginary * b.imaginary);

    where T: ComplexCompatible
);

gen_ops_ex!(
    <T>;
    types ComplexNumber<T>, T => ComplexNumber<T>;

    /// Scales a complex number
    /// 
    /// Returns a complex number
    for / call |vector: &ComplexNumber<T>, scalar:&T| ComplexNumber(vector.real / *scalar, vector.imaginary / *scalar);

    where T: ComplexCompatible
);

gen_ops_ex!(
    <T>;
    types T, ComplexNumber<T> => ComplexNumber<T>;

    /// Divides the given real number by a complex number
    /// 
    /// Returns a complex number
    for / call |real:&T, complex:&ComplexNumber| (complex.conjugate() * real) / (complex.real * complex.real + complex.imaginary * complex.imaginary);

    where T: ComplexCompatible
);

gen_ops_comm_ex!(
    <T>;
    types ComplexNumber<T>, T => ComplexNumber<T>;

    /// Scales a complex number
    /// 
    /// Returns a complex number
    for * call |vector: &ComplexNumber<T>, scalar:&T| ComplexNumber(vector.real * *scalar, vector.imaginary * *scalar);

    where T: ComplexCompatible
);
