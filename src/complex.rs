/// Brayden Jonsson, 2023
/// https://github.com/BraydenJonsson/rust-matrix
///
/// Contains a struct and methods for representing a complex number
use gen_ops::gen_ops;
use num_traits::One;
use num_traits::Zero;
use trait_set::trait_set;
use std::cmp;

trait_set! {
    pub trait ComplexCompatible = num_traits::NumAssign
    + num_traits::sign::Signed
    + Copy;
}

/// Represents a complex number
#[derive(Debug, Copy)]
pub struct ComplexNumber<T>
where
    T: ComplexCompatible,
{
    pub real: T,
    pub imaginary: T,
}

impl<T> ComplexNumber<T>
where
    T: ComplexCompatible,
{
    pub fn conjugate(&self) -> Self {
        Self {
            real: self.real,
            imaginary: self.imaginary.neg(),
        }
    }
}

impl<T> One for ComplexNumber<T>
where
    T: ComplexCompatible,
{
    fn one() -> ComplexNumber<T> {
        ComplexNumber {
            real: T::one(),
            imaginary: T::zero(),
        }
    }
}

impl<T> Zero for ComplexNumber<T>
where
    T: ComplexCompatible,
{
    fn zero() -> ComplexNumber<T> {
        ComplexNumber {
            real: T::zero(),
            imaginary: T::zero(),
        }
    }

    fn is_zero(&self) -> bool {
        *self == Self::zero()
    }
}

gen_ops!(
    <T>;
    types ComplexNumber<T>, ComplexNumber<T> => ComplexNumber<T>;

    /// Add two complex numbers
    ///
    /// Returns a complex number
    for + call |a: &ComplexNumber<T>, b: &ComplexNumber<T>| ComplexNumber { real: a.real + b.real, imaginary: a.imaginary + b.imaginary};

    /// Subtract two complex numbers
    ///
    /// Returns a complex number
    for - call |a: &ComplexNumber<T>, b: &ComplexNumber<T>| ComplexNumber{ real: a.real - b.real, imaginary: a.imaginary - b.imaginary};

    /// Multiply two complex numbers
    ///
    /// Returns a complex number
    for * call |a: &ComplexNumber<T>, b: &ComplexNumber<T>| ComplexNumber{real: a.real * b.real - a.imaginary * b.imaginary, imaginary: a.real * b.imaginary + a.imaginary * b.real};

    /// Divide two complex numbers
    ///
    /// Returns a complex number
    for / call |a: &ComplexNumber<T>, b: &ComplexNumber<T>| (*a * b.conjugate()) / (b.real * b.real + b.imaginary * b.imaginary);

    where T: ComplexCompatible
);

gen_ops!(
    <T>;
    types ComplexNumber<T>, T => ComplexNumber<T>;

    /// Scales a complex number
    ///
    /// Returns a complex number
    for * call |vector: &ComplexNumber<T>, scalar:&T| ComplexNumber {real: vector.real * *scalar, imaginary: vector.imaginary * *scalar};
    
    where T: ComplexCompatible
);

gen_ops!(
    <T>;
    types ComplexNumber<T>, T => ComplexNumber<T>;

    /// Scales a complex number
    ///
    /// Returns a complex number
    for / call |vector: &ComplexNumber<T>, scalar:&T| ComplexNumber {real: vector.real / *scalar, imaginary: vector.imaginary / *scalar};

    where T: ComplexCompatible
);

gen_ops!(
    <T>;
    types ComplexNumber<T>, ComplexNumber<T>;

    /// Add assign two complex numbers
    for += call |a: &mut ComplexNumber<T>, b: &ComplexNumber<T>| (*a = a.clone() + *b);
    /// Sub assign two complex numbers
    for -= call |a: &mut ComplexNumber<T>, b: &ComplexNumber<T>| (*a = a.clone() - *b);
    /// Multiply assign two complex numbers
    for *= call |a: &mut ComplexNumber<T>, b: &ComplexNumber<T>| (*a = a.clone() * *b);
    /// Divide assign two complex numbers
    for /= call |a: &mut ComplexNumber<T>, b: &ComplexNumber<T>| (*a = a.clone() / *b);

    where T: ComplexCompatible
);

gen_ops!(
    <T>;
    types ComplexNumber<T>, T;

    /// Multiply assign a complex number with a scalar
    for *= call |a: &mut ComplexNumber<T>, b: &T| (*a = a.clone() * *b);

    /// Divide assign a complex number with a scalar
    for /= call |a: &mut ComplexNumber<T>, b: &T| (*a = a.clone() / *b);

    where T: ComplexCompatible
);

impl<T> cmp::PartialEq for ComplexNumber<T> where T: ComplexCompatible {
    /// Check if two complex numbers are equal
    ///
    /// Returns boolean
    fn eq(&self, other: &Self) -> bool {
        self.real == other.real && self.imaginary == other.imaginary
    }
}

impl<T> Clone for ComplexNumber<T> where T: ComplexCompatible {
    fn clone(&self) -> Self {
        Self { real: self.real.clone(), imaginary: self.imaginary.clone() }
    }
}
