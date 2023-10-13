/// Brayden Jonsson, 2023
/// https://github.com/BraydenJonsson/rust-matrix
///
/// Contains a struct and methods for representing a complex number

use bigdecimal::BigDecimal;
use gen_ops::gen_ops;
use num_traits;
use std::cmp;
use std::ops::Neg;
use std::str::FromStr;
use trait_set::trait_set;

pub trait ComplexFunctions {
    // required
    fn sqrt(&self) -> Self;
}

impl ComplexFunctions for f64 {
    fn sqrt(&self) -> Self {
        f64::sqrt(*self)
    }
}

impl ComplexFunctions for f32 {
    fn sqrt(&self) -> Self {
        f32::sqrt(*self)
    }
}

impl ComplexFunctions for BigDecimal {
    fn sqrt(&self) -> Self {
        BigDecimal::sqrt(&self).unwrap()
    }
}

trait_set! {
    pub trait ComplexCompatible = num_traits::NumAssign
    + num_traits::sign::Signed
    + Copy
    + ComplexFunctions
    + FromStr;
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

#[allow(unused)]
impl<T> ComplexNumber<T>
    where
        T: ComplexCompatible,
{
    // -------- CONSTRUCTORS ------------
    /// Returns a new Complex Number set to 0
    pub fn new() -> Self {
        Self {
            real: T::zero(),
            imaginary: T::zero(),
        }
    }

    /// Returns a Complex Number where the imaginary part is set to 1 and the real part is 0,
    /// meaning this number is equal to i
    pub fn i() -> Self {
        Self {
            real: T::zero(),
            imaginary: T::one(),
        }
    }

    // -------- PUBLIC METHODS ------------
    /// Returns the conjugate of this Complex Number
    pub fn conjugate(&self) -> Self {
        Self {
            real: self.real,
            imaginary: self.imaginary.neg(),
        }
    }

    /// Returns the magnitude of this Complex Number (treated as a vector)
    pub fn magnitude(&self) -> T {
        (self.real * self.real + self.imaginary * self.imaginary).sqrt()
    }

    /// Normalizes this vector to a magnitude of 1
    pub fn normalize(&self) -> Self {
        *self / self.magnitude()
    }

    /// Resizes this vector to the given magnitude
    pub fn resize(&self, magnitude: T) -> Self {
        self.normalize() * magnitude
    }
}

impl<T> num_traits::One for ComplexNumber<T>
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

impl<T> num_traits::Zero for ComplexNumber<T>
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

impl<T> num_traits::Num for ComplexNumber<T> where T: ComplexCompatible {
    type FromStrRadixErr = T::FromStrRadixErr;

    /// Only works on the real portion
    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        let result: Result<T, T::FromStrRadixErr> = T::from_str_radix(str, radix);
        let output: Result<ComplexNumber<T>, T::FromStrRadixErr>;

        if result.is_ok() {
            output = Ok(ComplexNumber {
                real: result.ok().unwrap(),
                imaginary: T::zero(),
            });
        } else {
            output = Err(result.err().unwrap());
        }

        output
    }
}

impl<T> Neg for ComplexNumber<T> where T: ComplexCompatible {
    type Output = ComplexNumber<T>;

    fn neg(self) -> ComplexNumber<T> {
        ComplexNumber {
            real: self.real.neg(),
            imaginary: self.imaginary.neg(),
        }
    }
}

impl<T> num_traits::sign::Signed for ComplexNumber<T> where T: ComplexCompatible {
    /// Equal to the magnitude/distance from the origin
    fn abs(&self) -> ComplexNumber<T> {
        ComplexNumber {
            real: self.magnitude(),
            imaginary: T::zero(),
        }
    }

    /// Compares magnitudes
    fn abs_sub(&self, other: &Self) -> ComplexNumber<T> {
        let a: ComplexNumber<T> = ComplexNumber {
            real: self.magnitude(),
            imaginary: T::zero(),
        };

        let b: ComplexNumber<T> = ComplexNumber {
            real: other.magnitude(),
            imaginary: T::zero(),
        };

        a - b
    }

    /// Equal to z/abs(z)
    fn signum(&self) -> Self {
        *self / self.abs()
    }

    /// Works only based on the real portion
    fn is_positive(&self) -> bool {
        self.real.is_positive()
    }

    /// Works only based on the real portion
    fn is_negative(&self) -> bool {
        self.real.is_negative()
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

    /// Find the remainder of two complex numbers
    ///
    /// This isn't well defined, we just take the % modulus during the final step of scaling by 1/(c^2+d^2) for (a+bi)/(c+di)
    ///
    /// Returns a complex number
    for % call |a: &ComplexNumber<T>, b: &ComplexNumber<T>| (*a * b.conjugate()) % (b.real * b.real + b.imaginary * b.imaginary);

    where T: ComplexCompatible
);

gen_ops!(
    <T>;
    types ComplexNumber<T>, T => ComplexNumber<T>;

    /// Scales a complex number
    ///
    /// Returns a complex number
    for * call |vector: &ComplexNumber<T>, scalar:&T| ComplexNumber {real: vector.real * *scalar, imaginary: vector.imaginary * *scalar};

    /// Scales a complex number
    ///
    /// Returns a complex number
    for / call |vector: &ComplexNumber<T>, scalar:&T| ComplexNumber {real: vector.real / *scalar, imaginary: vector.imaginary / *scalar};

    /// This applies the modulus operator to both parts of the number simulataneously
    ///
    /// Returns a complex number
    for % call |vector: &ComplexNumber<T>, scalar:&T| ComplexNumber {real: vector.real % *scalar, imaginary: vector.imaginary % *scalar};
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

impl<T> cmp::PartialEq for ComplexNumber<T>
    where
        T: ComplexCompatible,
{
    /// Check if two complex numbers are equal
    ///
    /// Returns boolean
    fn eq(&self, other: &Self) -> bool {
        self.real == other.real && self.imaginary == other.imaginary
    }
}

impl<T> Clone for ComplexNumber<T>
    where
        T: ComplexCompatible,
{
    fn clone(&self) -> Self {
        Self {
            real: self.real.clone(),
            imaginary: self.imaginary.clone(),
        }
    }
}

#[derive(Debug)]
pub struct ParseComplexNumberError;

impl<T> FromStr for ComplexNumber<T>
    where T:ComplexCompatible,
{
    type Err = ParseComplexNumberError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let parts: Vec<&str> = s.split(&['+','i']).collect();

        let real_result = parts[0].trim().parse::<T>();
        let imaginary_result = parts[1].trim().parse::<T>();

        if real_result.is_err() || imaginary_result.is_err() {
            Err(ParseComplexNumberError)
        } else {
            Ok(
                Self {
                    real: real_result.unwrap_or(T::zero()),
                    imaginary: imaginary_result.unwrap_or(T::zero()),
                }
            )
        }
    }
}
