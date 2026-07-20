use eunomia::FloatElement;
use leto_ops::{CsrMatrix, RealScalar};

pub(crate) fn nonsymmetric_matrix<T>() -> Result<CsrMatrix<T>, leto::LetoError>
where
    T: RealScalar + FloatElement,
{
    CsrMatrix::from_parts(
        vec![
            T::from_f64(4.0),
            T::from_f64(1.0),
            T::from_f64(2.0),
            T::from_f64(3.0),
            T::from_f64(1.0),
            T::from_f64(1.0),
            T::from_f64(2.0),
        ],
        vec![0, 1, 0, 1, 2, 1, 2],
        vec![0, 2, 5, 7],
        3,
        3,
    )
}
