//! CPU linear operators.

mod csr;
mod dense;
mod rectangular_csr;

pub use csr::CsrOperator;
pub use dense::BorrowedDenseOperator;
pub use rectangular_csr::RectangularCsrOperator;
