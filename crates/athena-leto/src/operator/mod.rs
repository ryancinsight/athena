//! CPU linear operators.

mod csr;
mod dense;

pub use csr::CsrOperator;
pub use dense::BorrowedDenseOperator;
