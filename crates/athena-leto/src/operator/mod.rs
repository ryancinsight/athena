//! CPU linear operators.

mod borrowed_csr;
mod csr;
mod dense;
mod rectangular_csr;
mod stencil;

pub use borrowed_csr::BorrowedCsrOperator;
pub use csr::CsrOperator;
pub use dense::BorrowedDenseOperator;
pub use rectangular_csr::RectangularCsrOperator;
pub use stencil::PoissonStencilOperator;
