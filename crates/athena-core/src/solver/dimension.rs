use crate::SolveError;

pub(crate) fn validate_dimension<E>(
    context: &'static str,
    expected: usize,
    actual: usize,
) -> Result<(), SolveError<E>> {
    if expected == actual {
        Ok(())
    } else {
        Err(SolveError::DimensionMismatch {
            context,
            expected,
            actual,
        })
    }
}
