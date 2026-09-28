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

pub(crate) fn validate_dimensions<E>(
    checks: &[(&'static str, usize, usize)],
) -> Result<(), SolveError<E>> {
    for (context, expected, actual) in checks {
        validate_dimension(*context, *expected, *actual)?;
    }
    Ok(())
}
