use base_types::Algorithm;
use crate::Error;

pub(crate) fn validate(alg: &Algorithm) -> Result<(), Error> {
    // Check memory layout doesn't overlap
    let mut ranges = vec![];
    for offset in &alg.state.unit_scratch_offsets {
        ranges.push(*offset..*offset + alg.state.unit_scratch_size);
    }
    ranges.push(
        alg.state.shared_data_offset..alg.state.shared_data_offset + alg.state.shared_data_size,
    );
    ranges.push(alg.state.gpu_offset..alg.state.gpu_offset + alg.state.gpu_size);

    ranges.sort_by_key(|r| r.start);
    for window in ranges.windows(2) {
        if window[0].end > window[1].start {
            return Err(Error::InvalidConfig("Memory regions overlap".into()));
        }
    }

    // Check assignments are valid
    for &assignment in &alg.simd_assignments {
        if assignment != 255 && assignment as usize >= alg.units.simd_units {
            return Err(Error::InvalidConfig("Invalid unit assignment".into()));
        }
    }

    // Check scratch offsets match unit count
    if alg.state.unit_scratch_offsets.len() != alg.units.simd_units {
        return Err(Error::InvalidConfig(
            "Scratch offsets count doesn't match SIMD units".into(),
        ));
    }

    Ok(())
}
