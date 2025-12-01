//! Audio capture and processing

mod capture;
mod resample;

pub use capture::{AudioCapture, AudioCaptureEvent};
pub use resample::{resample_to_16khz, stereo_to_mono};
