//! Microphone audio capture using cpal

use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::sync::mpsc::{self, Receiver};
use std::sync::{Arc, Mutex};

/// Events from audio capture
#[derive(Debug)]
pub enum AudioCaptureEvent {
    /// Audio samples received
    Samples(Vec<f32>),
    /// Capture error
    Error(String),
    /// Capture stopped
    Stopped,
}

/// Audio capture from the default input device
pub struct AudioCapture {
    stream: Option<cpal::Stream>,
    sample_rate: u32,
    receiver: Receiver<AudioCaptureEvent>,
    buffer: Arc<Mutex<Vec<f32>>>,
}

impl AudioCapture {
    /// Create a new audio capture instance
    pub fn new() -> Result<Self> {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .context("No input device available")?;

        tracing::info!("Using input device: {}", device.name().unwrap_or_default());

        let config = device
            .default_input_config()
            .context("Failed to get default input config")?;

        let sample_rate = config.sample_rate().0;
        tracing::info!("Input sample rate: {} Hz", sample_rate);

        let (_sender, receiver) = mpsc::channel();
        let buffer = Arc::new(Mutex::new(Vec::new()));

        Ok(Self {
            stream: None,
            sample_rate,
            receiver,
            buffer,
        })
    }

    /// Get the input sample rate
    pub fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    /// Start recording audio
    pub fn start(&mut self) -> Result<()> {
        let host = cpal::default_host();
        let device = host
            .default_input_device()
            .context("No input device available")?;

        let config = device.default_input_config()?;
        let sample_format = config.sample_format();
        let config: cpal::StreamConfig = config.into();

        let buffer = Arc::clone(&self.buffer);
        let (sender, receiver) = mpsc::channel();
        self.receiver = receiver;

        let err_sender = sender.clone();

        let stream = match sample_format {
            cpal::SampleFormat::F32 => device.build_input_stream(
                &config,
                move |data: &[f32], _: &_| {
                    let mut buf = buffer.lock().unwrap();
                    buf.extend_from_slice(data);
                },
                move |err| {
                    let _ = err_sender.send(AudioCaptureEvent::Error(err.to_string()));
                },
                None,
            )?,
            cpal::SampleFormat::I16 => {
                let buffer = Arc::clone(&self.buffer);
                device.build_input_stream(
                    &config,
                    move |data: &[i16], _: &_| {
                        let samples: Vec<f32> = data
                            .iter()
                            .map(|&s| s as f32 / i16::MAX as f32)
                            .collect();
                        let mut buf = buffer.lock().unwrap();
                        buf.extend_from_slice(&samples);
                    },
                    move |err| {
                        let _ = sender.send(AudioCaptureEvent::Error(err.to_string()));
                    },
                    None,
                )?
            }
            _ => anyhow::bail!("Unsupported sample format: {:?}", sample_format),
        };

        stream.play()?;
        self.stream = Some(stream);
        self.buffer.lock().unwrap().clear();

        tracing::info!("Audio capture started");
        Ok(())
    }

    /// Stop recording and return captured samples
    pub fn stop(&mut self) -> Result<Vec<f32>> {
        if let Some(stream) = self.stream.take() {
            drop(stream);
        }

        let samples = std::mem::take(&mut *self.buffer.lock().unwrap());
        tracing::info!("Audio capture stopped, {} samples", samples.len());
        Ok(samples)
    }

    /// Check if currently recording
    pub fn is_recording(&self) -> bool {
        self.stream.is_some()
    }

    /// Get events from the capture (non-blocking)
    pub fn try_recv(&self) -> Option<AudioCaptureEvent> {
        self.receiver.try_recv().ok()
    }
}

impl Drop for AudioCapture {
    fn drop(&mut self) {
        let _ = self.stop();
    }
}
