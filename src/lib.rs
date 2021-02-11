#[macro_use]
extern crate vst;
extern crate rand;

mod sinc;

use std::collections::VecDeque;
use std::f64::consts::PI;
use std::sync::Arc;

use vst::api::{Events, Supported};
use vst::buffer::AudioBuffer;
use vst::event::Event;
use vst::plugin::{CanDo, Category, Info, Plugin, PluginParameters};

use Rust_WORLD::rsworld::{cheaptrick, d4c, dio, stonemask, synthesis};
use Rust_WORLD::rsworld_sys::{CheapTrickOption, D4COption, DioOption};

const BUFFER_SIZE: usize = 1024 * 16;
const COMPOSE_SIZE: usize = 1024;

// for note input
pub const TAU: f64 = PI * 2.0;
fn midi_pitch_to_freq(pitch: u8) -> f64 {
    const A4_PITCH: i8 = 69;
    const A4_FREQ: f64 = 440.0;

    // Midi notes can be 0-127
    ((f64::from(pitch as i8 - A4_PITCH)) / 12.).exp2() * A4_FREQ
}

struct VoiceChange {
    sample_rate: i32,
    params: Arc<VoiceChangeParameters>,
    source_buffer: VecDeque<f64>,
    compose_buffer: Vec<f64>,
    dc_offset: f64,
    comp_level: f64,
    notes: u8,
    pitch: u8,
}

impl Default for VoiceChange {
    fn default() -> Self {
        Self {
            sample_rate: 44100,
            params: Arc::new(VoiceChangeParameters::default()),
            source_buffer: VecDeque::default(),
            compose_buffer: vec![0f64; COMPOSE_SIZE],
            dc_offset: 0f64,
            comp_level: 1f64,
            notes: 0,
            pitch: 0,
        }
    }
}

impl VoiceChange {
    #[allow(dead_code)]
    fn time_per_sample(&self) -> f64 {
        f64::from(1.0 / self.sample_rate as f64)
    }

    fn process_midi_event(&mut self, data: [u8; 3]) {
        match data[0] {
            // if note on, increment our counter
            144 => self.notes += 1u8,
            // if note off, decrement our counter
            128 => self.notes -= 1u8,
            _ => (),
        }
        self.pitch = data[1];
        // self.velocity = data[2];
    }

    fn stream_process(&mut self, samples: usize, input: Vec<f64>) -> Vec<f64> {
        // Fetch parameters
        let f0_rate = self.params.get_f0_rate();
        let f0_sqr_rate = self.params.get_f0_exp();
        let sp_rate = self.params.get_sp_rate();
        let sp_limit = self.params.get_sp_limit();
        let pitch_freq = if self.notes == 0 {
            0.0
        } else {
            midi_pitch_to_freq(self.pitch)
        };

        // Buffering
        let mut buffer = self.source_buffer.clone();
        if buffer.capacity() < BUFFER_SIZE + samples {
            buffer.reserve(BUFFER_SIZE + samples - buffer.capacity());
            buffer.resize(BUFFER_SIZE, 0f64);
        }
        buffer.append(&mut VecDeque::from(input));
        let buffer_size = buffer.len();
        if buffer_size > BUFFER_SIZE {
            buffer.drain(0..buffer_size - BUFFER_SIZE);
        }
        let compose_idx = buffer.len() - COMPOSE_SIZE;
        let dest_idx = compose_idx - samples;
        // self.buffer.replace(buffer);
        self.source_buffer = buffer;

        // Analyze voice features
        let x: Vec<f64> = self.source_buffer.iter().cloned().collect();
        let dio_option = DioOption::new();
        let (tpos, f0) = dio(&x, self.sample_rate, &dio_option);
        let mut f0 = stonemask(&x, self.sample_rate, &tpos, &f0);
        let mut ct_option = CheapTrickOption::new(self.sample_rate);
        let sp = cheaptrick(&x, self.sample_rate, &tpos, &f0, &mut ct_option);
        let d4c_option = D4COption::new();
        let ap = d4c(&x, self.sample_rate, &tpos, &f0, &d4c_option);

        // self.params.debug.set((f0[0] * 0.0001) as f32);

        // change voice here!
        let mut last_freq = 100.0;
        for fi in f0.iter_mut() {
            *fi = if self.notes > 0 {
                pitch_freq
            } else {
                if *fi > 100.0 {
                    last_freq = ((*fi - 100.0).powf(f0_sqr_rate) + 100.0) * f0_rate;
                }
                last_freq
            };
        }

        let sp2: Vec<Vec<f64>> = sp
            .iter()
            .map(|frm| {
                let sp_limit = (frm.len() as f64 * sp_limit).floor() as usize;
                let sp_range = (frm.len() as f64 * sp_rate).floor() as usize;
                frm.iter()
                    .enumerate()
                    .map(|(f, elem)| -> f64 {
                        if f < sp_range && f < sp_limit {
                            // warn: conv_size is 0, so linear interpolating
                            sinc::interpolate(frm, f as f64 / sp_rate, 0f64)
                        } else {
                            *elem
                        }
                    })
                    .collect()
            })
            .collect();

        // let ap2: Vec<Vec<f64>> = ap
        //     .iter()
        //     .map(|frm| {
        //         let ap_range = frm.len() as f64;
        //         frm.iter()
        //             .enumerate()
        //             .map(|(f, elem)| -> f64 { *elem * (f as f64 / ap_range).powf(0.1) })
        //             // .map(|(f, elem)| -> f64 { *elem * (f as f64 / ap_range) })
        //             // .map(|(f, elem)| -> f64 { 0f64 })
        //             .collect()
        //     })
        //     .collect();

        // synthesize voice
        let mut out = synthesis(&f0, &sp2, &ap, dio_option.frame_period, self.sample_rate);
        let mut dc_offset = self.dc_offset;
        for y in &mut out {
            dc_offset = 0.98 * dc_offset + 0.02 * *y;
            *y -= dc_offset;
        }

        // let mut dc_offset = self.dc_offset;
        let mut comp_level = self.comp_level;
        let dest: Vec<f64> = out[dest_idx..compose_idx]
            .iter()
            .enumerate()
            .map(|(i, y)| -> f64 {
                // composing
                let mut yout = if i < COMPOSE_SIZE {
                    let ratio = i as f64 / COMPOSE_SIZE as f64;
                    let prev_y = (1.0f64 - ratio) * self.compose_buffer[i];
                    let curr_y = ratio * *y;
                    prev_y + curr_y
                } else {
                    *y
                };
                // remove dc offset (easy-algo..)
                // TODO: improve dc cutter
                // dc_offset = 0.98 * dc_offset + 0.02 * yout;
                // yout = yout - dc_offset;
                // compressor (likely hard limitter)
                comp_level = f64::min(0.1f64 + 0.9f64 * comp_level, 1f64 / yout.abs());
                yout *= comp_level;
                return yout;
            })
            .collect();
        self.compose_buffer = out[compose_idx..out.len()].iter().cloned().collect();
        self.dc_offset = dc_offset;
        self.comp_level = comp_level;
        return dest;
    }
}

struct VoiceChangeParameters {
    // Must be f32 params and range is 0~1
    gain: vst::util::AtomicFloat,
    f0_rate: vst::util::AtomicFloat,
    sp_rate: vst::util::AtomicFloat,
    sp_limit: vst::util::AtomicFloat,
    f0_exp: vst::util::AtomicFloat,
}

impl VoiceChangeParameters {
    fn get_f0_rate(&self) -> f64 {
        2.0f32.powf(2.0 * self.f0_rate.get() - 1.0) as f64
    }
    fn get_sp_rate(&self) -> f64 {
        (0.75 + self.sp_rate.get()) as f64
    }
    fn get_sp_limit(&self) -> f64 {
        self.sp_limit.get() as f64
    }
    fn get_f0_exp(&self) -> f64 {
        (0.4 * self.f0_exp.get() + 0.8) as f64
    }
}

impl Default for VoiceChangeParameters {
    fn default() -> Self {
        Self {
            gain: vst::util::AtomicFloat::new(1.0),
            f0_rate: vst::util::AtomicFloat::new(0.5),
            sp_rate: vst::util::AtomicFloat::new(0.25),
            sp_limit: vst::util::AtomicFloat::new(1.0),
            f0_exp: vst::util::AtomicFloat::new(0.5),
        }
    }
}

impl PluginParameters for VoiceChangeParameters {
    fn get_parameter_label(&self, index: i32) -> String {
        match index {
            0 => "x".to_string(),
            1 => "x".to_string(),
            2 => "x".to_string(),
            3 => "x".to_string(),
            _ => "".to_string(),
        }
    }
    fn get_parameter_text(&self, index: i32) -> String {
        match index {
            0 => format!("{:.3}", self.gain.get()),
            1 => format!("{:.3}", self.get_f0_rate()),
            2 => format!("{:.3}", self.get_sp_rate()),
            3 => format!("{:.3}", self.get_sp_limit()),
            4 => format!("{:.3}", self.get_f0_exp()),
            _ => format!(""),
        }
    }
    fn get_parameter_name(&self, index: i32) -> String {
        match index {
            0 => "Output Gain".to_string(),
            1 => "Pitch Shift".to_string(),
            2 => "Formant Shift".to_string(),
            3 => "Formant Effective Range".to_string(),
            4 => "Intonation".to_string(),
            _ => "".to_string(),
        }
    }
    fn get_parameter(&self, index: i32) -> f32 {
        match index {
            0 => self.gain.get(),
            1 => self.f0_rate.get(),
            2 => self.sp_rate.get(),
            3 => self.sp_limit.get(),
            4 => self.f0_exp.get(),
            _ => 0.0,
        }
    }
    fn set_parameter(&self, index: i32, value: f32) {
        match index {
            0 => self.gain.set(value),
            1 => self.f0_rate.set(value),
            2 => self.sp_rate.set(value),
            3 => self.sp_limit.set(value),
            4 => self.f0_exp.set(value),
            _ => (),
        }
    }
}

impl Plugin for VoiceChange {
    fn get_info(&self) -> Info {
        Info {
            name: "Voice Changer".to_string(),
            vendor: "sabamotto".to_string(),
            unique_id: 0x300001,
            version: 102,
            category: Category::Effect,

            parameters: 5,
            inputs: 1,
            outputs: 1,
            f64_precision: true,
            silent_when_stopped: true,

            ..Info::default()
        }
    }

    fn process_events(&mut self, events: &Events) {
        for event in events.events() {
            match event {
                Event::Midi(ev) => self.process_midi_event(ev.data),
                _ => (),
            }
        }
    }

    fn set_sample_rate(&mut self, rate: f32) {
        self.sample_rate = rate as i32
    }

    fn process(&mut self, buffer: &mut AudioBuffer<f32>) {
        let samples = buffer.samples();
        let (input_buffer, mut output_buffer) = buffer.split();
        let input = input_buffer.get(0);
        let output = output_buffer.get_mut(0);
        let gain = self.params.gain.get() as f64;

        // convert floating precision
        let input_vec: Vec<f64> = input.iter().map(|&y| y as f64).collect();
        let synthesized = self.stream_process(samples, input_vec);
        for idx in 0..samples {
            output[idx] = (gain * synthesized[idx]) as f32;
        }
    }

    fn process_f64(&mut self, buffer: &mut AudioBuffer<f64>) {
        let samples = buffer.samples();
        let (input_buffer, mut output_buffer) = buffer.split();
        let input = input_buffer.get(0);
        let output = output_buffer.get_mut(0);
        let gain = self.params.gain.get() as f64;

        let synthesized = self.stream_process(samples, input.to_vec());
        for idx in 0..samples {
            output[idx] = gain * synthesized[idx];
        }
    }

    fn can_do(&self, can_do: CanDo) -> Supported {
        match can_do {
            CanDo::ReceiveMidiEvent => Supported::Yes,
            _ => Supported::Maybe,
        }
    }

    fn get_parameter_object(&mut self) -> Arc<dyn PluginParameters> {
        Arc::clone(&self.params) as Arc<dyn PluginParameters>
    }
}

plugin_main!(VoiceChange);
