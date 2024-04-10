use std::f64::consts::PI;

use num_complex::ComplexFloat;
use rayon::prelude::*;
use rustitude::prelude::*;
use sphrs::{ComplexSH, SHEval};

use crate::utils::{Frame, Part, Reflectivity, Wave};

pub struct Ylm {
    wave: Wave,
    frame: Frame,
    data: Vec<Complex64>,
}
impl Ylm {
    pub fn new(wave: Wave, frame: Frame) -> Self {
        Self {
            wave,
            frame,
            data: Vec::default(),
        }
    }
}
impl Node for Ylm {
    fn parameters(&self) -> Option<Vec<String>> {
        None
    }

    fn precalculate(&mut self, dataset: &Dataset) {
        self.data = dataset
            .par_iter()
            .map(|event| {
                let resonance = event.daughter_p4s[0] + event.daughter_p4s[1];
                let daughter = event.daughter_p4s[0];
                let (_, _, _, p) = self.frame.coordinates(&resonance, &daughter, event);
                ComplexSH::Spherical.eval(self.wave.l(), self.wave.m(), &p)
            })
            .collect();
    }

    fn calculate(&self, _parameters: &[f64], event: &Event) -> Complex64 {
        self.data[event.index]
    }
}

pub struct Zlm {
    wave: Wave,
    reflectivity: Reflectivity,
    part: Part,
    frame: Frame,
    data: Vec<Complex64>,
}
impl Zlm {
    pub fn new(wave: Wave, reflectivity: Reflectivity, part: Part, frame: Frame) -> Self {
        Self {
            wave,
            reflectivity,
            part,
            frame,
            data: Vec::default(),
        }
    }
}
impl Node for Zlm {
    fn precalculate(&mut self, dataset: &Dataset) {
        self.data = dataset
            .par_iter()
            .map(|event| {
                let resonance = event.daughter_p4s[0] + event.daughter_p4s[1];
                let daughter = event.daughter_p4s[0];
                let (_, y, _, p) = self.frame.coordinates(&resonance, &daughter, event);
                let ylm = ComplexSH::Spherical.eval(self.wave.l(), self.wave.m(), &p);
                let big_phi = y.dot(&event.eps).atan2(
                    event
                        .beam_p4
                        .momentum()
                        .normalize()
                        .dot(&event.eps.cross(&y)),
                );
                let pgamma = event.eps.norm();

                let phase = Complex64::cis(-big_phi);
                let zlm = ylm * phase;
                let zlm_part: Complex64 = match self.part {
                    Part::Real => zlm.re.into(),
                    Part::Imag => zlm.im.into(),
                    Part::Both => zlm,
                };
                match self.reflectivity {
                    Reflectivity::Positive => (1.0 + pgamma).sqrt() * zlm_part,
                    Reflectivity::Negative => (1.0 - pgamma).sqrt() * zlm_part,
                }
            })
            .collect()
    }
    fn calculate(&self, _parameters: &[f64], event: &Event) -> Complex64 {
        self.data[event.index]
    }
    fn parameters(&self) -> Option<Vec<String>> {
        None
    }
}

pub struct OnePS {
    reflectivity: Reflectivity,
    part: Part,
    frame: Frame,
    data: Vec<Complex64>,
}
impl OnePS {
    pub fn new(reflectivity: Reflectivity, part: Part, frame: Frame) -> Self {
        Self {
            reflectivity,
            part,
            frame,
            data: Vec::default(),
        }
    }
}
impl Node for OnePS {
    fn precalculate(&mut self, dataset: &Dataset) {
        self.data = dataset
            .par_iter()
            .map(|event| {
                let resonance = event.daughter_p4s[0] + event.daughter_p4s[1];
                let daughter = event.daughter_p4s[0];
                let (_, y, _, _) = self.frame.coordinates(&resonance, &daughter, event);
                let pol_angle = event.eps[0].acos();
                let big_phi = y.dot(&event.eps).atan2(
                    event
                        .beam_p4
                        .momentum()
                        .normalize()
                        .dot(&event.eps.cross(&y)),
                );
                let pgamma = event.eps.norm();
                let phase = Complex64::cis(-(pol_angle + big_phi));
                let phase_part: Complex64 = match self.part {
                    Part::Real => phase.re.into(),
                    Part::Imag => phase.im.into(),
                    Part::Both => phase,
                };
                match self.reflectivity {
                    Reflectivity::Positive => (1.0 + pgamma).sqrt() * phase_part,
                    Reflectivity::Negative => (1.0 - pgamma).sqrt() * phase_part,
                }
            })
            .collect()
    }

    fn calculate(&self, _parameters: &[f64], event: &Event) -> Complex64 {
        self.data[event.index]
    }

    fn parameters(&self) -> Option<Vec<String>> {
        None
    }
}

pub struct TwoPS {
    wave: Wave,
    reflectivity: Reflectivity,
    part: Part,
    frame: Frame,
    data: Vec<Complex64>,
}
impl TwoPS {
    pub fn new(wave: Wave, reflectivity: Reflectivity, part: Part, frame: Frame) -> Self {
        Self {
            wave,
            reflectivity,
            part,
            frame,
            data: Vec::default(),
        }
    }
}
impl Node for TwoPS {
    fn precalculate(&mut self, dataset: &Dataset) {
        self.data = dataset
            .par_iter()
            .map(|event| {
                let resonance = event.daughter_p4s[0] + event.daughter_p4s[1];
                let daughter = event.daughter_p4s[0];
                let (_, _, _, p) = self.frame.coordinates(&resonance, &daughter, event);
                let ylm_p = ComplexSH::Spherical
                    .eval(self.wave.l(), self.wave.m(), &p)
                    .conj();
                let ylm_m = ComplexSH::Spherical
                    .eval(self.wave.l(), -self.wave.m(), &p)
                    .conj();
                let m_refl = (if self.wave.m() % 2 == 0 {
                    self.reflectivity as isize
                } else {
                    -(self.reflectivity as isize)
                }) as f64;
                let big_theta = match self.wave.m().cmp(&0) {
                    std::cmp::Ordering::Less => 0.0,
                    std::cmp::Ordering::Equal => 0.5,
                    std::cmp::Ordering::Greater => f64::sqrt(0.5),
                };
                let wigner_d_lm0_m =
                    f64::sqrt(4.0 * PI / (2.0 * self.wave.l() as f64 + 1.0)) * ylm_m;
                let amp = big_theta * ylm_p - m_refl * wigner_d_lm0_m;
                let amp_part: Complex64 = match self.part {
                    Part::Real => amp.re.into(),
                    Part::Imag => amp.im.into(),
                    Part::Both => amp,
                };
                amp_part
            })
            .collect()
    }

    fn calculate(&self, _parameters: &[f64], event: &Event) -> Complex64 {
        self.data[event.index]
    }

    fn parameters(&self) -> Option<Vec<String>> {
        None
    }
}
