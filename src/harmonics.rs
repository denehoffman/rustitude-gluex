use std::f64::consts::PI;
use std::fmt::Display;

use nalgebra::Vector3;
use num_complex::ComplexFloat;
use rayon::prelude::*;
use rustitude::prelude::*;
use sphrs::{ComplexSH, Coordinates, SHEval};

#[derive(Clone, Copy, Debug)]
pub enum Part {
    Real,
    Imag,
    Both,
}

#[derive(Clone, Copy, Default)]
#[rustfmt::skip]
pub enum Wave {
    #[default]
    S,
    S0,
    Pn1, P0, P1, P,
    Dn2, Dn1, D0, D1, D2, D,
    Fn3, Fn2, Fn1, F0, F1, F2, F3, F,
}

#[rustfmt::skip]
impl Wave {
    pub fn l(&self) -> i64 {
        match self {
            Self::S0 | Self::S => 0,
            Self::Pn1 | Self::P0 | Self::P1 | Self::P => 1,
            Self::Dn2 | Self::Dn1 | Self::D0 | Self::D1 | Self::D2 | Self::D => 2,
            Self::Fn3 | Self::Fn2 | Self::Fn1 | Self::F0 | Self::F1 | Self::F2 | Self::F3 | Self::F => 3,
        }
    }
    pub fn m(&self) -> i64 {
        match self {
            Self::S | Self::P | Self::D | Self::F => 0,
            Self::S0 | Self::P0 | Self::D0 | Self::F0 => 0,
            Self::Pn1 | Self::Dn1 | Self::Fn1 => -1,
            Self::P1 | Self::D1 | Self::F1 => 1,
            Self::Dn2 | Self::Fn2 => -2,
            Self::D2 | Self::F2 => 2,
            Self::Fn3 => -3,
            Self::F3 => 3,
        }
    }
}

impl Display for Wave {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let l_string = match self.l() {
            0 => "S",
            1 => "P",
            2 => "D",
            3 => "F",
            _ => unimplemented!(),
        };
        write!(f, "{} {:+}", l_string, self.m())
    }
}

pub enum Frame {
    Helicity,
    GottfriedJackson,
}

impl Frame {
    pub fn coordinates(
        &self,
        resonance: &FourMomentum,
        daughter: &FourMomentum,
        event: &Event,
    ) -> (Vector3<f64>, Vector3<f64>, Vector3<f64>, Coordinates<f64>) {
        match self {
            Frame::Helicity => {
                let daughter_vec = daughter.boost_along(resonance).momentum();
                let z = resonance.momentum().normalize();
                let y = event
                    .beam_p4
                    .momentum()
                    .cross(&(resonance.momentum()))
                    .normalize();
                let x = y.cross(&z);
                (
                    x,
                    y,
                    z,
                    Coordinates::cartesian(
                        daughter_vec.dot(&x),
                        daughter_vec.dot(&y),
                        daughter_vec.dot(&z),
                    ),
                )
            }
            Frame::GottfriedJackson => {
                let daughter_vec = daughter.boost_along(resonance).momentum();
                let z = event.beam_p4.boost_along(resonance).momentum().normalize();
                let y = event
                    .beam_p4
                    .momentum()
                    .cross(&(resonance.momentum()))
                    .normalize();
                let x = y.cross(&z);
                (
                    x,
                    y,
                    z,
                    Coordinates::cartesian(
                        daughter_vec.dot(&x),
                        daughter_vec.dot(&y),
                        daughter_vec.dot(&z),
                    ),
                )
            }
        }
    }
}

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

#[derive(Copy, Clone)]
pub enum Reflectivity {
    Positive = 1,
    Negative = -1,
}

impl Display for Reflectivity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Reflectivity::Positive => write!(f, "+"),
            Reflectivity::Negative => write!(f, "-"),
        }
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
