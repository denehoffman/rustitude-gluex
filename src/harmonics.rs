use std::fmt::Display;

use num_complex::Complex64;
use rayon::prelude::*;
use rustitude::prelude::*;
use sphrs::SHEval;
use sphrs::{ComplexSH, Coordinates};

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

pub struct Ylm(Wave, Vec<Complex64>);
impl Ylm {
    pub fn new(wave: Wave) -> Self {
        Self(wave, Vec::default())
    }
}
impl Node for Ylm {
    fn parameters(&self) -> Option<Vec<String>> {
        None
    }

    fn precalculate(&mut self, dataset: &Dataset) {
        self.1 = dataset
            .par_iter()
            .map(|event| {
                let resonance = event.daughter_p4s[0] + event.daughter_p4s[1];
                let p1 = event.daughter_p4s[0];
                let recoil_res = event.recoil_p4.boost_along(&resonance);
                let p1_res = p1.boost_along(&resonance);
                let z = -1.0 * recoil_res.momentum().normalize();
                let y = event
                    .beam_p4
                    .momentum()
                    .cross(&(-1.0 * event.recoil_p4.momentum()));
                let x = y.cross(&z);
                let p1_vec = p1_res.momentum();
                let p = Coordinates::cartesian(p1_vec.dot(&x), p1_vec.dot(&y), p1_vec.dot(&z));
                ComplexSH::Spherical.eval(self.0.l(), self.0.m(), &p)
            })
            .collect();
    }

    fn calculate(&self, _parameters: &[f64], event: &Event) -> Complex64 {
        self.1[event.index]
    }
}

pub enum Reflectivity {
    Positive,
    Negative,
}

impl Display for Reflectivity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Reflectivity::Positive => write!(f, "+"),
            Reflectivity::Negative => write!(f, "-"),
        }
    }
}

pub struct ReZlm(pub Wave, pub Reflectivity, Vec<f64>);
impl ReZlm {
    pub fn new(wave: Wave, reflectivity: Reflectivity) -> Self {
        Self(wave, reflectivity, Vec::default())
    }
}
impl Node for ReZlm {
    fn precalculate(&mut self, dataset: &Dataset) {
        self.2 = dataset
            .par_iter()
            .map(|event| {
                let resonance = event.daughter_p4s[0] + event.daughter_p4s[1];
                let p1 = event.daughter_p4s[0];
                let recoil_res = event.recoil_p4.boost_along(&resonance);
                let p1_res = p1.boost_along(&resonance);
                let z = -1.0 * recoil_res.momentum().normalize();
                let y = event
                    .beam_p4
                    .momentum()
                    .cross(&(-1.0 * event.recoil_p4.momentum()));
                let x = y.cross(&z);
                let p1_vec = p1_res.momentum();
                let p = Coordinates::cartesian(p1_vec.dot(&x), p1_vec.dot(&y), p1_vec.dot(&z));
                let ylm = ComplexSH::Spherical.eval(self.0.l(), self.0.m(), &p);
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
                match self.1 {
                    Reflectivity::Positive => (1.0 + pgamma).sqrt() * zlm.re,
                    Reflectivity::Negative => (1.0 - pgamma).sqrt() * zlm.re,
                }
            })
            .collect()
    }
    fn calculate(&self, _parameters: &[f64], event: &Event) -> Complex64 {
        self.2[event.index].into()
    }
    fn parameters(&self) -> Option<Vec<String>> {
        None
    }
}

pub struct ImZlm(pub Wave, pub Reflectivity, Vec<f64>);
impl ImZlm {
    pub fn new(wave: Wave, reflectivity: Reflectivity) -> Self {
        Self(wave, reflectivity, Vec::default())
    }
}
impl Node for ImZlm {
    fn precalculate(&mut self, dataset: &Dataset) {
        self.2 = dataset
            .par_iter()
            .map(|event| {
                let resonance = event.daughter_p4s[0] + event.daughter_p4s[1];
                let p1 = event.daughter_p4s[0];
                let recoil_res = event.recoil_p4.boost_along(&resonance);
                let p1_res = p1.boost_along(&resonance);
                let z = -1.0 * recoil_res.momentum().normalize();
                let y = event
                    .beam_p4
                    .momentum()
                    .cross(&(-1.0 * event.recoil_p4.momentum()));
                let x = y.cross(&z);
                let p1_vec = p1_res.momentum();
                let p = Coordinates::cartesian(p1_vec.dot(&x), p1_vec.dot(&y), p1_vec.dot(&z));
                let ylm = ComplexSH::Spherical.eval(self.0.l(), self.0.m(), &p);
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
                match self.1 {
                    Reflectivity::Positive => (1.0 - pgamma).sqrt() * zlm.im,
                    Reflectivity::Negative => (1.0 + pgamma).sqrt() * zlm.im,
                }
            })
            .collect()
    }
    fn calculate(&self, _parameters: &[f64], event: &Event) -> Complex64 {
        self.2[event.index].into()
    }
    fn parameters(&self) -> Option<Vec<String>> {
        None
    }
}
