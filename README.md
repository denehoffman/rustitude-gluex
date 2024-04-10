<p align="center">
  <img
    width="800"
    src="https://raw.githubusercontent.com/denehoffman/rustitude-gluex/main/media/logo.png"
  />
</p>
<p align="center">
    <h1 align="center">GlueX Amplitudes for Rustitude</h1>
</p>

<p align="center">
  <a href="https://github.com/denehoffman/rustitude-gluex/commits/main/" alt="Lastest Commits">
    <img src="https://img.shields.io/github/last-commit/denehoffman/rustitude-gluex/main" /></a>
  <a href="https://github.com/denehoffman/rustitude-gluex/actions" alt="Build Status">
    <img src="https://img.shields.io/github/actions/workflow/status/denehoffman/rustitude-gluex/rust.yml" /></a>
  <a href="LICENSE" alt="License">
    <img src="https://img.shields.io/github/license/denehoffman/rustitude-gluex" /></a>
  <a href="https://crates.io/crates/rustitude-gluex" alt="Rustitude-GlueX on crates.io">
    <img src="https://img.shields.io/crates/v/rustitude-gluex" /></a>
  <a href="https://docs.rs/rustitude-gluex/latest/rustitude_gluex/" alt="Rustitude-GlueX documentation on docs.rs">
    <img src="https://img.shields.io/docsrs/rustitude-gluex" /></a>
</p>


### Note: This project is still very much under development and not recommended for use in actual research projects (yet)

### Table of Contents
- [Overview](#Overview)
- [Installation](#Installation)
- [Usage](#Usage)

## Overview
This is a library of amplitudes which are commonly used in analyses of [GlueX](http://gluex.org/) data

## Installation

Cargo provides the usual command for including this crate in a project. You should also include the [rustitude](https://github.com/denehoffman/rustitude/) library, since this is just a support library to provide amplitudes to `rustitude`: 
```sh
cargo add rustitude
cargo add rustitude-gluex
```

## Usage
See [rustitude](https://github.com/denehoffman/rustitude?tab=readme-ov-file#usage)'s documentation for more information on how to use this library.

## TODOs

There are a few amplitudes from [halld_sim](https://github.com/JeffersonLab/halld_sim/tree/master/src/libraries/AMPTOOLS_AMPS) which have been implemented, several which need to be implemented, and a few which don't need to be implemented (like [Uniform.cc](https://github.com/JeffersonLab/halld_sim/blob/master/src/libraries/AMPTOOLS_AMPS/Uniform.cc)). Additionally, I see a few areas of improvement when it comes to providing a consistent interface across all amplitudes (selection of daughter particles varies a lot across amplitudes written by different people, for instance), and there are possibly a few amplitudes which will become more performant due to the abilities of `rustitude`. For example, the `Ylm`,`ZlmRe`, and `ZlmIm` implementations of [Ylm.cc](https://github.com/JeffersonLab/halld_sim/blob/master/src/libraries/AMPTOOLS_AMPS/Ylm.cc) and [Zlm.cc](https://github.com/JeffersonLab/halld_sim/blob/master/src/libraries/AMPTOOLS_AMPS/Zlm.cc) no longer calculate anything on the compute step, since the entire amplitude can be computed ahead of time. Here is a list of the identified `halld_sim` amplitudes:

| `halld_sim` Amplitude | `rustitude` Equivalent | Priority |
|---|---|:---:|
| [BreitWigner.cc](https://github.com/JeffersonLab/halld_sim/blob/master/src/libraries/AMPTOOLS_AMPS/BreitWigner.cc) | `rustitude-gluex::resonances::BreitWigner` | :white_check_mark: |
| [BreitWigner3body.cc](https://github.com/JeffersonLab/halld_sim/blob/master/src/libraries/AMPTOOLS_AMPS/BreitWigner3body.cc) |  | :x: |
| [ComplexCoeff.cc](https://github.com/JeffersonLab/halld_sim/blob/master/src/libraries/AMPTOOLS_AMPS/ComplexCoeff.cc) | `rustitude::amplitude::ComplexScalar` | :white_check_mark: |
| [Compton.cc](https://github.com/JeffersonLab/halld_sim/blob/master/src/libraries/AMPTOOLS_AMPS/Compton.cc) |  | :heavy_exclamation_mark: |
| [DblRegge_FastEta.cc](https://github.com/JeffersonLab/halld_sim/blob/master/src/libraries/AMPTOOLS_AMPS/DblRegge_FastEta.cc) |  | :heavy_exclamation_mark: |
| [DblRegge_FastPi.cc](https://github.com/JeffersonLab/halld_sim/blob/master/src/libraries/AMPTOOLS_AMPS/DblRegge_FastPi.cc) |  | :heavy_exclamation_mark: |
| [EtaPb_tdist.cc](https://github.com/JeffersonLab/halld_sim/blob/master/src/libraries/AMPTOOLS_AMPS/EtaPb_tdist.cc) |  | :heavy_exclamation_mark: |
| [Flatte.cc](https://github.com/JeffersonLab/halld_sim/blob/master/src/libraries/AMPTOOLS_AMPS/Flatte.cc) |  | :bangbang: |
| [Hist2D.cc](https://github.com/JeffersonLab/halld_sim/blob/master/src/libraries/AMPTOOLS_AMPS/Hist2D.cc) |  | :x: |
| [Lambda1520Angles.cc](https://github.com/JeffersonLab/halld_sim/blob/master/src/libraries/AMPTOOLS_AMPS/Lambda1520Angles.cc) |  | :x: |
| [Lambda1520tdist.cc](https://github.com/JeffersonLab/halld_sim/blob/master/src/libraries/AMPTOOLS_AMPS/Lambda1520tdist.cc) |  | :x: |
| [LowerVertexDelta.cc](https://github.com/JeffersonLab/halld_sim/blob/master/src/libraries/AMPTOOLS_AMPS/LowerVertexDelta.cc) |  | :x: |
| [OmegaDalitz.cc](https://github.com/JeffersonLab/halld_sim/blob/master/src/libraries/AMPTOOLS_AMPS/OmegaDalitz.cc) | `rustitude-gluex::dalitz::OmegaDalitz` | :white_check_mark: |
| [PhaseOffset.cc](https://github.com/JeffersonLab/halld_sim/blob/master/src/libraries/AMPTOOLS_AMPS/PhaseOffset.cc) |  | :bangbang: |
| [Pi0Regge.cc](https://github.com/JeffersonLab/halld_sim/blob/master/src/libraries/AMPTOOLS_AMPS/Pi0Regge.cc) |  | :heavy_exclamation_mark: |
| [Pi0ReggeModel.cc](https://github.com/JeffersonLab/halld_sim/blob/master/src/libraries/AMPTOOLS_AMPS/Pi0ReggeModel.cc) |  | :x: |
| [Pi0SAID.cc](https://github.com/JeffersonLab/halld_sim/blob/master/src/libraries/AMPTOOLS_AMPS/Pi0SAID.cc) |  | :x: |
| [PiPlusRegge.cc](https://github.com/JeffersonLab/halld_sim/blob/master/src/libraries/AMPTOOLS_AMPS/PiPlusRegge.cc) |  | :x: |
| [Piecewise.cc](https://github.com/JeffersonLab/halld_sim/blob/master/src/libraries/AMPTOOLS_AMPS/Piecewise.cc) |  | :bangbang: |
| [SinglePS.cc](https://github.com/JeffersonLab/halld_sim/blob/master/src/libraries/AMPTOOLS_AMPS/SinglePS.cc) | `rustitude-gluex::harmonics::OnePS` | :white_check_mark: |
| [ThreePiAngles.cc](https://github.com/JeffersonLab/halld_sim/blob/master/src/libraries/AMPTOOLS_AMPS/ThreePiAngles.cc) |  | :bangbang: |
| [ThreePiAnglesSchilling.cc](https://github.com/JeffersonLab/halld_sim/blob/master/src/libraries/AMPTOOLS_AMPS/ThreePiAnglesSchilling.cc) |  | :bangbang: |
| [TwoLeptonAngles.cc](https://github.com/JeffersonLab/halld_sim/blob/master/src/libraries/AMPTOOLS_AMPS/TwoLeptonAngles.cc) |  | :heavy_exclamation_mark: |
| [TwoLeptonAnglesGJ.cc](https://github.com/JeffersonLab/halld_sim/blob/master/src/libraries/AMPTOOLS_AMPS/TwoLeptonAnglesGJ.cc) |  | :heavy_exclamation_mark: |
| [TwoPSAngles.cc](https://github.com/JeffersonLab/halld_sim/blob/master/src/libraries/AMPTOOLS_AMPS/TwoPSAngles.cc) | `rustitude-gluex::harmonics::TwoPS` | :white_check_mark: |
| [TwoPSHelicity.cc](https://github.com/JeffersonLab/halld_sim/blob/master/src/libraries/AMPTOOLS_AMPS/TwoPSHelicity.cc) | `rustitude-gluex::harmonics::TwoPS` | :white_check_mark: |
| [TwoPiAngles.cc](https://github.com/JeffersonLab/halld_sim/blob/master/src/libraries/AMPTOOLS_AMPS/TwoPiAngles.cc) |  | :bangbang: |
| [TwoPiAngles_amp.cc](https://github.com/JeffersonLab/halld_sim/blob/master/src/libraries/AMPTOOLS_AMPS/TwoPiAngles_amp.cc) |  | :heavy_exclamation_mark: |
| [TwoPiAngles_primakoff.cc](https://github.com/JeffersonLab/halld_sim/blob/master/src/libraries/AMPTOOLS_AMPS/TwoPiAngles_primakoff.cc) |  | :bangbang: |
| [TwoPiEtas_tdist.cc](https://github.com/JeffersonLab/halld_sim/blob/master/src/libraries/AMPTOOLS_AMPS/TwoPiEtas_tdist.cc) |  | :x: |
| [TwoPiNC_tdist.cc](https://github.com/JeffersonLab/halld_sim/blob/master/src/libraries/AMPTOOLS_AMPS/TwoPiNC_tdist.cc) |  | :x: |
| [TwoPiW_brokenetas.cc](https://github.com/JeffersonLab/halld_sim/blob/master/src/libraries/AMPTOOLS_AMPS/TwoPiW_brokenetas.cc) |  | :x: |
| [TwoPiWt_primakoff.cc](https://github.com/JeffersonLab/halld_sim/blob/master/src/libraries/AMPTOOLS_AMPS/TwoPiWt_primakoff.cc) |  | :heavy_exclamation_mark: |
| [TwoPiWt_sigma.cc](https://github.com/JeffersonLab/halld_sim/blob/master/src/libraries/AMPTOOLS_AMPS/TwoPiWt_sigma.cc) |  | :x: |
| [TwoPitdist.cc](https://github.com/JeffersonLab/halld_sim/blob/master/src/libraries/AMPTOOLS_AMPS/TwoPitdist.cc) |  | :x: |
| [Uniform.cc](https://github.com/JeffersonLab/halld_sim/blob/master/src/libraries/AMPTOOLS_AMPS/Uniform.cc) | N/A | :white_check_mark: |
| [VecRadiative_SDME.cc](https://github.com/JeffersonLab/halld_sim/blob/master/src/libraries/AMPTOOLS_AMPS/VecRadiative_SDME.cc) |  | :bangbang: |
| [Vec_ps_refl.cc](https://github.com/JeffersonLab/halld_sim/blob/master/src/libraries/AMPTOOLS_AMPS/Vec_ps_refl.cc) |  | :bangbang: |
| [Ylm.cc](https://github.com/JeffersonLab/halld_sim/blob/master/src/libraries/AMPTOOLS_AMPS/Ylm.cc) | `rustitude-gluex::harmonics::Ylm` | :white_check_mark: |
| [Zlm.cc](https://github.com/JeffersonLab/halld_sim/blob/master/src/libraries/AMPTOOLS_AMPS/Zlm.cc) | `rustitude-gluex::harmonics::{ZlmRe, ZlmIm}` | :white_check_mark: |
| _not yet implemented_ | `rustitude-gluex::resonances::{KMatrixF0, KMatrixF2, KMatrixA0, KMatrixA2, KMatrixRho, KMatrixPi1}` | :white_check_mark: |
