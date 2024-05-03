pub mod dalitz;
pub mod harmonics;
pub mod resonances;
pub mod sdmes;
pub mod utils;

use pyo3::prelude::*;

#[pymodule]
fn gluex(m: &Bound<'_, PyModule>) -> PyResult<()> {
    dalitz::register_module(m)?;
    harmonics::register_module(m)?;
    resonances::register_module(m)?;
    sdmes::register_module(m)?;
    Ok(())
}

pub fn register_module(parent: &Bound<'_, PyModule>) -> PyResult<()> {
    let m = PyModule::new_bound(parent.py(), "rustitude.gluex")?;
    gluex(&m)?;
    parent.add_submodule(&m)?;
    parent
        .py()
        .import_bound("sys")?
        .getattr("modules")?
        .set_item("rustitude.gluex", &m)?;
    Ok(())
}
