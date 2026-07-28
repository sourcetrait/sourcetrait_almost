//! Generating a training set from the syllabus tree into a railroad.
use crate::*;

/// `bquest syllabus emit`: one set, one railroad, one committed REV.
pub(crate) fn syllabus_emit(args: &SyllabusEmitArgs) -> BquestResult<()> {
    let started = std::time::Instant::now();
    let railroad = match &args.railroad {
        Some(dir) => lib::railroad::Railroad::at(dir)?,
        None => lib::railroad::Railroad::lay()?,
    };
    let emitted = lib::syllabus::emit(&args.root, &railroad)?;
    let summary = lib::nu::Value::record(
        lib::nu::record! {
            "nom" => v_str(railroad.nom().as_str()),
            "railroad" => v_str(&railroad.dir().display().to_string()),
            "rev" => v_int(emitted.rev as i64),
            "methods" => v_int(emitted.methods as i64),
            "cases" => v_int(emitted.cases as i64),
            "seconds" => v_float(started.elapsed().as_secs_f64()),
        },
        span(),
    );
    println!("{}", lib::nu::to_nuon_text(&summary)?);
    Ok(())
}
