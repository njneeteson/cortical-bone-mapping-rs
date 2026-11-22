//! Writing meshes as legacy binary VTK PolyData (the "simple legacy format"
//! of the VTK file formats; big-endian binary `double` data, one
//! `POLYGONS` cell per triangle, point data as `SCALARS` and `VECTORS`), for
//! viewing in ParaView or reading with PyVista.
//!
//! # References
//!
//! * W. Schroeder, K. Martin, B. Lorensen, *The Visualization Toolkit: an
//!   Object-Oriented Approach to 3D Graphics*, 4th ed., Kitware, 2006
//!   (file formats appendix).

use std::io::Write;

use super::mesh::Mesh;
use crate::util::V3;

impl Mesh {
    /// Write the mesh as a legacy binary VTK PolyData file, with optional
    /// per-point scalar and 3-vector arrays.
    pub fn write_vtk(
        &self,
        path: &std::path::Path,
        scalars: &[(&str, &[f64])],
        vectors: &[(&str, &[V3])],
    ) -> std::io::Result<()> {
        let mut w = std::io::BufWriter::new(std::fs::File::create(path)?);
        let n = self.vertices.len();
        writeln!(w, "# vtk DataFile Version 3.0")?;
        writeln!(w, "cortical-bone-mapping")?;
        writeln!(w, "BINARY")?;
        writeln!(w, "DATASET POLYDATA")?;
        writeln!(w, "POINTS {n} double")?;
        for v in &self.vertices {
            for c in v {
                w.write_all(&c.to_be_bytes())?;
            }
        }
        writeln!(w)?;
        writeln!(w, "POLYGONS {} {}", self.faces.len(), self.faces.len() * 4)?;
        for f in &self.faces {
            w.write_all(&3i32.to_be_bytes())?;
            for &i in f {
                w.write_all(&(i as i32).to_be_bytes())?;
            }
        }
        writeln!(w)?;
        if !scalars.is_empty() || !vectors.is_empty() {
            writeln!(w, "POINT_DATA {n}")?;
        }
        for (name, data) in scalars {
            assert_eq!(data.len(), n, "scalar array {name} has the wrong length");
            writeln!(w, "SCALARS {name} double 1")?;
            writeln!(w, "LOOKUP_TABLE default")?;
            for x in *data {
                w.write_all(&x.to_be_bytes())?;
            }
            writeln!(w)?;
        }
        for (name, data) in vectors {
            assert_eq!(data.len(), n, "vector array {name} has the wrong length");
            writeln!(w, "VECTORS {name} double")?;
            for v in *data {
                for c in v {
                    w.write_all(&c.to_be_bytes())?;
                }
            }
            writeln!(w)?;
        }
        w.flush()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vtk_writes() {
        let m = Mesh::uv_sphere(1.0, 6, 4);
        let dir = std::env::temp_dir().join("cbm_vtk_test.vtk");
        let s: Vec<f64> = (0..m.vertices.len()).map(|i| i as f64).collect();
        m.write_vtk(&dir, &[("idx", &s)], &[("n", &m.vertex_normals())])
            .unwrap();
        let bytes = std::fs::read(&dir).unwrap();
        assert!(bytes.starts_with(b"# vtk DataFile"));
        let _ = std::fs::remove_file(dir);
    }
}
