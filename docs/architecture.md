# Architecture

How the code is organised, how the modules call each other, and what data
flows between them. Each module's own documentation (`cargo doc`, or the
source) gives its equations and references; this page connects them.

## Layers

```mermaid
flowchart TB
    user["Python user code<br/>(NumPy arrays, file I/O)"]
    subgraph py["Python package: cortical_bone_mapping_rs"]
        opts["options.py<br/>Options, RefineOptions:<br/>dataclasses, validation, presets"]
        api["mapping.py, segmentation.py, mesh.py<br/>one wrapper per entry point"]
        bones["bones.py<br/>several bones: labels, shared scanner,<br/>crop, exclude, uncrop"]
        res["results.py<br/>CbmResult, RefineResult"]
    end
    subgraph core["Extension module: _core (python/src)"]
        conv["convert.rs, options.rs<br/>arrays in place, option dict to Options"]
        logs["logs.rs<br/>log records to Python logging"]
    end
    crate["Rust crate: cortical-bone-mapping"]
    user --> api
    user --> bones
    bones --> api
    opts --> api
    api -->|"arrays, flat option dict"| conv
    conv -->|"ArrayView3, Options<br/>(GIL released)"| crate
    crate -->|"CbmResult, RefineResult, masks"| conv
    conv -->|"dict of arrays"| res
    crate -.->|"log records"| logs
```

| layer | does | does not |
|---|---|---|
| Python package | validates options, offers presets, orchestrates several bones (crop, exclude, shared scanner) | compute anything per node or voxel |
| `_core` | converts arrays without copying, converts options, releases the GIL, forwards logs | hold state between calls |
| crate | every computation, multithreaded with rayon | read or write image files |

Images of any common numeric type are read in place; masks are `uint8`.
Coordinates are physical and in array-axis order everywhere: voxel
`(i, j, k)` is at `origin + (i, j, k) * spacing`.

## Crate modules

```mermaid
flowchart LR
    pipeline["pipeline<br/>run()"]
    refine["refine<br/>refine_segmentation()"]
    seg["segmentation<br/>otsu, bones, hull, buie,<br/>morphology"]
    surface["surface<br/>nets, mesh (Taubin), levelset,<br/>voxelize, closest, geodesic"]
    fit["fit<br/>fit_surface()"]
    prof["fit::profiles"]
    dens["fit::density, fit::psf"]
    loc["fit::local"]
    glob["fit::global<br/>Gauss–Newton / PCG solver"]
    mg["fit::multigrid"]
    reg["fit::regularized"]
    vol["fit::volumetric<br/>band, reach, problem,<br/>reliability"]
    ops["operators<br/>neighbours, smoothing ω,<br/>curvature, multigrid A"]
    base["model, volume, kdtree,<br/>util, logging"]
    pipeline --> surface
    pipeline --> fit
    refine --> surface
    refine --> fit
    fit --> prof
    fit --> dens
    fit --> loc
    fit --> mg
    fit --> reg
    fit --> vol
    dens --> loc
    mg --> glob
    reg --> glob
    reg --> loc
    vol --> glob
    vol --> dens
    mg --> ops
    reg --> ops
    vol --> ops
    seg --> base
    surface --> base
    fit --> base
    ops --> base
```

| module | role |
|---|---|
| `pipeline` | `run()`: mask → surface → candidate nodes → `fit_surface` |
| `surface` | surface nets (`nets`), the mesh and its Taubin smoothing (`mesh`), the level set of refinement (`levelset`), rasterising a mesh to a mask (`voxelize`), closest points and surface distances |
| `fit` | `fit_surface`: the stages common to every method, then the method |
| `fit::profiles` | sampling the image along each node's inward normal |
| `fit::density`, `fit::psf` | the cortical density `ρc` and the blur (fixed, a PSF, or estimated) |
| `fit::local` | one bounded Levenberg–Marquardt fit per profile |
| `fit::global` | the Gauss–Newton / conjugate-gradient solver shared by the global methods, behind the `LeastSquares` trait |
| `fit::multigrid`, `fit::regularized` | the 1D global methods: problems for `fit::global` |
| `fit::volumetric` | the volumetric model: its voxel band, its problem for `fit::global`, its reliability outputs |
| `operators` | sparse operators on the nodes: neighbourhoods, the smoothing `ω`, curvature penalties, multigrid interpolation |
| `refine` | segmentation refinement: the move–refit loop, freezing, scanner parameters, trust and stall diagnosis |
| `segmentation` | starting segmentations: Otsu thresholds, bone labels, hulls, Buie's method, morphology |
| `phantoms` | synthetic images with known truth, for the tests |
| `model`, `volume`, `kdtree`, `util`, `logging` | the profile model, image geometry and sampling, neighbour queries, numerics, logging |

## Data flow of a fit: `run()`

```mermaid
flowchart TB
    img[("image<br/>ArrayView3")]
    mask[("mask<br/>ArrayView3")]
    nets["surface nets<br/>Mesh (vertices, faces), border flags"]
    taubin["Taubin smoothing"]
    cand["candidate nodes<br/>not on border caps, near sub-mask"]
    mask --> nets --> taubin --> cand
    cand --> which{"method"}
    subgraph oned["1D methods: local, multigrid, regularised"]
        select["nodes and sampling normals<br/>(normal constraint)"]
        sample["Profiles: n × M samples<br/>(trilinear, f32), exclusions"]
        rhoc["ρc: fixed, estimated or per node"]
        blur["blur: free, fixed, PSF, estimated PSF"]
        g1["one model for all profiles<br/>(fit::global, all parameters global)"]
        meth["the method<br/>local / multigrid / regularised"]
        resid["residuals; per-node densities scaled back"]
        select --> sample --> rhoc --> blur --> g1 --> meth --> resid
    end
    subgraph vold["volumetric model"]
        vrho["ρc from the profiles<br/>(1D estimate)"]
        band["Band: voxels within [−outside, inside],<br/>grouped by closest face"]
        g2["global stage: one model for all voxels"]
        field["field stage: m, t, ρs, ρb per node,<br/>PSF global (fit::global)"]
        rel["outputs: parameters along d,<br/>noise, standard errors, data share"]
        vrho --> band --> g2 --> field --> rel
    end
    which -->|"local, multigrid, regularised"| select
    which -->|"volumetric"| vrho
    img --> sample
    img --> band
    resid --> out[("CbmResult")]
    rel --> out
```

What each stage produces, and roughly what it costs:

| stage | output | size |
|---|---|---|
| surface nets | `Mesh` | one vertex per grid cell the boundary crosses (about one per boundary voxel); two triangles per vertex |
| profiles (1D) | `Profiles` | `n × M` `f32`, `M = (outside + inside) / resolution + 1` |
| band (volumetric) | `Band` | 24 bytes per voxel within `[−outside, inside]` of the surface |
| smoothing operator `ω` | `SmoothingOperator` | 8 bytes per neighbour per node (`reg_neighbours`) |
| solver state | variables, gradient, blocks | tens of doubles per node |
| result | `CbmResult` | 5 parameters, residual, point and normal per node; the surface |

## The global solver

Every global fit, the one-model initialisations, multigrid, regularised and
volumetric, is a `LeastSquares` problem for `fit::global::minimise`. The
problem supplies the cost, the gradient, Gauss–Newton blocks and products;
the solver never forms the Hessian.

```mermaid
flowchart TB
    start["x clamped into the bounds;<br/>linearise: cost, gradient, blocks"]
    test{"scaled gradient ≤ gtol,<br/>or nothing free?"}
    pcg["solve (JᵀJ + μD) δ = −g<br/>by PCG, block-Jacobi preconditioner,<br/>to relative residual η"]
    proj["project x + δ onto the bounds;<br/>gain ratio ρ"]
    acc{"ρ > 10⁻⁴?"}
    damp["μ ← μν, ν ← 2ν"]
    accept["accept; relax μ; linearise"]
    stop{"ftol, edge_tol,<br/>stall, xtol?"}
    budget{"iterations or<br/>CG budget spent?"}
    done(["converged"])
    unconv(["stopped unconverged"])
    start --> test
    test -->|yes| done
    test -->|no| budget
    budget -->|yes| unconv
    budget -->|no| pcg --> proj --> acc
    acc -->|no| damp --> budget
    acc -->|yes| accept --> stop
    stop -->|yes| done
    stop -->|no| test
```

The cost of a solve is dominated by the Gauss–Newton products inside PCG,
one per CG iteration: for the 1D methods, a pass over the nodes' 5 × 5
blocks plus the operators; for the volumetric model, a pass over the faces'
blocks (the voxels are visited only when the blocks are formed, once per
outer iteration).

## Refinement: `refine_segmentation()`

```mermaid
flowchart TB
    s0["start: surface from the mask, level set,<br/>first full fit (scanner parameters estimated)"]
    disp["displacements: trusted nodes toward m − t/2 (×α);<br/>harmonic fill for the rest; freezing updated"]
    meas["measure: RMS, maximum, regional RMS"]
    stop{"converged, out of moves,<br/>or stalled?"}
    rescan{"another scanner pass?<br/>(re-fit changed by > tolerance)"}
    move["move: level set advanced,<br/>new mesh extracted, state transferred"]
    refit["refit: inexact fit of the moving nodes<br/>(frozen neighbours held), or a full fit<br/>to verify a settled surface"]
    fin["finish: output fit if needed;<br/>periosteal, endosteal, cortical masks"]
    s0 --> disp --> meas --> stop
    stop -->|no| move --> refit --> disp
    stop -->|yes| rescan
    rescan -->|"yes: full fit with the new values"| disp
    rescan -->|no| fin
    fin --> rr[("RefineResult")]
```

The state carried from fit to fit (`refine::state::NodeState`) is per
surface vertex: warm-start parameters (with `m` shifted by the move), how
long the node has been still, whether it is frozen or held. When the level
set extracts a new mesh, every new vertex takes the state of the nearest
moved old vertex.

The scanner parameters (`ρc`, and the volumetric model's PSF) are held
through a pass and re-fitted between passes (`refine::scanner`): every fit
inside a pass uses the same values, so the edges do not shift because the
blur did.

## Several bones: `refine_bones()` (Python)

```mermaid
flowchart TB
    labels[("labels: one per bone<br/>(bone_hulls)")]
    shared["shared_scanner: one fit of all bones together<br/>→ ρc, PSF as option overrides"]
    order["bones from smallest to largest"]
    crop["crop the bone's box + sampling range;<br/>exclude every other bone's region"]
    rs["_core.refine → refine_segmentation_excluding"]
    paste["masks pasted back at full size;<br/>the bone's region becomes its refined mask"]
    labels --> shared --> order --> crop --> rs --> paste
    paste -->|"next bone"| crop
    paste --> out[("{label: RefineResult}")]
```

The bones of one scan share its blur and apparent cortical density, so those
are estimated once on all of them and held in each bone's refinement. A
bone's surface can never move into another's region (zero weight for
profile samples there, and a hard constraint on the surface).

## Threads, memory and determinism

- Parallel loops use rayon's global pool, or a pool of `threads` workers if
  that option is set. The bindings release the GIL for the whole call.
- Every sum over nodes or faces runs in fixed-size chunks in a fixed order,
  so results are bitwise identical across runs and thread counts.
- Memory scales with the number of nodes (profiles, operators, solver
  state) and, for the volumetric model, with the band's voxels. Refinement
  adds a level-set grid over the bone's box. Cropping each bone
  (`refine_bones`) keeps the image and the masks small.

## Logging

Every stage logs through the [`log`](https://docs.rs/log) facade (module
`cortical_bone_mapping::logging`), and the Python package forwards the
records to Python's `logging`:

```python
import cortical_bone_mapping_rs as cbm
cbm.log_to("run.log", level="DEBUG")  # or configure logging.getLogger("cortical_bone_mapping")
```

| level | what |
|---|---|
| INFO | one line per call (fit, refinement, segmentation step) and its outcome |
| DEBUG | each stage when it starts and when it ends (sizes, duration), the options in use, every solver iteration and why the solve stopped, every refinement move and fit |
| TRACE | details within stages, e.g. every conjugate-gradient solve |
| WARNING | an unconverged solve, a refinement that did not converge, an estimate that fell back |

A stage logs when it starts, so the log of a run that stalls shows which
stage it is in. Logger names follow the Rust modules
(`cortical_bone_mapping.fit.volumetric.reliability`, …), so one part can be
followed alone. Nothing is logged inside per-node or per-voxel loops.
