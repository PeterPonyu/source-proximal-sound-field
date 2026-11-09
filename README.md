# Sound field estimation next to loudspeakers with point sources shared across frequency

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24067791.svg)](https://doi.org/10.5281/zenodo.24067791)

Code and per-problem results for the paper *Sound Field Estimation Next to Loudspeakers from Microphones Alone Using
Point Sources Shared Across Frequency*. Every version is archived at Zenodo;
[doi:10.5281/zenodo.24067791](https://doi.org/10.5281/zenodo.24067791) resolves to the latest one.

The estimator adds point-source fields to a homogeneous Helmholtz Gaussian process. A greedy evidence scan, summed
over the bins of a frequency band, places points whose coordinates are shared across the band; cross-validation over
the microphones chooses how many to keep, at most five; the kept points are then refined into spherical source
regions, and the homogeneous process models the rest of the room's field. This repository holds the estimator, its
comparators, the runners and analyses of every study in the paper, their per-problem results and run receipts, and the
two scripts that regenerate every number, table and figure of the paper from those results.

This is version 1.1.0, published on 9 November 2026 as [doi:10.5281/zenodo.24075698](https://doi.org/10.5281/zenodo.24075698). It adds two post hoc analyses to version 1.0.0, a single-point and order decomposition and a point-plus-dipole baseline ([doi:10.5281/zenodo.24067792](https://doi.org/10.5281/zenodo.24067792)).

## Layout

Files keep the paths they have in the project repository, so the registered code keeps the SHA-256 listed in
`FROZEN-INPUTS.json`. Below, `acoustic/` stands for the study root `archive/2026-11-05-taslp/acoustic/`.

| Path | What it holds |
|---|---|
| `acoustic/common/` | library: fields and room model, problem builders, the estimator and the generic comparators (`v2.py`, `methods_v2.py`), source-region kernels and their adaptations (`mo2025.py`, `prox*.py`), the guard of the sparse Bayesian comparators, known-answer checks and the input checks run before every registered study (`release.py`) |
| `acoustic/run_v2.py`, `analyze_v2.py` | the simulation study (`TB-S2`) and the room-grid study on MeshRIR (`TB-M2`) |
| `acoustic/run_followups.py`, `analyze_followups.py` | follow-ups on the room grid: target-guarded sparse Bayesian comparators (`TB-M2-SBLFIX`) and the published adaptive-kernel code (`TB-M2-RIB`) |
| `acoustic/run_exact*.py`, `analyze_exact*.py` | follow-ups that rerun the five-ball variants with the exact spherical-wave series (`TB-M2-EXACT`, `TB-M2-EXACT-HI`) |
| `acoustic/run_prox*.py`, `analyze_prox*.py` | the adaptations of the two published source-region methods (`TB-PROX`, `TB-PROX-S2`) |
| `acoustic/studies/TB-M3-MIRACLE/` | the anechoic study on MIRACLE: extraction, runner, analysis and coverage check |
| `acoustic/studies/<study>/` | per study: per-problem results (`rows.jsonl`), summaries, run receipts and side analyses |
| `acoustic/posthoc/` | post hoc analyses: refits, reproductions of the published configurations, a Gaussian source density, an order cap of eight, fixed orders, the Cramér–Rao check, margins and geometry, the single-point and order decomposition, and the point-plus-dipole baseline |
| `acoustic/studies/AUDITS/` | numerical checks of the ball kernel |
| `acoustic/V2-dev/` | development runs on the development dataset, which selected the estimator |
| `acoustic/B3-measured/` | preparation of the measured datasets and their integrity records |
| `acoustic/comparators/ribeiro2024/` | bridge to the published adaptive-kernel code of Ribeiro et al. (Julia) |
| `acoustic/remote/` | capsule build, data download, cross-machine check and return verification for runs on a rented machine |
| `acoustic/B0-setup/`, `acoustic/B1-estimators/`, `acoustic/run_sim.py`, `acoustic/run_measured.py` | setup checks, reproduction checks of a published estimator, and the runners of the two earlier single-frequency studies, whose results are not released |
| `archive/2026-10-2*/acoustic-estimator-robustness/` | three modules of earlier acoustic code from the same project that the library imports: the modal room model, image-source walls and relocation controls |
| `papers/acoustic-source-proximal/tools/` | `make_numbers_v2.py` writes every number and table of the paper; `make_figures_v2.py` draws every figure and checks its fonts and layout |
| `REGISTRATION.md` | every registered protocol and analysis plan, with the time of its commit and its SHA-256 |

Method keys in the results: `j_cv_ext` is the proposed estimator, `ind_cv_ext` the same fitted to each bin alone,
`v1` the earlier single-frequency estimator, `sinc_ev` the sinc kernel, `adk_standin` the adaptive kernel,
`gsmf` and `mo26_cap5` the variants that fit five balls shared across the band or per bin, `two_stage_point` and
`two_stage_region` the arrival-time baselines, `mo26_prox`, `mo26_prox_s2` and `gsmf_prox` the adaptations of the
sparse and group-sparse source-region methods, and `ribeiro_official` the published adaptive-kernel code.

## Regenerating the paper's numbers, tables and figures

Python 3.13 with the versions in `requirements.txt`; the figure script also needs the Arial TrueType fonts. From the
repository root:

```bash
pip install -r requirements.txt
python3 -B papers/acoustic-source-proximal/tools/make_numbers_v2.py archive/2026-11-05-taslp/acoustic papers/acoustic-source-proximal/generated
python3 -B papers/acoustic-source-proximal/tools/make_figures_v2.py
```

The first script writes `v2-numbers.tex`, every number of the paper as a LaTeX macro, and the tables to
`papers/acoustic-source-proximal/generated/`; it also reads `ka-table.tex` there, the known-answer checks of the two
earlier studies. The second writes the figures and a font and layout report to `papers/acoustic-source-proximal/figures/`.
With these package versions both outputs are byte-identical to those of the paper.

## Rerunning the studies

This package supports regeneration from saved results. It does not support the registered study reruns as shipped:
the frozen-input checks also require the registration documents, which are not public, full dataset extracts, and the original
registration history or a capsule made from that history. Downloading the datasets alone does not satisfy those checks.
The public checksum listing documents the inputs; it does not replace the registration-history check. The registered
runners and their checks are preserved unchanged. Manuscript LaTeX sources are not included in this package.

The saved outputs of the decomposition and dipole analyses are included and suffice for regenerating their numbers in the paper. Rerunning
`acoustic/posthoc/review4/saved.py` additionally needs the full MeshRIR extract for its low-frequency power diagnostic;
`acoustic/posthoc/review4/dipole.py` needs both MIRACLE scenario extracts. The preparation scripts in the table below
create these extracts. These post hoc reruns are separate from the saved-result regeneration commands above.

The datasets are not redistributed. `acoustic/remote/fetch_data.sh` downloads the files and checks their MD5:

| Dataset | Record and licence | Files (MD5) | Extract written by |
|---|---|---|---|
| MeshRIR 1.2 | [doi:10.5281/zenodo.10852693](https://doi.org/10.5281/zenodo.10852693), CC BY 4.0 | `S32-M441_npy.zip` (`9818fc66b36513590e7abd071243d8e9`) | `acoustic/B3-measured/meshrir_prep.py` |
| MP-RIR 1.0, development | [doi:10.5281/zenodo.11148712](https://doi.org/10.5281/zenodo.11148712), CC BY 4.0 | `Mxyz.npy` (`d1aad2e13b7b32b6f5484df5b4ac8d93`), `Setup.npz` (`6ed01cadfdc6b4ed51236dbdd2af3f54`), `S4_Mrir.npy` (`a705e3f028cfab4303511c52f75749ef`), `S7_Mrir.npy` (`f33f021d2550711d1371b6d65c5abbfb`), `S8_Mrir.npy` (`25b297c591162d71059f53b6e955057c`) | `acoustic/V2-dev/measured_dev_prep.py`, `acoustic/B3-measured/heldout_prep.py` |
| MIRACLE | [doi:10.14279/depositonce-18318](https://doi.org/10.14279/depositonce-18318), CC BY-NC-SA 4.0 | `A1.h5` (`44ded79a5129083a9ac61edf8e6009e2`), `R2.h5` (`4f2d5a9bf4c2402b0335eb2c1da34ffb`) | `acoustic/studies/TB-M3-MIRACLE/miracle_prep.py` |

The SHA-256 of every extract is recorded in the JSON receipt next to it and in `FROZEN-INPUTS.json`. Of the MeshRIR
extract this repository keeps only the microphone and loudspeaker positions, which the geometry figure reads;
`meshrir_prep.py` writes the full extract at the same path. `acoustic/common/mprir.py` and
`acoustic/B3-measured/integrity_check.py` read MP-RIR from a fixed local path (`DATA`); set it to your copy.

Each runner's docstring gives its command line. The registered runners first check their frozen inputs, against the
registration commit in the project repository or against the checksum list of a capsule
(`acoustic/common/release.py`, `acoustic/remote/make_capsule.py`); the post hoc scripts have no such check. The bridge in
`acoustic/comparators/ribeiro2024/` runs the original authors' Julia library, `skoyamalab/PhysicsConstrainedNeuralKernel` (MIT),
at commit `7a1e5c9b9187bef00c864942a4587b2eb6b71e23` with Julia 1.11.9; `Project.toml` and `Manifest.toml` pin the Julia
packages. The docstrings name the protocols and plans in `PROTOCOLS/` and `PLAN*.md`, which are listed in
`REGISTRATION.md` by commit time and SHA-256.

## Citation

If you use this code or the results, please cite the archived version, [doi:10.5281/zenodo.24067791](https://doi.org/10.5281/zenodo.24067791);
`CITATION.cff` holds the same reference.

## Licence

MIT for the code. The result files hold numbers computed from the datasets; the datasets keep their own licences.
