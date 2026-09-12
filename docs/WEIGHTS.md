# Weight provenance and DIMER hosting

- Upstream: `facebook/sam2.1-hiera-small`
- Immutable revision: `ee5bba1d82bb8749febdf90f45e84b687142ba03`
- Weight format: SafeTensors (`model.safetensors`, 184,305,280 bytes)
- Manifest: `weights/sam2.1-hiera-small/dimer-base-manifest.json` (7 files, 184,336,196 bytes total, per-file SHA-256). The snapshot also carries the native `sam2.1_hiera_s.yaml` and `video_preprocessor_config.json`; both are verified by digest and unused by this package.
- Upstream weight license: Apache-2.0
- DIMER hosting: Apache-2.0 permits use, modification, distribution, and commercial use subject to preservation of the license and notices. The Git repository does not vendor the checkpoint (`weights/**/*.safetensors` is git-ignored); DIMER may mirror the pinned snapshot in its model store under the upstream license.
- Fresh clone: `stage_missing_files(allow_download=True)` fetches only the manifest-listed files absent on disk, at the pinned revision, into the snapshot directory; `verify_snapshot()` then checks every file before any load.
- Loader trust boundary: Transformers `Sam2Model` / `Sam2Processor` (image mode) with `trust_remote_code=False`, `local_files_only=True` from the verified directory. `config.json` names `Sam2VideoModel`; the image class loads the shared weights and Transformers logs a model-type notice on load.
