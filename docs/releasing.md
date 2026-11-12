# Releasing

1. Update the data: `tools/build_recipes.py`, `tools/ingest_speed.py` (new measured runs), then
   `python3 tools/build_site.py --readme-only` and commit.
2. Tag: `git tag v0.1.0 && git push --tags`. The `release` workflow builds `bitino_bench-linux-x86_64`,
   `bitino_bench-linux-aarch64` (native ARM runner, static glibc), `bitino_bench-android-arm64` and
   `bitino-sdk-android-arm64.tar.gz` from the tagged tree and attaches them, with `SHA256SUMS`, to a **draft** release.
3. Models: `python3 tools/release_models.py --artifacts <dir with artifacts/> --out dist/models --tag v0.1.0
   --write-registry` writes the redistributable converted files (explicit permissive weights licence only), one
   `<model-id>.LICENSE.txt` per model, `SHA256SUMS` and `manifest.json` (qualified models with the licence evidence,
   excluded ones with the reason). Commit `registry/release_models.json` (used by `bitino fetch --prebuilt`) and
   upload `dist/models/*` to the draft release, e.g. `gh release upload v0.1.0 dist/models/*`.
4. Website: `python3 tools/build_site.py --out <site dir>` and upload the folder (relative links only; it works
   under any sub-path).
5. Publish the draft release.
6. Python wheels: the `wheels` workflow builds and tests the sdist and the wheels of the tag (Linux x86-64 / aarch64,
   macOS arm64, Windows x64) and keeps them as workflow artifacts. `bitino.load()` downloads the redistributable
   models from this release, so publish the release (step 5) first. PyPI upload is manual: Actions -> wheels ->
   Run workflow on the tag with `publish` ticked, then approve the `pypi` environment (one-time setup and details:
   [python.md](python.md#releasing-wheels)).

Locally, `scripts/build_release.sh dist/bin` builds the same binaries (on an x86-64 host the linux-aarch64 binary
falls back to a static NDK build unless an `aarch64-linux-gnu` cross compiler is installed; `BUILDINFO.txt` says
which).
