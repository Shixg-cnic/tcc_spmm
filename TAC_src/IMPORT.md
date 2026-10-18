# TAC source import

Imported from the user-provided `TAC_src.tar.gz`, retaining the `TAC_src/TAC/` directory layout.

- Archive SHA-256: `15fb88f8deebb42f0eae1002ad6128cf7040e4502e5571264cd415e665ac19f7`
- Operator sources: `TAC/training_backend/spmm_cnic/`, `TAC/training_backend/c++_frontend/3rd/`, and `TAC/sampling_server/3rd/acc-spmm/`.
- Training, sampling, IPC support, build scripts, and historical source variants under `TAC/backup/` are included.
- Generated build/dist files, binaries, profiling traces, datasets, caches, and editor settings are omitted. The original archive remains local and is ignored by Git.
- Source file contents were verified byte-for-byte against the archive. Relative symbolic links are preserved (trailing directory slashes may be normalized during extraction).

This is a source import only. The imported operators have not been compiled or integrated into the current benchmark. The original nested `.gitignore` is preserved; the initial import explicitly stages source files that it would otherwise ignore.
