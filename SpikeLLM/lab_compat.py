"""Compatibility fixes for Hugging Face caches on Windows lab machines.

Some Windows environments report os.symlink() success without creating a link.
Force huggingface_hub to use its copy fallback so snapshot directories contain
real files. This does not affect model or quantization numerics.
"""

try:
    import huggingface_hub.file_download as _hf_file_download

    def _symlinks_unsupported(*args, **kwargs):
        return False

    _hf_file_download.are_symlinks_supported = _symlinks_unsupported
except (ImportError, AttributeError):
    # Older/newer hub versions may not expose this private helper. Their normal
    # behavior is left untouched rather than making model startup fail.
    pass
