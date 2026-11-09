# amanuensis 0.2.0b2

- Windows: `transcribe()` no longer ends the process holding a folder's `transcribe.lock` when it checks whether that process is still running.
- Tests no longer depend on today's date, or on `HOME` on Windows.
- Windows: an instructions file name ending in a space or a dot is now reported as not found, not as a wrong extension.
