Fixed
An abbreviated `--json` (`--js`, `--jso`, `--j`), which argparse already
accepts, now also turns usage errors into the one `{"ok": false, "error",
"usage"}` object on stdout instead of plain usage text on stderr (#666).
