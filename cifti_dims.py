#!/usr/bin/env python3
"""
cifti_dims.py - print the dimensions of CIFTI / NIfTI files without nibabel.

CIFTI-2 files (.ptseries.nii, .dtseries.nii, .pconn.nii) are NIfTI-2 containers,
and NIfTI-2 stores its dimensions at a fixed header offset. Reading the first
80 bytes is enough, so this needs nothing outside the standard library.

NIfTI-2 header layout:
    offset  0   int32    sizeof_hdr  (540; 348 would mean NIfTI-1)
    offset  4   char[8]  magic
    offset 12   int16    datatype
    offset 14   int16    bitpix
    offset 16   int64[8] dim

For CIFTI, dim[0]=6 and the matrix dimensions sit in dim[5] and dim[6].
For a ptseries that is (timepoints, parcels).

Usage:
    python3 cifti_dims.py <file-or-glob> [...]
    python3 cifti_dims.py '/dataset/.../func/*seg-Gordon*.ptseries.nii'
"""

import glob
import gzip
import struct
import sys


def read_header(path):
    opener = open
    with open(path, "rb") as fh:
        if fh.read(2) == b"\x1f\x8b":
            opener = gzip.open
    with opener(path, "rb") as fh:
        return fh.read(80)


def dims(path):
    hdr = read_header(path)
    if len(hdr) < 80:
        return None, "file too short to hold a NIfTI header"

    # Endianness is whichever reading of sizeof_hdr gives a known value.
    for endian in ("<", ">"):
        size = struct.unpack(endian + "i", hdr[0:4])[0]
        if size == 540:
            d = struct.unpack(endian + "8q", hdr[16:80])
            return d, None
        if size == 348:
            return None, "NIfTI-1 file, not CIFTI-2"
    return None, "not a recognised NIfTI file"


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    paths = []
    for arg in sys.argv[1:]:
        hits = sorted(glob.glob(arg))
        paths.extend(hits if hits else [arg])

    if not paths:
        print("No files matched.")
        sys.exit(1)

    for p in paths:
        try:
            d, err = dims(p)
        except OSError as e:
            print(f"  ERROR  {p}: {e}")
            continue
        name = p.split("/")[-1]
        if err:
            print(f"  {'-':>12}  {name}   [{err}]")
            continue
        # dim[0] is the number of dimensions; CIFTI uses dim[5], dim[6].
        if d[0] >= 6:
            print(f"  {d[5]:>6} x {d[6]:<6}  {name}")
        else:
            print(f"  {'x'.join(str(v) for v in d[1:d[0]+1]):>14}  {name}")

    print()
    print("  For a .ptseries.nii the first number is TIMEPOINTS, the second PARCELS.")


if __name__ == "__main__":
    main()
