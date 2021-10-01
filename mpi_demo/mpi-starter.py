#!/usr/bin/env python

import argparse
import glob

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--mpi", action="store_true", help="use mpi")
    parser.add_argument("--max-frames", type=int, default=0, help="maximum number of frames to process")
    args = parser.parse_args()

    if args.mpi:
        # initialize mpi if requested
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.rank
        size = comm.size
    else:
        # otherewise, setup placeholder values
        comm = None
        rank = 0
        size = 1

    print(f"Hello from rank {rank} of {size}")

    # Initialize list of frames on rank 0
    if rank == 0:
        fnames = glob.glob('/nobackupp12/chedges/tess/sector01/camera1/ccd1/*ffic.fits.gz')
        if args.max_frames > 0:
            fnames = fnames[:args.max_frames]
    else:
        fnames = None
    
    # Broadcast filenames to all other ranks
    if comm is not None:
        fnames = comm.bcast(fnames, root=0)

    # Each rank reads a different filename
    for fname in fnames[rank::size]:
        print(f"rank {rank} will read: {fname}")


if __name__ == "__main__":
    main()
