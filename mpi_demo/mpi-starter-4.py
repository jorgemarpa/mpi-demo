#!/usr/bin/env python

import argparse
import glob
import os
import time

import numpy as np

N = 2000

def load_frame(fname):
    #- dummy function since I can't access the actual files
    return fname * np.ones((N, N), dtype='i')

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--mpi", action="store_true", help="use mpi")
    parser.add_argument("--max-frames", type=int, default=0, help="maximum number of frames to process")
    parser.add_argument("--frames-per-rank", type=int, default=1, help="number of frames per rank to use in each batch")
    parser.add_argument("--buffer-gather", action="store_true", help="use buffer objects to gather frames")
    parser.add_argument("--interleave-io", action="store_true", help="interleave io")
    args = parser.parse_args()

    time_start = time.time()
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
    time_mpi_init = time.time()
    if rank == 0:
        print(f"time mpi init: {time_mpi_init-time_start:.4f} s")

    # Initialize list of frames on rank 0
    if rank == 0:
        # fnames = glob.glob('/nobackupp12/chedges/tess/sector01/camera1/ccd1/*ffic.fits.gz')
        fnames = list(range(1234))
        if args.max_frames > 0:
            fnames = fnames[:args.max_frames]

        # initialize back drop here?
    else:
        fnames = None
    time_fnames_init = time.time()
    if rank == 0:
        print(f"time fnames init: {time_fnames_init-time_mpi_init:.4f} s")

    # Broadcast filenames to all other ranks
    if comm is not None:
        fnames = comm.bcast(fnames, root=0)
    else:
        # nothing to do if we are not using mpi
        pass
    time_fnames_bcast = time.time()
    if rank == 0:
        print(f"time fnames bcast: {time_fnames_bcast-time_fnames_init:.4f} s")

    if args.interleave_io:
        # use all ranks except for root
        num_io_ranks = size - 1
        iorank = rank - 1
    else:
        # use all ranks for io
        num_io_ranks = size
        iorank = rank

    num_files = len(fnames)
    frames_per_rank = args.frames_per_rank
    batch_size = frames_per_rank*num_io_ranks
    # round up in case batch_size does not evenly divide num_files
    num_batches = (num_files + batch_size - 1) // batch_size

    # read and process files in batches
    for batch_index in range(num_batches):
        time_batch_start = time.time()
        batch_start = batch_index * batch_size
        batch_stop = min((batch_index + 1) * batch_size, num_files)
        batch_fnames = fnames[batch_start:batch_stop]

        if args.buffer_gather:
            sendbuf = np.zeros((frames_per_rank, N, N), dtype='i')
            recvbuf = None
            if rank == 0:
                recvbuf = np.empty([size, frames_per_rank, N, N], dtype='i')
        else:
            frames = []


        if iorank < len(batch_fnames) and iorank >= 0:
            # Each rank reads a different filename
            for i, fname in enumerate(batch_fnames[iorank::size]):
                print(f"{batch_index=} rank {rank} will read: {fname}")
                frame = load_frame(fname)
                # simulate slow IO
                time.sleep(0.5)
                if args.buffer_gather:
                    sendbuf[i] = frame
                else:
                    frames.append(frame)
        else:
            # if batch_size does not evenly divide num_files, the final
            # batch will have some ranks that do no have any files to read in
            pass

        # this barrier is for timing measurement only, other timing statements
        # do not have this barrrier because they are performed after collective operations
        # or only relevant to rank 0
        if comm is not None and not args.interleave_io:
            # don't want barrier if we're interleaving io
            comm.barrier()
        time_batch_load = time.time()
        if rank == 0:
            print(f"time batch load: {time_batch_load-time_batch_start:.4f} s")

        # gather frames to rank 0
        if comm is not None:
            if args.buffer_gather:
                comm.Gather(sendbuf, recvbuf, root=0)
                if rank == 0:
                    # transpose outermost dimensions, then flatten batch dimension
                    if args.interleave_io:
                        # skip empty buffer from root rank when interleaving io
                        recvbuf = recvbuf[1:]
                    frames = recvbuf.reshape(batch_size, N, N)
            else:
                # frames on rank 0 is a list of lists
                frames = comm.gather(frames, root=0)
                # unpack/flatten list of lists
                if rank == 0:
                    # transpose, flatten, and stack all in one line!
                    frames = np.stack([item for sublist in frames for item in sublist])
                else:
                    # other ranks have nothing to do
                    pass
        else:
            frames = np.stack(frames)
    
        time_batch_gather = time.time()
        if rank == 0:
            print(f"time batch gather: {time_batch_gather-time_batch_load:.4f} s")

        if rank == 0:
            # process frames on rank 0 / perform back drop here
            result = np.average(frames, axis=(-2, -1))
            # simulate slow process
            time.sleep(0.5)
            print(f"{batch_index=} {result}")
            # write results here?
        else:
            # other ranks have nothing to do
            pass

        time_batch_process = time.time()
        if rank == 0:
            print(f"time batch process: {time_batch_process-time_batch_gather:.4f} s")

    time_end = time.time()
    if rank == 0:
        print(f"time total: {time_end-time_start:.4f} s")


if __name__ == "__main__":
    main()
