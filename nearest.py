import lwvlib
import sys
import argparse

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Nearest in several models')
    parser.add_argument("models", nargs="+", help='At leas one bin file.')
    parser.add_argument("--max-rank-mem", type=int, default=None, help='Max vectors in memory')
    parser.add_argument("--max-rank", type=int, default=None, help='Max vectors total')
    args = parser.parse_args()

    models=[lwvlib.load(m,max_rank_mem=args.max_rank_mem,max_rank=args.max_rank) for m in args.models]
    while True:
        w=input("> ")
        nearest=[m.nearest(w,10) for m in models]
        for nn in zip(*nearest):
            for sim,neighb in nn:
                print("{:.2f} {:15s}".format(sim,neighb),"       ",end="")
            print()
        print()
        print()
        
            
    
