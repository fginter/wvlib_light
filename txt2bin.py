import lwvlib
import sys

if __name__=="__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Convert .txt into .bin')
    parser.add_argument("input", nargs="?", default=sys.stdin, help='Input file name or nothing for stdin.')
    parser.add_argument("output", nargs="?", default=sys.stdout.buffer, help='Output file name or nothing for stdout.')
    parser.add_argument('--max', type=int, default=0, help='How many vectors to preserve? 0 for all. default: %(default)d')
    args = parser.parse_args()

    lwvlib.txt2bin(args.input,args.output,args.max)
    
