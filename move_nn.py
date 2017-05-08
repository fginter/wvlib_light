import lwvlib
import argparse
import numpy as np
import sys
import six
assert six.PY3, "please run me with Python3"

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Move neighbor vectors')
    parser.add_argument('pre', metavar='binfile', help='Model in its original form')
    parser.add_argument('post', metavar='binfile', help='Model in its post-training form')
    parser.add_argument('tomove', metavar='binfile', help='Model to move (efficiency noted if move is same as pre)')
    parser.add_argument('out', metavar='binfile', help='Model to save to')
    parser.add_argument('--mbsize', type=int, default=1000, help='Minibatch size. Default: %(default)d')
    parser.add_argument('--max-pre', type=int, default=0, help='How many vectors to read for pre? 0 -> as many as possible. Default: %(default)d')
    parser.add_argument('--max-tomove', type=int, default=0, help='How many vectors to read for tomove? 0 -> as many as possible. Default: %(default)d')

    args = parser.parse_args()
    if args.max_pre==0:
        args.max_pre=None
    if args.max_tomove==0:
        args.max_tomove=None
    
    pre=lwvlib.load(args.pre,args.max_pre,args.max_pre)
    post=lwvlib.load(args.post,args.max_pre,args.max_pre)
    pre_vec=pre.vectors.astype(np.double)

    post_vec=post.vectors.astype(np.double)
    assert len(pre_vec)>=len(post_vec)
    post_vec=post_vec[:len(pre_vec),]
    diff=post_vec-pre_vec
    diff_mag=np.linalg.norm(diff,axis=1,ord=2)
    diff_mag/=max(diff_mag) #Magnitude of the difference, used as a scaling factor for its effect
    tomove=lwvlib.load(args.tomove,args.max_tomove,args.max_tomove)
    tomove_vec=tomove.vectors.astype(np.double)
    tomove_words,vec_dim=tomove_vec.shape
    #Padding
    mb_size=args.mbsize
    if (tomove_words%mb_size)>0:
        stretch=np.zeros((mb_size-tomove_words%mb_size,vec_dim),dtype=tomove_vec.dtype)
        tomove_vec=np.concatenate((tomove_vec,stretch))
        stretch=np.ones((mb_size-tomove_words%mb_size),dtype=tomove.norm_constants.dtype)
        tomove.norm_constants=np.concatenate((tomove.norm_constants,stretch))
    assert len(tomove_vec)%mb_size==0
    assert len(tomove.norm_constants)%mb_size==0

    tomove_batched=tomove_vec.reshape(len(tomove_vec)//mb_size,mb_size,vec_dim)
    tomove_norm_constants_batched=tomove.norm_constants.reshape(len(tomove.norm_constants)//mb_size,mb_size,1)
    for batch in range(len(tomove_batched)):
        distances=tomove_batched[batch].dot(pre_vec.T)/tomove_norm_constants_batched[batch]/pre.norm_constants[None,:] #Similarities to the pre vectors
        distances=2.0*distances-1.0  #distances in range 0.5...1 are scaled to 0...1, everything else is negative
        np.clip(distances,0.0,1.0,distances) #clip to the 0...1 range this is how much each neighbor affects
        distances*=diff_mag[None,:]
        moves=distances.dot(diff) #diffs summed up and rescaled by distances
        sum_scales=np.sum(distances,axis=1)[:,None] #sum of the scales for every row, completes weighted average
        sum_scales[sum_scales==0.0]=1.0 #avoid division by zero
        moves/=sum_scales #weighted average of moves
        #So these are now my moves
        tomove_batched[batch]+=moves
        print("Batch",batch,"/",len(tomove_batched),file=sys.stderr,flush=True)
    tomove.vectors=(tomove_batched.reshape(len(tomove_batched)*mb_size,vec_dim)[:tomove_words]).astype(np.float32)
    if args.out.endswith(".vector") or args.out.endswith(".vectors"):
        tomove.save_txt(args.out)
    elif args.out.endswith(".bin"):
        tomove.save_bin(args.out)
    
