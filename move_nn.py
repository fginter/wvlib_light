import lwvlib
import argparse
import numpy as np

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Move neighbor vectors')
    parser.add_argument('pre', metavar='binfile', help='Model in its original form')
    parser.add_argument('post', metavar='binfile', help='Model in its post-training form')
    parser.add_argument('tomove', metavar='binfile', help='Model to move (efficiency noted if move is same as pre)')
    parser.add_argument('out', metavar='binfile', help='Model to save to')
    

    args = parser.parse_args()

    pre=lwvlib.load(args.pre,12000,12000)
    post=lwvlib.load(args.post,12000,12000)

    pre_vec=pre.vectors
    post_vec=post.vectors
    diff=post_vec-pre_vec

    tomove=lwvlib.load(args.tomove,12000,12000)
    tomove_vec=tomove.vectors

    tomove_words,vec_dim=tomove_vec.shape
    mb_size=100
    tomove_batched=tomove_vec.reshape(tomove_words//mb_size,mb_size,vec_dim)
    tomove_norm_constants_batched=tomove.norm_constants.reshape(tomove_words//mb_size,mb_size,1)
    for batch in range(1): #range(len(move_batched))
        distances=tomove_batched[batch].dot(pre_vec.T)/tomove_norm_constants_batched[batch]/pre.norm_constants[None,:] #Similarities to the pre vectors
        distances=2.0*distances-1.0  #distances in range 0.5...1 are scaled to 0...1, everything else is negative
        np.clip(distances,0.0,1.0,distances) #clip to the 0...1 range this is how much each neighbor affects
        moves=distances.dot(diff) #diffs summed up and rescaled by distances
        sum_scales=np.sum(distances,axis=1)[:,None] #sum of the scales for every row, completes weighted average
        sum_scales[sum_scales==0.0]=1.0 #avoid division by zero
        moves/=sum_scales #weighted average of moves
        #So these are now my moves
        tomove_batched[batch]+=moves
    tomove.vectors=tomove_batched.reshape(tomove_words,vec_dim)
    tomove.save(args.out)
    
