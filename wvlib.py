"""
wv=WV.load("somefile.bin",10000,500000)
"""

import numpy
import mmap
import os
import StringIO

class WV(object):

    @staticmethod
    def read_word(inp):
        """
        Reads a single word from the input file
        """
        chars=[]
        while True:
            c = inp.read(1)
            if c == ' ':
                break
            if not c:
                raise ValueError("preliminary end of file")
            chars.append(c)
        wrd=''.join(chars).strip()
        try:
            return unicode(wrd,"utf-8")
        except UnicodeDecodeError:
            #Not a utf-8, shoots, what now?
            #maybe I should warn here TODO
            return unicode(wrd,"utf-8","replace")
        
    
    @classmethod
    def load(cls,file_name,max_rank_mem=None,max_rank=None,float_type=numpy.float32):
        """
        Loads a w2v bin file. 
        `inp` an open file or a file name
        `max_rank_mem` read up to this many vectors into an internal matrix, the rest is memory-mapped
        `max_rank` read up to this many vectors, memory-mapping whatever above max_rank_mem
        `float_type` the type of the vector matrix
        """
        f=open(file_name,"r+b")
        #Read the size line
        try:
            l=f.readline().strip()
            wcount,vsize=l.split()
            wcount,vsize=int(wcount),int(vsize)
        except ValueError:
            raise ValueError("Size line in the file is malformed: '%s'. Maybe this is not a w2v binary file?"%l)

        if max_rank is None or max_rank>wcount:
            max_rank=wcount

        if max_rank_mem is None or max_rank_mem>max_rank:
            max_rank_mem=max_rank

        #offsets: byte offsets at which the vectors start
        offsets=[]
        #words: the words themselves
        words=[]
        #data: the vector matrix for the first max_rank vectors
        data=numpy.zeros((max_rank_mem,vsize),float_type)

        #Now read one word at a time, fill into the matrix
        for idx in range(max_rank_mem):
            words.append(cls.read_word(f))
            offsets.append(f.tell())
            data[idx,:]=numpy.fromfile(f,numpy.float32,vsize)
        #Keep reading, but only remember the offsets
        for idx in range(max_rank_mem,max_rank):
            words.append(cls.read_word(f))
            offsets.append(f.tell())
            f.seek(vsize*4,os.SEEK_CUR) #seek over the vector (4 is the size of float32)
        fm=mmap.mmap(f.fileno(),0)
        return cls(words,data,fm,offsets)
    
    def __init__(self,words,vector_matrix,mm_file,offsets):
        """
        `words`: list of words
        `vector_matrix`: numpy matrix
        `mm_file`: memory-mapped .bin file with the vectors
        `offsets`: for every word, the offset at which its vector starts
        """
        self.vectors=vector_matrix #Numpy matrix
        self.words=words #The words to go with them
        self.w_to_dim=dict((w,i) for i,w in enumerate(self.words))
        self.mm_file=mm_file
        self.offsets=offsets
        self.max_rank_mem,self.vsize=self.vectors.shape
        #normalization constants for every row
        self.norm_constants=numpy.linalg.norm(x=self.vectors,ord=None,axis=1)#.reshape(self.max_rank,1) #Column vector of norms

    def w_to_normv(self,wrd):
        #Return a normalized vector for wrd if you can, None if you cannot
        wrd_dim=self.w_to_dim.get(wrd)
        if wrd_dim is None:
            return None #We know nothing of this word, sorry
        if wrd_dim<self.max_rank_mem: #We have the vector loaded in memory
            return self.vectors[wrd_dim]/self.norm_constants[wrd_dim]
        else: #We don't have the vector loaded in memory, grab it from the file
            vec=numpy.fromstring(self.mm_file[self.offsets[wrd_dim]:self.offsets[wrd_dim]+self.vsize*4],numpy.float32,self.vsize).astype(self.vectors.dtype)
            vec/=numpy.linalg.norm(x=vec,ord=None)
            return vec
        
    def nearest(self,wrd,N=10):
        wrd_vec_norm=self.w_to_normv(wrd)
        if wrd_vec_norm is None:
            return
        sims=self.vectors.dot(wrd_vec_norm)/self.norm_constants
        #http://stackoverflow.com/questions/6910641/how-to-get-indices-of-n-maximum-values-in-a-numpy-array
        return sorted(((sims[idx],self.words[idx]) for idx in numpy.argpartition(sims,-N)[-N:]), reverse=True)[1:]

if __name__=="__main__":
    wv=WV.load("pb34_wf_200_v2.bin",50000,200000)
    print wv.nearest(u"ruoka")
        
        
    
