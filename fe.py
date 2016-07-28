import numpy as np
import sidekit as sk
import h5py
import os
import sys
from multiprocessing import Pool, cpu_count

def mfccParallel(args):
    nchunk, chunk, path, lf, hf, nceps, spec, mspec = args
    mfccf = h5py.File(os.path.join(path, 'mfcc.hdf'))
    logef = h5py.File(os.path.join(path, 'loge.hdf'))
    if spec:
        specf = h5py.File(os.path.join(path, 'spec.hdf'))
    if mspec:
        mspecf = h5py.File(os.path.join(path, 'mspec.hdf'))

    n = 'chunk'+str(nchunk)
    print(n)

    for f in chunk:
        name = os.path.split(f)[-1].split('.')[0]
        data, sr = sk.frontend.io.read_audio(f, 16000)
        feat = sk.frontend.features.mfcc(input_sig=data, fs=sr, lowfreq=lf, maxfreq=hf, nlinfilt=0, nlogfilt=24, nwin=0.025, nceps=nceps, get_spec=spec, get_mspec=mspec)
        mfccf[n].create_dataset(name=f, data=feat[0])
        logef[n].create_dataset(name=f, data=feat[1])
        if spec:
            specf[n].create_dataset(name=f, data=feat[2])
        if mspec:
            mspecf[n].create_dataset(name=f, data=feat[3])

    mfccf.close()
    logef.close()
    if spec:
        specf.close()
    if mspec:
        mspecf.close()

def mfccSave(inpath='/home/cilsat/data/perisalah/studio/wav-studio-read/1.3', outpath='.', ncpu=4, lf=150, hf=6000, nceps=12, spec=True, mspec=True):
    files = [os.path.join(p, n) for p,_,f in os.walk(inpath) for n in f if n.endswith('.wav')]
    chunksize = int(len(files)/ncpu)
    chunks = [files[i:i+chunksize] for i in range(0, len(files), chunksize)]


    mfccf = h5py.File(os.path.join(outpath, 'mfcc.hdf'))
    logef = h5py.File(os.path.join(outpath, 'loge.hdf'))
    if spec:
        specf = h5py.File(os.path.join(outpath, 'spec.hdf'))
    if mspec:
        mspecf = h5py.File(os.path.join(outpath, 'mspec.hdf'))
    try: 
        for n in range(ncpu):
            mfccf.create_group('chunk'+str(n))
            logef.create_group('chunk'+str(n))
            specf.create_group('chunk'+str(n))
            mspecf.create_group('chunk'+str(n))
    except:
        pass
    mfccf.close()
    logef.close()
    if spec:
        specf.close()
    if mspec:
        mspecf.close()

    #[mfccParallel([nchunk, chunk, outpath, lf, hf, nceps, spec, mspec]) for nchunk, chunk in enumerate(chunks)]
    pool = Pool(cpu_count())
    pool.map(mfccParallel, [[nchunk, chunk, outpath, lf, hf, nceps, spec, mspec] for nchunk, chunk in enumerate(chunks)])
    pool.close()
    pool.terminate()
    pool.join()
