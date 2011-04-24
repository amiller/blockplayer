import numpy as np
import opencl
import config


def carve_opencl(*args, **kwargs):
    kwargs['use_opencl']=True
    return carve(*args, **kwargs)


def carve(xfix=None, zfix=None, use_opencl=True):
    global gridinds, inds, grid
    gridmin = np.zeros((4,),'f')
    gridmax = np.zeros((4,),'f')
    gridmin[:3] = config.bounds[0]
    gridmax[:3] = config.bounds[1]

    if xfix is None or zfix is None:
        import lattice
        xfix, zfix = lattice.meanx, lattice.meanz

    if use_opencl:
        opencl.compute_gridinds(xfix,zfix,
                                config.LW, config.LH,
                                gridmin, gridmax)
        gridinds = opencl.get_gridinds()
        inds = gridinds[gridinds[:,0,3]!=0,:,:3]
    else:
        global X,Y,Z, XYZ
        X,Y,Z,face = np.rollaxis(opencl.get_modelxyz(),1)
        XYZ = np.array((X,Y,Z)).transpose()
        fix = np.array((xfix,0,zfix))
        cxyz = np.frombuffer(np.array(face).data,
                             dtype='i1').reshape(-1,4)[:,:3]
        global cx,cy,cz
        cx,cy,cz = np.rollaxis(cxyz,1)
        f1 = cxyz*0.5

        mod = np.array([config.LW, config.LH, config.LW])
        gi = np.floor(-gridmin[:3] + (XYZ-fix)/mod + f1)
        gi = np.array((gi, gi - cxyz))
        gi = np.rollaxis(gi, 1)

        def get_gridinds():
            (L,T),(R,B) = opencl.rect
            length = opencl.length
            return (gridinds[:length,:,:].reshape(T-B,R-L,2,3),
                    gridinds[length:,:,:].reshape(T-B,R-L,2,3))

        gridinds = gi
        grid = get_gridinds()
        inds = gridinds[np.any(cxyz!=0,1),:,:]

    if len(inds) == 0:
        return None

    bins = [np.arange(0,gridmax[i]-gridmin[i]+1)
            for i in range(3)]
    global occ, vac
    occH,_ = np.histogramdd(inds[:,0,:], bins)
    vacH,_ = np.histogramdd(inds[:,1,:], bins)

    vac = vacH>30
    occ = occH>30

    return occ, vac
