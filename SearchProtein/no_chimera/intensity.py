import numpy as np
import sys
from scipy import spatial, ndimage


def cal_intensity_temp(pdb):
    side = 72
    nsamples = 16
    voxel = side / nsamples
    halfside = side / 2
    n = int(side / voxel)
    # want n to be even for speed/memory optimization with the FFT, ideally a power of 2, but wont enforce that
    if n % 2 == 1: n += 1
    dx = side / n
    x_ = np.linspace(-halfside, halfside, n)
    x, y, z = np.meshgrid(x_, x_, x_, indexing='ij')
    xyz = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    rho = gauss(pdb, xyz=xyz, sigma=10, mode="slow")
    return rho, side


def gauss(pdb,xyz,sigma,mode="slow",eps=1e-6):
    """Simple isotropic gaussian sum at coordinate locations.

    Fast mode uses KDTree to only calculate density at grid points with
    a density above a threshold.
    see https://stackoverflow.com/questions/52208434"""
    n = int(round(xyz.shape[0]**(1/3.)))
    sigma /= 4.
    dx = xyz[1,2] - xyz[0,2]
    shift = np.ones(3)*dx/2.
    values = np.zeros((xyz.shape[0]))
    for i in range(pdb.coords.shape[0]):
        sys.stdout.write("\r% 5i / % 5i atoms" % (i+1,pdb.coords.shape[0]))
        sys.stdout.flush()
        dist = spatial.distance.cdist(pdb.coords[None,i]-shift, xyz)
        dist *= dist
        values += pdb.nelectrons[i]*1./np.sqrt(2*np.pi*sigma**2) * np.exp(-dist[0]/(2*sigma**2))
    return values.reshape(n,n,n)


def cal_intensity(pdb, out_iq_path = 'out_iq_path.dat'):
    ns = 1
    threshold = 0.0
    rho, side = cal_intensity_temp(pdb)
    if rho.shape[0] % 2 == 1:
        rho = rho[:-1, :-1, :-1]
    rho = np.copy(rho[::ns, ::ns, ::ns])
    rho[rho <= threshold] = 0
    halfside = side / 2
    nx, ny, nz = rho.shape[0], rho.shape[1], rho.shape[2]
    n = nx
    voxel = side / n
    # want n to be even for speed/memory optimization with the FFT, ideally a power of 2, but wont enforce that
    # store n for later use if needed
    n_orig = n
    dx = side / n
    dV = dx ** 3
    V = side ** 3
    x_ = np.linspace(-halfside, halfside, n)
    x, y, z = np.meshgrid(x_, x_, x_, indexing='ij')
    df = 1 / side
    qx_ = np.fft.fftfreq(x_.size) * n * df * 2 * np.pi
    qz_ = np.fft.rfftfreq(x_.size) * n * df * 2 * np.pi
    qx, qy, qz = np.meshgrid(qx_, qx_, qx_, indexing='ij')
    qr = np.sqrt(qx ** 2 + qy ** 2 + qz ** 2)
    qmax = np.max(qr)
    qstep = np.min(qr[qr > 0])
    nbins = int(qmax / qstep)
    qbins = np.linspace(0, nbins * qstep, nbins*10)
    # print(qbins)
    # create modified qbins and put qbins in center of bin rather than at left edge of bin.
    qbinsc = np.copy(qbins)
    qbinsc[1:] += qstep / 2.
    # create an array labeling each voxel according to which qbin it belongs
    qbin_labels = np.searchsorted(qbins, qr, "right")
    qbin_labels -= 1
    # create list of qbin indices just in region of data for later F scaling
    qbin_args = np.copy(qbinsc)
    F = np.fft.fftn(rho)
    I3D = np.abs(F) ** 2
    # Imean = ndimage.mean(I3D, labels=qbin_labels, index=np.arange(0, qbin_labels.max() + 1))
    Imean = ndimage.mean(I3D, labels=qbin_labels, index=np.unique(qbin_labels))
    print('Default dq = %.4f' % (2 * np.pi / side))
    dq = None
    n = None
    if dq is not None or n is not None:
        # padded to get desired dq value (or near it)
        if n is not None:
            n = n
        else:
            current_dq = 2 * np.pi / side
            desired_dq = dq
            if desired_dq > current_dq:
                print
                "desired dq must be smaller than dq calculated from map (which is %f)" % current_dq
                print
                "Resetting desired dq to current dq..."
                desired_dq = current_dq
            # what side would give us desired dq?
            desired_side = 2 * np.pi / desired_dq
            # what n, given the existing voxel size, would give us closest to desired_side
            desired_n = desired_side / voxel
            n = int(desired_n)
            if n % 2 == 1: n += 1
        side = voxel * n
        halfside = side / 2
        dx = side / n
        dV = dx ** 3
        V = side ** 3
        x_ = np.linspace(-halfside, halfside, n)
        x, y, z = np.meshgrid(x_, x_, x_, indexing='ij')
        df = 1 / side
        print(n, 2 * np.pi * df)
        qx_ = np.fft.fftfreq(x_.size) * n * df * 2 * np.pi
        qx, qy, qz = np.meshgrid(qx_, qx_, qx_, indexing='ij')
        qr = np.sqrt(qx ** 2 + qy ** 2 + qz ** 2)
        qmax = np.max(qr)
        qstep = np.min(qr[qr > 0])
        nbins = int(qmax / qstep)
        qbins = np.linspace(0, nbins * qstep, nbins + 1)
        # create modified qbins and put qbins in center of bin rather than at left edge of bin.
        qbinsc = np.copy(qbins)
        qbinsc[1:] += qstep / 2.
        # create an array labeling each voxel according to which qbin it belongs
        qbin_labels = np.searchsorted(qbins, qr, "right")
        qbin_labels -= 1
        # create list of qbin indices just in region of data for later F scaling
        qbin_args = np.copy(qbinsc)
        rho_pad = np.zeros((n, n, n), dtype=np.float32)
        a = n / 2 - n_orig / 2
        b = n / 2 + n_orig / 2
        rho_pad[a:b, a:b, a:b] = rho
        F = np.fft.fftn(rho_pad)
        I3D = np.abs(F) ** 2
        Imean = ndimage.mean(I3D, labels=qbin_labels, index=np.arange(0, qbin_labels.max() + 1))

    qmax_to_use = np.max(qx_)
    qbinsc_to_use = qbinsc[qbinsc<qmax_to_use]
    isize = np.shape(Imean)
    qbinsc = qbinsc[:isize[0]]
    Imean_to_use = Imean[qbinsc<qmax_to_use]
    qbinsc = np.copy(qbinsc_to_use)
    Imean = np.copy(Imean_to_use)
    Iq = np.vstack((qbinsc, Imean, Imean * .03)).T
    np.savetxt(out_iq_path, Iq, delimiter=' ', fmt='% .16e')
    # print(Iq)
    return Iq

# import handle
# import os
# if __name__ == "__main__":
#     os.system('sastbx.she structure=/Users/wyf/Desktop/sbw/auto_fit822/correct_test/dat/t0_t0.pdb experimental_data=/Users/wyf/Desktop/sbw/auto_fit822/correct_test/dat/t0_t0_saxs.dat output=/Users/wyf/Desktop/sbw/auto_fit822/correct_test/dat/tbx33333333.dat ')
    # pdb_path = 'correct_test/dat/t0_t0.pdb'
    # pdb = handle.PDB(pdb_path)
    # iq = cal_intensity(pdb)
    # np.savetxt('correct_test/dat/iq_denss.dat', iq, delimiter=' ', fmt='% .16e')