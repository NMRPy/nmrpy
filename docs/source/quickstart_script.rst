The full script for the quickstart tutorial:

.. code:: python

    import nmrpy
    import os
    from matplotlib import pyplot as plt

    fname = os.path.join(os.path.dirname(nmrpy.__file__), 'tests',
                         'test_data', 'test1.fid')
    fid_array = nmrpy.from_path(fid_path=fname)
    fid_array.emhz_fids()
    #fid_array.fid00.plot_ppm()
    fid_array.ft_fids()
    #fid_array.fid00.plot_ppm()
    #fid_array.fid00.phaser()
    fid_array.phase_correct_fids()
    #fid_array.fid00.plot_ppm()
    fid_array.real_fids()
    fid_array.norm_fids()
    #fid_array.plot_array()
    #fid_array.plot_array(upper_ppm=7, lower_ppm=-1, filled=True, azim=-76, elev=23)
    #fid_array.calibrate()
    
    peaks = [ 4.73,  4.63,  4.15,  0.55]
    ranges = [[ 5.92,  3.24], [ 1.19, -0.01]]
    for fid in fid_array.get_fids():
        fid.peaks = peaks
        fid.ranges = ranges
    
    fid_array.deconv_fids()

    #fid_array.fid10.plot_deconv(upper_ppm=5.5, lower_ppm=3.5)
    #fid_array.fid10.plot_deconv(upper_ppm=0.9, lower_ppm=0.2)
    #fid_array.plot_deconv_array(upper_ppm=6, lower_ppm=3)
    
    integrals = fid_array.deconvoluted_integrals.transpose()
    
    g6p = integrals[0] + integrals[1]
    f6p = integrals[2]
    tep = integrals[3]
    
    #scale species by internal standard tep at 5 mM
    g6p = 5.0*g6p/tep.mean()
    f6p = 5.0*f6p/tep.mean()
    tep = 5.0*tep/tep.mean()
    
    species = {'g6p': g6p,
               'f6p': f6p,
               'tep': tep}
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    for k, v in species.items():
        ax.plot(fid_array.t, v, label=k)
    ax.set_xlabel('min')
    ax.set_ylabel('mM')
    ax.legend(loc=0, frameon=False)
    plt.show()
    
    print([f.id for f in fid_array.get_fids()])

    #delete selected Fids from array
    for fid_id in [f.id for f in fid_array.get_fids()][::4]:
        fid_array.del_fid(fid_id)
    print([f.id for f in fid_array.get_fids()])
    print(['{:.2f}'.format(i) for i in fid_array.t])
    #fid_array.plot_array(upper_ppm=7, lower_ppm=-1, filled=True, azim=-68, elev=25)
    
    #fid_array.save_to_file(filename='fidarray.nmrpy')
    #fid_array = nmrpy.from_path(fid_path='fidarray.nmrpy')
