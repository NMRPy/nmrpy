import numpy
import scipy
import pylab
import numbers
from datetime import datetime
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection

class Plot():

    _plot_id_num = 0

    def __init__(self):
        self._time = datetime.now()
        self.id = 'plot_{}'.format(Plot._plot_id_num)
        Plot._plot_id_num += 1
        self.fig = None

    @property
    def id(self):
        return self.__id

    @id.setter
    def id(self, id):
        self.__id = id

    @property
    def fig(self):
        return self._fig

    @fig.setter
    def fig(self, fig):
        if fig is None or isinstance(fig, Figure):
            self._fig = fig
        else:
            raise AttributeError('fig must be of type matplotlib.figure.Figure.')

    def _plot_ppm(self, data, params):
        if not Plot._is_flat_iter(data): 
            raise AttributeError('data must be flat iterable.')
        sw_left = params['sw_left']
        sw = params['sw']
        ppm = numpy.linspace(sw_left-sw, sw_left, len(data))[::-1]
        self.fig = pylab.figure(figsize=[10,5])
        ax = self.fig.add_subplot(111)
        ax.plot(ppm, data)
        ax.invert_xaxis()
        ax.set_xlim([sw_left, sw_left-sw])
        ax.grid()
        ax.set_xlabel('PPM (%.2f MHz)'%(params['reffrq']))
        #self.fig.show()
        
    def _plot_array(self, data, params, upper_index=None, lower_index=None, upper_ppm=None, lower_ppm=None, lw=0.3, azim=-90, elev=40, filled=False, labels=None, filename=None):
        if not Plot._is_iter_of_iters(data): 
            raise AttributeError('data must be 2D.')
        if upper_ppm is not None and lower_ppm is not None:
            if upper_ppm == lower_ppm or upper_ppm < lower_ppm:
                raise ValueError('ppm range specified is invalid.')
        if upper_index is not None and lower_index is not None:
            if upper_index == lower_index or upper_index < lower_index:
                raise ValueError('index range specified is invalid.')


        sw_left = params['sw_left']
        sw = params['sw']

        if upper_index is None:
            upper_index = -1
        if lower_index is None:
            lower_index = 0
        
        if upper_ppm is None:
            upper_ppm = sw_left
        if lower_ppm is None:
            lower_ppm = sw_left-sw

        ppm = numpy.linspace(sw_left-sw, sw_left, data.shape[1])[::-1]
        ppm_bool_index = (ppm < upper_ppm) * (ppm > lower_ppm)
        ppm = ppm[ppm_bool_index]
        data = data[lower_index:upper_index, ppm_bool_index]

        bh = abs(data.min()) 

        acqtime = params['acqtime'][0]
        minutes = numpy.arange(len(data))*acqtime
        self.fig = pylab.figure(figsize=[15,5])
        ax = self.fig.add_subplot(111, projection='3d', azim=azim, elev=elev)

        if not filled:
            for datum, minute in zip(data, minutes):
                ax.plot(ppm, len(datum)*[minute], datum, color='k', lw=lw)
        if filled:
            verts = []
            plot_data = data+bh 
            for datum in plot_data:
                datum[0], datum[-1] = 0, 0
                verts.append(list(zip(ppm, datum)))
            
            poly = PolyCollection(verts, facecolors=['w']*len(verts), linewidths=[lw]*len(verts))
            ax.add_collection3d(poly, zs=minutes, zdir='y')
            ax.set_zlim([0, 1.1*plot_data.max()])

        ax.invert_xaxis()
        ax.set_xlim([upper_ppm, lower_ppm])
        ax.set_ylim([minutes[0], minutes[-1]])
        ax.set_xlabel('PPM (%.2f MHz)'%(params['reffrq']))
        ax.set_ylabel('min.')
        #self.fig.show()

    @classmethod
    def _is_iter(cls, i):
        try:
            iter(i)
            return True
        except TypeError:
            return False

    @classmethod
    def _is_iter_of_iters(cls, i):
        if i == []:
            return False
        elif cls._is_iter(i) and all(cls._is_iter(j) for j in i):
            return True
        return False

    @classmethod
    def _is_flat_iter(cls, i):
        if i == []:
            return True
        elif cls._is_iter(i) and not any(cls._is_iter(j) for j in i):
            return True
        return False

if __name__ == '__main__':
    pass
