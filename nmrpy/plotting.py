import numpy
import scipy
import pylab
import numbers
from datetime import datetime
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

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
        
    def _plot_array(self, data, params):
        if not Plot._is_iter_of_iters(data): 
            raise AttributeError('data must be 2D.')
        sw_left = params['sw_left']
        sw = params['sw']
        ppm = numpy.linspace(sw_left-sw, sw_left, len(data[0]))[::-1]
        acqtime = params['acqtime'][0]
        minutes = numpy.arange(len(data))*acqtime
        self.fig = pylab.figure(figsize=[10,5])
        ax = self.fig.add_subplot(111, projection='3d')
        for datum, minute in zip(data, minutes):
            ax.plot(ppm, len(datum)*[minute], datum, 'b')
        ax.invert_xaxis()
        ax.set_xlim([sw_left, sw_left-sw])
        ax.set_xlabel('PPM (%.2f MHz)'%(params['reffrq']))
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
