import numpy
import scipy
import pylab
import numbers
from datetime import datetime
from matplotlib.figure import Figure

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
        

if __name__ == '__main__':
    pass
