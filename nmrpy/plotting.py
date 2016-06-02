import numpy
import scipy
import pylab
import numbers
from datetime import datetime
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection

from matplotlib.mlab import dist
from matplotlib.patches import Circle, Rectangle
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory
from matplotlib.widgets import Cursor

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
            raise TypeError('fig must be of type matplotlib.figure.Figure.')

    def _plot_ppm(self, data, params, 
            upper_ppm=None, 
            lower_ppm=None, 
            color='k', 
            lw=1,
            filename=None):
        if not Plot._is_flat_iter(data): 
            raise AttributeError('data must be flat iterable.')
        if upper_ppm is not None and lower_ppm is not None:
            if upper_ppm == lower_ppm or upper_ppm < lower_ppm:
                raise ValueError('ppm range specified is invalid.')
        sw_left = params['sw_left']
        sw = params['sw']

        if upper_ppm is None:
            upper_ppm = sw_left
        if lower_ppm is None:
            lower_ppm = sw_left-sw

        ppm = numpy.linspace(sw_left-sw, sw_left, len(data))[::-1]
        ppm_bool_index = (ppm < upper_ppm) * (ppm > lower_ppm)
        ppm = ppm[ppm_bool_index]
        data = data[ppm_bool_index]

        self.fig = pylab.figure(figsize=[10,5])
        ax = self.fig.add_subplot(111)
        ax.plot(ppm, data, color=color, lw=lw)
        ax.invert_xaxis()
        ax.set_xlim([upper_ppm, lower_ppm])
        ax.grid()
        ax.set_xlabel('PPM (%.2f MHz)'%(params['reffrq']))
        #self.fig.show()
        if filename is not None:
            self.fig.savefig(filename, format='pdf')
        
    def _plot_deconv(self, data, params, peakshapes,
            upper_ppm=None, 
            lower_ppm=None, 
            colour='k', 
            peak_colour='b', 
            summed_peak_colour='r', 
            residual_colour='g', 
            lw=1):
        if not Plot._is_flat_iter(data): 
            raise AttributeError('data must be flat iterable.')
        if not Plot._is_iter_of_iters(peakshapes): 
            raise AttributeError('data must be flat iterable.')
        if upper_ppm is not None and lower_ppm is not None:
            if upper_ppm == lower_ppm or upper_ppm < lower_ppm:
                raise ValueError('ppm range specified is invalid.')
        sw_left = params['sw_left']
        sw = params['sw']

        if upper_ppm is None:
            upper_ppm = sw_left
        if lower_ppm is None:
            lower_ppm = sw_left-sw

        ppm = numpy.linspace(sw_left-sw, sw_left, len(data))[::-1]
        ppm_bool_index = (ppm <= upper_ppm) * (ppm >= lower_ppm)
        ppm = ppm[ppm_bool_index]
        data = data[ppm_bool_index]
        peakshapes = peakshapes[:, ppm_bool_index]
        summed_peaks = peakshapes.sum(0)
        residual = data-summed_peaks

        self.fig = pylab.figure(figsize=[10,5])
        ax = self.fig.add_subplot(111)
        ax.plot(ppm, residual, color=residual_colour, lw=lw)
        ax.plot(ppm, data, color=colour, lw=lw)
        ax.plot(ppm, summed_peaks, '--', color=summed_peak_colour, lw=lw)
        for peak in peakshapes:
            ax.plot(ppm, peak, '-', color=peak_colour, lw=lw)
        ax.invert_xaxis()
        ax.set_xlim([upper_ppm, lower_ppm])
        ax.grid()
        ax.set_xlabel('PPM (%.2f MHz)'%(params['reffrq']))
        #self.fig.show()
        
    def _plot_array(self, data, params, 
                upper_index=None, 
                lower_index=None, 
                upper_ppm=None, 
                lower_ppm=None, 
                lw=0.3, 
                azim=-90, 
                elev=40, 
                filled=False, 
                show_zticks=False, 
                labels=None, 
                colour=True,
                filename=None,
                ):

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

        if colour:
            cl = pylab.cm.viridis(numpy.linspace(0, 1, len(data)))
        bh = abs(data.min()) 

        acqtime = params['acqtime'][0]
        minutes = numpy.arange(len(data))*acqtime
        self.fig = pylab.figure(figsize=[15,7.5])
        ax = self.fig.add_subplot(111, projection='3d', azim=azim, elev=elev)

        if not filled:
            #spectra are plotted in reverse for zorder
            for n in range(len(data))[::-1]:
                datum = data[n]
                minute = minutes[n]
                clr = 'k' 
                if colour:
                    clr = cl[n]
                ax.plot(ppm, len(datum)*[minute], datum, color=clr, lw=lw)
        if filled:
            verts = []
            plot_data = data+bh 
            for datum in plot_data:
                datum[0], datum[-1] = 0, 0
                verts.append(list(zip(ppm, datum)))
             
            fclr, eclr = ['w']*len(data), ['k']*len(data)
            if colour:
                fclr = cl
            poly = PolyCollection(verts, 
                facecolors=fclr,
                edgecolors=eclr,
                linewidths=[lw]*len(verts))
            ax.add_collection3d(poly, zs=minutes, zdir='y')
            ax.set_zlim([0, 1.1*plot_data.max()])

        ax.invert_xaxis()
        ax.set_xlim([upper_ppm, lower_ppm])
        ax.set_ylim([minutes[0], minutes[-1]])
        ax.set_xlabel('PPM (%.2f MHz)'%(params['reffrq']))
        ax.set_ylabel('min.')
        if not show_zticks:
            ax.set_zticklabels([])
        #self.fig.show()
        if filename is not None:
            self.fig.savefig(filename, format='pdf')

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

class Phaser(object):
    """Interactive phase-correction widget"""
    def __init__(self, fid):
        if not Plot._is_flat_iter(fid.data): 
            raise ValueError('data must be flat iterable.')
        if fid.data == [] or fid.data == None:
            raise ValueError('data must exist.')
        self.fid = fid
        self.fig = pylab.figure(figsize=[15, 7.5])
        self.phases = numpy.array([0.0, 0.0])
        self.y = 0.0
        self.ax = self.fig.add_subplot(111)
        self.ax.plot(self.fid.data, color='k', linewidth=1.0)
        self.ax.hlines(0, 0, len(self.fid.data)-1)
        self.ax.set_xlim([0, len(self.fid.data)])
        xtcks = numpy.linspace(0,1,10)*len(self.fid.data)
        xtcks[-1] = xtcks[-1]-1
        self.ax.set_xticks(xtcks)
        self.ax.set_xlabel('PPM (%.2f MHz)'%(self.fid._params['reffrq']))
        self.ax.set_xticklabels([numpy.round(self.fid._ppm[int(i)], 1) for i in xtcks])
        ylims = numpy.array([-1, 1])*numpy.array([max(self.ax.get_ylim())]*2)
        self.ax.set_ylim(ylims)
        self.ax.grid()
        self.visible = True
        self.canvas = self.ax.figure.canvas
        self.canvas.mpl_connect('motion_notify_event', self.onmove)
        self.canvas.mpl_connect('button_press_event', self.press)
        self.canvas.mpl_connect('button_release_event', self.release)
        self.pressv = None
        self.buttonDown = False
        self.prev = (0, 0)
        self.ax.text(0.05 *self.ax.get_xlim()[1],0.7 *self.ax.get_ylim()[1],'phasing\nleft - zero-order\nright - first order')
        cursor = Cursor(self.ax, useblit=True, color='k', linewidth=0.5)
        cursor.horizOn = False
        self.fig.show()

    def press(self, event):
        tb = pylab.get_current_fig_manager().toolbar
        if tb.mode == '':
            x, y = event.xdata, event.ydata
            if event.inaxes is not None:
                self.buttonDown = True
                self.button = event.button
                self.y = y

    def release(self, event):
        self.buttonDown = False
        print('p0: {} p1: {}'.format(*self.phases))
        return False

    def onmove(self, event):
        if self.buttonDown is False or event.inaxes is None:
                return
        x = event.xdata
        y = event.ydata
        dy = y-self.y
        self.y = y
        if self.button == 1:
                self.phases[0] = 100*dy/self.ax.get_ylim()[1]
        if self.button == 3:
                self.phases[1] = 100*dy/self.ax.get_ylim()[1]
        self.fid.ps(p0=self.phases[0], p1=self.phases[1])
        self.ax.lines[0].set_data(numpy.array([numpy.arange(len(self.fid.data)), self.fid.data]))
        self.canvas.draw()  # _idle()
        return False

class PeakPicker:
    """Interactive peak-picking widget"""
    def __init__(self, data, params):
        self.fig = pylab.figure(figsize=[15, 7.5])
        self.data = numpy.array(data)
        self.ax = self.fig.add_subplot(111)
        if len(self.data.shape)==1:
            ppm = numpy.mgrid[params['sw_left']-params['sw']:params['sw_left']:complex(data.shape[0])]
            self.ax.plot(ppm[::-1], data, color='k', lw=1)
        elif len(self.data.shape)==2:
            cl = dict(zip(range(len(data)), pylab.cm.viridis(numpy.linspace(0,1,len(data)))))
            ppm = numpy.mgrid[params['sw_left']-params['sw']:params['sw_left']:complex(data.shape[1])]
            inc_orig = 0.5*data.max()/len(data)
            inc = inc_orig.copy()
            for i,j in zip(range(len(data)), data[::-1]):
                self.ax.plot(ppm[::-1], j+inc, color=cl[i], lw=1)
                inc += inc_orig
        self.ax.set_xlabel('ppm')
        self.rectprops = dict(facecolor='0.5', alpha=0.2)
        self.visible = True
        self.canvas = self.ax.figure.canvas
        self.canvas.mpl_connect('motion_notify_event', self.onmove)
        self.canvas.mpl_connect('button_press_event', self.press)
        self.canvas.mpl_connect('button_release_event', self.release)
        self.minspan = 0
        self.rect = None
        self.pressv = None
        self.buttonDown = False
        self.prev = (0, 0)
        trans = blended_transform_factory(
            self.ax.transData,
            self.ax.transAxes)
        w, h = 0, 1
        self.rect = Rectangle([0, 0], w, h,
                              transform=trans,
                              visible=False,
                              **self.rectprops
                              )
        self.ax.add_patch(self.rect)
        self.ranges = []
        self.peaks = []
        self.ylims = numpy.array([self.ax.get_ylim()[0], self.data.max() + abs(self.ax.get_ylim()[0])])
        self.ax.set_ylim([self.ax.get_ylim()[0], self.data.max()*1.1])
        self.ax_lims = self.ax.get_ylim()
        self.xlims = [ppm[-1], ppm[0]]
        self.ax.set_xlim(self.xlims)
        self.ax.text(
            0.95 *
            self.ax.get_xlim()[0],
            0.7 *
            self.ax.get_ylim()[1],
            'Peak picking\nLeft - select peak\nMiddle - delete last peak\nDrag Right - select range')
        cursor = Cursor(self.ax, useblit=True, color='k', linewidth=0.5)
        cursor.horizOn = False
        pylab.show()

    def press(self, event):
        tb = pylab.get_current_fig_manager().toolbar
        if tb.mode == '':
            x = numpy.round(event.xdata, 2)
            if event.button == 2:
                self.peaks = self.peaks[:-1]
                self.ax.lines = self.ax.lines[:-1]
            if event.button == 3:
                self.buttonDown = True
                self.pressv = event.xdata
            if event.button == 1 and (x >= self.xlims[1]) and (x <= self.xlims[0]):
                self.peaks.append(x)
                self.ax.vlines(x,self.ax_lims[0],self.ax_lims[1], color='#CC0000',lw=0.5)
                print(x)
                self.peaks = sorted(self.peaks)[::-1]
            self.canvas.draw()

    def release(self, event):
        if self.pressv is None or not self.buttonDown:
            return
        self.buttonDown = False
        self.rect.set_visible(False)
        vmin = numpy.round(self.pressv, 2)
        vmax = numpy.round(event.xdata or self.prev[0], 2)
        if vmin > vmax:
            vmin, vmax = vmax, vmin
        span = vmax - vmin
        self.pressv = None
        spantest = False
        print('%.2f - %.2f' % (vmax,vmin))
        if len(self.ranges) > 0:
            for i in self.ranges:
                if (vmin >= i[0]) and (vmin <= i[1]):
                    spantest = True
                    if (vmax >= i[0]) and (vmax <= i[1]):
                         spantest = True
        if span > self.minspan and spantest is False:
            self.ranges.append([numpy.round(vmin, 2), numpy.round(vmax, 2)])
            self.ax.bar(left=vmin,
                        height=sum(abs(self.ylims)),
                        width=span,
                        bottom=self.ylims[0],
                        alpha=0.2,
                        color='0.5',
                        edgecolor='k')
        self.canvas.draw()
        self.ranges = [numpy.sort(i)[::-1] for i in self.ranges]
        return False

    def onmove(self, event):
        if self.pressv is None or self.buttonDown is False or event.inaxes is None:
                return
        self.rect.set_visible(self.visible)
        x, y = event.xdata, event.ydata
        self.prev = x, y
        v = x
        minv, maxv = v, self.pressv
        if minv > maxv:
                minv, maxv = maxv, minv
        self.rect.set_xy([minv, self.rect.xy[1]])
        self.rect.set_width(maxv-minv)
        vmin = self.pressv
        vmax = event.xdata  # or self.prev[0]
        if vmin > vmax:
                vmin, vmax = vmax, vmin
        self.canvas.draw_idle()
        return False



if __name__ == '__main__':
    pass
