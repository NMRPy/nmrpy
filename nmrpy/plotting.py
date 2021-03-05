import nmrpy.data_objects
import logging, traceback
import numpy
import scipy
from matplotlib import pyplot as plt
import numbers
from datetime import datetime
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.collections import PolyCollection
import copy

from matplotlib.patches import Circle, Rectangle
from matplotlib.lines import Line2D
from matplotlib.transforms import blended_transform_factory
from matplotlib.widgets import Cursor
from matplotlib.backend_bases import NavigationToolbar2, Event

from ipywidgets import FloatText, Output
from IPython.display import display
import asyncio

original_home = NavigationToolbar2.home
original_zoom = NavigationToolbar2.zoom

class Plot():
    """
    Basic 'plot' class containing functions for various types of plots.
    """

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

    def _plot_ppm(self, fid,
            upper_ppm=None, 
            lower_ppm=None, 
            color='k', 
            lw=1,
            filename=None):
        data = fid.data
        params = fid._params
        ft=fid._flags['ft']
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

        self.fig = plt.figure(figsize=[9,5])
        ax = self.fig.add_subplot(111)
        if ft:
            ax.plot(ppm, data, color=color, lw=lw)
            ax.invert_xaxis()
            ax.set_xlim([upper_ppm, lower_ppm])
            ax.grid()
            ax.set_xlabel('PPM (%.2f MHz)'%(params['reffrq']))
        elif not ft:
            at = params['at']*1000 # ms
            t = numpy.linspace(0, at, len(data))
            ax.plot(t, data, color=color, lw=lw)
            ax.set_xlim([0, at])
            ax.grid()
            ax.set_xlabel('Time (ms)')
        #self.fig.show()
        if filename is not None:
            self.fig.savefig(filename, format='pdf')

    def _deconv_generator(self, fid,
            upper_ppm=None, 
            lower_ppm=None, 
            ):

        data = fid.data
        params = fid._params

        if not Plot._is_flat_iter(data): 
            raise AttributeError('data must be flat iterable.')

        peakshapes = fid._f_pks_list(fid._deconvoluted_peaks, numpy.arange(len(data))) 

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
        return ppm, data, peakshapes, summed_peaks, residual, upper_ppm, lower_ppm

    def _plot_deconv(self, fid,
            upper_ppm=None, 
            lower_ppm=None, 
            colour='k', 
            peak_colour='b', 
            summed_peak_colour='r', 
            residual_colour='g', 
            lw=1):

        #validation takes place in self._deconv_generator
        ppm, data, peakshapes, summed_peaks, residual, upper_ppm, \
            lower_ppm = self._deconv_generator(fid, 
                                               upper_ppm=upper_ppm, 
                                               lower_ppm=lower_ppm)

        self.fig = plt.figure(figsize=[9,5])
        ax = self.fig.add_subplot(111)
        ax.plot(ppm, residual, color=residual_colour, lw=lw)
        ax.plot(ppm, data, color=colour, lw=lw)
        ax.plot(ppm, summed_peaks, '--', color=summed_peak_colour, lw=lw)
        label_pad = 0.02*peakshapes.max()
        for n in range(len(peakshapes)):
            peak = peakshapes[n]
            ax.plot(ppm, peak, '-', color=peak_colour, lw=lw)
            ax.text(ppm[numpy.argmax(peak)], label_pad+peak.max(), str(n), ha='center')
        ax.invert_xaxis()
        ax.set_xlim([upper_ppm, lower_ppm])
        ax.grid()
        ax.set_xlabel('PPM (%.2f MHz)'%(fid._params['reffrq']))
        
    def _plot_deconv_array(self, fids,
            upper_index=None, 
            lower_index=None, 
            upper_ppm=None, 
            lower_ppm=None, 
            data_colour='k', 
            summed_peak_colour='r', 
            residual_colour='g', 
            data_filled=False,
            summed_peak_filled=True,
            residual_filled=False,
            figsize=[9, 6],
            lw=0.3, 
            azim=-90, 
            elev=20, 
            filename=None):

        if lower_index is None:
            lower_index = 0
        if upper_index is None:
            upper_index = len(fids)
        if lower_index >= upper_index:
            raise ValueError('upper_index must exceed lower_index')
        fids = fids[lower_index: upper_index]
        generated_deconvs = []
        for fid in fids:
            generated_deconvs.append(self._deconv_generator(fid, upper_ppm=upper_ppm, lower_ppm=lower_ppm))
      
        params = fids[0]._params 
        ppm = generated_deconvs[0][0]
        data = [i[1] for i in generated_deconvs]
        peakshapes = [i[2] for i in generated_deconvs]
        summed_peaks = [i[3] for i in generated_deconvs]
        residuals = [i[4] for i in generated_deconvs]
        upper_ppm = generated_deconvs[0][5]
        lower_ppm = generated_deconvs[0][6]

        plot_data = numpy.array([
                    residuals, 
                    data, 
                    summed_peaks,
                    ])
        colours_list = [
                    [residual_colour]*len(residuals),
                    [data_colour]*len(data), 
                    [summed_peak_colour]*len(summed_peaks), 
                    ]
        filled_list = [
                    residual_filled,
                    data_filled, 
                    summed_peak_filled, 
                    ] 

        xlabel = 'PPM (%.2f MHz)'%(params['reffrq'])
        ylabel = 'min.'
        acqtime = fids[0]._params['acqtime']
        minutes = acqtime[lower_index:upper_index]
        self.fig = self._generic_array_plot(ppm, minutes, plot_data, 
                                            colours_list=colours_list,
                                            filled_list=filled_list,
                                            figsize=figsize, 
                                            xlabel=xlabel,
                                            ylabel=ylabel,
                                            lw=lw, 
                                            azim=azim, 
                                            elev=elev, 
                                            )
        if filename is not None:
            self.fig.savefig(filename, format='pdf')
        plt.show()
          
        

    def _plot_array(self, data, params, 
                upper_index=None, 
                lower_index=None, 
                upper_ppm=None, 
                lower_ppm=None,
                figsize=(9, 6),
                lw=0.3, 
                azim=-90, 
                elev=20, 
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
            upper_index = len(data)
        if lower_index is None:
            lower_index = 0
        
        if upper_ppm is None:
            upper_ppm = sw_left
        if lower_ppm is None:
            lower_ppm = sw_left-sw

        acqtime = params['acqtime']
        ppm = numpy.linspace(sw_left-sw, sw_left, data.shape[1])[::-1]
        ppm_bool_index = (ppm < upper_ppm) * (ppm > lower_ppm)
        ppm = ppm[ppm_bool_index]
        if len(data) > 1:
            data = data[lower_index:upper_index, ppm_bool_index]
            minutes = acqtime[lower_index:upper_index]
        else:
            data = data[:, ppm_bool_index]
            minutes = acqtime[0]

        if colour:
            colours_list = [plt.cm.viridis(numpy.linspace(0, 1, len(data)))]
        else:
            colours_list = None

        xlabel = 'PPM (%.2f MHz)'%(params['reffrq'])
        ylabel = 'min.'
        self.fig = self._generic_array_plot(ppm, minutes, [data], 
                                            colours_list=colours_list,
                                            filled_list=[filled],
                                            figsize=figsize, 
                                            xlabel=xlabel,
                                            ylabel=ylabel,
                                            lw=lw, 
                                            azim=azim, 
                                            elev=elev, 
                                            )
        if filename is not None:
            self.fig.savefig(filename, format='pdf')
        plt.show()

    @staticmethod
    def _interleave_datasets(data):
        """
        interleave a list of lists with equal dimensions
        """
        idata = []
        for y in range(len(data[0])):
            for x in range(len(data)):
                idata.append(data[x][y])
        return idata

    def _generic_array_plot(self, x, y, zlist, 
                colours_list=None, 
                filled_list=None, 
                upper_lim=None,
                lower_lim=None,
                lw=0.3, 
                azim=-90, 
                elev=20, 
                figsize=[5,5],
                show_zticks=False, 
                labels=None, 
                xlabel=None,
                ylabel=None,
                filename=None,
        ):
        """

        Generic function for plotting arrayed data on a set of 3D axes. x and y
        must be 1D arrays. zlist is a list of 2D data arrays, each of which will be
        plotted with the corresponding colours_list colours, and filled_lists filled
        state.

        """

        


        if colours_list is None:
            colours_list = [['k']*len(y)]*len(zlist)

        if filled_list is None:
            filled_list = [False]*len(zlist)


        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection='3d', azim=azim, elev=elev)

        for data_n in range(len(zlist)):
            data = zlist[data_n]
            bh = abs(data.min()) 
            filled = filled_list[data_n]
            cl = colours_list[data_n]
            if not filled:
                #spectra are plotted in reverse for zorder
                for n in range(len(data))[::-1]:
                    datum = data[n]
                    clr = cl[n]
                    ax.plot(x, len(datum)*[y[n]], datum, color=clr, lw=lw)
            if filled:
                verts = []
                plot_data = data+bh 
                for datum in plot_data:
                    datum[0], datum[-1] = 0, 0
                    verts.append(list(zip(x, datum)))
                 
                fclr, eclr = ['w']*len(data), ['k']*len(data)
                fclr = cl
                poly = PolyCollection(verts, 
                    facecolors=fclr,
                    edgecolors=eclr,
                    linewidths=[lw]*len(verts))
                ax.add_collection3d(poly, zs=y, zdir='y')
    
        ax.set_zlim([0, 1.1*max(numpy.array(zlist).flat)])
        ax.invert_xaxis()
        if upper_lim is None:
            upper_lim = x[0]
        if lower_lim is None:
            lower_lim = x[-1]
        ax.set_xlim([upper_lim, lower_lim])
        ax.set_ylim([y[0], y[-1]])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if not show_zticks:
            ax.set_zticklabels([])
        return fig
        

    @classmethod
    def _is_iter(cls, i):
        try:
            iter(i)
            return True
        except TypeError:
            return False

    @classmethod
    def _is_iter_of_iters(cls, i):
        if type(i) == list and len(i) == 0:
            return False
        elif cls._is_iter(i) and all(cls._is_iter(j) for j in i):
            return True
        return False

    @classmethod
    def _is_flat_iter(cls, i):
        if type(i) == list and len(i) == 0:
            return True
        elif cls._is_iter(i) and not any(cls._is_iter(j) for j in i):
            return True
        return False

class Phaser:
    """Interactive phase-correction widget"""
    def __init__(self, fid):
        if not Plot._is_flat_iter(fid.data): 
            raise ValueError('data must be flat iterable.')
        if fid.data is [] or fid.data is None:
            raise ValueError('data must exist.')
        self.fid = fid
        self.fig = plt.figure(figsize=[9, 6])
        self.phases = numpy.array([0.0, 0.0])
        self.cum_phases = numpy.array([0.0, 0.0])
        self.y = 0.0
        self.ax = self.fig.add_subplot(111)
        self.ax.plot(self.fid.data, color='k', linewidth=1.0)
        self.ax.hlines(0, 0, len(self.fid.data)-1)
        self.ax.set_xlim([0, len(self.fid.data)])
        xtcks = numpy.linspace(0,1,11)*len(self.fid.data)
        xtcks[-1] = xtcks[-1]-1
        self.ax.set_xticks(xtcks)
        self.ax.set_xlabel('PPM (%.2f MHz)'%(self.fid._params['reffrq']))
        self.ax.set_xticklabels([numpy.round(self.fid._ppm[int(i)], 1) for i in xtcks])
        ylims = numpy.array([-1.6, 1.6])*max(abs(numpy.array(self.ax.get_ylim())))
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
        self.fig.subplots_adjust(bottom=0.13)
        self.text1 = self.fig.text(0.12, 0.02, ' ', fontsize='large')
        plt.show()

    def press(self, event):
        tb = plt.get_current_fig_manager().toolbar
        if tb.mode == '':
            x, y = event.xdata, event.ydata
            if event.inaxes is not None:
                self.buttonDown = True
                self.button = event.button
                self.y = y

    def release(self, event):
        self.text1.set_text('cumulative    p0: {0:.1f}    p1: {1:.1f}'.format(*self.cum_phases))
        self.buttonDown = False
        return False

    def onmove(self, event):
        if self.buttonDown is False or event.inaxes is None:
            return
        x = event.xdata
        y = event.ydata
        dy = y-self.y
        self.y = y
        if self.button == 1:
            self.phases[0] = 50*dy/self.ax.get_ylim()[1]
            self.phases[1] = 0.0
        if self.button == 3:
            self.phases[1] = 50*dy/self.ax.get_ylim()[1]
            self.phases[0] = 0.0
        self.fid.ps(p0=self.phases[0], p1=self.phases[1])
        self.cum_phases += self.phases
        self.ax.lines[0].set_data(numpy.array([numpy.arange(len(self.fid.data)), self.fid.data]))
        self.canvas.draw()  # _idle()
        return False


class BaseSelectorMixin:

    def __init__(self):
        super().__init__()
       
    def press(self, event):
        pass

    def release(self, event):
        pass

    def onmove(self, event):
        pass

    def redraw(self):
        pass

    def change_visible(self):
        pass

class PolySelectorMixin(BaseSelectorMixin):

    def __init__(self):
        super().__init__()
        class Psm:
            pass
        self.psm = Psm()
        self.psm.btn_add = 1
        self.psm.btn_del = 1
        self.psm.btn_cls = 3
        self.psm.key_mod = 'control'
        self.psm.xs = []
        self.psm.ys = []
        self.psm._xs = []
        self.psm._ys = []
        self.psm._x = None
        self.psm._y = None
        self.psm.datax = None
        self.psm.datay = None
        self.psm.lines = []
        self.psm.data_lines = []
        self.psm.index_lines = []
        self.psm._visual_lines = []
        self.psm.line = None
        self.psm._yline = None
        self.psm.lw = 1
        self.blocking = False
        if not hasattr(self, 'show_tracedata'):
            self.show_tracedata = False

    def redraw(self):
        super().redraw()
        if hasattr(self, 'psm'):
            for i in self.psm._visual_lines:
                self.ax.draw_artist(i)
            if self.psm.line is not None:
                self.ax.draw_artist(self.psm.line)
            if self.psm._yline is not None:
                self.ax.draw_artist(self.psm._yline)

    def change_visible(self):
        super().change_visible()
        if hasattr(self, 'psm'):
            for i in self.psm._visual_lines:
                i.set_visible(not i.get_visible())
            if self.psm.line is not None:
                self.psm.line.set_visible(not self.psm.line.get_visible())

    def makepoly(self,
        xs=None,
        ys=None,
        lw=1,
        colour='r',
        ms='+',
        ls='-',
        ):
        if xs is not None and ys is not None:
            return self.ax.plot(
                xs,
                ys,
                lw=lw,
                color=colour,
                marker=ms,
                ls=ls,
                )
 
    def press(self, event):
        super().press(event)
        if self.check_mode() != '':
            return
        if event.xdata is None or event.ydata is None:
            return
        if event.button == self.psm.btn_add and event.key != self.psm.key_mod:
                self.psm.xs.append(event.xdata)
                self.psm.ys.append(event.ydata)
                if self.show_tracedata:
                    self.psm._xs, self.psm._ys = self.get_line_ydata(self.psm.xs, self.psm.ys)
                if self.psm.line is None:
                    self.psm.line, = self.makepoly(
                        self.psm.xs, 
                        self.psm.ys, 
                        lw=self.psm.lw,
                        )
                    self.blocking = True
                    if self.show_tracedata:
                        self.psm._yline, = self.makepoly(
                            self.psm._xs,
                            self.psm._ys,
                            lw=self.psm.lw,
                            ms='+',
                            ls='-',
                            colour='r',
                            )
                else:
                    self.psm.line.set_data(self.psm.xs, self.psm.ys)
                    if self.show_tracedata:
                        self.psm._yline.set_data(self.psm._xs, self.psm._ys)
        elif event.button == self.psm.btn_del and event.key == self.psm.key_mod:
            if len(self.psm._visual_lines) > 0:
                x = event.xdata
                y = event.ydata
                #trace_dist = [[i[0]-x, i[1]-y] for i in self.psm.lines]
                trace_dist = [[i[0]-x] for i in self.psm.lines]
                #delete_trace = numpy.argmin([min(numpy.sqrt(i[0]**2+i[1]**2))                 
                delete_trace = numpy.argmin([min(numpy.sqrt(i[0]**2)) for i in trace_dist])
                self.psm.lines.pop(delete_trace)
                self.psm.data_lines.pop(delete_trace)
                trace = self.psm._visual_lines.pop(delete_trace)
                trace.remove()
        elif event.button == self.psm.btn_cls and self.psm.line is not None:
            if len(self.psm.xs) > 1:
                self.psm._visual_lines.append(self.makepoly(
                        self.psm.xs, 
                        self.psm.ys, 
                        lw=self.psm.lw,
                        colour='b', 
                        )[0])
                self.psm.lines.append(numpy.array([self.psm.xs, self.psm.ys]))
                self.psm.xs, self.psm.ys = [], []
                self.psm.line.remove()
                self.psm.line = None
                self.psm._yline.remove()
                self.psm._yline = None
                self.psm.data_lines.append(self.get_polygon_neighbours_data(self.psm.lines[-1]))
                self.psm.index_lines.append(self.get_polygon_neighbours_indices(self.psm.lines[-1]))
                self.blocking = False
            else:
                self.psm.xs, self.psm.ys = [], []
                self.psm.line = None
        #self.redraw()
    
    def onmove(self, event):
        super().onmove(event)
        self.psm._x = event.xdata
        self.psm._y = event.ydata
        if self.psm.line is not None:
            xs = self.psm.xs+[self.psm._x]
            ys = self.psm.ys+[self.psm._y]
            self.psm.line.set_data(xs, ys)
            if self.show_tracedata:
                current_x_ydata = self.get_line_ydata(
                    [self.psm.xs[-1]]+[self.psm._x],
                    [self.psm.ys[-1]]+[self.psm._y],
                    )
                self.psm._yline.set_data(
                    self.psm._xs+current_x_ydata[0], 
                    self.psm._ys+current_x_ydata[1],
                    )

    def get_line_ydata(self, xs, ys):
        xdata = []
        ydata = []
        for i in range(len(xs)-1):
            current_xy_data = self.get_polygon_neighbours_data([
                xs[i:i+2],
                ys[i:i+2],
                ])
            xdata += current_xy_data[0]
            ydata += current_xy_data[1]
        return xdata, ydata

    def get_polygon_neighbours_data(self, line):
        """
        Returns the nearest datum in each spectrum as it is intersected by a
        polygonal line consisting of [[x coordinates], [y coordinates]].
        """
        line_xs = []
        line_ys = []
        for i in range(len(line[0])-1):
            x1, y1, x2, y2 = line[0][i], line[1][i], line[0][i+1], line[1][i+1]
            x, y, x_index, y_index = self.get_neighbours([x1, x2], [y1, y2])
            if x is not None and y is not None:
                line_xs = line_xs+list(x)
                line_ys = line_ys+list(y)
        return [line_xs, line_ys]

    def get_polygon_neighbours_indices(self, line):
        """
        Returns the nearest datum in each spectrum as it is intersected by a
        polygonal line consisting of [[x coordinates], [y coordinates]].
        """
        line_xs = []
        line_ys = []
        for i in range(len(line[0])-1):
            x1, y1, x2, y2 = line[0][i], line[1][i], line[0][i+1], line[1][i+1]
            x, y, x_index, y_index = self.get_neighbours([x1, x2], [y1, y2])
            if x_index is not None and y_index is not None:
                line_xs = line_xs+list(x_index)
                line_ys = line_ys+list(y_index)
        return [line_xs, line_ys]
            
    def get_neighbours(self, xs, ys):
        """
        For a pair of coordinates (xs = [x1, x2], ys = [y1, y2]), return the
        nearest datum in each spectrum for a line subtended between the two coordinate
        points which intersects the baseline of each spectrum.
        Returns three arrays, one of x-coordinates, one of y-coordinates, and a y index range
        """
        ymask = list((self.y_indices <= max(ys)) * (self.y_indices >= min(ys)))
        if True not in ymask:
            return None, None, None, None
        y_lo = ymask.index(True)
        y_hi = len(ymask)-ymask[::-1].index(True)
        x_neighbours = []
        y_neighbours = []
        y_indices = [i for i in range(y_lo, y_hi)]
        if ys[0] > ys[1]:
            y_indices = y_indices[::-1]
        x_indices = []
        for i in y_indices:
            x = [self.ppm[0], self.ppm[-1], xs[0], xs[1]]    
            y = [self.y_indices[i], self.y_indices[i], ys[0], ys[1]]    
            x, y = self.get_intersection(x, y)
            x = numpy.argmin(abs(self.ppm[::-1]-x))
            x_indices.append(x)
            x_neighbours.append(self.ppm[::-1][x])
            y_neighbours.append(self.data[i][x]+self.y_indices[i])
        return x_neighbours, y_neighbours, x_indices, y_indices

    @staticmethod
    def get_intersection(x, y):
        """
        This function take a set of two pairs of x/y coordinates, defining a
        pair of crossing lines, and returns the intersection. x = [x1, x2, x3, x4], y =
        [y1, y2, y3, y4], where [x1, y1] and [x2, y2] represent one line, and [x3, y3]
        and [x4, y4] represent the other. See
        https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection#Given_two_points_on_each_line
        """
        px = (((x[0]*y[1]-y[0]*x[1])*(x[2]-x[3])-(x[0]-x[1])*(x[2]*y[3]-y[2]*x[3]))/((x[0]-x[1])*(y[2]-y[3])-(y[0]-y[1])*(x[2]-x[3])))
        py = (((x[0]*y[1]-y[0]*x[1])*(y[2]-y[3])-(y[0]-y[1])*(x[2]*y[3]-y[2]*x[3]))/((x[0]-x[1])*(y[2]-y[3])-(y[0]-y[1])*(x[2]-x[3])))
        return px, py

class LineSelectorMixin(BaseSelectorMixin):

    def __init__(self):
        super().__init__()
        class Lsm:
            pass
        self.lsm = Lsm()
        self.lsm.btn_add = 1
        self.lsm.btn_del = 1
        self.lsm.key_mod = 'control'
        self.lsm.peaklines = {}
        self.lsm.peaks = []
        for x in self.peaks:
            self.lsm.peaks.append(x)
            self.lsm.peaklines[x] = self.makeline(x)
            #self.ax.draw_artist(self.lsm.peaklines[x])
        self.lsm.peaks = sorted(self.lsm.peaks)[::-1]
                
    def makeline(self, x):
        return self.ax.plot(
            [x, x], 
            self.ylims,
            color='#CC0000', 
            lw=1,
            #animated=True
            )[0]

    def redraw(self):
        super().redraw()
        if hasattr(self, 'lsm'):
            for i, j in self.lsm.peaklines.items():
                self.ax.draw_artist(j)

    def change_visible(self):
        super().change_visible()
        if hasattr(self, 'lsm'):
            for i, j in self.lsm.peaklines.items():
                j.set_visible(True)
                j.set_visible(not j.get_visible())

    def press(self, event):
        super().press(event)
        x = numpy.round(event.xdata, 2)
        # left
        if event.button == self.lsm.btn_add and \
                event.key != self.lsm.key_mod and \
                (x >= self.xlims[1]) and (x <= self.xlims[0]):
            with self.out:
                print('peak {}'.format(x))
            if x not in self.lsm.peaks:
                self.lsm.peaks.append(x)
                self.lsm.peaklines[x] = self.makeline(x)
                self.lsm.peaks = sorted(self.lsm.peaks)[::-1]
                #self.ax.draw_artist(self.lsm.peaklines[x])
        #Ctrl+left
        elif event.button == self.lsm.btn_del and event.key == self.lsm.key_mod:
            #find and delete nearest peakline
            if len(self.lsm.peaks) > 0:
                delete_peak = numpy.argmin([abs(i-x) for i in self.lsm.peaks])
                old_peak = self.lsm.peaks.pop(delete_peak)
                try: 
                    peakline = self.lsm.peaklines.pop(old_peak)
                    peakline.remove()
                except:
                    with self.out:
                        print('Could not remove peakline')
            self.canvas.draw()
        #self.redraw()

    def release(self, event):
        super().release(event)

    def onmove(self, event):
        super().onmove(event)


class SpanSelectorMixin(BaseSelectorMixin):

    def __init__(self):
        super().__init__()
        class Ssm:
            pass
        self.ssm = Ssm()
        self.ssm.btn_add = 3
        self.ssm.btn_del = 3
        self.ssm.key_mod = 'control'
        self.ssm.minspan = 0
        self.ssm.rect = None
        self.ssm.rangespans = []
        self.ssm.rectprops = dict(facecolor='0.5', alpha=0.2)
        self.ssm.ranges = self.ranges
        for rng in self.ssm.ranges:
            self.ssm.rangespans.append(self.makespan(rng[1], rng[0]-rng[1]))
        self.redraw()
        trans = blended_transform_factory(
            self.ax.transData,
            self.ax.transAxes)
        w, h = 0, 1
        self.ssm.rect = Rectangle([0, 0], w, h,
                              transform=trans,
                              visible=False,
                              animated=True,
                              **self.ssm.rectprops
                              )
        self.ax.add_patch(self.ssm.rect)

    def makespan(self, left, width):
        trans = blended_transform_factory(
            self.ax.transData,
            self.ax.transAxes)
        bottom, top = self.ylims
        height = top-bottom
        rect = Rectangle([left, bottom], width, height,
                              transform=trans,
                              visible=True,
                              #animated=True,
                              **self.ssm.rectprops
                              )
        self.ax.add_patch(rect)
        return rect

    def redraw(self):
        super().redraw()
        if hasattr(self, 'ssm'):
            for i in self.ssm.rangespans:
                self.ax.draw_artist(i)

    def change_visible(self):
        super().change_visible()
        if hasattr(self, 'ssm'):
            for i in self.ssm.rangespans:
                i.set_visible(not i.get_visible())

    def press(self, event):
        super().press(event)
        if self.blocking:
            return
        if event.button == self.ssm.btn_add and event.key != self.ssm.key_mod:
            self.buttonDown = True
            self.pressv = event.xdata
        elif event.button == self.ssm.btn_add and event.key == self.ssm.key_mod:
            #find and delete range
            if len(self.ssm.ranges) > 0:
                x = event.xdata
                rng = 0
                while rng < len(self.ssm.ranges):
                    if x >= (self.ssm.ranges[rng])[1] and x <= (self.ssm.ranges[rng])[0]:
                        self.ssm.ranges.pop(rng) 
                        rangespan = self.ssm.rangespans.pop(rng)
                        rangespan.remove()
                        break
                    rng += 1
                self.canvas.draw()

    def release(self, event):
        super().release(event)
        self.ssm.rect.set_visible(False)
        vmin = numpy.round(self.pressv, 2)
        vmax = numpy.round(event.xdata or self.prev[0], 2)
        if vmin > vmax:
            vmin, vmax = vmax, vmin
        span = vmax - vmin
        self.pressv = None
        spantest = False
        #if len(self.ssm.ranges) > 0:
        #    for i in self.ssm.ranges:
        #        if (vmin >= i[1]) and (vmin <= i[0]):
        #            spantest = True
        #        if (vmax >= i[1]) and (vmax <= i[0]):
        #            spantest = True
        if span > self.ssm.minspan and spantest is False:
            self.ssm.ranges.append([numpy.round(vmin, 2), numpy.round(vmax, 2)])
            self.ssm.rangespans.append(self.makespan(vmin, span))
            with self.out:
                print('range {} -> {}'.format(vmax, vmin))
        self.ssm.ranges = [numpy.sort(i)[::-1] for i in self.ssm.ranges]


    def onmove(self, event):
        super().onmove(event)
        if self.pressv is None or self.buttonDown is False:
            return
        if event.button == self.ssm.btn_add and event.key != self.ssm.key_mod:
            x, y = self.prev
            v = x
            minv, maxv = v, self.pressv
            if minv > maxv:
                    minv, maxv = maxv, minv
            vmin = self.pressv
            vmax = event.xdata  # or self.prev[0]
            if vmin > vmax:
                    vmin, vmax = vmax, vmin
            self.ssm.rect.set_visible(self.visible)
            self.ssm.rect.set_xy([minv, self.ssm.rect.xy[1]])
            self.ssm.rect.set_width(maxv-minv)
            self.ax.draw_artist(self.ssm.rect)

class PeakSelectorMixin(BaseSelectorMixin):

    def __init__(self):
        super().__init__()
        class Psm:
            pass
        self.psm = Psm()
        self.psm.btn_add = 1
        self.psm.peak = None
        self.psm.newx = None
                
    def makeline(self, x):
        return self.ax.plot(
            [x, x], 
            self.ylims,
            color='#CC0000', 
            lw=1,
            )[0]

    def press(self, event):
        super().press(event)
        x = numpy.round(event.xdata, 2)
        # left
        if event.button == self.psm.btn_add and (x >= self.xlims[1]) and (x <= self.xlims[0]):
            self.psm.peak = x
            self.makeline(x)
            self.process()

    def release(self, event):
        super().release(event)

    def onmove(self, event):
        super().onmove(event)
        
    def process(self):
        pass
    
class AssignMixin(BaseSelectorMixin):

    def __init__(self):
        super().__init__()
        class Am:
            pass
        self.am = Am()
        self.am.btn_assign = 3
        self.am.key_mod1 = 'ctrl+alt'
        self.am.key_mod2 = 'alt+control'

    def press(self, event):
        super().press(event)
        if event.button == self.am.btn_assign and (event.key == self.am.key_mod1 \
                                        or event.key == self.am.key_mod2):
            with self.out:
                print('assigned peaks and ranges')
            self.assign() 

    def assign(self):
        pass
        
#this is to catch 'home' events in the dataselector 
def dataselector_home(self, *args, **kwargs):
    s = 'home_event'
    event = Event(s, self)
    original_home(self, *args, **kwargs)
    self.canvas.callbacks.process(s, event)

#this is to catch 'zoom' events in the dataselector 
def dataselector_zoom(self, *args, **kwargs):
    s = 'zoom_event'
    event = Event(s, self)
    original_zoom(self, *args, **kwargs)
    self.canvas.callbacks.process(s, event)

class DataSelector():
    """
    Interactive selector widget. can inherit from various mixins for functionality:
        Line selection: :class:`~nmrpy.plotting.LineSelectorMixin`
        Span selection: :class:`~nmrpy.plotting.SpanSelectorMixin`
        Poly selection: :class:`~nmrpy.plotting.PolySelectorMixin`
        
    This class is not intended to be used without inheriting at least one mixin.
    """

    def __init__(self, 
                data, 
                params, 
                extra_data=None,
                extra_data_colour='k',
                peaks=None, 
                ranges=None, 
                title=None, 
                voff=0.001, 
                label=None,
                ):
        if not Plot._is_iter(data):
            raise AttributeError('data must be iterable.')
        self.data = numpy.array(data)
        self.extra_data = extra_data
        self.extra_data_colour = extra_data_colour
        self.params = params
        self.ranges = []
        self.peaks = []
        if peaks is not None:
            self.peaks = list(peaks)
        if ranges is not None:
            self.ranges = list(ranges)
        self.voff = voff
        self.title = title
        self.label = label

        self._make_basic_fig()
        self.out = Output()
        display(self.out)

        self.visible = True

        self.pressv = None
        self.buttonDown = False
        self.prev = (0, 0)
        self.blocking = False        
        #self.canvas.restore_region(self.background)
        super().__init__() #calling parent init
        #self.canvas.blit(self.ax.bbox) 

        NavigationToolbar2.home = dataselector_home
        NavigationToolbar2.zoom = dataselector_zoom

        self.cidmotion = self.canvas.mpl_connect('motion_notify_event', self.onmove)
        self.cidpress = self.canvas.mpl_connect('button_press_event', self.press)
        self.cidrelease = self.canvas.mpl_connect('button_release_event', self.release)
        self.cidhome = self.canvas.mpl_connect('home_event', self.on_home) 
        self.cidzoom = self.canvas.mpl_connect('zoom_event', self.on_zoom) 
        self.ciddraw = self.canvas.mpl_connect('draw_event', self.on_draw) 
        #cursor = Cursor(self.ax, useblit=True, color='k', linewidth=0.5)
        #cursor.horizOn = False
        self.canvas.draw()
        #self.redraw()
        plt.show()

    def disconnect(self):
        self.canvas.mpl_disconnect(self.cidmotion)
        self.canvas.mpl_disconnect(self.cidpress)
        self.canvas.mpl_disconnect(self.cidrelease)
        self.canvas.mpl_disconnect(self.cidhome)
        self.canvas.mpl_disconnect(self.cidzoom)
        self.canvas.mpl_disconnect(self.ciddraw)

    def _make_basic_fig(self, *args, **kwargs):
        self.fig = plt.figure(figsize=[9, 6])
        self.ax = self.fig.add_subplot(111)
        if len(self.data.shape)==1:
            self.ppm = numpy.mgrid[self.params['sw_left']-self.params['sw']:self.params['sw_left']:complex(self.data.shape[0])]
            #extra_data
            if self.extra_data is not None:
                self.ax.plot(self.ppm[::-1], self.extra_data, color=self.extra_data_colour, lw=1)
            #data
            self.ax.plot(self.ppm[::-1], self.data, color='k', lw=1)
        elif len(self.data.shape)==2:
            cl = dict(zip(range(len(self.data)), plt.cm.viridis(numpy.linspace(0,1,len(self.data)))))
            self.ppm = numpy.mgrid[self.params['sw_left']-self.params['sw']:self.params['sw_left']:complex(self.data.shape[1])]
            self.y_indices = numpy.arange(len(self.data))*self.voff*self.data.max()
            #this is reversed for zorder
            #extra_data
            if self.extra_data is not None:
                for i,j in zip(range(len(self.extra_data))[::-1], self.extra_data[::-1]):
                    self.ax.plot(self.ppm[::-1], j+self.y_indices[i], color=self.extra_data_colour, lw=1)
            #data
            for i,j in zip(range(len(self.data))[::-1], self.data[::-1]):
                self.ax.plot(self.ppm[::-1], j+self.y_indices[i], color=cl[i], lw=1)
        self.ax.set_xlabel('ppm')
        self.ylims = numpy.array(self.ax.get_ylim())#numpy.array([self.ax.get_ylim()[0], self.data.max() + abs(self.ax.get_ylim()[0])])
        #self.ax.set_ylim(self.ylims)#self.ax.get_ylim()[0], self.data.max()*1.1])
        self.ax_lims = self.ax.get_ylim()
        self.xlims = [self.ppm[-1], self.ppm[0]]
        self.ax.set_xlim(self.xlims)
        self.fig.suptitle(self.title, size=20)
        self.ax.text(
            0.95 *
            self.ax.get_xlim()[0],
            0.7 *
            self.ax.get_ylim()[1],
            self.label),
        self.ax.set_ylim(self.ylims)
        self.canvas = self.ax.figure.canvas
        self.canvas.draw()
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)

    def check_mode(self):
        tb = plt.get_current_fig_manager().toolbar
        return tb.mode

    def on_draw(self, event):
        self.background = self.canvas.copy_from_bbox(self.ax.bbox)
        pass

    def on_home(self, event):
        pass

    def on_zoom(self, event):
        pass

    def press(self, event):
        tb = plt.get_current_fig_manager().toolbar
        if tb.mode == '' and event.xdata is not None:
            x = numpy.round(event.xdata, 2)
            self.canvas.restore_region(self.background)
            try:
                super().press(event)
            except Exception as e:
                logging.error(traceback.format_exc())
            self.redraw()
            self.canvas.blit(self.ax.bbox) 

    def release(self, event):
        if self.pressv is None or not self.buttonDown:
            return
        self.buttonDown = False
        self.canvas.restore_region(self.background)
        try:
            super().release(event)
        except Exception as e:
            logging.error(traceback.format_exc())
        self.redraw()
        self.canvas.blit(self.ax.bbox) 

    def onmove(self, event):
        if event.inaxes is None:
                return
        x, y = event.xdata, event.ydata
        self.prev = x, y
        self.canvas.restore_region(self.background)
        try:
            super().onmove(event)
        except Exception as e:
            logging.error(traceback.format_exc())
        self.redraw()
        self.canvas.blit(self.ax.bbox) 

    def make_invisible(self):
        try:
             super().make_invisible()
        except Exception as e:
            logging.error(traceback.format_exc())

    def make_visible(self):
        try:
             super().make_visible()
        except Exception as e:
            logging.error(traceback.format_exc())

    def redraw(self):
        try:
             super().redraw()
        except Exception as e:
            logging.error(traceback.format_exc())
        
    def change_visible(self):
        try:
             super().change_visible()
        except Exception as e:
            logging.error(traceback.format_exc())

class IntegralDataSelector(DataSelector, PolySelectorMixin, AssignMixin):
    show_tracedata = True

class PeakTraceDataSelector(DataSelector, PolySelectorMixin, SpanSelectorMixin, AssignMixin):
    show_tracedata = True

class LineSpanDataSelector(DataSelector, LineSelectorMixin, SpanSelectorMixin, AssignMixin):
    pass

class PeakDataSelector(DataSelector, PeakSelectorMixin):
    pass
        
class SpanDataSelector(DataSelector, SpanSelectorMixin, AssignMixin):
    pass

class DataTraceSelector:
    """
    Interactive data-selection widget with traces and ranges. Traces are saved
    as self.data_traces (WRT data) and self.index_traces (WRT index).    
    """
    def __init__(self, fid_array,
            extra_data=None,
            extra_data_colour='b',
            voff=1e-3,
            lw=1,
            label=None,
            ):
        self.fid_array = fid_array
        if fid_array.data is [] or fid_array.data is None:
            raise ValueError('data must exist.')
        data = fid_array.data
        params = fid_array._params
        sw_left = params['sw_left']
        sw = params['sw']

        ppm = numpy.linspace(sw_left-sw, sw_left, data.shape[1])[::-1]
       
        self.integral_selector = IntegralDataSelector(
                extra_data,
                params,
                extra_data=data, 
                extra_data_colour=extra_data_colour,
                peaks=None, 
                ranges=None, 
                title='Integral trace selector', 
                voff=voff,
                label=label)
        self.integral_selector.assign = self.assign
        
    def assign(self):
        data_traces = self.integral_selector.psm.data_lines
        index_traces = self.integral_selector.psm.index_lines
        
        self.fid_array._data_traces = [dict(zip(i[1], i[0])) for i in data_traces]
        self.fid_array._index_traces = [dict(zip(i[1], i[0])) for i in index_traces]

        decon_peaks = []
        for i in self.fid_array._deconvoluted_peaks:
            if len(i):
                decon_peaks.append(i.transpose()[0])
            else:
                decon_peaks.append(None)

        trace_dict = {}
        for t in range(len(self.fid_array._index_traces)):
            trace = self.fid_array._index_traces[t]
            integrals = {}
            for fid, indx in trace.items():
                try:
                    integrals[fid] = numpy.argmin(abs(decon_peaks[fid]-indx))
                except:
                    integrals[fid] = None
            trace_dict[t] = integrals
        last_fid = (len(self.fid_array.get_fids())-1)
        for i in trace_dict:
            tmin = min(trace_dict[i])
            tminval = trace_dict[i][tmin]
            if tmin > 0:
                for j in range(0, tmin):
                    trace_dict[i][j] = tminval
            tmax = max(trace_dict[i])
            tmaxval = trace_dict[i][tmax]
            if tmax < last_fid:
                for j in range(tmax, last_fid+1):
                    trace_dict[i][j] = tmaxval
        self.fid_array.integral_traces = trace_dict
        plt.close(self.integral_selector.fig)

class DataTraceRangeSelector:
    """
    Interactive data-selection widget with traces and ranges. Traces are saved
    as self.data_traces (WRT data) and self.index_traces (WRT index). Spans are
    saves as self.spans.
    """
    def __init__(self, fid_array,
            peaks=None,
            ranges=None,
            voff=1e-3,
            lw=1,
            label=None,
            ):
        self.fid_array = fid_array
        if fid_array.data is [] or fid_array.data is None:
            raise ValueError('data must exist.')
        data = fid_array.data
        params = fid_array._params
        sw_left = params['sw_left']
        sw = params['sw']

        ppm = numpy.linspace(sw_left-sw, sw_left, data.shape[1])[::-1]
       
        self.peak_selector = PeakTraceDataSelector(
                data, 
                params,
                peaks=peaks, 
                ranges=ranges, 
                title='Peak and range trace selector', 
                voff=voff,
                label=label)
        self.peak_selector.assign = self.assign

    def assign(self):
        data_traces = self.peak_selector.psm.data_lines
        index_traces = self.peak_selector.psm.index_lines
        spans = self.peak_selector.ssm.ranges
        
        traces = [[i[0], j[1]] for i, j in zip(data_traces,  index_traces)]

        self.fid_array.traces = traces
        self.fid_array._trace_mask = self.fid_array._generate_trace_mask(traces)

        self.fid_array._set_all_peaks_ranges_from_traces_and_spans(
                traces, spans)
        plt.close(self.peak_selector.fig)

class DataPeakSelector:
    """
    Interactive data-selection widget with lines and ranges for a single Fid.
    Lines and spans are saved as self.peaks, self.ranges.
    """
    def __init__(self, fid,
            peaks=None,
            ranges=None,
            voff=1e-3,
            lw=1,
            label=None,
            title=None,
            ):
        self.fid = fid
        if fid.data is [] or fid.data is None:
            raise ValueError('data must exist.')
        data = fid.data
        params = fid._params
        sw_left = params['sw_left']
        sw = params['sw']
        ppm = numpy.linspace(sw_left-sw, sw_left, len(data))[::-1]

        if fid.peaks is not None:
            peaks = list(fid.peaks)
        if fid.ranges is not None:
            ranges = list(fid.ranges)   
       
        self.peak_selector = LineSpanDataSelector(
                data,
                params,
                peaks=peaks, 
                ranges=ranges, 
                title=title, 
                voff=voff,
                label=label)
        self.peak_selector.assign = self.assign
        
    def assign(self):
        if len(self.peak_selector.ssm.ranges) > 0 and len(self.peak_selector.lsm.peaks) > 0:
            self.fid.ranges = self.peak_selector.ssm.ranges
            peaks = []
            for peak in self.peak_selector.lsm.peaks:
                for rng in self.peak_selector.ssm.ranges:
                    if peak >= rng[1] and peak <= rng[0]:
                        peaks.append(peak)
            self.fid.peaks = peaks
        else:
            self.fid.peaks = None
            self.fid.ranges = None
        plt.close(self.peak_selector.fig)

class DataPeakRangeSelector:
    """Interactive data-selection widget with lines and ranges. Lines and spans are saved as self.peaks, self.ranges."""
    def __init__(self, fid_array,
            peaks=None,
            ranges=None,
            y_indices=None,
            aoti=True,
            voff=1e-3,
            lw=1,
            label=None,
            ):
        self.fid_array = fid_array
        self.fids = fid_array.get_fids()
        self.assign_only_to_index = aoti
        self.fid_number = y_indices
        if self.fid_number is not None:
            if not nmrpy.data_objects.Fid._is_iter(self.fid_number):
                self.fid_number = [self.fid_number]
        else:
            self.fid_number = range(len(self.fids))
        if fid_array.data is [] or fid_array.data is None:
            raise ValueError('data must exist.')
        data = fid_array.data
        if y_indices is not None:
            data = fid_array.data[numpy.array(self.fid_number)]
        params = fid_array._params
        sw_left = params['sw_left']
        sw = params['sw']

        ppm = numpy.linspace(sw_left-sw, sw_left, data.shape[1])[::-1]
       
        self.peak_selector = LineSpanDataSelector(
                data,
                params,
                peaks=peaks, 
                ranges=ranges, 
                title='Peak and range selector', 
                voff=voff,
                label=label)
        self.peak_selector.assign = self.assign
        
    def assign(self):
        self.peaks = self.peak_selector.lsm.peaks
        self.ranges = self.peak_selector.ssm.ranges
        
        if len(self.ranges) > 0 and len(self.peaks) > 0:
            ranges = self.ranges
            peaks = []
            for peak in self.peaks:
                for rng in ranges:
                    if peak >= rng[1] and peak <= rng[0]:
                        peaks.append(peak)
        else:
            peaks = None
            ranges = None

        if self.assign_only_to_index:
            for fid in [self.fids[i] for i in self.fid_number]:
                fid.peaks = peaks
                fid.ranges = ranges
        else:       
            for fid in self.fids:
                fid.peaks = peaks
                fid.ranges = ranges
        plt.close(self.peak_selector.fig)
  
class Calibrator:
    """
    Interactive data-selection widget for calibrating PPM of a spectrum.
    """
    def __init__(self, fid,
            lw=1,
            label=None,
            title=None,
            ):
        self.fid = fid
        if fid.data is [] or fid.data is None:
            raise ValueError('data must exist.')
        if not fid._flags['ft']:
            raise ValueError('Only Fourier-transformed data can be calibrated.')

        data = fid.data
        params = fid._params
        sw_left = params['sw_left']
        self.sw_left = sw_left
        sw = params['sw']
        ppm = numpy.linspace(sw_left-sw, sw_left, len(data))[::-1]

        self.peak_selector = PeakDataSelector(
                data,
                params,
                title=title, 
                label=label)
        self.peak_selector.process = self.process
        
        self.textinput = FloatText(value=0.0, description='New PPM:',
            disabled=False, continuous_update=False)
        
    def _wait_for_change(self, widget, value):
        future = asyncio.Future()
        def getvalue(change):
            # make the new value available
            future.set_result(change.new)
            widget.unobserve(getvalue, value)
        widget.observe(getvalue, value)
        return future
        
    def process(self):
        peak = self.peak_selector.psm.peak
        self.peak_selector.out.clear_output()
        with self.peak_selector.out:
            print('current peak ppm:    {}'.format(peak))
            display(self.textinput)
        async def f():
            newx = await self._wait_for_change(self.textinput, 'value')
            offset = newx - peak
            self.fid._params['sw_left'] = self.sw_left + offset
            with self.peak_selector.out:
                print('calibration done.')
            plt.close(self.peak_selector.fig)
        asyncio.ensure_future(f())

class RangeCalibrator:
    """
    Interactive data-selection widget for calibrating PPM of an
    array of spectra.
    """
    def __init__(self, fid_array,
            y_indices=None,
            aoti=True,
            voff=1e-3,
            lw=1,
            label=None,
            ):
        self.fid_array = fid_array
        self.fids = fid_array.get_fids()
        self.assign_only_to_index = aoti
        self.fid_number = y_indices
        if self.fid_number is not None:
            if not nmrpy.data_objects.Fid._is_iter(self.fid_number):
                self.fid_number = [self.fid_number]
        else:
            self.fid_number = range(len(self.fids))
        if fid_array.data is [] or fid_array.data is None:
            raise ValueError('data must exist.')
        if any (not fid._flags['ft'] for fid in self.fids):
            raise ValueError('Only Fourier-transformed data can be calibrated.')
        data = fid_array.data
        if y_indices is not None:
            data = fid_array.data[numpy.array(self.fid_number)]
        params = fid_array._params
        sw_left = params['sw_left']
        self.sw_left = sw_left
        sw = params['sw']
        ppm = numpy.linspace(sw_left-sw, sw_left, data.shape[1])[::-1]

        self.peak_selector = PeakDataSelector(
                data,
                params,
                title='FidArray calibration',
                voff = voff,
                label=label)
        self.peak_selector.process = self.process
        
        self.textinput = FloatText(value=0.0, description='New PPM:',
            disabled=False, continuous_update=False)
        
    def _wait_for_change(self, widget, value):
        future = asyncio.Future()
        def getvalue(change):
            # make the new value available
            future.set_result(change.new)
            widget.unobserve(getvalue, value)
        widget.observe(getvalue, value)
        return future
        
    def process(self):
        peak = self.peak_selector.psm.peak
        self.peak_selector.out.clear_output()
        with self.peak_selector.out:
            print('current peak ppm:    {}'.format(peak))
            display(self.textinput)
        async def f():
            newx = await self._wait_for_change(self.textinput, 'value')
            offset = newx - peak
            self._applycalibration(offset)
            with self.peak_selector.out:
                print('calibration done.')
            plt.close(self.peak_selector.fig)
        asyncio.ensure_future(f())

    def _applycalibration(self, offset):
        self.fid_array._params['sw_left'] = self.sw_left + offset
        
        if self.assign_only_to_index:
            for fid in [self.fids[i] for i in self.fid_number]:
                fid._params['sw_left'] = self.sw_left + offset
        else:       
            for fid in self.fids:
                fid._params['sw_left'] = self.sw_left + offset

class FidArrayRangeSelector:
    """Interactive data-selection widget with ranges. Spans are saved as self.ranges."""
    def __init__(self, 
            fid_array,
            ranges=None,
            y_indices=None,
            voff=1e-3,
            lw=1,
            title=None,
            label=None,
            ):
        self.fid_array = fid_array
        self.fids = fid_array.get_fids()
        data = fid_array.data
        params = fid_array._params
        if data is [] or data is None:
            raise ValueError('data must exist.')
        if y_indices is not None:
            data = data[numpy.array(y_indices)]
        sw_left = params['sw_left']
        sw = params['sw']

        ppm = numpy.linspace(sw_left-sw, sw_left, data.shape[1])[::-1]
       
        self.span_selector = SpanDataSelector(
                data,
                params,
                ranges=ranges, 
                title=title,
                voff=voff,
                label=label)
        self.span_selector.assign = self.assign

    def assign(self):
        self.ranges = self.span_selector.ssm.ranges
        for fid in self.fid_array.get_fids():
            bl_ppm = []
            for rng in self.ranges:
                peak_ind = (fid._ppm > rng[1]) * (fid._ppm < rng[0])
                cur_peaks = fid._ppm[peak_ind]
                bl_ppm.append(cur_peaks)
            bl_ppm = numpy.array([j for i in bl_ppm for j in i])
            fid._bl_ppm = bl_ppm
        plt.close(self.span_selector.fig)

class FidRangeSelector:
    """Interactive data-selection widget with ranges. Spans are saved as self.ranges."""
    def __init__(self, 
            fid,
            title=None,
            ranges=None,
            y_indices=None,
            voff=1e-3,
            lw=1,
            label=None,
            ):
        self.fid=fid
        data = fid.data
        params = fid._params
        if data is [] or data is None:
            raise ValueError('data must exist.')
        if y_indices is not None:
            data = data[numpy.array(y_indices)]
        sw_left = params['sw_left']
        sw = params['sw']

        self.ppm = numpy.linspace(sw_left-sw, sw_left, len(data))[::-1]
       
        self.span_selector = SpanDataSelector(
                data,
                params,
                ranges=ranges, 
                title=title,
                voff=voff,
                label=label)
        self.span_selector.assign = self.assign

    def assign(self):
        self.ranges = self.span_selector.ssm.ranges
        bl_ppm = []
        for rng in self.ranges:
            peak_ind = (self.ppm > rng[1]) * (self.ppm < rng[0])
            cur_peaks = self.ppm[peak_ind]
            bl_ppm.append(cur_peaks)
        bl_ppm = numpy.array([j for i in bl_ppm for j in i])
        self.fid._bl_ppm = bl_ppm
        plt.close(self.span_selector.fig)

if __name__ == '__main__':
    pass
