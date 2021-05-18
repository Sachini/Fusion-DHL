import matplotlib.cm as cm
import matplotlib.patches as patches
import numpy as np
from matplotlib.widgets import Slider, Button


class DraggablePoint:
    # Adopted from https://stackoverflow.com/questions/28001655/draggable-line-with-draggable-points

    lock = None  # only one can be animated at a time
    selected = None

    def __init__(self, index, point, parent):
        self.index = index
        self.point = point
        self.press = None
        self.background = None
        self.parent = parent

    def connect(self):
        'connect to all the events we need'
        self.cidpress = self.point.figure.canvas.mpl_connect('button_press_event', self.on_press)
        self.cidrelease = self.point.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cidmotion = self.point.figure.canvas.mpl_connect('motion_notify_event', self.on_motion)

    def on_press(self, event):
        if event.inaxes != self.point.axes: return
        if DraggablePoint.lock is not None: return
        contains, attrd = self.point.contains(event)
        if not contains: return
        self.press = (self.point.center), event.xdata, event.ydata
        DraggablePoint.lock = self
        DraggablePoint.selected = self

        canvas = self.point.figure.canvas
        axes = self.point.axes
        self.point.set_animated(True)
        canvas.draw()
        self.background = canvas.copy_from_bbox(self.point.axes.bbox)
        axes.draw_artist(self.point)
        canvas.blit(axes.bbox)

    def on_motion(self, event):
        if DraggablePoint.lock is not self:
            return
        if event.inaxes != self.point.axes: return
        self.point.center, xpress, ypress = self.press
        dx = event.xdata - xpress
        dy = event.ydata - ypress
        self.point.center = (self.point.center[0] + dx, self.point.center[1] + dy)

        canvas = self.point.figure.canvas
        axes = self.point.axes

        canvas.restore_region(self.background)
        axes.draw_artist(self.point)
        canvas.blit(axes.bbox)

    def on_release(self, event):
        'on release we reset the press data'
        if DraggablePoint.lock is not self:
            return

        self.press = None
        DraggablePoint.lock = None

        # turn off the rect animation property and reset the background
        self.point.set_animated(False)
        self.background = None

        # redraw the full figure
        self.point.figure.canvas.draw()

    def disconnect(self):
        'disconnect all the stored connection ids'
        self.point.figure.canvas.mpl_disconnect(self.cidpress)
        self.point.figure.canvas.mpl_disconnect(self.cidrelease)
        self.point.figure.canvas.mpl_disconnect(self.cidmotion)


class ScatterInteractor(object):
    """
    A scatter plot editor. Select points in left plot and adjust placement in right plot.
    """

    def __init__(self, ax, points, diameter, pos_constraints=None, canvas=None):
        self.ax = ax
        self.points = points
        self.colors = cm.rainbow(np.linspace(0, 1, len(points)))
        self.colors[:, -1] = 0.5

        self.selected_points = {}
        self.active_points = {}
        self.radius = diameter * 5
        self._selected_ind = None  # the active vert
        self._selected_circle = None

        self.canvas = canvas
        self.canvas.mpl_connect('pick_event', self.on_pick)
        self.canvas.mpl_connect('close_event', self.on_close)

        print('double click to select plot points')
        print('press \'d\' and \'r\' to delete and reset selected plot point on Left')
        print('press \'e\' and \'t\' to delete and reset selected plot point on Right')
        print('press \'i\' to specify the start-end-step of trajectory in command line.')

        ax_dim = 3 if ax.shape[1] == 2 else 0
        self.slider = Slider(ax[1, 0], 'Radius', self.radius / 10, self.radius * 5, valinit=self.radius,
                             valstep=self.radius / 4)
        self.slider.on_changed(self.update_point_radius)

        self.slider_c = Slider(ax[4 - ax_dim, ax_dim // 3], 'Radius', self.radius / 10, self.radius * 5,
                               valinit=self.radius, valstep=self.radius / 4)
        self.slider_c.on_changed(self.update_circle_radius)

        self.button = Button(ax[2, 0], 'Update slider', color='0.85', hovercolor='0.95')
        self.button.on_clicked(self._update_point_slider)

        self.button_c = Button(ax[5 - ax_dim, ax_dim // 3], 'Update slider', color='0.85', hovercolor='0.95')
        self.button_c.on_clicked(self._update_circle_slider)

        self.start, self.end, self.step = 0, len(self.points), 500
        self.constraint_plot = self.ax[3 - ax_dim, ax_dim // 3]
        self.draw_plot()
        self.ax[0, 0].plot(self.points[:, 0], self.points[:, 1], color='y')
        print('Adding points')
        if pos_constraints is None:
            self.add_new_point(0)
            self.add_new_point(len(points) - 1)
        else:
            self.add_points_batch(pos_constraints)

    def draw_plot(self):
        self.orig_plot = self.ax[0, 0].scatter(self.points[self.start:self.end:self.step, 0],
                                               self.points[self.start:self.end:self.step, 1],
                                               color=self.colors[self.start:self.end:self.step], picker=5, s=20)
        self._selected_ind = None
        self._reset_slider()
        if self.selected_points is not None:
            self.update_points_batch()

    def draw_plot_only(self):
        self.orig_plot = self.ax[0, 0].scatter(self.points[self.start:self.end:self.step, 0],
                                               self.points[self.start:self.end:self.step, 1],
                                               color=self.colors[self.start:self.end:self.step], picker=5, s=20)
        self._selected_ind = None
        self._reset_slider()
        if self.selected_points is not None:
            for key in self.selected_points.keys():
                if key >= self.start and key < self.end and (key - self.start) % self.step == 0:
                    i = self._ps(key)
                    self.orig_plot._facecolors[i, -1] = 0.8
                    self.orig_plot._edgecolors[i, :] = (0, 0, 0, 1)
        self.canvas.draw()

    def on_close(self, event):
        self.update_selected_points()

    def update_selected_points(self):
        self.selected_points.update(self.active_points)

    def _pi(self, i):
        # index of point given index of scatter plot
        return self.start + i * self.step

    def _ps_in_map(self, i):
        # index of scatter plot given index of point
        return i > self.start and (i - self.start) % self.step == 0

    def _ps(self, i):
        # index of scatter plot given index of point
        return (i - self.start) // self.step

    def on_pick(self, event):
        if event.mouseevent.dblclick:
            ind = self._pi(event.ind[0])
            self._selected_ind = ind
            if not (ind in self.active_points):
                self.add_new_point(ind)
            self._set_slider_value()

    def _set_slider_value(self):
        self.slider.set_val(self.active_points[self._selected_ind].point.get_radius())

    def _reset_slider(self, args=None, **kwargs):
        self.slider.reset()

    def _update_circle_slider(self, args=None, **kwargs):
        if DraggablePoint.selected:
            self.slider_c.set_val(DraggablePoint.selected.point.get_radius())
        else:
            self.slider_c.reset()

    def _update_point_slider(self, args=None, **kwargs):
        if self._selected_ind is not None:
            self.slider.set_val(self.active_points[self._selected_ind].point.get_radius())
        else:
            self.slider.reset()

    def add_new_point(self, ind):
        i = self._ps(ind)
        self.orig_plot._facecolors[i, -1] = 0.8
        self.orig_plot._edgecolors[i, :] = (0, 0, 0, 1)

        circle = patches.Circle((self.points[ind, 0], self.points[ind, 1]), self.radius, fc=self.colors[ind, :3],
                                alpha=0.8)
        self.constraint_plot.add_patch(circle)
        circle.figure.set_canvas(self.canvas)
        dr = DraggablePoint(ind, circle, self)
        dr.connect()
        self.active_points[ind] = dr
        self._reset_slider()
        self.canvas.draw()

    def add_points_batch(self, points):
        # Add multiplt points
        # points - [frame, x, y, radius]
        for i in range(len(points)):
            ind = int(points[i, 0])
            if self._ps_in_map(ind):
                s_in = self._ps(ind)
                self.orig_plot._facecolors[s_in, -1] = 0.8
                self.orig_plot._edgecolors[s_in, :] = (0, 0, 0, 1)

            circle = patches.Circle((points[i, 1], points[i, 2]), points[i, 3], fc=self.colors[ind, :3], alpha=0.8)
            self.constraint_plot.add_patch(circle)
            dr = DraggablePoint(ind, circle, self)
            dr.connect()
            self.active_points[ind] = dr
            self.canvas.draw()

    def update_points_batch(self):
        # Update points when the active range is refined
        for key in self.selected_points.keys():
            if key >= self.start and key < self.end and (key - self.start) % self.step == 0:
                i = self._ps(key)
                self.orig_plot._facecolors[i, -1] = 0.8
                self.orig_plot._edgecolors[i, :] = (0, 0, 0, 1)
                self.active_points[key] = self.selected_points[key]
                self.active_points[key].point.set_alpha(0.8)
            else:
                self.selected_points[key].point.set_alpha(0.3)
        self.canvas.draw()

    def delete_point(self, ind):
        circle = self.active_points[ind].point
        circle.remove()
        self.active_points.pop(ind)
        if ind in self.selected_points:
            self.selected_points.pop(ind)

        i = self._ps(ind)
        self.orig_plot._facecolors[i] = self.colors[ind]
        self.orig_plot._edgecolors[i] = self.colors[ind]
        self._reset_slider()
        self.canvas.draw()
        self._selected_ind = None

    def remove_circle(self, circle):
        index = circle.index
        circle.point.remove()
        if index in self.active_points:
            del self.active_points[index]
        if index in self.selected_points:
            del self.selected_points[index]

        if index == self._selected_ind:
            self._reset_slider()
            self._selected_ind = None
        if self._ps_in_map(index):
            i = self._ps(index)
            self.orig_plot._facecolors[i] = self.colors[index]
            self.orig_plot._edgecolors[i] = self.colors[index]
        self.canvas.draw()
        DraggablePoint.selected = None

    def remove_all_active_circles(self):
        for index in self.active_points:
            circle = self.active_points[index].point
            circle.remove()
            if index in self.selected_points:
                del self.selected_points[index]

            if index == self._selected_ind:
                self._reset_slider()
                self._selected_ind = None
            if self._ps_in_map(index):
                i = self._ps(index)
                self.orig_plot._facecolors[i] = self.colors[index]
                self.orig_plot._edgecolors[i] = self.colors[index]
            DraggablePoint.selected = None
        self.canvas.draw()
        print("Removed {} points".format(len(self.active_points)))
        self.active_points.clear()

    def reset_point(self, ind):
        circle = self.active_points[ind].point
        circle.center = (self.points[ind, 0], self.points[ind, 1])
        circle.set_radius(self.radius)
        self._reset_slider()
        self.canvas.draw()

    def reset_circle(self, circle):
        index = circle.index
        circle.center = (self.points[index, 0], self.points[index, 1])
        circle.point.set_radius(self.radius)
        if index == self._selected_ind:
            self._reset_slider()
        self.canvas.draw()

    def update_point_radius(self, val):
        if self._selected_ind is None:
            print('Select a point to update')
        elif self._selected_ind in self.active_points:
            self.active_points[self._selected_ind].point.set_radius(val)
            self.canvas.draw()
        else:
            print('Current point {} is not selected'.format(self._selected_ind))

    def update_circle_radius(self, val):
        if DraggablePoint.selected is None:
            print('Select a circle to update')
        else:
            DraggablePoint.selected.point.set_radius(val)
            self.canvas.draw()

    def key_press_callback(self, event):
        'whenever a key is pressed'
        if not event.inaxes:
            return
        elif event.key == 'd':
            if self._selected_ind is None:
                print('Select a point to delete')
            elif self._selected_ind in self.active_points:
                print('Deleting point {}'.format(self._selected_ind))
                self.delete_point(self._selected_ind)
            else:
                print('Current point {} is not selected'.format(self._selected_ind))
        elif event.key == 'e':
            if DraggablePoint.selected is None:
                print('Select a circle to delete')
            else:
                print('Deleting circle {}'.format(DraggablePoint.selected.index))
                self.remove_circle(DraggablePoint.selected)
        elif event.key == 'r':
            if self._selected_ind is None:
                print('Select a point to reset')
            elif self._selected_ind in self.active_points:
                print('Resetting point {}'.format(self._selected_ind))
                self.reset_point(self._selected_ind)
            else:
                print('Current point {} is not selected'.format(self._selected_ind))
        elif event.key == 't':
            if DraggablePoint.selected is None:
                print('Select a circle to reset')
            else:
                print('Resetting circle {}'.format(DraggablePoint.selected.index))
                self.reset_circle(DraggablePoint.selected)
        elif event.key == 'i':
            start, end, step = self.refine_plot_params(self.start, self.end, self.step)
            self.refine_plot(start, end, step)

    def key_press_callback_from_tk(self, key):
        'whenever a key is pressed'
        if key == 'd':
            if self._selected_ind is None:
                print('Select a point to delete')
            elif self._selected_ind in self.active_points:
                print('Deleting point {}'.format(self._selected_ind))
                self.delete_point(self._selected_ind)
            else:
                print('Current point {} is not selected'.format(self._selected_ind))
        elif key == 'e':
            if DraggablePoint.selected is None:
                print('Select a circle to delete')
            else:
                print('Deleting circle {}'.format(DraggablePoint.selected.index))
                self.remove_circle(DraggablePoint.selected)
        elif key == 'r':
            if self._selected_ind is None:
                print('Select a point to reset')
            elif self._selected_ind in self.active_points:
                print('Resetting point {}'.format(self._selected_ind))
                self.reset_point(self._selected_ind)
            else:
                print('Current point {} is not selected'.format(self._selected_ind))
        elif key == 't':
            if DraggablePoint.selected is None:
                print('Select a circle to reset')
            else:
                print('Resetting circle {}'.format(DraggablePoint.selected.index))
                self.reset_circle(DraggablePoint.selected)
        elif key == 'i':
            start, end, step = self.refine_plot_params(self.start, self.end, self.step)
            self.refine_plot(start, end, step)

    def refine_plot(self, start, end, step):
        self.selected_points.update(self.active_points)
        self.start, self.end, self.step = start, end, step
        self.active_points.clear()
        self.orig_plot.remove()
        self.canvas.draw()
        print('Update plots')

        self.draw_plot()
        self.canvas.draw()

    def refine_plot_params(self, start, end, step):
        print('Change plot points?\n Press \'n\' for to cancel, \'y\' to reset.\n If not input 3 integers specifying '
              'start, end, interval between points (-1 to keep the current value)')
        print('Current values [start, end, step] = [{}, {}, {}]'.format(start, end, step))
        u_input = input()
        if u_input == "N" or u_input == 'n':
            return start, end, step
        if u_input == "Y" or u_input == 'y':
            return 0, len(self.points), 1
        else:
            try:
                u_input = [int(a) for a in u_input.split()]
                u_input[0] = u_input[0] if u_input[0] >= 0 else start
                u_input[1] = u_input[1] if u_input[1] >= 0 else end
                u_input[2] = u_input[2] if u_input[2] >= 0 else step
                if u_input[0] >= u_input[1]:
                    print('Start index should be less then end. Retry')
                    return self.refine_plot_params(start, end, step)
                else:
                    return u_input[0], u_input[1], u_input[2]
            except:
                print('Error occurred. Retry')
                return self.refine_plot_params(start, end, step)
