import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def create_plot(image, dpi, traj, save_files=False, title='', calc_errors=False, filename=None, show_fig=False,
                pos_constraints=None, time_adjust=1):
    fig, ax = plt.subplots(1, 1, figsize=(20, 20))
    ax.imshow(image, extent=[-0.5, image.shape[1] / dpi, image.shape[0] / dpi, -0.5], cmap='Greys_r')

    cmap = matplotlib.cm.brg(np.linspace(0, 1, len(traj)))
    gt_idx, gt_idx_cmap = None, None
    if pos_constraints is not None:
        gt_idx = np.round(pos_constraints[:, 0] / time_adjust).astype(np.int)
        gt_idx_cmap = cmap[gt_idx]
        ax.scatter(pos_constraints[:, 1], pos_constraints[:, 2], color=gt_idx_cmap, s=350, alpha=0.4)

    ax.scatter(traj[:, 0], traj[:, 1], color=cmap, s=1)

    if pos_constraints is not None:
        ax.scatter(traj[gt_idx, 0], traj[gt_idx, 1], color=gt_idx_cmap, s=200, marker='*', edgecolor='b',
                   linewidth=0.01, alpha=0.6)
        ax.scatter(pos_constraints[:, 1], pos_constraints[:, 2], color='k', s=5)

    dist = -1
    if calc_errors and pos_constraints is not None:
        dist = calc_euclidian_error(gt_idx, pos_constraints[:, 1:3], traj, time_diff=time_adjust)
        title = '{} | error: {:.4f} m '.format(title, dist)

    if len(title) > 0:
        ax.set_title(title, fontsize=25)

    plt.tight_layout()
    if save_files:
        plt.savefig(filename, transparent=False)
    if show_fig:
        plt.show()
    plt.close('all')
    return dist


def calc_euclidian_error(gt_idx, gt_traj, trajectory, time_diff):
    if time_diff != 1:
        gt_idx = np.round(gt_idx / time_diff).astype(np.int)
    return np.mean(np.linalg.norm(trajectory[gt_idx] - gt_traj, axis=1))
