"""Animate the computed axial current in an unwrapped 3D cage mesh."""
import argparse
from pathlib import Path
import meshio
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter


def render(source):
    mesh = meshio.read(source)
    centers = mesh.points[mesh.cells_dict["tetra"]].mean(axis=1)
    select = (centers[:,2]>0)&(centers[:,2]<.054)&(np.hypot(centers[:,0],centers[:,1])>.0229)
    point = centers[select]
    field = (mesh.cell_data_dict["J_real"]["tetra"][select,2]+1j*mesh.cell_data_dict["J_imag"]["tetra"][select,2])/1e6
    fig,ax=plt.subplots(figsize=(7,3.4),constrained_layout=True)
    bound=max(abs(field))
    cloud=ax.scatter(np.rad2deg(np.arctan2(point[:,1],point[:,0])),1000*point[:,2],s=3,c=field.real,
                     cmap="coolwarm",vmin=-bound,vmax=bound)
    fig.colorbar(cloud,ax=ax,label="Осевой ток, А/мм²")
    ax.set(xlabel="Угол вокруг вала, градусы",ylabel="Вдоль пакета, мм")
    def update(frame):
        cloud.set_array((field*np.exp(2j*np.pi*frame/24)).real)
        ax.set_title(f"Клетка со скосом. Фаза тока {15*frame}°")
    movie=FuncAnimation(fig,update,frames=24)
    movie.save(source.with_suffix(".gif"),writer=PillowWriter(fps=12))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("source",type=Path)
    render(p.parse_args().source)
