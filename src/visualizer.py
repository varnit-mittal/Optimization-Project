import numpy as np
import matplotlib.pyplot as plt
import imageio
from matplotlib.colors import Normalize

class Visualizer:
    def __init__(self, cost_func, bounds, output_dir, filename, use_3d=False):
        self.cost_func = cost_func
        self.bounds = np.array(bounds)
        self.output_dir = output_dir
        self.filename = filename
        self.use_3d = use_3d

        assert len(bounds) == 2, "Visualizer only supports 2D problems."
        x = np.linspace(bounds[0][0], bounds[0][1], 500)
        y = np.linspace(bounds[1][0], bounds[1][1], 500)
        self.X, self.Y = np.meshgrid(x, y)
        points = np.stack([self.X.ravel(), self.Y.ravel()], axis=-1)
        self.Z = np.apply_along_axis(self.cost_func, 1, points).reshape(self.X.shape)

    def _render_frame(self, population, generation):
        if self.use_3d:
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')
            surf = ax.plot_surface(self.X, self.Y, self.Z, cmap="viridis", alpha=0.7, edgecolor="none")
            ax.scatter(
                population[:, 0],
                population[:, 1],
                np.apply_along_axis(self.cost_func, 1, population),
                c="red",
                s=10
            )
            ax.set_title(f"Generation {generation}", fontsize=14)
            ax.set_xlabel("X1")
            ax.set_ylabel("X2")
            ax.set_zlabel("f")
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

        else:
            fig, ax = plt.subplots(figsize=(8, 6))
            norm = Normalize(vmin=np.min(self.Z), vmax=np.max(self.Z))
            contour = ax.contourf(self.X, self.Y, self.Z, levels=50, cmap="viridis", norm=norm)
            ax.scatter(population[:, 0], population[:, 1], c="red", s=10)
            ax.set_title(f"Generation {generation}", fontsize=14)
            ax.set_xlim(self.bounds[0][0], self.bounds[0][1])
            ax.set_ylim(self.bounds[1][0], self.bounds[1][1])
            ax.set_xlabel("X1")
            ax.set_ylabel("X2")
            plt.colorbar(contour, ax=ax)

        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)
        return image

    def plot_population(self, generation, population):
        population = np.array(population)
        if population.ndim == 1:
            population = population.reshape(1, -1)
        return self._render_frame(population, generation)

    def create_gif(self, population_history):
        gif_path = f"{self.output_dir}/{self.filename}"
        frames = []
        for generation, pop, _, _ in population_history:
            frame = self.plot_population(generation, pop)
            frames.append(frame)

        imageio.mimsave(gif_path, frames, fps=2)
        print(f"GIF saved at {gif_path}")
