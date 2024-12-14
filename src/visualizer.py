import numpy as np
import matplotlib.pyplot as plt
import imageio
from matplotlib.colors import Normalize

class Visualizer:
    """
    A class to visualize the optimization process on 2D cost functions, with options for 3D or 2D visualization.
    Can generate a GIF of the optimization process over generations.
    """

    def __init__(self, cost_func, bounds, output_dir, filename, use_3d=False):
        """
        Initialize the visualizer.
        
        Args:
            cost_func (callable): The cost function to visualize.
            bounds (list of tuples): Bounds for the 2D search space [(x_min, x_max), (y_min, y_max)].
            output_dir (str): Directory to save the generated GIF.
            filename (str): Name of the output GIF file.
            use_3d (bool): Whether to use 3D plotting (default: False).
        """
        self.cost_func = cost_func
        self.bounds = np.array(bounds)  # Bounds for the 2D space
        self.output_dir = output_dir
        self.filename = filename
        self.use_3d = use_3d

        # Ensure bounds are for a 2D problem
        assert len(bounds) == 2, "Visualizer only supports 2D problems."
        
        # Generate grid points for evaluating the cost function
        x = np.linspace(bounds[0][0], bounds[0][1], 500)
        y = np.linspace(bounds[1][0], bounds[1][1], 500)
        self.X, self.Y = np.meshgrid(x, y)
        points = np.stack([self.X.ravel(), self.Y.ravel()], axis=-1)
        
        # Compute cost function values over the grid
        self.Z = np.apply_along_axis(self.cost_func, 1, points).reshape(self.X.shape)

    def _render_frame(self, population, generation):
        """
        Render a single frame of the optimization process, either in 2D or 3D.
        
        Args:
            population (ndarray): Current population of solutions.
            generation (int): Current generation number.
        
        Returns:
            image (ndarray): Image of the rendered frame.
        """
        if self.use_3d:
            # Create 3D plot
            fig = plt.figure(figsize=(10, 7))
            ax = fig.add_subplot(111, projection='3d')

            # Plot the surface
            surf = ax.plot_surface(self.X, self.Y, self.Z, cmap="viridis", alpha=0.7, edgecolor="none")
            
            # Scatter plot of the current population
            ax.scatter(
                population[:, 0],
                population[:, 1],
                np.apply_along_axis(self.cost_func, 1, population),
                c="red",
                s=10
            )
            
            # Add labels and title
            ax.set_title(f"Generation {generation}", fontsize=14)
            ax.set_xlabel("X1")
            ax.set_ylabel("X2")
            ax.set_zlabel("f")
            fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)  # Add colorbar

        else:
            # Create 2D contour plot
            fig, ax = plt.subplots(figsize=(8, 6))
            
            # Normalize Z values for consistent coloring
            norm = Normalize(vmin=np.min(self.Z), vmax=np.max(self.Z))
            
            # Plot contour map
            contour = ax.contourf(self.X, self.Y, self.Z, levels=50, cmap="viridis", norm=norm)
            
            # Scatter plot of the current population
            ax.scatter(population[:, 0], population[:, 1], c="red", s=10)
            
            # Add labels and title
            ax.set_title(f"Generation {generation}", fontsize=14)
            ax.set_xlim(self.bounds[0][0], self.bounds[0][1])
            ax.set_ylim(self.bounds[1][0], self.bounds[1][1])
            ax.set_xlabel("X1")
            ax.set_ylabel("X2")
            plt.colorbar(contour, ax=ax)  # Add colorbar

        # Convert plot to an image
        fig.canvas.draw()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        plt.close(fig)  # Close the figure to free memory
        return image

    def plot_population(self, generation, population):
        """
        Plot the current population on the cost function.
        
        Args:
            generation (int): Current generation number.
            population (ndarray): Current population of solutions.
        
        Returns:
            image (ndarray): Rendered frame as an image.
        """
        population = np.array(population)
        if population.ndim == 1:
            population = population.reshape(1, -1)  # Ensure population is 2D
        return self._render_frame(population, generation)

    def create_gif(self, population_history):
        """
        Generate a GIF of the optimization process over generations.
        
        Args:
            population_history (list of tuples): List of tuples containing (generation, population, _, _).
        """
        gif_path = f"{self.output_dir}/{self.filename}"
        frames = []

        # Generate frames for each generation
        for generation, pop, _, _ in population_history:
            frame = self.plot_population(generation, pop)
            frames.append(frame)

        # Save all frames as a GIF
        imageio.mimsave(gif_path, frames, fps=2)
        print(f"GIF saved at {gif_path}")
