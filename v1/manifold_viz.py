"""
Visualization Module for Manifold Projections
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Dict, Any
from manifold_learning import ManifoldProjection, TrajectoryAnalyzer
import warnings

warnings.filterwarnings('ignore')


class ManifoldVisualizer:
    """
    Visualization tools for manifold projections
    """
    
    def __init__(self, figsize=(10, 8)):
        self.figsize = figsize
        self.colors = plt.cm.tab10
    
    def plot_projection_2d(self, projection: ManifoldProjection,
                          labels: Optional[List] = None,
                          title: Optional[str] = None,
                          show_labels: bool = False,
                          alpha: float = 0.6,
                          s: float = 50):
        """
        Plot 2D manifold projection
        """
        if projection.target_dim != 2:
            raise ValueError("Projection must be 2D for this plot")
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        X = projection.embedding
        
        if labels is not None:
            # Color by labels
            unique_labels = sorted(set(labels))
            for i, label in enumerate(unique_labels):
                mask = np.array(labels) == label
                ax.scatter(X[mask, 0], X[mask, 1], 
                          c=[self.colors(i)], 
                          label=str(label),
                          alpha=alpha, 
                          s=s,
                          edgecolors='black',
                          linewidths=0.5)
            ax.legend()
        else:
            ax.scatter(X[:, 0], X[:, 1], 
                      alpha=alpha, 
                      s=s,
                      edgecolors='black',
                      linewidths=0.5)
        
        if show_labels:
            for i, entity_id in enumerate(projection.entity_ids):
                ax.annotate(entity_id, (X[i, 0], X[i, 1]), 
                           fontsize=8, alpha=0.7)
        
        if title is None:
            title = f"{projection.method} Projection"
            if 'explained_variance' in projection.metadata:
                var = projection.metadata['total_explained_variance']
                title += f" (variance explained: {var:.2%})"
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(f"Component 1", fontsize=12)
        ax.set_ylabel(f"Component 2", fontsize=12)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, ax
    
    def plot_projection_3d(self, projection: ManifoldProjection,
                          labels: Optional[List] = None,
                          title: Optional[str] = None,
                          alpha: float = 0.6,
                          s: float = 50):
        """
        Plot 3D manifold projection
        """
        if projection.target_dim != 3:
            raise ValueError("Projection must be 3D for this plot")
        
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        X = projection.embedding
        
        if labels is not None:
            unique_labels = sorted(set(labels))
            for i, label in enumerate(unique_labels):
                mask = np.array(labels) == label
                ax.scatter(X[mask, 0], X[mask, 1], X[mask, 2],
                          c=[self.colors(i)], 
                          label=str(label),
                          alpha=alpha, 
                          s=s)
            ax.legend()
        else:
            ax.scatter(X[:, 0], X[:, 1], X[:, 2],
                      alpha=alpha, 
                      s=s)
        
        if title is None:
            title = f"{projection.method} Projection (3D)"
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.set_zlabel("Component 3")
        
        plt.tight_layout()
        return fig, ax
    
    def compare_projections(self, projections: Dict[str, ManifoldProjection],
                           labels: Optional[List] = None,
                           ncols: int = 3):
        """
        Compare multiple projections side by side
        """
        n_projections = len(projections)
        nrows = (n_projections + ncols - 1) // ncols
        
        fig, axes = plt.subplots(nrows, ncols, figsize=(6*ncols, 5*nrows))
        
        if n_projections == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, (name, projection) in enumerate(projections.items()):
            if projection.target_dim != 2:
                continue
            
            ax = axes[idx]
            X = projection.embedding
            
            if labels is not None:
                unique_labels = sorted(set(labels))
                for i, label in enumerate(unique_labels):
                    mask = np.array(labels) == label
                    ax.scatter(X[mask, 0], X[mask, 1], 
                              c=[self.colors(i)],
                              alpha=0.6,
                              s=30,
                              edgecolors='black',
                              linewidths=0.3)
            else:
                ax.scatter(X[:, 0], X[:, 1], alpha=0.6, s=30)
            
            ax.set_title(projection.method, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        # Hide extra subplots
        for idx in range(n_projections, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        return fig, axes
    
    def plot_trajectory(self, trajectory_points: np.ndarray,
                       projection: ManifoldProjection,
                       entity_id: str,
                       show_points: bool = True,
                       show_arrows: bool = True,
                       color: str = 'blue'):
        """
        Plot a single entity's trajectory through manifold space
        """
        if projection.target_dim != 2:
            raise ValueError("Only 2D trajectories supported for now")
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot all entities as context
        X = projection.embedding
        ax.scatter(X[:, 0], X[:, 1], 
                  alpha=0.1, 
                  s=20, 
                  c='gray',
                  label='All entities')
        
        # Plot trajectory
        traj = trajectory_points
        
        if show_points:
            ax.scatter(traj[:, 0], traj[:, 1], 
                      c=color, 
                      s=100,
                      alpha=0.7,
                      edgecolors='black',
                      linewidths=1,
                      label='Trajectory points',
                      zorder=5)
        
        # Plot path
        ax.plot(traj[:, 0], traj[:, 1], 
               c=color, 
               linewidth=2, 
               alpha=0.7,
               zorder=4)
        
        if show_arrows and len(traj) > 1:
            # Add arrows to show direction
            for i in range(len(traj) - 1):
                ax.annotate('', 
                           xy=traj[i+1], 
                           xytext=traj[i],
                           arrowprops=dict(
                               arrowstyle='->',
                               color=color,
                               lw=1.5,
                               alpha=0.7
                           ),
                           zorder=6)
        
        # Mark start and end
        ax.scatter(traj[0, 0], traj[0, 1], 
                  c='green', 
                  s=200, 
                  marker='o',
                  edgecolors='black',
                  linewidths=2,
                  label='Start',
                  zorder=7)
        
        ax.scatter(traj[-1, 0], traj[-1, 1], 
                  c='red', 
                  s=200, 
                  marker='s',
                  edgecolors='black',
                  linewidths=2,
                  label='End',
                  zorder=7)
        
        ax.set_title(f"Trajectory: {entity_id}", fontsize=14, fontweight='bold')
        ax.set_xlabel("Component 1", fontsize=12)
        ax.set_ylabel("Component 2", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, ax
    
    def plot_multiple_trajectories(self, trajectories: Dict[str, np.ndarray],
                                   projection: ManifoldProjection,
                                   title: str = "Entity Trajectories"):
        """
        Plot multiple entity trajectories
        """
        if projection.target_dim != 2:
            raise ValueError("Only 2D trajectories supported for now")
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot all entities as context
        X = projection.embedding
        ax.scatter(X[:, 0], X[:, 1], 
                  alpha=0.1, 
                  s=20, 
                  c='gray')
        
        # Plot trajectories
        colors = plt.cm.tab10(np.linspace(0, 1, len(trajectories)))
        
        for idx, (entity_id, traj) in enumerate(trajectories.items()):
            color = colors[idx]
            
            # Plot path
            ax.plot(traj[:, 0], traj[:, 1], 
                   c=color, 
                   linewidth=2, 
                   alpha=0.7,
                   label=entity_id)
            
            # Mark start
            ax.scatter(traj[0, 0], traj[0, 1], 
                      c=[color], 
                      s=100, 
                      marker='o',
                      edgecolors='black',
                      linewidths=1,
                      zorder=5)
            
            # Mark end
            ax.scatter(traj[-1, 0], traj[-1, 1], 
                      c=[color], 
                      s=100, 
                      marker='s',
                      edgecolors='black',
                      linewidths=1,
                      zorder=5)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel("Component 1", fontsize=12)
        ax.set_ylabel("Component 2", fontsize=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, ax
    
    def plot_feature_importance(self, projection: ManifoldProjection,
                               component: int = 0,
                               top_n: int = 20):
        """
        Plot feature importance for PCA components
        (only works for PCA projections)
        """
        if projection.method != "PCA":
            print("Feature importance only available for PCA")
            return None, None
        
        # Would need access to the fitted model to get components
        # This is a placeholder for the structure
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.set_title(f"Feature Importance - Component {component}", 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel("Feature")
        ax.set_ylabel("Loading")
        
        plt.tight_layout()
        return fig, ax
    
    def plot_variance_explained(self, projection: ManifoldProjection):
        """
        Plot explained variance for PCA
        """
        if 'explained_variance_ratio' not in projection.metadata:
            print("Variance data not available for this projection")
            return None, None
        
        var_ratio = projection.metadata['explained_variance_ratio']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Individual variance
        components = range(1, len(var_ratio) + 1)
        ax1.bar(components, var_ratio)
        ax1.set_xlabel("Component", fontsize=12)
        ax1.set_ylabel("Variance Explained Ratio", fontsize=12)
        ax1.set_title("Variance Explained by Component", fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Cumulative variance
        cumsum = np.cumsum(var_ratio)
        ax2.plot(components, cumsum, marker='o', linewidth=2)
        ax2.axhline(y=0.8, color='r', linestyle='--', alpha=0.5, label='80% threshold')
        ax2.axhline(y=0.95, color='g', linestyle='--', alpha=0.5, label='95% threshold')
        ax2.set_xlabel("Number of Components", fontsize=12)
        ax2.set_ylabel("Cumulative Variance Explained", fontsize=12)
        ax2.set_title("Cumulative Variance Explained", fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, (ax1, ax2)
    
    def save_figure(self, fig, filepath: str, dpi: int = 300):
        """Save figure to file"""
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        print(f"Figure saved to {filepath}")


if __name__ == "__main__":
    print("Manifold Visualization Module")