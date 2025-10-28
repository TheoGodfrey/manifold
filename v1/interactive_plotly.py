"""
Interactive Plotly Visualizations for Manifold System
Provides interactive, zoomable, hoverable plots
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Any

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Plotly not available. Install with: pip install plotly")

from manifold_learning import ManifoldProjection


class InteractivePlotter:
    """
    Interactive visualization using Plotly
    """
    
    def __init__(self):
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly required. Install with: pip install plotly")
    
    def plot_projection_2d(self, projection: ManifoldProjection,
                          labels: Optional[List] = None,
                          hover_data: Optional[Dict] = None,
                          title: Optional[str] = None,
                          size: int = 8):
        """
        Interactive 2D scatter plot
        
        Args:
            projection: ManifoldProjection object
            labels: Optional labels for coloring
            hover_data: Additional data to show on hover (dict with entity_id as key)
            title: Plot title
            size: Marker size
        """
        if projection.target_dim != 2:
            raise ValueError("Projection must be 2D")
        
        X = projection.embedding
        entity_ids = projection.entity_ids
        
        # Prepare hover text
        hover_texts = []
        for i, eid in enumerate(entity_ids):
            text = f"Entity: {eid}<br>"
            text += f"Component 1: {X[i, 0]:.4f}<br>"
            text += f"Component 2: {X[i, 1]:.4f}"
            
            if hover_data and eid in hover_data:
                for key, value in hover_data[eid].items():
                    text += f"<br>{key}: {value}"
            
            hover_texts.append(text)
        
        # Create DataFrame
        df = pd.DataFrame({
            'Component 1': X[:, 0],
            'Component 2': X[:, 1],
            'Entity ID': entity_ids,
            'hover_text': hover_texts
        })
        
        if labels is not None:
            df['Label'] = labels
            color_col = 'Label'
        else:
            color_col = None
        
        # Create plot
        if title is None:
            title = f"{projection.method} Projection (Interactive)"
        
        fig = px.scatter(
            df,
            x='Component 1',
            y='Component 2',
            color=color_col,
            hover_data={'hover_text': True, 'Component 1': False, 'Component 2': False},
            title=title,
            width=900,
            height=700
        )
        
        # Update layout
        fig.update_traces(
            marker=dict(size=size, line=dict(width=0.5, color='white')),
            hovertemplate='%{customdata[0]}<extra></extra>'
        )
        
        fig.update_layout(
            hovermode='closest',
            showlegend=True if labels is not None else False,
            plot_bgcolor='rgba(240,240,240,0.9)',
            font=dict(size=12)
        )
        
        return fig
    
    def plot_projection_3d(self, projection: ManifoldProjection,
                          labels: Optional[List] = None,
                          hover_data: Optional[Dict] = None,
                          title: Optional[str] = None,
                          size: int = 5):
        """
        Interactive 3D scatter plot
        """
        if projection.target_dim != 3:
            raise ValueError("Projection must be 3D")
        
        X = projection.embedding
        entity_ids = projection.entity_ids
        
        # Prepare hover text
        hover_texts = []
        for i, eid in enumerate(entity_ids):
            text = f"Entity: {eid}<br>"
            text += f"C1: {X[i, 0]:.3f} | C2: {X[i, 1]:.3f} | C3: {X[i, 2]:.3f}"
            
            if hover_data and eid in hover_data:
                for key, value in hover_data[eid].items():
                    text += f"<br>{key}: {value}"
            
            hover_texts.append(text)
        
        # Create DataFrame
        df = pd.DataFrame({
            'C1': X[:, 0],
            'C2': X[:, 1],
            'C3': X[:, 2],
            'Entity ID': entity_ids,
            'hover_text': hover_texts
        })
        
        if labels is not None:
            df['Label'] = [str(l) for l in labels]
            color_col = 'Label'
        else:
            color_col = None
        
        # Create 3D plot
        if title is None:
            title = f"{projection.method} Projection (3D Interactive)"
        
        fig = px.scatter_3d(
            df,
            x='C1',
            y='C2',
            z='C3',
            color=color_col,
            hover_data={'hover_text': True, 'C1': False, 'C2': False, 'C3': False},
            title=title,
            width=1000,
            height=800
        )
        
        fig.update_traces(
            marker=dict(size=size, line=dict(width=0.3, color='white')),
            hovertemplate='%{customdata[0]}<extra></extra>'
        )
        
        fig.update_layout(
            scene=dict(
                xaxis_title='Component 1',
                yaxis_title='Component 2',
                zaxis_title='Component 3',
                bgcolor='rgba(240,240,240,0.9)'
            )
        )
        
        return fig
    
    def plot_trajectory(self, trajectory_points: np.ndarray,
                       projection: ManifoldProjection,
                       entity_id: str,
                       timestamps: Optional[List] = None,
                       show_all: bool = True):
        """
        Interactive trajectory plot
        
        Args:
            trajectory_points: Array of shape (n_timepoints, n_dims)
            projection: Full projection for context
            entity_id: ID of the entity being traced
            timestamps: Optional timestamps for each point
            show_all: Whether to show all entities as context
        """
        if projection.target_dim != 2:
            raise ValueError("Only 2D trajectories supported")
        
        fig = go.Figure()
        
        # Background: all entities
        if show_all:
            X = projection.embedding
            fig.add_trace(go.Scatter(
                x=X[:, 0],
                y=X[:, 1],
                mode='markers',
                marker=dict(size=5, color='lightgray', opacity=0.3),
                name='All entities',
                hoverinfo='skip'
            ))
        
        # Trajectory path
        traj = trajectory_points
        
        if timestamps:
            hover_text = [f"Time: {t}<br>C1: {traj[i,0]:.3f}<br>C2: {traj[i,1]:.3f}" 
                         for i, t in enumerate(timestamps)]
        else:
            hover_text = [f"Point {i}<br>C1: {traj[i,0]:.3f}<br>C2: {traj[i,1]:.3f}" 
                         for i in range(len(traj))]
        
        # Add trajectory line
        fig.add_trace(go.Scatter(
            x=traj[:, 0],
            y=traj[:, 1],
            mode='lines+markers',
            line=dict(color='blue', width=3),
            marker=dict(size=10, color='blue'),
            name=f'Trajectory: {entity_id}',
            text=hover_text,
            hovertemplate='%{text}<extra></extra>'
        ))
        
        # Mark start and end
        fig.add_trace(go.Scatter(
            x=[traj[0, 0]],
            y=[traj[0, 1]],
            mode='markers',
            marker=dict(size=15, color='green', symbol='circle', line=dict(width=2, color='black')),
            name='Start',
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Scatter(
            x=[traj[-1, 0]],
            y=[traj[-1, 1]],
            mode='markers',
            marker=dict(size=15, color='red', symbol='square', line=dict(width=2, color='black')),
            name='End',
            hoverinfo='skip'
        ))
        
        fig.update_layout(
            title=f"Trajectory: {entity_id}",
            xaxis_title="Component 1",
            yaxis_title="Component 2",
            hovermode='closest',
            width=900,
            height=700,
            plot_bgcolor='rgba(240,240,240,0.9)'
        )
        
        return fig
    
    def plot_multiple_trajectories(self, trajectories: Dict[str, np.ndarray],
                                   projection: ManifoldProjection,
                                   title: str = "Entity Trajectories"):
        """
        Plot multiple trajectories
        """
        if projection.target_dim != 2:
            raise ValueError("Only 2D trajectories supported")
        
        fig = go.Figure()
        
        # Background
        X = projection.embedding
        fig.add_trace(go.Scatter(
            x=X[:, 0],
            y=X[:, 1],
            mode='markers',
            marker=dict(size=5, color='lightgray', opacity=0.2),
            name='All entities',
            hoverinfo='skip'
        ))
        
        # Add trajectories
        colors = px.colors.qualitative.Set2
        
        for idx, (entity_id, traj) in enumerate(trajectories.items()):
            color = colors[idx % len(colors)]
            
            # Path
            fig.add_trace(go.Scatter(
                x=traj[:, 0],
                y=traj[:, 1],
                mode='lines+markers',
                line=dict(color=color, width=2),
                marker=dict(size=8, color=color),
                name=entity_id,
                hovertemplate=f'{entity_id}<br>C1: %{{x:.3f}}<br>C2: %{{y:.3f}}<extra></extra>'
            ))
            
            # Start marker
            fig.add_trace(go.Scatter(
                x=[traj[0, 0]],
                y=[traj[0, 1]],
                mode='markers',
                marker=dict(size=12, color=color, symbol='circle', line=dict(width=2, color='black')),
                showlegend=False,
                hoverinfo='skip'
            ))
            
            # End marker
            fig.add_trace(go.Scatter(
                x=[traj[-1, 0]],
                y=[traj[-1, 1]],
                mode='markers',
                marker=dict(size=12, color=color, symbol='square', line=dict(width=2, color='black')),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Component 1",
            yaxis_title="Component 2",
            hovermode='closest',
            width=1000,
            height=800,
            plot_bgcolor='rgba(240,240,240,0.9)'
        )
        
        return fig
    
    def compare_projections_interactive(self, projections: Dict[str, ManifoldProjection],
                                       labels: Optional[List] = None):
        """
        Interactive comparison of multiple projections
        """
        n_proj = len(projections)
        
        if n_proj == 0:
            raise ValueError("No projections provided")
        
        # Create subplots
        rows = (n_proj + 1) // 2
        cols = 2
        
        fig = make_subplots(
            rows=rows,
            cols=cols,
            subplot_titles=[proj.method for proj in projections.values()],
            vertical_spacing=0.15,
            horizontal_spacing=0.1
        )
        
        # Add each projection
        for idx, (name, proj) in enumerate(projections.items()):
            if proj.target_dim != 2:
                continue
            
            row = idx // 2 + 1
            col = idx % 2 + 1
            
            X = proj.embedding
            
            if labels is not None:
                unique_labels = sorted(set(labels))
                colors = px.colors.qualitative.Set1
                
                for i, label in enumerate(unique_labels):
                    mask = np.array(labels) == label
                    
                    fig.add_trace(
                        go.Scatter(
                            x=X[mask, 0],
                            y=X[mask, 1],
                            mode='markers',
                            marker=dict(size=6, color=colors[i % len(colors)]),
                            name=str(label),
                            legendgroup=str(label),
                            showlegend=(idx == 0),
                            hovertemplate=f'{proj.method}<br>C1: %{{x:.3f}}<br>C2: %{{y:.3f}}<extra></extra>'
                        ),
                        row=row,
                        col=col
                    )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=X[:, 0],
                        y=X[:, 1],
                        mode='markers',
                        marker=dict(size=6, color='blue'),
                        showlegend=False,
                        hovertemplate=f'{proj.method}<br>C1: %{{x:.3f}}<br>C2: %{{y:.3f}}<extra></extra>'
                    ),
                    row=row,
                    col=col
                )
        
        fig.update_layout(
            title="Projection Comparison",
            height=400 * rows,
            width=1200,
            showlegend=True
        )
        
        return fig


def show_interactive(fig):
    """Helper to show plotly figure"""
    fig.show()


def save_interactive(fig, filename: str):
    """Save plotly figure as HTML"""
    fig.write_html(filename)
    print(f"Saved interactive plot to {filename}")


if __name__ == "__main__":
    if PLOTLY_AVAILABLE:
        print("Interactive Plotly module loaded successfully")
        print("Use: from interactive_plotly import InteractivePlotter")
    else:
        print("Plotly not available. Install with: pip install plotly")
