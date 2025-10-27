"""
Manifold Learning Module
Implements various manifold learning algorithms for geometric projections
"""

import numpy as np
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

# Conditional imports for manifold learning libraries
try:
    from sklearn.manifold import (
        TSNE, Isomap, LocallyLinearEmbedding, 
        MDS, SpectralEmbedding
    )
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


@dataclass
class ManifoldProjection:
    """
    Stores a manifold projection result
    """
    method: str
    embedding: np.ndarray  # Low-dimensional representation
    entity_ids: List[str]
    feature_names: List[str]
    original_dim: int
    target_dim: int
    metadata: Dict[str, Any]
    
    def get_entity_embedding(self, entity_id: str) -> Optional[np.ndarray]:
        """Get embedding for a specific entity"""
        try:
            idx = self.entity_ids.index(entity_id)
            return self.embedding[idx]
        except ValueError:
            return None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'method': self.method,
            'embedding': self.embedding.tolist(),
            'entity_ids': self.entity_ids,
            'feature_names': self.feature_names,
            'original_dim': self.original_dim,
            'target_dim': self.target_dim,
            'metadata': self.metadata
        }


class ManifoldLearner:
    """
    Unified interface for various manifold learning algorithms
    """
    
    def __init__(self):
        self.projections: Dict[str, ManifoldProjection] = {}
        self.fitted_models: Dict[str, Any] = {}
    
    def check_dependencies(self):
        """Check which libraries are available"""
        deps = {
            'sklearn': SKLEARN_AVAILABLE,
            'umap': UMAP_AVAILABLE,
        }
        return deps
    
    def project_pca(self, X: np.ndarray, n_components: int = 2,
                   entity_ids: Optional[List[str]] = None,
                   feature_names: Optional[List[str]] = None,
                   name: str = "pca") -> ManifoldProjection:
        """
        Principal Component Analysis (linear)
        Fast, interpretable, good baseline
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for PCA")
        
        pca = PCA(n_components=n_components)
        embedding = pca.fit_transform(X)
        
        self.fitted_models[name] = pca
        
        projection = ManifoldProjection(
            method="PCA",
            embedding=embedding,
            entity_ids=entity_ids or [f"entity_{i}" for i in range(len(X))],
            feature_names=feature_names or [f"feature_{i}" for i in range(X.shape[1])],
            original_dim=X.shape[1],
            target_dim=n_components,
            metadata={
                'explained_variance_ratio': pca.explained_variance_ratio_.tolist(),
                'total_explained_variance': float(pca.explained_variance_ratio_.sum())
            }
        )
        
        self.projections[name] = projection
        return projection
    
    def project_isomap(self, X: np.ndarray, n_components: int = 2,
                      n_neighbors: int = 5,
                      entity_ids: Optional[List[str]] = None,
                      feature_names: Optional[List[str]] = None,
                      name: str = "isomap") -> ManifoldProjection:
        """
        Isomap - preserves geodesic distances
        Good for data with smooth, low-dimensional manifold structure
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for Isomap")
        
        isomap = Isomap(n_components=n_components, n_neighbors=n_neighbors)
        embedding = isomap.fit_transform(X)
        
        self.fitted_models[name] = isomap
        
        projection = ManifoldProjection(
            method="Isomap",
            embedding=embedding,
            entity_ids=entity_ids or [f"entity_{i}" for i in range(len(X))],
            feature_names=feature_names or [f"feature_{i}" for i in range(X.shape[1])],
            original_dim=X.shape[1],
            target_dim=n_components,
            metadata={
                'n_neighbors': n_neighbors,
                'reconstruction_error': float(isomap.reconstruction_error())
            }
        )
        
        self.projections[name] = projection
        return projection
    
    def project_lle(self, X: np.ndarray, n_components: int = 2,
                   n_neighbors: int = 5,
                   entity_ids: Optional[List[str]] = None,
                   feature_names: Optional[List[str]] = None,
                   name: str = "lle") -> ManifoldProjection:
        """
        Locally Linear Embedding
        Preserves local neighborhood structure
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for LLE")
        
        lle = LocallyLinearEmbedding(
            n_components=n_components,
            n_neighbors=n_neighbors
        )
        embedding = lle.fit_transform(X)
        
        self.fitted_models[name] = lle
        
        projection = ManifoldProjection(
            method="LLE",
            embedding=embedding,
            entity_ids=entity_ids or [f"entity_{i}" for i in range(len(X))],
            feature_names=feature_names or [f"feature_{i}" for i in range(X.shape[1])],
            original_dim=X.shape[1],
            target_dim=n_components,
            metadata={
                'n_neighbors': n_neighbors,
                'reconstruction_error': float(lle.reconstruction_error_)
            }
        )
        
        self.projections[name] = projection
        return projection
    
    def project_tsne(self, X: np.ndarray, n_components: int = 2,
                    perplexity: float = 30.0,
                    learning_rate: float = 200.0,
                    max_iter: int = 1000,
                    entity_ids: Optional[List[str]] = None,
                    feature_names: Optional[List[str]] = None,
                    name: str = "tsne") -> ManifoldProjection:
        """
        t-SNE - preserves local structure, great for visualization
        Non-deterministic, can be slow for large datasets
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for t-SNE")
        
        tsne = TSNE(
            n_components=n_components,
            perplexity=perplexity,
            learning_rate=learning_rate,
            max_iter=max_iter,
            random_state=42
        )
        embedding = tsne.fit_transform(X)
        
        self.fitted_models[name] = tsne
        
        projection = ManifoldProjection(
            method="t-SNE",
            embedding=embedding,
            entity_ids=entity_ids or [f"entity_{i}" for i in range(len(X))],
            feature_names=feature_names or [f"feature_{i}" for i in range(X.shape[1])],
            original_dim=X.shape[1],
            target_dim=n_components,
            metadata={
                'perplexity': perplexity,
                'learning_rate': learning_rate,
                'max_iter': max_iter,
                'kl_divergence': float(tsne.kl_divergence_)
            }
        )
        
        self.projections[name] = projection
        return projection
    
    def project_umap(self, X: np.ndarray, n_components: int = 2,
                    n_neighbors: int = 15,
                    min_dist: float = 0.1,
                    metric: str = 'euclidean',
                    entity_ids: Optional[List[str]] = None,
                    feature_names: Optional[List[str]] = None,
                    name: str = "umap") -> ManifoldProjection:
        """
        UMAP - preserves both local and global structure
        Fast, scalable, often better than t-SNE
        """
        if not UMAP_AVAILABLE:
            raise ImportError("umap-learn required. Install with: pip install umap-learn")
        
        umap_model = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            metric=metric,
            random_state=42
        )
        embedding = umap_model.fit_transform(X)
        
        self.fitted_models[name] = umap_model
        
        projection = ManifoldProjection(
            method="UMAP",
            embedding=embedding,
            entity_ids=entity_ids or [f"entity_{i}" for i in range(len(X))],
            feature_names=feature_names or [f"feature_{i}" for i in range(X.shape[1])],
            original_dim=X.shape[1],
            target_dim=n_components,
            metadata={
                'n_neighbors': n_neighbors,
                'min_dist': min_dist,
                'metric': metric
            }
        )
        
        self.projections[name] = projection
        return projection
    
    def project_mds(self, X: np.ndarray, n_components: int = 2,
                   metric: bool = True,
                   entity_ids: Optional[List[str]] = None,
                   feature_names: Optional[List[str]] = None,
                   name: str = "mds") -> ManifoldProjection:
        """
        Multidimensional Scaling - preserves pairwise distances
        """
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn required for MDS")
        
        mds = MDS(n_components=n_components, metric=metric, random_state=42)
        embedding = mds.fit_transform(X)
        
        self.fitted_models[name] = mds
        
        projection = ManifoldProjection(
            method="MDS",
            embedding=embedding,
            entity_ids=entity_ids or [f"entity_{i}" for i in range(len(X))],
            feature_names=feature_names or [f"feature_{i}" for i in range(X.shape[1])],
            original_dim=X.shape[1],
            target_dim=n_components,
            metadata={
                'metric': metric,
                'stress': float(mds.stress_)
            }
        )
        
        self.projections[name] = projection
        return projection
    
    def compare_methods(self, X: np.ndarray,
                       methods: List[str] = None,
                       n_components: int = 2,
                       entity_ids: Optional[List[str]] = None,
                       feature_names: Optional[List[str]] = None) -> Dict[str, ManifoldProjection]:
        """
        Compare multiple manifold learning methods
        """
        if methods is None:
            methods = ['pca', 'isomap', 'lle', 'tsne']
            if UMAP_AVAILABLE:
                methods.append('umap')
        
        results = {}
        
        for method in methods:
            try:
                if method == 'pca':
                    proj = self.project_pca(X, n_components, entity_ids, feature_names, f"compare_{method}")
                elif method == 'isomap':
                    proj = self.project_isomap(X, n_components, entity_ids=entity_ids, feature_names=feature_names, name=f"compare_{method}")
                elif method == 'lle':
                    proj = self.project_lle(X, n_components, entity_ids=entity_ids, feature_names=feature_names, name=f"compare_{method}")
                elif method == 'tsne':
                    proj = self.project_tsne(X, n_components, entity_ids=entity_ids, feature_names=feature_names, name=f"compare_{method}")
                elif method == 'umap':
                    proj = self.project_umap(X, n_components, entity_ids=entity_ids, feature_names=feature_names, name=f"compare_{method}")
                elif method == 'mds':
                    proj = self.project_mds(X, n_components, entity_ids=entity_ids, feature_names=feature_names, name=f"compare_{method}")
                else:
                    print(f"Unknown method: {method}")
                    continue
                
                results[method] = proj
                print(f"✓ {method.upper()} completed")
                
            except Exception as e:
                print(f"✗ {method.upper()} failed: {str(e)}")
        
        return results
    
    def get_projection(self, name: str) -> Optional[ManifoldProjection]:
        """Retrieve a saved projection"""
        return self.projections.get(name)
    
    def list_projections(self) -> List[str]:
        """List all saved projections"""
        return list(self.projections.keys())


class TrajectoryAnalyzer:
    """
    Analyze entity trajectories in manifold space
    """
    
    def __init__(self, projection: ManifoldProjection):
        self.projection = projection
    
    def compute_velocity(self, trajectory_points: np.ndarray) -> np.ndarray:
        """
        Compute velocity vectors between consecutive points
        trajectory_points: array of shape (n_timepoints, n_dims)
        """
        if len(trajectory_points) < 2:
            return np.array([])
        
        velocities = np.diff(trajectory_points, axis=0)
        return velocities
    
    def compute_speed(self, trajectory_points: np.ndarray) -> np.ndarray:
        """Compute speed (magnitude of velocity)"""
        velocities = self.compute_velocity(trajectory_points)
        if len(velocities) == 0:
            return np.array([])
        
        speeds = np.linalg.norm(velocities, axis=1)
        return speeds
    
    def compute_distance_traveled(self, trajectory_points: np.ndarray) -> float:
        """Total distance traveled along trajectory"""
        velocities = self.compute_velocity(trajectory_points)
        if len(velocities) == 0:
            return 0.0
        
        distances = np.linalg.norm(velocities, axis=1)
        return float(distances.sum())
    
    def compute_acceleration(self, trajectory_points: np.ndarray) -> np.ndarray:
        """Compute acceleration (change in velocity)"""
        velocities = self.compute_velocity(trajectory_points)
        if len(velocities) < 2:
            return np.array([])
        
        accelerations = np.diff(velocities, axis=0)
        return accelerations
    
    def trajectory_summary(self, trajectory_points: np.ndarray) -> Dict[str, Any]:
        """Get comprehensive trajectory statistics"""
        return {
            'n_points': len(trajectory_points),
            'total_distance': self.compute_distance_traveled(trajectory_points),
            'mean_speed': float(self.compute_speed(trajectory_points).mean()) if len(trajectory_points) > 1 else 0.0,
            'max_speed': float(self.compute_speed(trajectory_points).max()) if len(trajectory_points) > 1 else 0.0,
            'start_position': trajectory_points[0].tolist() if len(trajectory_points) > 0 else [],
            'end_position': trajectory_points[-1].tolist() if len(trajectory_points) > 0 else [],
        }


if __name__ == "__main__":
    print("Manifold Learning Module")
    learner = ManifoldLearner()
    print(f"Available dependencies: {learner.check_dependencies()}")