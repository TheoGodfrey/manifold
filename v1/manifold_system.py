"""
Manifold v1 - Data Entity Processing System
A flexible system for processing entities with many variables using manifold learning
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json


@dataclass
class Entity:
    """
    Core data entity representation
    """
    entity_id: str
    timestamp: datetime
    features: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_vector(self, feature_names: List[str]) -> np.ndarray:
        """Convert entity to feature vector"""
        return np.array([self.features.get(name, np.nan) for name in feature_names])
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'entity_id': self.entity_id,
            'timestamp': self.timestamp.isoformat(),
            'features': self.features,
            'metadata': self.metadata
        }


@dataclass
class EntityTrajectory:
    """
    Represents an entity's path through time
    """
    entity_id: str
    snapshots: List[Entity] = field(default_factory=list)
    
    def add_snapshot(self, entity: Entity):
        """Add a temporal snapshot"""
        self.snapshots.append(entity)
        self.snapshots.sort(key=lambda x: x.timestamp)
    
    def get_feature_series(self, feature_name: str) -> List[tuple]:
        """Get time series for a specific feature"""
        return [(e.timestamp, e.features.get(feature_name)) 
                for e in self.snapshots]


class DataLoader:
    """
    Flexible data loader for various formats
    """
    
    def __init__(self):
        self.loaders = {
            'csv': self._load_csv,
            'json': self._load_json,
            'parquet': self._load_parquet,
        }
    
    def load(self, path: Union[str, Path], format: str = 'csv', **kwargs) -> pd.DataFrame:
        """Load data from various sources"""
        path = Path(path)
        
        if format not in self.loaders:
            raise ValueError(f"Unsupported format: {format}")
        
        return self.loaders[format](path, **kwargs)
    
    def _load_csv(self, path: Path, **kwargs) -> pd.DataFrame:
        return pd.read_csv(path, **kwargs)
    
    def _load_json(self, path: Path, **kwargs) -> pd.DataFrame:
        return pd.read_json(path, **kwargs)
    
    def _load_parquet(self, path: Path, **kwargs) -> pd.DataFrame:
        return pd.read_parquet(path, **kwargs)


class FeatureProcessor:
    """
    Process and transform features
    """
    
    def __init__(self):
        self.transformers: Dict[str, Callable] = {}
        self.feature_stats: Dict[str, Dict] = {}
    
    def register_transformer(self, feature_name: str, transformer: Callable):
        """Register a custom transformer for a feature"""
        self.transformers[feature_name] = transformer
    
    def process_features(self, df: pd.DataFrame, 
                        numeric_features: Optional[List[str]] = None,
                        categorical_features: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Process features with basic transformations
        """
        df = df.copy()
        
        # Auto-detect if not specified
        if numeric_features is None:
            numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if categorical_features is None:
            categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Apply custom transformers
        for feat, transformer in self.transformers.items():
            if feat in df.columns:
                df[feat] = transformer(df[feat])
        
        # Store statistics
        for feat in numeric_features:
            if feat in df.columns:
                self.feature_stats[feat] = {
                    'mean': df[feat].mean(),
                    'std': df[feat].std(),
                    'min': df[feat].min(),
                    'max': df[feat].max(),
                    'type': 'numeric'
                }
        
        return df
    
    def normalize(self, df: pd.DataFrame, features: Optional[List[str]] = None) -> pd.DataFrame:
        """Normalize numeric features to [0, 1]"""
        df = df.copy()
        
        if features is None:
            features = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for feat in features:
            if feat in df.columns:
                min_val = df[feat].min()
                max_val = df[feat].max()
                if max_val > min_val:
                    df[feat] = (df[feat] - min_val) / (max_val - min_val)
        
        return df
    
    def standardize(self, df: pd.DataFrame, features: Optional[List[str]] = None) -> pd.DataFrame:
        """Standardize numeric features (z-score)"""
        df = df.copy()
        
        if features is None:
            features = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for feat in features:
            if feat in df.columns:
                mean_val = df[feat].mean()
                std_val = df[feat].std()
                if std_val > 0:
                    df[feat] = (df[feat] - mean_val) / std_val
        
        return df


class EntityManager:
    """
    Manage collections of entities and trajectories
    """
    
    def __init__(self):
        self.entities: Dict[str, List[Entity]] = {}
        self.trajectories: Dict[str, EntityTrajectory] = {}
    
    def add_entity(self, entity: Entity):
        """Add an entity snapshot"""
        entity_id = entity.entity_id
        
        if entity_id not in self.entities:
            self.entities[entity_id] = []
            self.trajectories[entity_id] = EntityTrajectory(entity_id)
        
        self.entities[entity_id].append(entity)
        self.trajectories[entity_id].add_snapshot(entity)
    
    def from_dataframe(self, df: pd.DataFrame, 
                      entity_id_col: str,
                      timestamp_col: Optional[str] = None,
                      feature_cols: Optional[List[str]] = None,
                      metadata_cols: Optional[List[str]] = None):
        """
        Convert a dataframe to entities
        """
        if feature_cols is None:
            # Use all columns except entity_id and timestamp
            exclude_cols = [entity_id_col]
            if timestamp_col:
                exclude_cols.append(timestamp_col)
            if metadata_cols:
                exclude_cols.extend(metadata_cols)
            feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        for _, row in df.iterrows():
            entity_id = str(row[entity_id_col])
            
            # Handle timestamp
            if timestamp_col and timestamp_col in row:
                ts = pd.to_datetime(row[timestamp_col])
            else:
                ts = datetime.now()
            
            # Extract features
            features = {col: row[col] for col in feature_cols if col in row}
            
            # Extract metadata
            metadata = {}
            if metadata_cols:
                metadata = {col: row[col] for col in metadata_cols if col in row}
            
            entity = Entity(
                entity_id=entity_id,
                timestamp=ts,
                features=features,
                metadata=metadata
            )
            
            self.add_entity(entity)
    
    def get_trajectory(self, entity_id: str) -> Optional[EntityTrajectory]:
        """Get trajectory for an entity"""
        return self.trajectories.get(entity_id)
    
    def get_all_trajectories(self) -> List[EntityTrajectory]:
        """Get all trajectories"""
        return list(self.trajectories.values())
    
    def to_feature_matrix(self, feature_names: Optional[List[str]] = None,
                         timestamp: Optional[datetime] = None) -> tuple:
        """
        Convert entities to feature matrix
        Returns: (feature_matrix, entity_ids)
        """
        if not self.entities:
            return np.array([]), []
        
        # Get latest snapshot for each entity
        latest_entities = []
        entity_ids = []
        
        for entity_id, entity_list in self.entities.items():
            if timestamp:
                # Get snapshot at or before timestamp
                valid_entities = [e for e in entity_list if e.timestamp <= timestamp]
                if valid_entities:
                    latest_entities.append(max(valid_entities, key=lambda x: x.timestamp))
                    entity_ids.append(entity_id)
            else:
                # Get most recent
                latest_entities.append(max(entity_list, key=lambda x: x.timestamp))
                entity_ids.append(entity_id)
        
        if not latest_entities:
            return np.array([]), []
        
        # Get all feature names if not specified
        if feature_names is None:
            feature_names = list(latest_entities[0].features.keys())
        
        # Build matrix
        matrix = np.array([e.to_vector(feature_names) for e in latest_entities])
        
        return matrix, entity_ids, feature_names


class ManifoldSystem:
    """
    Main manifold processing system
    """
    
    def __init__(self):
        self.loader = DataLoader()
        self.processor = FeatureProcessor()
        self.entity_manager = EntityManager()
        self.config = {}
    
    def load_data(self, path: Union[str, Path], format: str = 'csv', **kwargs) -> pd.DataFrame:
        """Load data"""
        return self.loader.load(path, format, **kwargs)
    
    def ingest_dataframe(self, df: pd.DataFrame,
                        entity_id_col: str,
                        timestamp_col: Optional[str] = None,
                        feature_cols: Optional[List[str]] = None,
                        metadata_cols: Optional[List[str]] = None,
                        preprocess: bool = True):
        """
        Ingest a dataframe into the system
        """
        if preprocess:
            df = self.processor.process_features(df)
        
        self.entity_manager.from_dataframe(
            df,
            entity_id_col=entity_id_col,
            timestamp_col=timestamp_col,
            feature_cols=feature_cols,
            metadata_cols=metadata_cols
        )
    
    def get_feature_matrix(self, normalize: bool = False, 
                          standardize: bool = False) -> tuple:
        """
        Get feature matrix for all entities
        Returns: (feature_matrix, entity_ids, feature_names)
        """
        matrix, entity_ids, feature_names = self.entity_manager.to_feature_matrix()
        
        if matrix.size == 0:
            return matrix, entity_ids, feature_names
        
        # Convert to dataframe for processing
        df = pd.DataFrame(matrix, columns=feature_names)
        
        if normalize:
            df = self.processor.normalize(df)
        elif standardize:
            df = self.processor.standardize(df)
        
        return df.values, entity_ids, feature_names
    
    def get_trajectories(self) -> Dict[str, EntityTrajectory]:
        """Get all entity trajectories"""
        return self.entity_manager.trajectories
    
    def summary(self) -> Dict[str, Any]:
        """Get system summary statistics"""
        return {
            'num_entities': len(self.entity_manager.entities),
            'num_trajectories': len(self.entity_manager.trajectories),
            'feature_stats': self.processor.feature_stats,
        }


if __name__ == "__main__":
    # Example usage
    print("Manifold System v1 initialized")
    system = ManifoldSystem()
    print(f"System ready: {system.summary()}")