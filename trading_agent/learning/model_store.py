"""
Model Storage & Retrieval

Utilities for saving, loading, and versioning ML models.
"""

import os
import json
import pickle
import logging
from typing import Optional
from pathlib import Path
from datetime import datetime

from learning.prediction_engine import PredictionEngine


logger = logging.getLogger(__name__)


class ModelStore:
    """
    Centralized model storage with versioning and metadata.
    """
    
    def __init__(self, base_path: str = "models"):
        """
        Initialize model store.
        
        Args:
            base_path: Base directory for storing models.
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def save_model(
        self,
        model: PredictionEngine,
        model_name: str,
        version: Optional[str] = None,
        tags: Optional[dict] = None,
    ) -> str:
        """
        Save a trained model with metadata.
        
        Args:
            model: Trained PredictionEngine model.
            model_name: Name of the model (e.g., "xgboost_5bar_20bar").
            version: Version string (e.g., "v1.0", "2024-01-15"). 
                     If None, auto-generates timestamp.
            tags: Optional dict of metadata tags.
            
        Returns:
            Path where model was saved.
        """
        # Generate version if not provided
        if version is None:
            version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create model directory
        model_dir = self.base_path / model_name / version
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        model_path = str(model_dir / "model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        self.logger.info(f"Saved model to {model_path}")
        
        # Save metadata
        metadata = {
            "model_name": model_name,
            "version": version,
            "saved_at": datetime.now().isoformat(),
            "tags": tags or {},
        }
        
        # Add model metadata if available
        if hasattr(model, "metadata") and model.metadata:
            metadata["training_period"] = {
                "start": str(model.metadata.get("training_start")),
                "end": str(model.metadata.get("training_end")),
            }
            metadata["features"] = model.metadata.get("features", [])
            metadata["horizons"] = model.metadata.get("horizons", [])
            metadata["regimes"] = model.metadata.get("regimes", [])
            metadata["metrics"] = model.metadata.get("metrics", {})
        
        metadata_path = str(model_dir / "metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, default=str)
        self.logger.info(f"Saved metadata to {metadata_path}")
        
        return str(model_dir)
    
    def load_model(
        self,
        model_name: str,
        version: Optional[str] = None,
    ) -> Optional[PredictionEngine]:
        """
        Load a trained model.
        
        Args:
            model_name: Name of the model.
            version: Version to load. If None, loads latest.
            
        Returns:
            PredictionEngine model or None if not found.
        """
        model_dir = self.base_path / model_name
        
        if not model_dir.exists():
            self.logger.error(f"Model {model_name} not found")
            return None
        
        # Find version to load
        if version is None:
            # Get latest version (most recent timestamp)
            versions = sorted([d.name for d in model_dir.iterdir() if d.is_dir()])
            if not versions:
                self.logger.error(f"No versions found for model {model_name}")
                return None
            version = versions[-1]
            self.logger.info(f"Loading latest version: {version}")
        
        model_path = model_dir / version / "model.pkl"
        
        if not model_path.exists():
            self.logger.error(f"Model file not found: {model_path}")
            return None
        
        try:
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            self.logger.info(f"Loaded model from {model_path}")
            return model
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return None
    
    def list_models(self) -> dict:
        """
        List all available models and their versions.
        
        Returns:
            Dict mapping model_name -> list of versions.
        """
        models = {}
        
        if not self.base_path.exists():
            return models
        
        for model_dir in self.base_path.iterdir():
            if not model_dir.is_dir():
                continue
            
            model_name = model_dir.name
            versions = sorted([d.name for d in model_dir.iterdir() if d.is_dir()])
            
            if versions:
                models[model_name] = versions
        
        return models
    
    def delete_model(
        self,
        model_name: str,
        version: Optional[str] = None,
    ) -> bool:
        """
        Delete a model or model version.
        
        Args:
            model_name: Name of the model.
            version: Version to delete. If None, deletes all versions.
            
        Returns:
            True if successful.
        """
        model_dir = self.base_path / model_name
        
        if not model_dir.exists():
            self.logger.error(f"Model {model_name} not found")
            return False
        
        if version is None:
            # Delete entire model directory
            import shutil
            try:
                shutil.rmtree(model_dir)
                self.logger.info(f"Deleted model {model_name}")
                return True
            except Exception as e:
                self.logger.error(f"Failed to delete model: {e}")
                return False
        else:
            # Delete specific version
            version_dir = model_dir / version
            
            if not version_dir.exists():
                self.logger.error(f"Version {version} not found for model {model_name}")
                return False
            
            import shutil
            try:
                shutil.rmtree(version_dir)
                self.logger.info(f"Deleted version {version} of model {model_name}")
                return True
            except Exception as e:
                self.logger.error(f"Failed to delete version: {e}")
                return False
    
    def get_metadata(
        self,
        model_name: str,
        version: Optional[str] = None,
    ) -> Optional[dict]:
        """
        Get metadata for a model without loading the full model.
        
        Args:
            model_name: Name of the model.
            version: Version to load. If None, loads latest.
            
        Returns:
            Metadata dict or None if not found.
        """
        model_dir = self.base_path / model_name
        
        if not model_dir.exists():
            self.logger.error(f"Model {model_name} not found")
            return None
        
        # Find version
        if version is None:
            versions = sorted([d.name for d in model_dir.iterdir() if d.is_dir()])
            if not versions:
                return None
            version = versions[-1]
        
        metadata_path = model_dir / version / "metadata.json"
        
        if not metadata_path.exists():
            return None
        
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            return metadata
        except Exception as e:
            self.logger.error(f"Failed to load metadata: {e}")
            return None


def save_model(
    model: PredictionEngine,
    model_name: str,
    base_path: str = "models",
    version: Optional[str] = None,
    tags: Optional[dict] = None,
) -> str:
    """
    Convenience function to save a model.
    
    Args:
        model: Trained PredictionEngine model.
        model_name: Name of the model.
        base_path: Base directory for models.
        version: Version string (auto-generated if None).
        tags: Optional metadata tags.
        
    Returns:
        Path where model was saved.
    """
    store = ModelStore(base_path=base_path)
    return store.save_model(model, model_name, version=version, tags=tags)


def load_model(
    model_name: str,
    base_path: str = "models",
    version: Optional[str] = None,
) -> Optional[PredictionEngine]:
    """
    Convenience function to load a model.
    
    Args:
        model_name: Name of the model.
        base_path: Base directory for models.
        version: Version to load (latest if None).
        
    Returns:
        PredictionEngine model or None.
    """
    store = ModelStore(base_path=base_path)
    return store.load_model(model_name, version=version)


if __name__ == "__main__":
    # Example usage
    store = ModelStore(base_path="models")
    
    # List all models
    print("Available models:")
    for model_name, versions in store.list_models().items():
        print(f"  {model_name}:")
        for version in versions:
            print(f"    - {version}")
    
    # Get latest model metadata
    metadata = store.get_metadata("xgboost_model")
    if metadata:
        print(f"\nLatest model metadata: {metadata}")
