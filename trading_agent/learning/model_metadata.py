"""
Model Metadata & Versioning

Tracks model lineage, hyperparameters, grading results, and deployment status.
"""

import json
import logging
from typing import Dict, Optional, Any
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ModelMetadata:
    """Comprehensive model metadata and versioning."""
    
    # Identification
    model_name: str
    model_version: str  # e.g., "1.0.0"
    model_type: str  # e.g., "xgboost", "lightgbm"
    
    # Training info
    training_date: str
    training_features: list  # Feature names
    training_horizons: list  # e.g., [5, 20, 60]
    training_data_period: Dict[str, str]  # {"start": "2020-01-01", "end": "2023-12-31"}
    training_n_samples: int
    
    # Hyperparameters
    hyperparameters: Dict[str, Any]
    
    # Grading results
    prediction_grade: Optional[str] = None
    prediction_metrics: Optional[Dict[str, float]] = None
    grading_date: Optional[str] = None
    
    # Deployment status
    deployment_status: str = "NOT_GRADED"  # NOT_GRADED, BLOCKED, INCUBATION, PAPER, LIVE
    deployment_date: Optional[str] = None
    deployment_notes: Optional[str] = None
    
    # Performance tracking (live)
    live_performance: Optional[Dict[str, float]] = None
    live_last_updated: Optional[str] = None
    
    # Lineage & reproducibility
    parent_model: Optional[str] = None  # Previous version
    experiment_id: Optional[str] = None
    commit_hash: Optional[str] = None  # Git commit
    
    # Comments
    comments: Optional[str] = None


class ModelRegistry:
    """
    Central registry for model metadata, versioning, and deployment tracking.
    """
    
    def __init__(self, registry_path: str = "models/registry.json"):
        """
        Initialize model registry.
        
        Args:
            registry_path: Path to registry JSON file.
        """
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.models: Dict[str, list] = self._load_registry()
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    def _load_registry(self) -> Dict[str, list]:
        """Load registry from disk."""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Failed to load registry: {e}")
                return {}
        return {}
    
    def _save_registry(self) -> None:
        """Save registry to disk."""
        try:
            with open(self.registry_path, 'w') as f:
                json.dump(self.models, f, indent=2, default=str)
            self.logger.info(f"Saved registry to {self.registry_path}")
        except Exception as e:
            self.logger.error(f"Failed to save registry: {e}")
    
    def register_model(
        self,
        metadata: ModelMetadata,
        save_immediately: bool = True,
    ) -> None:
        """
        Register a new model version.
        
        Args:
            metadata: ModelMetadata object.
            save_immediately: Whether to save to disk immediately.
        """
        model_name = metadata.model_name
        
        if model_name not in self.models:
            self.models[model_name] = []
        
        # Convert to dict for storage
        metadata_dict = asdict(metadata)
        
        # Check for duplicate version
        for existing in self.models[model_name]:
            if existing['model_version'] == metadata.model_version:
                self.logger.warning(
                    f"Model {model_name} v{metadata.model_version} already exists. Updating."
                )
                existing.update(metadata_dict)
                if save_immediately:
                    self._save_registry()
                return
        
        # Add new version
        self.models[model_name].append(metadata_dict)
        self.logger.info(f"Registered {model_name} v{metadata.model_version}")
        
        if save_immediately:
            self._save_registry()
    
    def update_grading(
        self,
        model_name: str,
        model_version: str,
        grade: str,
        metrics: Dict[str, float],
        comments: Optional[str] = None,
    ) -> None:
        """
        Update model grading results.
        
        Args:
            model_name: Model name.
            model_version: Model version.
            grade: Letter grade (A/B/C/D).
            metrics: Dict of grading metrics.
            comments: Optional comments.
        """
        model = self._get_model(model_name, model_version)
        
        if model is None:
            self.logger.error(f"Model {model_name} v{model_version} not found")
            return
        
        model['prediction_grade'] = grade
        model['prediction_metrics'] = metrics
        model['grading_date'] = pd.Timestamp.now().isoformat()
        
        if comments:
            model['comments'] = comments
        
        # Update deployment status based on grade
        if grade == "A":
            model['deployment_status'] = "PAPER"
        elif grade == "B":
            model['deployment_status'] = "INCUBATION"
        elif grade == "C":
            model['deployment_status'] = "INCUBATION"
        else:
            model['deployment_status'] = "BLOCKED"
        
        self._save_registry()
        self.logger.info(f"Updated grading for {model_name} v{model_version}: {grade}")
    
    def update_deployment(
        self,
        model_name: str,
        model_version: str,
        status: str,  # LIVE, PAPER, INCUBATION, BLOCKED
        notes: Optional[str] = None,
    ) -> None:
        """
        Update deployment status.
        
        Args:
            model_name: Model name.
            model_version: Model version.
            status: Deployment status.
            notes: Optional deployment notes.
        """
        model = self._get_model(model_name, model_version)
        
        if model is None:
            self.logger.error(f"Model {model_name} v{model_version} not found")
            return
        
        model['deployment_status'] = status
        model['deployment_date'] = pd.Timestamp.now().isoformat()
        
        if notes:
            model['deployment_notes'] = notes
        
        self._save_registry()
        self.logger.info(f"Updated deployment for {model_name} v{model_version}: {status}")
    
    def update_live_performance(
        self,
        model_name: str,
        model_version: str,
        performance_metrics: Dict[str, float],
    ) -> None:
        """
        Update live trading performance metrics.
        
        Args:
            model_name: Model name.
            model_version: Model version.
            performance_metrics: Dict of live performance metrics.
        """
        model = self._get_model(model_name, model_version)
        
        if model is None:
            self.logger.error(f"Model {model_name} v{model_version} not found")
            return
        
        model['live_performance'] = performance_metrics
        model['live_last_updated'] = pd.Timestamp.now().isoformat()
        
        self._save_registry()
        self.logger.info(f"Updated live performance for {model_name} v{model_version}")
    
    def get_model(
        self,
        model_name: str,
        model_version: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Get model metadata.
        
        Args:
            model_name: Model name.
            model_version: Specific version (latest if None).
            
        Returns:
            Model metadata dict or None if not found.
        """
        return self._get_model(model_name, model_version)
    
    def _get_model(
        self,
        model_name: str,
        model_version: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        """Internal helper to get model metadata."""
        if model_name not in self.models:
            return None
        
        versions = self.models[model_name]
        
        if not versions:
            return None
        
        if model_version is None:
            # Return latest
            return versions[-1]
        
        # Find specific version
        for model in versions:
            if model['model_version'] == model_version:
                return model
        
        return None
    
    def list_models(self) -> Dict[str, list]:
        """
        List all models and versions.
        
        Returns:
            Dict mapping model_name -> list of versions.
        """
        result = {}
        
        for model_name, versions in self.models.items():
            result[model_name] = [v['model_version'] for v in versions]
        
        return result
    
    def get_deployment_candidates(self, min_grade: str = "B") -> list:
        """
        Get models eligible for deployment.
        
        Args:
            min_grade: Minimum grade for eligibility (A, B, C, D).
            
        Returns:
            List of (model_name, version, grade) tuples.
        """
        grade_order = {"A": 0, "B": 1, "C": 2, "D": 3}
        min_rank = grade_order.get(min_grade, 3)
        
        candidates = []
        
        for model_name, versions in self.models.items():
            for version in versions:
                if version.get('prediction_grade'):
                    grade = version['prediction_grade']
                    if grade_order.get(grade, 3) <= min_rank:
                        candidates.append((
                            model_name,
                            version['model_version'],
                            grade,
                            version.get('deployment_status', 'UNKNOWN'),
                        ))
        
        return candidates
    
    def get_live_models(self) -> list:
        """
        Get all models currently in live trading.
        
        Returns:
            List of (model_name, version, metrics) tuples.
        """
        live_models = []
        
        for model_name, versions in self.models.items():
            for version in versions:
                if version.get('deployment_status') == 'LIVE':
                    live_models.append((
                        model_name,
                        version['model_version'],
                        version.get('live_performance', {}),
                    ))
        
        return live_models
    
    def print_registry_summary(self) -> None:
        """Print summary of registry."""
        print("\n" + "=" * 80)
        print("MODEL REGISTRY SUMMARY")
        print("=" * 80)
        
        for model_name, versions in self.models.items():
            print(f"\n{model_name}:")
            
            for version in versions:
                grade = version.get('prediction_grade', 'UNGRADED')
                status = version.get('deployment_status', 'UNKNOWN')
                grading_date = version.get('grading_date', 'N/A')
                
                print(f"  v{version['model_version']:10s} | Grade: {grade:1s} | "
                      f"Status: {status:12s} | Graded: {grading_date}")
        
        print("\n" + "=" * 80)


if __name__ == "__main__":
    # Example usage
    registry = ModelRegistry()
    
    # Create and register a model
    metadata = ModelMetadata(
        model_name="xgboost_v1",
        model_version="1.0.0",
        model_type="xgboost",
        training_date=pd.Timestamp.now().isoformat(),
        training_features=["sma_5", "sma_20", "volatility"],
        training_horizons=[5, 20],
        training_data_period={
            "start": "2020-01-01",
            "end": "2023-12-31",
        },
        training_n_samples=1000,
        hyperparameters={
            "max_depth": 6,
            "learning_rate": 0.1,
        },
    )
    
    registry.register_model(metadata)
    registry.update_grading("xgboost_v1", "1.0.0", "A", {"roc_auc": 0.62})
    registry.print_registry_summary()
