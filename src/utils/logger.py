"""
Logging utilities for graph licensing optimization algorithms.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


class AlgorithmLogger:
    """
    Centralized logging for debugging graph licensing optimization algorithms.
    """

    def __init__(self, algorithm_name: str = "Algorithm", log_level: int = logging.INFO, log_to_file: bool = False, log_dir: str = "logs"):
        """
        Initialize logger for an algorithm.

        Args:
            algorithm_name: Name of the algorithm for log identification
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            log_to_file: Whether to save logs to file
            log_dir: Directory for log files
        """
        self.algorithm_name = algorithm_name
        self.logger = logging.getLogger(f"graph_licensing.{algorithm_name}")

        # Prevent duplicate handlers
        if self.logger.handlers:
            self.logger.handlers.clear()

        self.logger.setLevel(log_level)

        # Create formatter
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%H:%M:%S")

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        # File handler (optional)
        if log_to_file:
            self._setup_file_handler(log_dir, formatter)

    def _setup_file_handler(self, log_dir: str, formatter: logging.Formatter):
        """Setup file logging handler."""
        log_path = Path(log_dir)
        log_path.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_path / f"{self.algorithm_name}_{timestamp}.log"

        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        self.info(f"Logging to file: {log_file}")

    def debug(self, message: str, **kwargs):
        """Log debug message with optional context."""
        if kwargs:
            context = ", ".join(f"{k}={v}" for k, v in kwargs.items())
            message = f"{message} | {context}"
        self.logger.debug(message)

    def info(self, message: str, **kwargs):
        """Log info message with optional context."""
        if kwargs:
            context = ", ".join(f"{k}={v}" for k, v in kwargs.items())
            message = f"{message} | {context}"
        self.logger.info(message)

    def warning(self, message: str, **kwargs):
        """Log warning message with optional context."""
        if kwargs:
            context = ", ".join(f"{k}={v}" for k, v in kwargs.items())
            message = f"{message} | {context}"
        self.logger.warning(message)

    def error(self, message: str, **kwargs):
        """Log error message with optional context."""
        if kwargs:
            context = ", ".join(f"{k}={v}" for k, v in kwargs.items())
            message = f"{message} | {context}"
        self.logger.error(message)

    def log_algorithm_start(self, graph_nodes: int, graph_edges: int, **params):
        """Log algorithm start with graph and parameter info."""
        self.info(f"Starting {self.algorithm_name}")
        self.info(f"Graph: {graph_nodes} nodes, {graph_edges} edges")
        if params:
            param_str = ", ".join(f"{k}={v}" for k, v in params.items())
            self.info(f"Parameters: {param_str}")

    def log_iteration(self, iteration: int, best_cost: float, current_cost: Optional[float] = None):
        """Log iteration progress."""
        if current_cost is not None:
            self.debug(f"Iteration {iteration}: best={best_cost:.2f}, current={current_cost:.2f}")
        else:
            self.debug(f"Iteration {iteration}: best={best_cost:.2f}")

    def log_solution_found(self, cost: float, groups: int, is_valid: bool):
        """Log final solution details."""
        status = "VALID" if is_valid else "INVALID"
        self.info(f"Solution found: cost={cost:.2f}, groups={groups}, status={status}")

    def log_improvement(self, old_cost: float, new_cost: float, iteration: Optional[int] = None):
        """Log cost improvement."""
        improvement = old_cost - new_cost
        improvement_pct = (improvement / old_cost) * 100 if old_cost > 0 else 0

        if iteration is not None:
            self.info(f"Improvement at iteration {iteration}: {old_cost:.2f} → {new_cost:.2f} (Δ=-{improvement:.2f}, {improvement_pct:.1f}%)")
        else:
            self.info(f"Improvement: {old_cost:.2f} → {new_cost:.2f} (Δ=-{improvement:.2f}, {improvement_pct:.1f}%)")

    def log_validation_failure(self, reason: str, **details):
        """Log validation failure with details."""
        self.warning(f"Validation failed: {reason}")
        if details:
            for key, value in details.items():
                self.warning(f"  {key}: {value}")

    def log_exception(self, exception: Exception, context: str = ""):
        """Log exception with context."""
        if context:
            self.error(f"Exception in {context}: {type(exception).__name__}: {str(exception)}")
        else:
            self.error(f"Exception: {type(exception).__name__}: {str(exception)}")


class DebugContext:
    """
    Context manager for debug logging sections.
    """

    def __init__(self, logger: AlgorithmLogger, operation: str):
        self.logger = logger
        self.operation = operation
        self.start_time = None

    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.debug(f"Starting {self.operation}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = datetime.now() - self.start_time
        if exc_type is None:
            self.logger.debug(f"Completed {self.operation} in {duration.total_seconds():.3f}s")
        else:
            self.logger.error(f"Failed {self.operation} after {duration.total_seconds():.3f}s: {exc_val}")


def get_logger(algorithm_name: str, debug: bool = False, log_to_file: bool = False) -> AlgorithmLogger:
    """
    Factory function to create configured logger.

    Args:
        algorithm_name: Name of the algorithm
        debug: Enable debug logging
        log_to_file: Save logs to file

    Returns:
        Configured AlgorithmLogger instance
    """
    log_level = logging.DEBUG if debug else logging.INFO
    return AlgorithmLogger(algorithm_name=algorithm_name, log_level=log_level, log_to_file=log_to_file)
