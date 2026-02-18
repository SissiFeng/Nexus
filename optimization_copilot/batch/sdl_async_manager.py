"""Asynchronous experiment submission manager for Self-Driving Labs (SDL).

Manages concurrent experiment execution with support for:
- Out-of-order result arrival
- Experiment queuing and prioritization
- Resource-aware scheduling
- Integration with physical lab equipment
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable

from optimization_copilot.batch.async_executor import AsyncBatchExecutor, TrialState


class ExperimentPriority(Enum):
    """Priority levels for experiment scheduling."""
    CRITICAL = auto()      # Must run immediately
    HIGH = auto()          # Important experiments
    NORMAL = auto()        # Standard priority
    LOW = auto()           # Can be deferred
    BACKGROUND = auto()    # Fill-in experiments when resources idle


class ExperimentStatus(Enum):
    """Extended status for SDL experiments."""
    PENDING = "pending"
    QUEUED = "queued"              # Waiting for resources
    PREPARING = "preparing"        # Sample prep, robot moving
    RUNNING = "running"
    ANALYZING = "analyzing"        # Post-processing, QC
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"              # Temporarily suspended


@dataclass
class SDLExperiment:
    """An experiment in the SDL queue.
    
    Attributes:
        experiment_id: Unique identifier
        parameters: Parameter configuration
        priority: Scheduling priority
        estimated_duration: Estimated runtime in seconds
        resource_requirements: Required equipment/resources
        dependencies: Other experiments that must complete first
        status: Current execution status
        created_at: Timestamp when experiment was created
        started_at: Timestamp when execution began
        completed_at: Timestamp when execution finished
        result: Experimental result
        metadata: Additional experiment metadata
    """
    experiment_id: str
    parameters: dict[str, float]
    priority: ExperimentPriority = ExperimentPriority.NORMAL
    estimated_duration: float = 3600.0  # Default 1 hour
    resource_requirements: list[str] = field(default_factory=list)
    dependencies: list[str] = field(default_factory=list)
    status: ExperimentStatus = ExperimentStatus.PENDING
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    result: dict[str, Any] | None = None
    error_message: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @property
    def wait_time(self) -> float:
        """Time spent waiting before execution started."""
        if self.started_at is None:
            return time.time() - self.created_at
        return self.started_at - self.created_at
    
    @property
    def execution_time(self) -> float:
        """Time spent executing."""
        if self.started_at is None:
            return 0.0
        end = self.completed_at or time.time()
        return end - self.started_at
    
    @property
    def total_time(self) -> float:
        """Total time from creation to completion."""
        end = self.completed_at or time.time()
        return end - self.created_at


@dataclass
class ResourceState:
    """State of a physical resource (robot, instrument, etc.).
    
    Attributes:
        resource_id: Unique identifier
        resource_type: Type of resource
        is_available: Whether resource is currently available
        current_experiment: ID of experiment currently using resource
        scheduled_until: When resource will be free
        capabilities: What this resource can do
    """
    resource_id: str
    resource_type: str
    is_available: bool = True
    current_experiment: str | None = None
    scheduled_until: float = 0.0
    capabilities: list[str] = field(default_factory=list)


class SDLAsyncManager:
    """Manages asynchronous experiment execution for Self-Driving Labs.
    
    Integrates with physical lab equipment to:
    - Queue experiments based on priority and resource availability
    - Handle out-of-order result arrival
    - Manage experiment dependencies
    - Optimize resource utilization
    - Provide real-time status updates
    """

    def __init__(
        self,
        max_concurrent: int = 4,
        default_priority: ExperimentPriority = ExperimentPriority.NORMAL,
    ) -> None:
        self.max_concurrent = max_concurrent
        self.default_priority = default_priority
        
        # Experiment storage
        self._experiments: dict[str, SDLExperiment] = {}
        self._queue: list[str] = []  # Ordered list of experiment IDs
        
        # Resource management
        self._resources: dict[str, ResourceState] = {}
        
        # Callbacks
        self._on_complete: list[Callable[[SDLExperiment], None]] = []
        self._on_fail: list[Callable[[SDLExperiment], None]] = []
        
        # Statistics
        self._stats = {
            "submitted": 0,
            "completed": 0,
            "failed": 0,
            "cancelled": 0,
            "total_wait_time": 0.0,
            "total_execution_time": 0.0,
        }

    # -- Resource Management ------------------------------------------------

    def register_resource(
        self,
        resource_id: str,
        resource_type: str,
        capabilities: list[str] | None = None,
    ) -> None:
        """Register a physical resource (robot, instrument, etc.)."""
        self._resources[resource_id] = ResourceState(
            resource_id=resource_id,
            resource_type=resource_type,
            capabilities=capabilities or [],
        )

    def get_available_resources(self, capability: str | None = None) -> list[ResourceState]:
        """Get currently available resources, optionally filtered by capability."""
        available = [
            r for r in self._resources.values()
            if r.is_available and (capability is None or capability in r.capabilities)
        ]
        return available

    # -- Experiment Submission ----------------------------------------------

    def submit_experiment(
        self,
        parameters: dict[str, float],
        priority: ExperimentPriority | None = None,
        estimated_duration: float | None = None,
        resource_requirements: list[str] | None = None,
        dependencies: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Submit a new experiment to the queue.
        
        Args:
            parameters: Parameter configuration for the experiment
            priority: Scheduling priority (default: NORMAL)
            estimated_duration: Estimated runtime in seconds
            resource_requirements: Required resource capabilities
            dependencies: IDs of experiments that must complete first
            metadata: Additional experiment metadata
            
        Returns:
            experiment_id: Unique identifier for the experiment
        """
        import uuid
        
        exp_id = f"exp_{uuid.uuid4().hex[:12]}"
        
        experiment = SDLExperiment(
            experiment_id=exp_id,
            parameters=dict(parameters),
            priority=priority or self.default_priority,
            estimated_duration=estimated_duration or 3600.0,
            resource_requirements=resource_requirements or [],
            dependencies=dependencies or [],
            status=ExperimentStatus.QUEUED,
            metadata=metadata or {},
        )
        
        self._experiments[exp_id] = experiment
        self._insert_into_queue(exp_id)
        self._stats["submitted"] += 1
        
        return exp_id

    def submit_batch(
        self,
        parameter_list: list[dict[str, float]],
        priority: ExperimentPriority | None = None,
        shared_resources: list[str] | None = None,
    ) -> list[str]:
        """Submit a batch of experiments.
        
        Args:
            parameter_list: List of parameter configurations
            priority: Priority for all experiments in batch
            shared_resources: Resources shared by all experiments
            
        Returns:
            List of experiment IDs
        """
        exp_ids = []
        for params in parameter_list:
            exp_id = self.submit_experiment(
                parameters=params,
                priority=priority,
                resource_requirements=shared_resources,
            )
            exp_ids.append(exp_id)
        return exp_ids

    def _insert_into_queue(self, exp_id: str) -> None:
        """Insert experiment into queue based on priority."""
        experiment = self._experiments[exp_id]
        
        # Priority order: CRITICAL > HIGH > NORMAL > LOW > BACKGROUND
        priority_order = {
            ExperimentPriority.CRITICAL: 0,
            ExperimentPriority.HIGH: 1,
            ExperimentPriority.NORMAL: 2,
            ExperimentPriority.LOW: 3,
            ExperimentPriority.BACKGROUND: 4,
        }
        
        new_priority = priority_order[experiment.priority]
        
        # Find insertion point
        insert_idx = len(self._queue)
        for i, queued_id in enumerate(self._queue):
            queued_exp = self._experiments[queued_id]
            queued_priority = priority_order[queued_exp.priority]
            
            if new_priority < queued_priority:
                insert_idx = i
                break
            elif new_priority == queued_priority:
                # Same priority: FIFO (check creation time)
                if experiment.created_at < queued_exp.created_at:
                    insert_idx = i
                    break
                    
        self._queue.insert(insert_idx, exp_id)

    # -- Experiment Lifecycle -----------------------------------------------

    def start_experiment(
        self,
        experiment_id: str,
        resource_id: str | None = None,
    ) -> bool:
        """Mark an experiment as started.
        
        Args:
            experiment_id: ID of experiment to start
            resource_id: Resource assigned to this experiment
            
        Returns:
            True if successfully started, False otherwise
        """
        if experiment_id not in self._experiments:
            return False
            
        experiment = self._experiments[experiment_id]
        
        # Check dependencies
        for dep_id in experiment.dependencies:
            if dep_id in self._experiments:
                dep = self._experiments[dep_id]
                if dep.status not in (ExperimentStatus.COMPLETED,):
                    return False  # Dependencies not met
                    
        experiment.status = ExperimentStatus.RUNNING
        experiment.started_at = time.time()
        
        # Remove from queue since it's now running
        if experiment_id in self._queue:
            self._queue.remove(experiment_id)
        
        # Update resource state
        if resource_id and resource_id in self._resources:
            resource = self._resources[resource_id]
            resource.is_available = False
            resource.current_experiment = experiment_id
            resource.scheduled_until = time.time() + experiment.estimated_duration
            
        return True

    def complete_experiment(
        self,
        experiment_id: str,
        result: dict[str, Any],
    ) -> None:
        """Mark an experiment as completed with results.
        
        Args:
            experiment_id: ID of completed experiment
            result: Experimental result dictionary
        """
        if experiment_id not in self._experiments:
            raise ValueError(f"Unknown experiment: {experiment_id}")
            
        experiment = self._experiments[experiment_id]
        experiment.status = ExperimentStatus.COMPLETED
        experiment.completed_at = time.time()
        experiment.result = dict(result)
        
        # Free resources
        self._free_resources(experiment_id)
        
        # Update stats
        self._stats["completed"] += 1
        self._stats["total_wait_time"] += experiment.wait_time
        self._stats["total_execution_time"] += experiment.execution_time
        
        # Trigger callbacks
        for callback in self._on_complete:
            callback(experiment)

    def fail_experiment(
        self,
        experiment_id: str,
        error_message: str = "",
    ) -> None:
        """Mark an experiment as failed.
        
        Args:
            experiment_id: ID of failed experiment
            error_message: Description of failure
        """
        if experiment_id not in self._experiments:
            raise ValueError(f"Unknown experiment: {experiment_id}")
            
        experiment = self._experiments[experiment_id]
        experiment.status = ExperimentStatus.FAILED
        experiment.completed_at = time.time()
        experiment.error_message = error_message
        
        # Free resources
        self._free_resources(experiment_id)
        
        # Update stats
        self._stats["failed"] += 1
        
        # Trigger callbacks
        for callback in self._on_fail:
            callback(experiment)

    def cancel_experiment(self, experiment_id: str) -> bool:
        """Cancel a pending or queued experiment.
        
        Args:
            experiment_id: ID of experiment to cancel
            
        Returns:
            True if cancelled, False if not found or already running
        """
        if experiment_id not in self._experiments:
            return False
            
        experiment = self._experiments[experiment_id]
        
        if experiment.status in (ExperimentStatus.RUNNING, ExperimentStatus.COMPLETED, ExperimentStatus.FAILED):
            return False  # Cannot cancel running or completed experiments
            
        experiment.status = ExperimentStatus.CANCELLED
        experiment.completed_at = time.time()
        
        # Remove from queue
        if experiment_id in self._queue:
            self._queue.remove(experiment_id)
            
        self._stats["cancelled"] += 1
        return True

    def _free_resources(self, experiment_id: str) -> None:
        """Free resources assigned to an experiment."""
        for resource in self._resources.values():
            if resource.current_experiment == experiment_id:
                resource.is_available = True
                resource.current_experiment = None
                resource.scheduled_until = 0.0

    # -- Status Queries -----------------------------------------------------

    def get_experiment(self, experiment_id: str) -> SDLExperiment | None:
        """Get experiment by ID."""
        return self._experiments.get(experiment_id)

    def get_queue_status(self) -> dict[str, Any]:
        """Get current queue status."""
        by_status: dict[ExperimentStatus, list[str]] = {
            status: [] for status in ExperimentStatus
        }
        
        for exp_id, exp in self._experiments.items():
            by_status[exp.status].append(exp_id)
            
        return {
            "queued_experiments": len(self._queue),
            "running_experiments": len(by_status[ExperimentStatus.RUNNING]),
            "completed_experiments": len(by_status[ExperimentStatus.COMPLETED]),
            "failed_experiments": len(by_status[ExperimentStatus.FAILED]),
            "pending_experiments": len(by_status[ExperimentStatus.PENDING]),
            "total_experiments": len(self._experiments),
            "estimated_wait_time": self._estimate_wait_time(),
        }

    def _estimate_wait_time(self) -> float:
        """Estimate wait time for new experiments."""
        if not self._queue:
            return 0.0
            
        running = sum(
            1 for e in self._experiments.values()
            if e.status == ExperimentStatus.RUNNING
        )
        
        if running < self.max_concurrent:
            return 0.0  # Can start immediately
            
        # Estimate based on remaining time of running experiments
        remaining_times = []
        for exp in self._experiments.values():
            if exp.status == ExperimentStatus.RUNNING:
                elapsed = time.time() - (exp.started_at or time.time())
                remaining = max(0, exp.estimated_duration - elapsed)
                remaining_times.append(remaining)
                
        if remaining_times:
            return min(remaining_times)  # Next resource available
        return 0.0

    def get_next_experiments(self, n: int = 1) -> list[SDLExperiment]:
        """Get the next n experiments ready to run.
        
        Returns experiments that:
        1. Are at the front of the queue
        2. Have all dependencies satisfied
        3. Have required resources available
        """
        ready = []
        
        for exp_id in self._queue:
            if len(ready) >= n:
                break
                
            experiment = self._experiments[exp_id]
            
            # Check dependencies
            deps_satisfied = all(
                self._experiments.get(dep_id, SDLExperiment("", {})).status == ExperimentStatus.COMPLETED
                for dep_id in experiment.dependencies
            )
            if not deps_satisfied:
                continue
                
            # Check resources
            if experiment.resource_requirements:
                available = self.get_available_resources()
                has_resources = any(
                    all(req in r.capabilities for req in experiment.resource_requirements)
                    for r in available
                )
                if not has_resources:
                    continue
                    
            ready.append(experiment)
            
        return ready

    def poll_completed(self) -> list[SDLExperiment]:
        """Get all completed experiments that haven't been polled yet.
        
        (For integration with optimization loop)
        """
        completed = [
            exp for exp in self._experiments.values()
            if exp.status == ExperimentStatus.COMPLETED
            and not exp.metadata.get("polled", False)
        ]
        
        # Mark as polled
        for exp in completed:
            exp.metadata["polled"] = True
            
        return completed

    # -- Callbacks ----------------------------------------------------------

    def on_complete(self, callback: Callable[[SDLExperiment], None]) -> None:
        """Register callback for experiment completion."""
        self._on_complete.append(callback)

    def on_fail(self, callback: Callable[[SDLExperiment], None]) -> None:
        """Register callback for experiment failure."""
        self._on_fail.append(callback)

    # -- Statistics ---------------------------------------------------------

    def get_statistics(self) -> dict[str, Any]:
        """Get execution statistics."""
        stats = dict(self._stats)
        
        if stats["completed"] > 0:
            stats["average_wait_time"] = stats["total_wait_time"] / stats["completed"]
            stats["average_execution_time"] = stats["total_execution_time"] / stats["completed"]
        else:
            stats["average_wait_time"] = 0.0
            stats["average_execution_time"] = 0.0
            
        stats.update(self.get_queue_status())
        
        return stats

    def get_resource_utilization(self) -> dict[str, Any]:
        """Get resource utilization statistics."""
        utilization = {}
        
        for resource_id, resource in self._resources.items():
            utilization[resource_id] = {
                "type": resource.resource_type,
                "is_available": resource.is_available,
                "current_experiment": resource.current_experiment,
                "capabilities": resource.capabilities,
            }
            
        return utilization
