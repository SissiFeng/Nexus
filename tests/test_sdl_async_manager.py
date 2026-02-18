"""Tests for SDL async experiment manager."""

import time
import pytest

from optimization_copilot.batch.sdl_async_manager import (
    SDLAsyncManager,
    SDLExperiment,
    ExperimentPriority,
    ExperimentStatus,
    ResourceState,
)


def test_submit_experiment():
    """Test experiment submission."""
    manager = SDLAsyncManager()
    
    exp_id = manager.submit_experiment(
        parameters={"x": 0.5, "y": 0.3},
        priority=ExperimentPriority.HIGH,
    )
    
    assert exp_id.startswith("exp_")
    assert exp_id in manager._experiments
    
    exp = manager.get_experiment(exp_id)
    assert exp is not None
    assert exp.parameters == {"x": 0.5, "y": 0.3}
    assert exp.priority == ExperimentPriority.HIGH
    assert exp.status == ExperimentStatus.QUEUED


def test_priority_queue_ordering():
    """Test that higher priority experiments are queued first."""
    manager = SDLAsyncManager()
    
    # Submit in reverse priority order
    low_id = manager.submit_experiment(parameters={"x": 1}, priority=ExperimentPriority.LOW)
    normal_id = manager.submit_experiment(parameters={"x": 2}, priority=ExperimentPriority.NORMAL)
    high_id = manager.submit_experiment(parameters={"x": 3}, priority=ExperimentPriority.HIGH)
    critical_id = manager.submit_experiment(parameters={"x": 4}, priority=ExperimentPriority.CRITICAL)
    
    # Queue should be ordered by priority
    assert manager._queue[0] == critical_id
    assert manager._queue[1] == high_id
    assert manager._queue[2] == normal_id
    assert manager._queue[3] == low_id


def test_fifo_within_same_priority():
    """Test FIFO ordering within same priority."""
    manager = SDLAsyncManager()
    
    # Small delay to ensure different timestamps
    id1 = manager.submit_experiment(parameters={"x": 1}, priority=ExperimentPriority.NORMAL)
    time.sleep(0.001)
    id2 = manager.submit_experiment(parameters={"x": 2}, priority=ExperimentPriority.NORMAL)
    time.sleep(0.001)
    id3 = manager.submit_experiment(parameters={"x": 3}, priority=ExperimentPriority.NORMAL)
    
    # Should be FIFO
    idx1 = manager._queue.index(id1)
    idx2 = manager._queue.index(id2)
    idx3 = manager._queue.index(id3)
    
    assert idx1 < idx2 < idx3


def test_experiment_lifecycle():
    """Test complete experiment lifecycle."""
    manager = SDLAsyncManager()
    
    exp_id = manager.submit_experiment(parameters={"x": 0.5})
    
    # Start experiment
    started = manager.start_experiment(exp_id)
    assert started
    
    exp = manager.get_experiment(exp_id)
    assert exp.status == ExperimentStatus.RUNNING
    assert exp.started_at is not None
    
    # Complete experiment
    manager.complete_experiment(exp_id, {"objective": 5.0})
    
    exp = manager.get_experiment(exp_id)
    assert exp.status == ExperimentStatus.COMPLETED
    assert exp.result == {"objective": 5.0}
    assert exp.completed_at is not None


def test_experiment_failure():
    """Test experiment failure handling."""
    manager = SDLAsyncManager()
    
    exp_id = manager.submit_experiment(parameters={"x": 0.5})
    manager.start_experiment(exp_id)
    
    manager.fail_experiment(exp_id, error_message="Equipment malfunction")
    
    exp = manager.get_experiment(exp_id)
    assert exp.status == ExperimentStatus.FAILED
    assert exp.error_message == "Equipment malfunction"


def test_cancel_pending_experiment():
    """Test cancelling a pending experiment."""
    manager = SDLAsyncManager()
    
    exp_id = manager.submit_experiment(parameters={"x": 0.5})
    
    cancelled = manager.cancel_experiment(exp_id)
    assert cancelled
    
    exp = manager.get_experiment(exp_id)
    assert exp.status == ExperimentStatus.CANCELLED
    assert exp_id not in manager._queue


def test_cannot_cancel_running_experiment():
    """Test that running experiments cannot be cancelled."""
    manager = SDLAsyncManager()
    
    exp_id = manager.submit_experiment(parameters={"x": 0.5})
    manager.start_experiment(exp_id)
    
    cancelled = manager.cancel_experiment(exp_id)
    assert not cancelled
    
    exp = manager.get_experiment(exp_id)
    assert exp.status == ExperimentStatus.RUNNING


def test_resource_management():
    """Test resource registration and management."""
    manager = SDLAsyncManager()
    
    manager.register_resource(
        resource_id="robot_1",
        resource_type="liquid_handler",
        capabilities=["pipetting", "mixing"],
    )
    
    assert "robot_1" in manager._resources
    resource = manager._resources["robot_1"]
    assert resource.resource_type == "liquid_handler"
    assert "pipetting" in resource.capabilities


def test_get_available_resources():
    """Test querying available resources."""
    manager = SDLAsyncManager()
    
    manager.register_resource(
        resource_id="robot_1",
        resource_type="liquid_handler",
        capabilities=["pipetting"],
    )
    manager.register_resource(
        resource_id="robot_2",
        resource_type="liquid_handler",
        capabilities=["pipetting", "heating"],
    )
    
    # Use one resource
    manager._resources["robot_1"].is_available = False
    
    available = manager.get_available_resources()
    assert len(available) == 1
    assert available[0].resource_id == "robot_2"


def test_get_available_resources_filtered():
    """Test querying resources by capability."""
    manager = SDLAsyncManager()
    
    manager.register_resource(
        resource_id="robot_1",
        resource_type="liquid_handler",
        capabilities=["pipetting"],
    )
    manager.register_resource(
        resource_id="heater_1",
        resource_type="heater",
        capabilities=["heating"],
    )
    
    pipetting = manager.get_available_resources(capability="pipetting")
    assert len(pipetting) == 1
    assert pipetting[0].resource_id == "robot_1"


def test_queue_status():
    """Test queue status reporting."""
    manager = SDLAsyncManager()
    
    # Submit experiments
    for i in range(5):
        manager.submit_experiment(parameters={"x": i * 0.1})
    
    # Start some
    for i, exp_id in enumerate(list(manager._queue)[:2]):
        manager.start_experiment(exp_id)
    
    status = manager.get_queue_status()
    
    assert status["queued_experiments"] == 3  # 5 - 2 started
    assert status["running_experiments"] == 2
    assert status["total_experiments"] == 5


def test_poll_completed():
    """Test polling for completed experiments."""
    manager = SDLAsyncManager()
    
    exp_id = manager.submit_experiment(parameters={"x": 0.5})
    manager.start_experiment(exp_id)
    manager.complete_experiment(exp_id, {"obj": 1.0})
    
    # First poll should return the experiment
    completed = manager.poll_completed()
    assert len(completed) == 1
    assert completed[0].experiment_id == exp_id
    
    # Second poll should return empty (already polled)
    completed = manager.poll_completed()
    assert len(completed) == 0


def test_dependencies():
    """Test experiment dependencies."""
    manager = SDLAsyncManager()
    
    exp1_id = manager.submit_experiment(parameters={"x": 0.1})
    exp2_id = manager.submit_experiment(
        parameters={"x": 0.2},
        dependencies=[exp1_id],
    )
    
    # exp2 should not be ready until exp1 completes
    ready = manager.get_next_experiments(n=10)
    ready_ids = [e.experiment_id for e in ready]
    assert exp1_id in ready_ids
    assert exp2_id not in ready_ids
    
    # Complete exp1
    manager.start_experiment(exp1_id)
    manager.complete_experiment(exp1_id, {"obj": 1.0})
    
    # Now exp2 should be ready
    ready = manager.get_next_experiments(n=10)
    ready_ids = [e.experiment_id for e in ready]
    assert exp2_id in ready_ids


def test_statistics():
    """Test statistics tracking."""
    manager = SDLAsyncManager()
    
    # Submit and complete some experiments
    for i in range(3):
        exp_id = manager.submit_experiment(parameters={"x": i * 0.1})
        manager.start_experiment(exp_id)
        manager.complete_experiment(exp_id, {"obj": float(i)})
    
    # Submit and fail one
    exp_id = manager.submit_experiment(parameters={"x": 0.5})
    manager.start_experiment(exp_id)
    manager.fail_experiment(exp_id)
    
    stats = manager.get_statistics()
    
    assert stats["submitted"] == 4
    assert stats["completed"] == 3
    assert stats["failed"] == 1
    assert stats["average_wait_time"] >= 0
    assert stats["average_execution_time"] >= 0


def test_callbacks():
    """Test completion and failure callbacks."""
    manager = SDLAsyncManager()
    
    completed_events = []
    failed_events = []
    
    def on_complete(exp):
        completed_events.append(exp.experiment_id)
    
    def on_fail(exp):
        failed_events.append(exp.experiment_id)
    
    manager.on_complete(on_complete)
    manager.on_fail(on_fail)
    
    # Complete an experiment
    exp1 = manager.submit_experiment(parameters={"x": 0.1})
    manager.start_experiment(exp1)
    manager.complete_experiment(exp1, {"obj": 1.0})
    
    # Fail an experiment
    exp2 = manager.submit_experiment(parameters={"x": 0.2})
    manager.start_experiment(exp2)
    manager.fail_experiment(exp2)
    
    assert exp1 in completed_events
    assert exp2 in failed_events


def test_experiment_timing_properties():
    """Test experiment timing calculations."""
    manager = SDLAsyncManager()
    
    exp_id = manager.submit_experiment(parameters={"x": 0.5})
    exp = manager.get_experiment(exp_id)
    
    # Before starting
    assert exp.wait_time >= 0
    assert exp.execution_time == 0.0
    
    # After starting
    manager.start_experiment(exp_id)
    time.sleep(0.01)  # Small delay
    assert exp.execution_time >= 0.01
    
    # After completing
    manager.complete_experiment(exp_id, {"obj": 1.0})
    assert exp.total_time >= 0.01
