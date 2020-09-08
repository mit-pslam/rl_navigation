"""Run a single, default test."""
import rl_navigation._cli_tool as cli_tool
import rl_navigation.subcommands.train as rlnav_train
import rl_navigation.subcommands.plot as rlnav_plot
import rl_navigation.subcommands.video as rlnav_video
from unittest import mock
import inspect
import sys


def compare_args(mock_function, expected_args, num_extra=1):
    """Make sure that all necessary args are covered by the CLI."""
    dispatch_args = [arg for arg in mock_function.call_args[1]]

    # kwargs in required because we have some stuff in the cli to handle sub-commands
    assert "kwargs" in expected_args.parameters
    for param_name in expected_args.parameters:
        assert (param_name == "kwargs") ^ (param_name in dispatch_args)

    assert len(dispatch_args) - len(expected_args.parameters) == num_extra


@mock.patch.object(sys, "argv", ["rl_navigation", "train"])
def test_train_arguments():
    """Test that the train cli argument match what run_train expects."""
    expected_args = inspect.signature(rlnav_train.run_train)
    with mock.patch("rl_navigation.subcommands.train.run_train") as func_mock:
        cli_tool.main()
        assert func_mock.called
        compare_args(func_mock, expected_args)


@mock.patch.object(sys, "argv", ["rl_navigation", "plot"])
def test_plot_arguments():
    """Test that the plot cli argument match what run_plot expects."""
    expected_args = inspect.signature(rlnav_plot.run_plot)
    with mock.patch("rl_navigation.subcommands.plot.run_plot") as func_mock:
        cli_tool.main()
        assert func_mock.called
        compare_args(func_mock, expected_args)


@mock.patch.object(sys, "argv", ["rl_navigation", "video", "some_fake_policy_file"])
def test_video_arguments():
    """Test that the video cli argument match what run_video expects."""
    expected_args = inspect.signature(rlnav_video.run_video)
    with mock.patch("rl_navigation.subcommands.video.run_video") as func_mock:
        cli_tool.main()
        assert func_mock.called
        compare_args(func_mock, expected_args)
