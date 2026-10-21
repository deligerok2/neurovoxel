from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, sentinel

import pandas as pd
import pytest

import neurovoxel.ui.model_runner as runner


class FakeSessionState(dict):
    """Support both dictionary and attribute access."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


@pytest.mark.parametrize(
    ("overrides", "expected_n_jobs", "expected_random_state"),
    [
        ({}, -1, 42),
        ({"n_jobs": 2, "random_state": 123}, 2, 123),
    ],
    ids=["default-options", "custom-options"],
)
def test_render_model_runner(
    monkeypatch,
    tmp_path,
    overrides,
    expected_n_jobs,
    expected_random_state,
):
    input_tbl = pd.DataFrame({"subject": ["01"]})
    output_tbl = pd.DataFrame({"subject": ["01"], "included": [True]})
    images = [sentinel.image]

    analysis = {
        "query": "gm ~ age",
        "inference_terms": ["age"],
        "smoothing_fwhm": 6,
        "voxel_size": 2,
        "n_perm": 100,
        "tfce": True,
        "handle_zero_voxels": "error",
        "handle_multiple_sessions": "all",
        **overrides,
    }

    layout = Mock()
    layout.get.return_value = images

    state = FakeSessionState(
        entity_df=pd.DataFrame(
            [
                {"name": "wm", "suffix": "probseg", "label": "WM",
                 "task": "rest"},
                {"name": "gm", "suffix": "probseg", "label": "GM",
                 "task": None},
            ]
        ),
        layout=layout,
        paths={
            "mask": str(tmp_path / "mask.nii.gz"),
            "bids_root": str(tmp_path),
        },
        analysis=analysis,
        tbl=input_tbl,
    )

    spinner = MagicMock()
    fake_st = SimpleNamespace(session_state=state, spinner=spinner)

    get_masker = Mock(return_value=sentinel.masker)
    run_query = Mock(return_value=(sentinel.result, output_tbl))

    monkeypatch.setattr(runner, "st", fake_st)
    monkeypatch.setattr(runner, "get_masker", get_masker)
    monkeypatch.setattr(runner, "run_query", run_query)

    returned = runner.render_model_runner("gm")

    # Select the requested entity and remove name/missing columns.
    layout.get.assert_called_once_with(suffix="probseg", label="GM")

    get_masker.assert_called_once_with(
        str(tmp_path / "mask.nii.gz"),
        6,
        2,
        n_jobs=expected_n_jobs,
    )

    run_query.assert_called_once()
    args, kwargs = run_query.call_args

    assert args[0] == "gm ~ age"
    assert args[1] == tmp_path
    assert args[2] == ["age"]
    assert args[3] is input_tbl
    assert args[4] is images
    assert args[5] is sentinel.masker

    assert kwargs == {
        "n_perm": 100,
        "n_jobs": expected_n_jobs,
        "random_state": expected_random_state,
        "tfce": True,
        "handle_zero_voxels": "error",
        "handle_multiple_sessions": "all",
    }

    # Save the analysis outputs into session state.
    assert state.masker is sentinel.masker
    assert state.result is sentinel.result
    assert state.tbl is output_tbl
    assert returned is None

    spinner.assert_called_once_with("Running analysis...")
    spinner.return_value.__enter__.assert_called_once()
    spinner.return_value.__exit__.assert_called_once()