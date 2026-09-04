from unittest import mock

import pytest

from cli.utils import select_orchestration_mode


@pytest.mark.parametrize("selected", ["static", "teacher"])
def test_orchestration_menu_returns_selected_runtime_mode(selected: str) -> None:
    prompt = mock.MagicMock()
    prompt.ask.return_value = selected
    with mock.patch("cli.utils.questionary.select", return_value=prompt) as select:
        assert select_orchestration_mode() == selected

    choices = select.call_args.kwargs["choices"]
    assert [choice.value for choice in choices] == ["static", "teacher"]
