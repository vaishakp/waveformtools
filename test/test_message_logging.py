import logging

from waveformtools import waveformtools as wt

CALLER_NAME = f"{__name__}.emit_message"


def emit_message(*args, **kwargs):
    return wt.message(*args, **kwargs)


def test_message_preserves_return_value_and_prints_caller_name(capsys):
    result = emit_message(
        "hello",
        "world",
        message_verbosity=2,
        print_verbosity=2,
        log_verbosity=-1,
    )

    captured = capsys.readouterr()

    assert result == 1
    assert captured.out == f"[{CALLER_NAME}] [INFO] hello world\n"


def test_message_honors_print_like_sep_end_and_flush(capsys):
    emit_message(
        "left",
        "right",
        sep="|",
        end="!",
        flush=True,
        message_verbosity=2,
        print_verbosity=2,
        log_verbosity=-1,
    )

    captured = capsys.readouterr()

    assert captured.out == f"[{CALLER_NAME}] [INFO] left|right!"


def test_message_runtime_print_verbosity_is_dynamic(monkeypatch, capsys):
    monkeypatch.setattr(wt.vlconf, "print_verbosity", 1, raising=False)

    emit_message("hidden", message_verbosity=2, log_verbosity=-1)
    assert capsys.readouterr().out == ""

    monkeypatch.setattr(wt.vlconf, "print_verbosity", 2, raising=False)
    emit_message("visible", message_verbosity=2, log_verbosity=-1)

    assert "visible" in capsys.readouterr().out


def test_message_emits_to_logging_with_caller_name(caplog):
    caplog.set_level(logging.DEBUG, logger=__name__)

    emit_message(
        "captured",
        message_verbosity=2,
        print_verbosity=-1,
        log_verbosity=2,
    )

    assert len(caplog.records) == 1
    record = caplog.records[0]
    assert record.name == CALLER_NAME
    assert record.levelname == "INFO"
    assert record.getMessage() == "captured"


def test_message_writes_bracketed_file_log(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    wt._reset_message_logging_for_tests()

    emit_message(
        "file",
        "log",
        message_verbosity=1,
        print_verbosity=-1,
        log_verbosity=1,
    )
    wt._reset_message_logging_for_tests()

    log_files = list((tmp_path / "logs").glob("*.log"))
    assert len(log_files) == 1
    log_text = log_files[0].read_text(encoding="utf-8")
    assert f"[{CALLER_NAME}] [WARNING] file log" in log_text
