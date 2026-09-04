from training.scheduler.batch_generate import build_jobs, run_batch, unstarted_jobs


def _task(task_id: str, role: str) -> dict:
    return {
        "task_id": task_id,
        "ticker": task_id,
        "seed_family": "quiet_control",
        "collection_role": role,
    }


def test_builds_two_jobs_for_paired_and_one_for_teacher_only(tmp_path) -> None:
    jobs = build_jobs(
        [_task("AAA", "paired"), _task("BBB", "teacher_only")],
        output_root=tmp_path,
        run_prefix="run",
    )

    assert [(job.task_id, job.mode) for job in jobs] == [
        ("AAA", "static"),
        ("AAA", "teacher"),
        ("BBB", "teacher"),
    ]


def test_filters_jobs_that_have_already_started(tmp_path) -> None:
    jobs = build_jobs([_task("AAA", "paired")], output_root=tmp_path, run_prefix="run")
    started = jobs[0]
    (tmp_path / "paired" / "01-aaa-quiet-control" / "static").mkdir(parents=True)

    assert unstarted_jobs(jobs) == [jobs[1]]
    assert started not in unstarted_jobs(jobs)


def test_run_batch_records_incremental_results(monkeypatch, tmp_path) -> None:
    jobs = build_jobs([_task("AAA", "paired")], output_root=tmp_path, run_prefix="run")

    def fake_run(job, **_kwargs):
        return {**job.__dict__, "status": "accepted", "returncode": 0}

    monkeypatch.setattr("training.scheduler.batch_generate.run_job", fake_run)
    manifest = run_batch(
        jobs,
        repo=tmp_path,
        tasks_path=tmp_path / "tasks.jsonl",
        output_root=tmp_path,
        max_workers=2,
    )

    assert manifest["status"] == "completed"
    assert manifest["planned_jobs"] == 2
    assert manifest["status_counts"] == {"accepted": 2}
    assert (tmp_path / "batch_manifest.json").exists()
