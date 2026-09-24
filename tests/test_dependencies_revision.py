"""Dependency revision: PATCH /jobs/{id}/dependencies (service + HTTP).

Covers: wholesale replacement with dedup/sort normalisation, empty-array
detach, cross-tenant references, QUEUED-only enforcement, rejection of
unknown ids / self-dependence / cycles with the original edges preserved,
the next preview/batch deciding on the revised graph, and preservation of
the first submission's idempotency fingerprint (a replay returns the
revised job without restoring the submitted dependencies).
"""
from __future__ import annotations

import pytest

from app import db, scheduler

from tests.conftest import at, iso  # noqa: F401


def submit(tenant, gpus=1, minutes=10, weight=1.0, depends_on=None,
           when=None):
    job = scheduler.submit_job(
        tenant, gpus, minutes, weight, depends_on=depends_on
    )
    if when is not None:
        db.get_conn().execute(
            "UPDATE jobs SET queued_at = ? WHERE id = ?", (when, job["id"])
        )
    return scheduler.get_job(job["id"])


def deps(job_id):
    return scheduler.get_job(job_id)["depends_on"]


# ---------------------------------------------------------------------------
# Service level
# ---------------------------------------------------------------------------

def test_replace_returns_fresh_job_deduped_and_sorted(conn):
    p1 = submit("base")                       # J000001
    p2 = submit("base")                       # J000002
    p3 = submit("other")                      # J000003 (different tenant)
    job = submit("t", depends_on=[p1["id"]])  # J000004

    # Duplicates and out-of-order input normalise to the sorted unique list;
    # cross-tenant references are allowed.
    revised = scheduler.revise_dependencies(
        job["id"], [p3["id"], p2["id"], p3["id"]]
    )
    assert revised["id"] == job["id"]
    assert revised["status"] == "QUEUED"
    assert revised["depends_on"] == [p2["id"], p3["id"]]
    assert revised["blocked_by"] == [
        {"id": p2["id"], "status": "QUEUED"},
        {"id": p3["id"], "status": "QUEUED"},
    ]
    # The change is persisted, not just returned.
    assert deps(job["id"]) == [p2["id"], p3["id"]]


def test_empty_array_detaches_every_dependency(conn):
    p1 = submit("base")
    job = submit("t", depends_on=[p1["id"]])
    assert scheduler.get_job(job["id"])["blocked_by"] != []

    revised = scheduler.revise_dependencies(job["id"], [])
    assert revised["depends_on"] == []
    assert revised["blocked_by"] == []
    assert deps(job["id"]) == []


def test_unknown_job_is_404(conn):
    with pytest.raises(scheduler.NotFound) as exc:
        scheduler.revise_dependencies("J000999", [])
    assert exc.value.status_code == 404


def test_only_queued_jobs_can_be_revised(conn):
    p1 = submit("base")
    job = submit("t", gpus=1, minutes=10, depends_on=None, when=iso(0))
    scheduler.run_batch(now=at(1))
    assert scheduler.get_job(job["id"])["status"] == "RUNNING"
    with pytest.raises(scheduler.Conflict) as exc:
        scheduler.revise_dependencies(job["id"], [p1["id"]])
    assert exc.value.status_code == 409
    assert deps(job["id"]) == []

    scheduler.complete_job(job["id"])
    with pytest.raises(scheduler.Conflict):
        scheduler.revise_dependencies(job["id"], [p1["id"]])
    assert deps(job["id"]) == []

    other = submit("t2", depends_on=[p1["id"]])
    scheduler.cancel_job(other["id"])
    with pytest.raises(scheduler.Conflict):
        scheduler.revise_dependencies(other["id"], [])
    # A cancelled job's edges are untouched by the rejected revision.
    assert deps(other["id"]) == [p1["id"]]


def test_unknown_prerequisite_rejected_originals_kept(conn):
    p1 = submit("base")
    job = submit("t", depends_on=[p1["id"]])
    with pytest.raises(scheduler.SchedulerError) as exc:
        scheduler.revise_dependencies(job["id"], [p1["id"], "J000999"])
    assert exc.value.status_code == 400
    assert "unknown depends_on" in str(exc.value)
    # The failed revision changed nothing.
    assert deps(job["id"]) == [p1["id"]]


def test_self_dependency_rejected_originals_kept(conn):
    p1 = submit("base")
    job = submit("t", depends_on=[p1["id"]])
    with pytest.raises(scheduler.SchedulerError) as exc:
        scheduler.revise_dependencies(job["id"], [job["id"]])
    assert exc.value.status_code == 400
    assert "itself" in str(exc.value)
    assert deps(job["id"]) == [p1["id"]]


def test_cycle_rejected_originals_kept(conn):
    a = submit("a")                    # J000001
    b = submit("b", depends_on=[a["id"]])   # J000002: b -> a
    c = submit("c", depends_on=[b["id"]])   # J000003: c -> b -> a

    # Direct 2-cycle: a -> b while b -> a.
    with pytest.raises(scheduler.SchedulerError) as exc:
        scheduler.revise_dependencies(a["id"], [b["id"]])
    assert "dependency cycle" in str(exc.value)
    assert deps(a["id"]) == []

    # Longer cycle: a -> c -> b -> a.
    with pytest.raises(scheduler.SchedulerError) as exc:
        scheduler.revise_dependencies(a["id"], [c["id"]])
    assert "dependency cycle" in str(exc.value)
    # Every rejected attempt left the whole graph untouched.
    assert deps(a["id"]) == []
    assert deps(b["id"]) == [a["id"]]
    assert deps(c["id"]) == [b["id"]]


def test_invalid_depends_on_shape_rejected(conn):
    job = submit("t")
    for bad in ("J000001", [""], ["   "], [123], None):
        with pytest.raises(scheduler.SchedulerError) as exc:
            scheduler.revise_dependencies(job["id"], bad)
        assert exc.value.status_code == 400
    assert deps(job["id"]) == []


def test_revision_does_not_touch_other_jobs(conn):
    p1 = submit("base")
    p2 = submit("base")
    j1 = submit("t1", depends_on=[p1["id"]])
    j2 = submit("t2", depends_on=[p1["id"], p2["id"]])

    scheduler.revise_dependencies(j1["id"], [p2["id"]])
    assert deps(j1["id"]) == [p2["id"]]
    assert deps(j2["id"]) == [p1["id"], p2["id"]]


def test_next_batch_uses_revised_dependencies(conn):
    pre = submit("a", when=iso(0))                       # J000001
    job = submit("b", depends_on=[pre["id"]], when=iso(1))  # J000002

    # Blocked: only the prerequisite is dispatched.
    r1 = scheduler.run_batch(now=at(2))
    assert r1["selection_order"] == [pre["id"]]
    assert scheduler.get_job(job["id"])["status"] == "QUEUED"

    # Detaching the dependency releases the job for the very next batch.
    scheduler.revise_dependencies(job["id"], [])
    r2 = scheduler.run_batch(now=at(3))
    assert r2["selection_order"] == [job["id"]]


def test_revision_can_block_a_ready_job(conn):
    pre = submit("a", when=iso(0))
    job = submit("b", when=iso(1))
    scheduler.revise_dependencies(job["id"], [pre["id"]])
    assert scheduler.get_job(job["id"])["blocked_by"] == [
        {"id": pre["id"], "status": "QUEUED"}
    ]
    r = scheduler.run_batch(now=at(2))
    assert r["selection_order"] == [pre["id"]]
    assert scheduler.get_job(job["id"])["status"] == "QUEUED"


def test_preview_uses_revised_dependencies_and_writes_nothing(conn):
    pre = submit("a", when=iso(0))
    job = submit("b", depends_on=[pre["id"]], when=iso(1))

    p1 = scheduler.preview_batch(now=at(2))
    assert p1["selection_order"] == [pre["id"]]

    scheduler.revise_dependencies(job["id"], [])
    p2 = scheduler.preview_batch(now=at(2))
    assert p2["preview"] is True
    assert p2["selection_order"] == [pre["id"], job["id"]]

    # The preview itself persisted nothing: both jobs are still QUEUED and
    # the committed batch then makes the same decision on the new graph.
    assert scheduler.get_job(job["id"])["status"] == "QUEUED"
    r = scheduler.run_batch(now=at(2))
    assert r["selection_order"] == [pre["id"], job["id"]]


def test_idempotency_fingerprint_preserved_after_revision(conn):
    p1 = submit("base")   # J000001
    p2 = submit("base")   # J000002
    job, created = scheduler.submit_job_idempotent(
        "t", 1, 10, 1.0, [p1["id"]], idempotency_key="k1"
    )
    assert created is True
    assert job["depends_on"] == [p1["id"]]

    # Revise the dependencies out of band.
    scheduler.revise_dependencies(job["id"], [p2["id"]])

    # Replaying the ORIGINAL submission (same key, original parameters)
    # returns the revised job; it does not restore the submitted edges.
    replay, created2 = scheduler.submit_job_idempotent(
        "t", 1, 10, 1.0, [p1["id"]], idempotency_key="k1"
    )
    assert created2 is False
    assert replay["id"] == job["id"]
    assert replay["depends_on"] == [p2["id"]]
    assert deps(job["id"]) == [p2["id"]]

    # The stored fingerprint is still the first submission's: diverging
    # parameters keep conflicting with 409.
    with pytest.raises(scheduler.Conflict):
        scheduler.submit_job_idempotent(
            "t", 1, 10, 1.0, [p2["id"]], idempotency_key="k1"
        )
    assert deps(job["id"]) == [p2["id"]]


# ---------------------------------------------------------------------------
# HTTP level
# ---------------------------------------------------------------------------

def _post(client, tenant, **kw):
    body = {"tenant_id": tenant, "gpus": 1, "duration_min": 10}
    body.update(kw)
    r = client.post("/jobs", json=body)
    assert r.status_code == 201, r.text
    return r.json()


def test_http_patch_replaces_dependencies(client):
    p1 = _post(client, "base")
    p2 = _post(client, "other")
    job = _post(client, "t", depends_on=[p1["id"]])

    r = client.patch(
        f"/jobs/{job['id']}/dependencies",
        json={"depends_on": [p2["id"], p1["id"], p2["id"]]},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["id"] == job["id"]
    assert body["depends_on"] == [p1["id"], p2["id"]]
    assert body["blocked_by"] == [
        {"id": p1["id"], "status": "QUEUED"},
        {"id": p2["id"], "status": "QUEUED"},
    ]
    # Persisted: a plain GET shows the same graph.
    assert client.get(f"/jobs/{job['id']}").json()["depends_on"] == [
        p1["id"], p2["id"]
    ]


def test_http_patch_empty_array_detaches(client):
    p1 = _post(client, "base")
    job = _post(client, "t", depends_on=[p1["id"]])
    r = client.patch(f"/jobs/{job['id']}/dependencies",
                     json={"depends_on": []})
    assert r.status_code == 200
    assert r.json()["depends_on"] == []
    assert r.json()["blocked_by"] == []


def test_http_patch_unknown_job_404(client):
    r = client.patch("/jobs/J000999/dependencies", json={"depends_on": []})
    assert r.status_code == 404


def test_http_patch_non_queued_409(client):
    pre = _post(client, "base")
    job = _post(client, "t")
    client.post("/scheduler/batch")
    assert client.get(f"/jobs/{job['id']}").json()["status"] == "RUNNING"
    r = client.patch(f"/jobs/{job['id']}/dependencies",
                     json={"depends_on": [pre["id"]]})
    assert r.status_code == 409
    assert client.get(f"/jobs/{job['id']}").json()["depends_on"] == []


def test_http_patch_unknown_prerequisite_400_originals_kept(client):
    p1 = _post(client, "base")
    job = _post(client, "t", depends_on=[p1["id"]])
    r = client.patch(f"/jobs/{job['id']}/dependencies",
                     json={"depends_on": ["J000999"]})
    assert r.status_code == 400
    assert "unknown depends_on" in r.json()["detail"]
    assert client.get(f"/jobs/{job['id']}").json()["depends_on"] == [p1["id"]]


def test_http_patch_self_dependency_400(client):
    job = _post(client, "t")
    r = client.patch(f"/jobs/{job['id']}/dependencies",
                     json={"depends_on": [job["id"]]})
    assert r.status_code == 400
    assert client.get(f"/jobs/{job['id']}").json()["depends_on"] == []


def test_http_patch_cycle_400_originals_kept(client):
    a = _post(client, "a")
    b = _post(client, "b", depends_on=[a["id"]])
    r = client.patch(f"/jobs/{a['id']}/dependencies",
                     json={"depends_on": [b["id"]]})
    assert r.status_code == 400
    assert "dependency cycle" in r.json()["detail"]
    assert client.get(f"/jobs/{a['id']}").json()["depends_on"] == []
    assert client.get(f"/jobs/{b['id']}").json()["depends_on"] == [a["id"]]


def test_http_patch_invalid_body_422(client):
    job = _post(client, "t")
    assert client.patch(
        f"/jobs/{job['id']}/dependencies", json={}).status_code == 422
    assert client.patch(
        f"/jobs/{job['id']}/dependencies",
        json={"depends_on": "J000001"}).status_code == 422
    assert client.patch(
        f"/jobs/{job['id']}/dependencies",
        json={"depends_on": [""]}).status_code == 422
    assert client.patch(
        f"/jobs/{job['id']}/dependencies",
        json={"depends_on": [123]}).status_code == 422
    assert client.get(f"/jobs/{job['id']}").json()["depends_on"] == []


def test_http_revision_guides_next_batch(client):
    pre = _post(client, "a")
    job = _post(client, "b", depends_on=[pre["id"]])

    r = client.post("/scheduler/batch").json()
    assert r["selection_order"] == [pre["id"]]

    client.patch(f"/jobs/{job['id']}/dependencies", json={"depends_on": []})
    r = client.post("/scheduler/batch").json()
    assert r["selection_order"] == [job["id"]]


def test_http_idempotent_replay_returns_revised_job(client):
    p1 = _post(client, "base")
    p2 = _post(client, "base")
    headers = {"Idempotency-Key": "rev-1"}
    body = {"tenant_id": "t", "gpus": 1, "duration_min": 10,
            "depends_on": [p1["id"]]}
    r = client.post("/jobs", json=body, headers=headers)
    assert r.status_code == 201
    job_id = r.json()["id"]

    client.patch(f"/jobs/{job_id}/dependencies",
                 json={"depends_on": [p2["id"]]})

    # Same key + original parameters -> 200 replay of the revised job; the
    # first submission's fingerprint is untouched and nothing is reset.
    r2 = client.post("/jobs", json=body, headers=headers)
    assert r2.status_code == 200
    assert r2.json()["id"] == job_id
    assert r2.json()["depends_on"] == [p2["id"]]
    assert len(client.get("/jobs").json()) == 3
