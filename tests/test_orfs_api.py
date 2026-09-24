"""开放阅读框（ORF）检索接口测试。"""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "test.db"))
    with TestClient(app) as c:
        yield c


def _create(client, sequence, topology="linear"):
    resp = client.post(
        "/digests",
        json={
            "sequence": sequence,
            "topology": topology,
            "enzymes": [
                {"name": "EcoRI", "pattern": "GAATTC", "cut_top": 1, "cut_bottom": 5}
            ],
        },
    )
    assert resp.status_code == 201
    return resp.json()["id"]


def _orfs(client, digest_id, min_aa_length=1, expect=200):
    resp = client.post(
        f"/digests/{digest_id}/orfs", json={"min_aa_length": min_aa_length}
    )
    assert resp.status_code == expect, resp.text
    return resp.json() if expect == 200 else resp


# ---------- 线性：正链基本场景 ----------


def test_linear_plus_strand_basic(client):
    # ATG@2，同框 AAA、TAA：ORF [2,11)，2 个氨基酸（Met + Lys，终止不计）
    digest_id = _create(client, "CCATGAAATAAGG")
    data = _orfs(client, digest_id)
    assert data["id"] == digest_id
    assert data["topology"] == "linear"
    assert data["sequence_length"] == 13
    assert data["min_aa_length"] == 1
    assert data["orf_count"] == 1
    assert data["orfs"] == [
        {
            "strand": "+",
            "start": 2,
            "end": 11,
            "wraps_origin": False,
            "sequence": "ATGAAATAA",
            "aa_length": 2,
        }
    ]


def test_nested_orfs_sharing_same_stop_all_kept(client):
    # 同框 ATG@0 与 ATG@6 共用 TAA@9：两个嵌套 ORF 都保留
    digest_id = _create(client, "ATGAAAATGTAA")
    data = _orfs(client, digest_id, min_aa_length=1)
    assert [(o["start"], o["end"], o["aa_length"]) for o in data["orfs"]] == [
        (0, 12, 3),
        (6, 12, 1),
    ]
    assert [o["sequence"] for o in data["orfs"]] == ["ATGAAAATGTAA", "ATGTAA"]


def test_min_aa_length_filters_nested_orfs(client):
    digest_id = _create(client, "ATGAAAATGTAA")
    assert _orfs(client, digest_id, min_aa_length=1)["orf_count"] == 2
    assert _orfs(client, digest_id, min_aa_length=2)["orf_count"] == 1
    assert _orfs(client, digest_id, min_aa_length=3)["orf_count"] == 1
    assert _orfs(client, digest_id, min_aa_length=4)["orf_count"] == 0


def test_minimal_orf_is_single_met(client):
    # ATG 紧邻终止密码子：氨基酸数为 1（仅起始 Met）
    digest_id = _create(client, "ATGTAA")
    data = _orfs(client, digest_id, min_aa_length=1)
    assert data["orfs"] == [
        {
            "strand": "+",
            "start": 0,
            "end": 6,
            "wraps_origin": False,
            "sequence": "ATGTAA",
            "aa_length": 1,
        }
    ]
    assert _orfs(client, digest_id, min_aa_length=2)["orfs"] == []


def test_first_in_frame_stop_ends_orf(client):
    # 同框第一个终止密码子 TAA@3 结束 ORF，不延伸到后面的 TAG
    digest_id = _create(client, "ATGTAATAG")
    orfs = _orfs(client, digest_id)["orfs"]
    assert len(orfs) == 1
    assert orfs[0]["sequence"] == "ATGTAA"
    assert orfs[0]["end"] == 6


def test_all_three_stop_codons_recognized(client):
    for stop in ("TAA", "TAG", "TGA"):
        digest_id = _create(client, "ATGAAA" + stop)
        data = _orfs(client, digest_id, min_aa_length=2)
        assert data["orf_count"] == 1, stop
        assert data["orfs"][0]["sequence"] == "ATGAAA" + stop
        assert data["orfs"][0]["aa_length"] == 2


# ---------- 简并符号不充当起止信号 ----------


def test_degenerate_start_codon_not_recognized(client):
    # ATN@0 不是起点；字面量 ATG@3 才是，ORF 起点为 3
    digest_id = _create(client, "ATNATGTAA")
    orfs = _orfs(client, digest_id)["orfs"]
    assert len(orfs) == 1
    assert orfs[0]["start"] == 3
    assert orfs[0]["sequence"] == "ATGTAA"


def test_degenerate_stop_codon_not_recognized(client):
    # TAR 含简并符号不是终止信号，ORF 延伸到后面的字面量 TAA
    digest_id = _create(client, "ATGTARTAA")
    orfs = _orfs(client, digest_id, min_aa_length=2)["orfs"]
    assert len(orfs) == 1
    assert orfs[0]["sequence"] == "ATGTARTAA"
    assert orfs[0]["aa_length"] == 2


def test_degenerate_codons_pass_through_inside_orf(client):
    # 简并密码子 NNN 不是终止信号，但作为普通密码子留在 ORF 序列内
    digest_id = _create(client, "ATGNNNTAA")
    orfs = _orfs(client, digest_id, min_aa_length=2)["orfs"]
    assert len(orfs) == 1
    assert orfs[0]["sequence"] == "ATGNNNTAA"
    assert orfs[0]["aa_length"] == 2


# ---------- 线性：反向互补链 ----------


def test_minus_strand_orf_reported_in_forward_coordinates(client):
    # 反链 ORF：正链 [2,11) 区段 TTATTTCAT 的反向互补即 ATGAAATAA
    seq = "CCTTATTTCATGG"
    digest_id = _create(client, seq)
    data = _orfs(client, digest_id, min_aa_length=2)
    assert data["orfs"] == [
        {
            "strand": "-",
            "start": 2,
            "end": 11,
            "wraps_origin": False,
            "sequence": "ATGAAATAA",
            "aa_length": 2,
        }
    ]
    # 坐标为原始正链半开区间；定向序列为反链 5'->3'（ATG 开头）
    assert seq[2:11] == "TTATTTCAT"


def test_results_sorted_by_strand_then_start(client):
    # 正链 ORF [0,6) 与反链 ORF [11,20)：正链在前，链内按起点升序
    seq = "ATGTAA" + "CCC" + "CCTTATTTCATGG"
    digest_id = _create(client, seq)
    data = _orfs(client, digest_id)
    assert data["orfs"] == [
        {
            "strand": "+",
            "start": 0,
            "end": 6,
            "wraps_origin": False,
            "sequence": "ATGTAA",
            "aa_length": 1,
        },
        {
            "strand": "-",
            "start": 11,
            "end": 20,
            "wraps_origin": False,
            "sequence": "ATGAAATAA",
            "aa_length": 2,
        },
    ]
    keys = [(o["strand"], o["start"], o["aa_length"]) for o in data["orfs"]]
    assert keys == sorted(keys)


# ---------- 线性：边界与空结果 ----------


def test_linear_orf_cannot_extend_past_boundary(client):
    # 终止密码子只剩两个碱基（不完整）：线性不得越界，无 ORF
    digest_id = _create(client, "CCATGAAATA")
    assert _orfs(client, digest_id)["orfs"] == []
    # 完全没有终止密码子
    digest_id2 = _create(client, "ATGAAA")
    assert _orfs(client, digest_id2)["orfs"] == []


def test_no_atg_returns_empty_array(client):
    digest_id = _create(client, "CCCGGG")
    data = _orfs(client, digest_id)
    assert data["orfs"] == []
    assert data["orf_count"] == 0


def test_min_aa_length_larger_than_possible_returns_empty(client):
    digest_id = _create(client, "ATGTAA")
    data = _orfs(client, digest_id, min_aa_length=100)
    assert data["orfs"] == []
    assert data["orf_count"] == 0


# ---------- 环形 ----------


def test_circular_orf_wraps_origin(client):
    # ATG@9 跨原点读到 TAA@0：end 取模为 3，wraps_origin 为 true
    digest_id = _create(client, "TAACCCCCCATG", topology="circular")
    data = _orfs(client, digest_id)
    assert data["topology"] == "circular"
    assert data["orfs"] == [
        {
            "strand": "+",
            "start": 9,
            "end": 3,
            "wraps_origin": True,
            "sequence": "ATGTAA",
            "aa_length": 1,
        }
    ]


def test_circular_start_codon_itself_wraps_origin(client):
    # 起点密码子 ATG 跨原点（位置 7,8,0），同框 TAA 在位置 1..3
    digest_id = _create(client, "GTAACCCAT", topology="circular")
    orfs = _orfs(client, digest_id)["orfs"]
    assert orfs == [
        {
            "strand": "+",
            "start": 7,
            "end": 4,
            "wraps_origin": True,
            "sequence": "ATGTAA",
            "aa_length": 1,
        }
    ]


def test_circular_minus_strand_orf_wraps_origin(client):
    # 反链模板为 TAACCCCCCATG：反链 ATG 跨原点，坐标映射回正链
    digest_id = _create(client, "CATGGGGGGTTA", topology="circular")
    orfs = _orfs(client, digest_id)["orfs"]
    assert orfs == [
        {
            "strand": "-",
            "start": 9,
            "end": 3,
            "wraps_origin": True,
            "sequence": "ATGTAA",
            "aa_length": 1,
        }
    ]


def test_circular_orf_exactly_one_turn_allowed(client):
    # 完整 ORF（含终止密码子）长度恰为一圈：允许，end 取模回到 start
    digest_id = _create(client, "ATGCCCCCCTAA", topology="circular")
    orfs = _orfs(client, digest_id, min_aa_length=3)["orfs"]
    assert orfs == [
        {
            "strand": "+",
            "start": 0,
            "end": 0,
            "wraps_origin": False,
            "sequence": "ATGCCCCCCTAA",
            "aa_length": 3,
        }
    ]


def test_circular_orf_longer_than_one_turn_excluded(client):
    # 同框终止密码子在位置 9..0（跨原点），完整 ORF 需 12 nt > 一圈 11 nt：排除
    digest_id = _create(client, "ATGCCCCCCTA", topology="circular")
    assert _orfs(client, digest_id)["orfs"] == []


def test_circular_vs_linear_same_sequence(client):
    # 同序列：线性下 ATG@9 后无同框终止密码子（越界），环形跨原点成 ORF
    seq = "TAACCCCCCATG"
    linear_id = _create(client, seq, topology="linear")
    circular_id = _create(client, seq, topology="circular")
    assert _orfs(client, linear_id)["orfs"] == []
    assert _orfs(client, circular_id)["orf_count"] == 1


# ---------- 状态码与校验 ----------


def test_missing_record_returns_404(client):
    resp = client.post("/digests/999/orfs", json={"min_aa_length": 1})
    assert resp.status_code == 404


@pytest.mark.parametrize("value", [0, -1, -100])
def test_non_positive_min_aa_length_returns_422(client, value):
    digest_id = _create(client, "ATGTAA")
    resp = client.post(f"/digests/{digest_id}/orfs", json={"min_aa_length": value})
    assert resp.status_code == 422


@pytest.mark.parametrize("value", ["abc", 1.5, None, [1]])
def test_non_integer_min_aa_length_returns_422(client, value):
    digest_id = _create(client, "ATGTAA")
    resp = client.post(f"/digests/{digest_id}/orfs", json={"min_aa_length": value})
    assert resp.status_code == 422


def test_missing_min_aa_length_returns_422(client):
    digest_id = _create(client, "ATGTAA")
    resp = client.post(f"/digests/{digest_id}/orfs", json={})
    assert resp.status_code == 422


# ---------- 只读 ----------


def test_search_is_read_only(client):
    digest_id = _create(client, "CCATGAAATAAGG")
    before = client.get("/digests").json()
    record_before = client.get(f"/digests/{digest_id}").json()
    _orfs(client, digest_id, min_aa_length=1)
    after = client.get("/digests").json()
    record_after = client.get(f"/digests/{digest_id}").json()
    assert after == before
    assert record_after == record_before


def test_search_ignores_digest_fragments(client):
    # 酶 EcoRI 在序列上没有位点：ORF 检索仍基于原始序列正常工作
    digest_id = _create(client, "CCATGAAATAAGG")
    record = client.get(f"/digests/{digest_id}").json()
    assert record["cuts"] == []
    assert len(record["fragments"]) == 1  # 未酶切的整条分子
    data = _orfs(client, digest_id)
    assert data["orf_count"] == 1
    assert data["orfs"][0]["sequence"] == "ATGAAATAA"
