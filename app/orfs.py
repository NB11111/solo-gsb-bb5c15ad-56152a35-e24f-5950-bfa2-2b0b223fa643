"""开放阅读框（ORF）检索（只读，不依赖酶切片段）。

输入
----
已保存记录的原始 IUPAC 序列与线性/环形拓扑，以及最短氨基酸数
``min_aa_length``（不含终止密码子）。分析只读取记录内容，不写数据库。

检索口径
--------
* 分别在**正链**（提交序列 5'->3'）与**反向互补链**（5'->3'）上逐碱基
  寻找**字面量** ``ATG`` 作为起点；
* 从各起点按三联体读取，遇到同框第一个**字面量**终止密码子
  （``TAA`` / ``TAG`` / ``TGA``）才形成 ORF；
* 含简并符号的密码子不充当起止信号（如 ``ATN`` 不是起点、``TAR``
  不是终止），只作为普通密码子留在 ORF 内部；
* **线性**：ORF 不得越出模板边界；**环形**：ORF 可跨原点，但完整 ORF
  （含终止密码子）长度不得超过模板一圈；
* 同一终止密码子可被同框多个 ``ATG`` 共用，所有满足最短氨基酸数的
  嵌套 ORF 都保留（氨基酸数不含终止密码子，起始 Met 计入）。

坐标与序列
----------
* ``start`` / ``end`` 为完整 ORF（含终止密码子）在**原始正链**上的
  0-based 半开区间端点，与链方向无关；反链 ORF 的翻译方向为从
  ``end`` 向 ``start``（右向左）。环形 ORF 跨原点时 ``end`` 为取模后
  坐标（可能小于 ``start``），``wraps_origin`` 为 true；
* ``sequence`` 为 ORF 在**自身链**上 5'->3' 的定向核苷酸序列
  （以 ``ATG`` 开头、终止密码子结尾）；
* ``aa_length`` 为氨基酸数（不含终止密码子）。

结果按（链方向、起点、氨基酸数）稳定排序；无结果时返回空数组。
"""

from __future__ import annotations

from .iupac import reverse_complement

START_CODON = "ATG"
STOP_CODONS = frozenset({"TAA", "TAG", "TGA"})
CODON = 3


def _scan_strand(template: str, circular: bool, min_aa_length: int):
    """在一条 5'->3' 模板上枚举全部 ORF，以链局部坐标生成。

    起点范围为 ``[0, L)``：线性模板的起点密码子不得越界，环形模板的
    起点密码子可跨原点（在复制串上判定）。从每个 ``ATG`` 起点按三联体
    读取，同框第一个字面量终止密码子结束 ORF；完整 ORF（含终止密码子）
    长度线性不超过 ``L - start``（不越界）、环形不超过 ``L``（一圈）。
    生成 ``{"start", "nt_length", "aa_length", "sequence"}``。
    """
    length = len(template)
    haystack = template + template if circular else template
    last_start = length if circular else length - CODON + 1
    for start in range(max(last_start, 0)):
        if haystack[start : start + CODON] != START_CODON:
            continue
        limit = length if circular else length - start
        offset = CODON
        while offset + CODON <= limit:
            if haystack[start + offset : start + offset + CODON] in STOP_CODONS:
                nt_length = offset + CODON
                aa_length = offset // CODON
                if aa_length >= min_aa_length:
                    yield {
                        "start": start,
                        "nt_length": nt_length,
                        "aa_length": aa_length,
                        "sequence": haystack[start : start + nt_length],
                    }
                break  # 每个起点只取同框第一个终止密码子
            offset += CODON


def find_orfs(sequence: str, topology: str, min_aa_length: int) -> list[dict]:
    """枚举两条链上全部满足最短氨基酸数的 ORF。

    返回的每个 ORF 含 ``strand`` / ``start`` / ``end`` / ``wraps_origin`` /
    ``sequence`` / ``aa_length``，按（链方向、起点、氨基酸数）稳定排序。
    调用方负责输入校验（序列为合法 IUPAC、``min_aa_length >= 1``）。
    """
    length = len(sequence)
    circular = topology == "circular"
    orfs = []
    for strand, template in (
        ("+", sequence),
        ("-", reverse_complement(sequence)),
    ):
        for hit in _scan_strand(template, circular, min_aa_length):
            stop_end = hit["start"] + hit["nt_length"]  # 链局部未取模终点
            if strand == "+":
                start = hit["start"]
                end = stop_end
            else:
                # 反链区间 [a, a+nt) 映射回正链半开区间 [L-a-nt, L-a)
                start = length - stop_end
                end = length - hit["start"]
            if circular:
                start %= length
                end %= length
            orfs.append(
                {
                    "strand": strand,
                    "start": start,
                    "end": end,
                    "wraps_origin": circular and stop_end > length,
                    "sequence": hit["sequence"],
                    "aa_length": hit["aa_length"],
                }
            )
    orfs.sort(key=lambda orf: (orf["strand"], orf["start"], orf["aa_length"]))
    return orfs
