"""
tests.test_blip_reranker - BLIP 交叉注意力精排（PLAN5-A）

分两层：
- **纯逻辑**（归一化、融合、降级契约）：不加载模型，任何机器上都能跑
- **需要模型**：标注在 `TestItmScoring` 里，模型不可用时自动 skip

重点守住三件事：
1. **ITM 绝对分跨查询不可比**（实测差一个数量级），所以融合前必须集合内归一化；
2. **算不出来 ≠ 不匹配**：ITM 为 None 时不能零填充降权；
3. **降级必须可见**：模型不可用时 `available` 为 False，调用方据此如实标注。
"""

from __future__ import annotations

import pytest

from src.retrieval.blip_reranker import BlipItmReranker, reranker_status


# ============================================================
# 纯逻辑：归一化
# ============================================================


class TestNormalize:
    def test_brings_different_queries_to_same_scale(self):
        """跨查询不可比的绝对分，归一化后量纲一致

        实测：同一张图，"白色轿车"给 0.0036、"黑色卡车"给 0.0293。
        不归一化的话，固定融合权重在不同查询下效果完全不同。
        """
        a = BlipItmReranker.normalize([0.0036, 0.0028, 0.0018])
        b = BlipItmReranker.normalize([0.0293, 0.0216, 0.0137])
        assert a == pytest.approx([1.0, 0.5556, 0.0], abs=1e-3)
        assert b == pytest.approx([1.0, 0.5064, 0.0], abs=1e-3)
        # 两者都落在 [0,1] 且首项为 1、末项为 0 —— 相对次序一致
        for out in (a, b):
            assert min(out) == 0.0 and max(out) == 1.0

    def test_all_equal_returns_neutral(self):
        """全相等时返回中性 0.5 —— 表示该维无区分度，而不是把它们全判成 0"""
        assert BlipItmReranker.normalize([0.5, 0.5, 0.5]) == [0.5, 0.5, 0.5]

    def test_preserves_none(self):
        """None 必须原样保留（它表示"没算出来"，不是 0）"""
        out = BlipItmReranker.normalize([0.9, None, 0.1])
        assert out[0] == 1.0 and out[1] is None and out[2] == 0.0

    def test_all_none(self):
        assert BlipItmReranker.normalize([None, None]) == [None, None]

    def test_empty(self):
        assert BlipItmReranker.normalize([]) == []


# ============================================================
# 纯逻辑：融合
# ============================================================


class TestFuse:
    def test_zero_weight_is_pure_vector(self):
        vec = [0.9, 0.5, 0.1]
        assert BlipItmReranker.fuse(vec, [1.0, 0.0, 0.5], 0.0) == pytest.approx(vec)

    def test_one_weight_is_pure_itm(self):
        out = BlipItmReranker.fuse([0.9, 0.5, 0.1], [1.0, 0.5, 0.0], 1.0)
        assert out == pytest.approx([1.0, 0.5, 0.0])

    def test_none_itm_keeps_vector_score(self):
        """**关键**：ITM 为 None 时保留纯向量分，不得零填充降权

        零填充会把"这张图没算出来"变成"这张图不匹配"，直接把候选压到末尾 ——
        那是把缺失当成证据。
        """
        out = BlipItmReranker.fuse([0.9, 0.8, 0.7], [0.5, None, 0.1], 0.5)
        assert out[1] == pytest.approx(0.8), "None 项应保留纯向量分"

    def test_weight_is_clamped(self):
        """越界权重被夹到 [0,1]，不抛异常"""
        vec = [0.5, 0.5]
        assert BlipItmReranker.fuse(vec, [1.0, 0.0], 5.0) == pytest.approx([1.0, 0.0])
        assert BlipItmReranker.fuse(vec, [1.0, 0.0], -3.0) == pytest.approx([0.5, 0.5])

    def test_normalize_can_be_disabled(self):
        """关掉归一化时按原值融合（供对照实验用）"""
        out = BlipItmReranker.fuse([0.9], [0.0036], 0.5, normalize_itm=False)
        assert out[0] == pytest.approx(0.5 * 0.9 + 0.5 * 0.0036)


# ============================================================
# 降级契约
# ============================================================


class TestDegradationContract:
    def test_score_returns_none_when_unavailable(self):
        """模型不可用时返回全 None，**不抛异常也不回填 0**"""
        r = BlipItmReranker(model_name="__not_a_real_model__")
        assert r.available is False
        assert r.load_error is not None
        assert r.score(["a.jpg", "b.jpg"], "白色轿车") == [None, None]

    def test_empty_input_short_circuits(self):
        r = BlipItmReranker(model_name="__not_a_real_model__")
        assert r.score([], "白色轿车") == []

    def test_blank_text_returns_none(self):
        r = BlipItmReranker(model_name="__not_a_real_model__")
        assert r.score(["a.jpg"], "   ") == [None]

    def test_availability_is_lazy(self):
        """构造时不应加载模型（沿用项目懒加载惯例，服务启动不受影响）"""
        r = BlipItmReranker()
        assert r._model is None and r._tried is False

    def test_status_reports_error(self):
        st = reranker_status(model_name="__not_a_real_model__")
        assert st["available"] is False
        assert st["error"]
        assert st["model"] == "__not_a_real_model__"


# ============================================================
# 需要真实模型（不可用时 skip）
# ============================================================


@pytest.fixture(scope="module")
def real_reranker():
    r = BlipItmReranker()
    if not r.available:
        pytest.skip(f"BLIP ITM 不可用: {r.load_error}")
    return r


def _a_real_crop():
    """取一张真实车辆裁剪图"""
    from pathlib import Path

    from src.storage.datastore import load_results

    data = load_results()
    if not data:
        return None
    for det in data.get("detections", []):
        p = det.get("crop_path")
        if p and Path(p).exists():
            return p
    return None


class TestItmScoring:
    def test_scores_are_valid_probabilities(self, real_reranker):
        """得分应是 [0,1] 的概率，不是 logits（取 softmax 后第 1 类）"""
        path = _a_real_crop()
        if path is None:
            pytest.skip("没有可用裁剪图")
        s = real_reranker.score([path], "白色轿车")[0]
        assert s is not None
        assert 0.0 <= s <= 1.0

    def test_missing_image_returns_none(self, real_reranker):
        """读不到的图返回 None，不抛异常也不填 0"""
        out = real_reranker.score(["__no_such_file__.jpg"], "白色轿车")
        assert out == [None]

    def test_batch_mixed_valid_and_invalid(self, real_reranker):
        """一批里混有坏图时，好的照常打分、坏的为 None（失败隔离）"""
        good = _a_real_crop()
        if good is None:
            pytest.skip("没有可用裁剪图")
        out = real_reranker.score([good, "__no_such_file__.jpg"], "白色轿车")
        assert out[0] is not None and out[1] is None

    def test_measured_discrimination_is_unusable(self, real_reranker):
        """
        **记录一个实测的负结果**：BLIP ITM 在本数据集的裁剪图上不可用于精排

        2026-09-21 两次独立测量（正确颜色描述 vs 随机错误颜色描述）：

        | 样本 | 判别正确率 |
        |---|---|
        | 80 张 | **51.2%**（≈ 随机） |
        | 40 张（另一抽样） | **22.5%**（**显著低于**随机） |

        两次都远离 50% 且方向相反 —— 说明分数**不是由"图文是否匹配"驱动的**，
        而是被**短语层面的偏置**主导。直接证据：同一辆白色轿车，
        `"白色轿车"` 打 0.0036，`"黑色卡车"` 却打 0.0293（高一个数量级）。

        根因与项目早先测到的"CLIP 向量精排不带来增益"同源：本数据集裁剪图中位数
        仅 **118×98 像素**，而 BLIP 在 COCO 高分辨率图上训练 —— 域差太大。

        **本测试的意义是防止有人（包括未来的我）在没有重新测量的情况下，
        宣称"接了 BLIP 精排所以效果更好"。** 断言是**单侧**的：要求它不能稳定地
        好于随机；一旦真的变好，这里会失败，提示重新测量并更新 PLAN5。
        """
        from pathlib import Path

        from src.storage.datastore import load_results

        data = load_results()
        if not data:
            pytest.skip("数据不可用")

        COLORS = ["白色", "黑色", "蓝色", "红色", "银色", "灰色"]
        VEHTYPES = ["轿车", "SUV", "卡车", "面包车", "皮卡"]

        pool = []
        for det in data["detections"]:
            a = det.get("attributes") or {}
            if a.get("color") in COLORS and a.get("vehicle_type") in VEHTYPES:
                p = det.get("crop_path")
                if p and Path(p).exists():
                    pool.append((p, a["color"], a["vehicle_type"]))
            if len(pool) >= 200:      # 采样上限，避免测试过慢
                break
        if len(pool) < 20:
            pytest.skip("样本不足")

        import random
        random.seed(42)
        sample = random.sample(pool, 40)

        wins = total = 0
        for path, color, vtype in sample:
            wrong = random.choice([c for c in COLORS if c != color])
            good = real_reranker.score([path], f"{color}{vtype}")[0]
            bad = real_reranker.score([path], f"{wrong}{vtype}")[0]
            if good is None or bad is None:
                continue
            total += 1
            wins += good > bad

        if total < 20:
            pytest.skip("有效样本不足")

        rate = wins / total
        assert rate < 0.70, (
            f"判别率 {rate:.1%}（{wins}/{total}）已明显好于随机 —— "
            f"说明 BLIP ITM 现在**在这份数据上有区分力了**。\n"
            f"若确实变了（换了数据 / 换了更大的模型 / 换了预处理），"
            f"请重新完整测量并更新 PLAN5 与本注释。\n"
            f"历史基线：80 样本 51.2%、40 样本 22.5%（2026-09-21，均不可用）。"
        )
