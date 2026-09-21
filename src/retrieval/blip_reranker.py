"""
src.retrieval.blip_reranker - BLIP 交叉注意力精排

## 它在检索流水线里的位置

    属性粗筛 → CLIP 双塔向量召回(top-N) → **BLIP ITM 精排(本模块)** → 融合 → 按轨迹去重

第一段是**双塔**：图像与文本各自编码，图像侧可离线全库预计算，因此能快速召回。
代价是两个塔之间没有交互，细粒度差异（同为白色轿车的两辆不同车）分不开。

第二段是**交叉注意力**：把图文拼在一起过模型，`ITM`（Image-Text Matching）头
输出匹配概率。代价是**必须图文成对输入，无法对全库扫描** —— 这正是它只能放在
第二阶段、只能作用在少量候选上的原因。

## 三个必须知道的坑（都踩过，写下来免得重踩）

1. **`BlipProcessor` 已从 transformers 顶层移除**（本项目实测 4.57.6）。
   `from transformers import BlipProcessor` 会失败；但
   `AutoProcessor.from_pretrained(...)` 仍能构造出它。模型类则要从
   `transformers.models.blip` 直接导入。

2. **ITM 必须图文一一对应**。传 1 张图配 N 条文本，会让视觉侧 batch=1、
   文本侧 batch=N，然后在 `modeling_blip_text.py` 抛一个**完全无关的 shape 错误**
   （`shape '[2, -1, 12, 64]' is invalid for input of size 443136`）。
   正确做法是 `images=[img] * len(texts)`。

3. **必须 `padding=True, truncation=True`**。多文本长度不一时，不 padding 会在
   组 batch 时直接抛 ValueError。

## 降级契约

`available` 为 False 时（未装依赖 / 模型缺失 / 加载失败），`score()` 返回全 None，
调用方**必须**退回纯向量检索，**并在结果里标明已降级** —— 不能静默换一套算法，
那是本项目反复清理过的错误（见 PLAN2 的 E6、PLAN3 的 A3）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from src.common.logger import get_logger

logger = get_logger("retrieval.blip_reranker")

# 默认权重：BLIP ITM base（COCO 上训练）。换更大的一档改这里即可。
DEFAULT_MODEL_NAME = "Salesforce/blip-itm-base-coco"


class BlipItmReranker:
    """
    BLIP Image-Text Matching 精排器

    使用方式:
        reranker = BlipItmReranker()
        if reranker.available:
            scores = reranker.score(["a.jpg", "b.jpg"], "白色轿车")
        # scores 里可能是 None（单张失败），调用方需逐项判空

    惰性加载：构造时不加载模型（沿用项目里"模块级懒加载"的惯例），
    首次 `score()` 才载入 —— 服务启动不受影响。
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: Optional[str] = None,
        batch_size: int = 8,
    ) -> None:
        """
        Args:
            model_name: HuggingFace 模型名或本地路径
            device: "cuda" / "cpu"；None 时自动选择
            batch_size: 单次前向的图文对数量（显存敏感）
        """
        self.model_name = model_name
        self.batch_size = max(1, batch_size)
        self._device = device
        self._model: Any = None
        self._processor: Any = None
        self._load_error: Optional[str] = None
        self._tried = False

    # ---------------- 加载 ----------------

    def _ensure_loaded(self) -> None:
        """首次调用时加载模型；失败记录下来（不抛），由 available 暴露"""
        if self._tried:
            return
        self._tried = True
        try:
            import torch
            # 坑 1：BlipProcessor 不在顶层命名空间，但 AutoProcessor 能构造它；
            #        模型类要从子模块直接导入。
            from transformers import AutoProcessor
            from transformers.models.blip import BlipForImageTextRetrieval

            if self._device is None:
                self._device = "cuda" if torch.cuda.is_available() else "cpu"

            self._processor = AutoProcessor.from_pretrained(self.model_name)
            self._model = BlipForImageTextRetrieval.from_pretrained(self.model_name)
            self._model.to(self._device)
            self._model.eval()
            logger.info(
                "BLIP ITM 精排器已加载: model=%s device=%s",
                self.model_name, self._device,
            )
        except Exception as e:  # 依赖缺失 / 权重不存在 / 显存不足，都走降级
            self._load_error = f"{type(e).__name__}: {e}"
            logger.warning(
                "BLIP ITM 精排器不可用，检索将退回纯向量路径: %s", self._load_error
            )

    @property
    def available(self) -> bool:
        """精排是否真的可用 —— 调用方必须据此决定要不要标注"已降级\""""
        self._ensure_loaded()
        return self._model is not None and self._processor is not None

    @property
    def load_error(self) -> Optional[str]:
        """加载失败的原因（可用时为 None）"""
        self._ensure_loaded()
        return self._load_error

    @property
    def device(self) -> Optional[str]:
        self._ensure_loaded()
        return self._device

    # ---------------- 打分 ----------------

    def score(
        self,
        image_paths: List[str],
        text: str,
    ) -> List[Optional[float]]:
        """
        对每个「图像 × 同一段文本」给出 ITM 匹配概率

        Args:
            image_paths: 图像文件路径列表（通常是候选的裁剪图）
            text: 查询文本（如 "白色轿车"）

        Returns:
            与 image_paths 等长的列表，每项为匹配概率 [0,1]；
            **读图失败、解码失败或模型不可用时该项为 None**（调用方需判空，
            不要用 0 冒充 —— 0 表示"确认不匹配"，与"没算出来"是两回事）。
        """
        if not self.available:
            return [None] * len(image_paths)
        if not image_paths:
            return []
        if not text or not text.strip():
            return [None] * len(image_paths)

        import torch
        from PIL import Image

        results: List[Optional[float]] = [None] * len(image_paths)

        # 逐批处理：显存与失败隔离（一批里某张图坏了，不该毁掉整批）
        for start in range(0, len(image_paths), self.batch_size):
            chunk = image_paths[start:start + self.batch_size]
            images, keep = [], []
            for i, path in enumerate(chunk):
                img = self._read_image(path, Image)
                if img is not None:
                    images.append(img)
                    keep.append(start + i)

            if not images:
                continue

            try:
                # 坑 2：图文一一对应 —— 每张图配一条同样的文本
                texts = [text] * len(images)
                # 坑 3：多文本需 padding/truncation 才能组 batch
                inputs = self._processor(
                    images=images,
                    text=texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )
                inputs = {k: v.to(self._device) for k, v in inputs.items()}

                with torch.no_grad():
                    out = self._model(**inputs, use_itm_head=True)

                # ITM 头输出 [batch, 2]（不匹配 / 匹配），取匹配的概率
                probs = torch.softmax(out.itm_score, dim=-1)[:, 1]
                for slot, prob in zip(keep, probs.tolist()):
                    results[slot] = float(prob)
            except Exception as e:
                logger.warning("BLIP ITM 批量打分失败（该批返回 None）: %s", e)

        return results

    @staticmethod
    def _read_image(path: str, image_mod) -> Any:
        """读图并转 RGB；失败返回 None（不抛，交由调用方判空）"""
        try:
            return image_mod.open(path).convert("RGB")
        except Exception:
            return None

    # ---------------- 融合 ----------------

    @staticmethod
    def normalize(itm_scores: List[Optional[float]]) -> List[Optional[float]]:
        """
        在**本次候选集内**做 min-max 归一化

        ## 为什么必须归一化（实测发现）

        BLIP ITM 的**绝对分跨查询不可比** —— 实测同一张图：
            查询 "白色轿车" → 0.0036
            查询 "黑色卡车" → 0.0293
        相差一个数量级。若直接按固定权重融合，有的查询里 ITM 几乎不起作用
        （`0.7*vec + 0.3*0.003` ≈ `0.7*vec`），有的查询里又明显起作用 ——
        融合行为随查询漂移，不可控。

        而精排的实际用途是**在同一查询内给候选排序**，所以集合内相对大小就够了。
        归一化后每个查询的 ITM 分都落在 [0,1] 且量纲一致，融合权重才有确定含义。

        Args:
            itm_scores: ITM 分（可含 None）

        Returns:
            归一化后的分（None 保持 None）；全部相等或只有一个非空时，非空项返回 0.5
            （表示"这一维没有区分度"，而不是把它们全判成 0）
        """
        valid = [s for s in itm_scores if s is not None]
        if not valid:
            return list(itm_scores)
        lo, hi = min(valid), max(valid)
        if hi - lo < 1e-12:
            return [None if s is None else 0.5 for s in itm_scores]
        return [None if s is None else (s - lo) / (hi - lo) for s in itm_scores]

    @classmethod
    def fuse(
        cls,
        vector_scores: List[float],
        itm_scores: List[Optional[float]],
        weight: float,
        normalize_itm: bool = True,
    ) -> List[float]:
        """
        把向量分与 ITM 分融合成最终排序分

        `weight` 是 **ITM 的权重**（0=纯向量，1=纯 ITM）。某项 ITM 为 None 时，
        该候选用纯向量分（**不做零填充** —— 那会把"没算出来"当成"不匹配"降权）。

        Args:
            vector_scores: 向量召回分（与候选一一对应）
            itm_scores: ITM 分，可为 None
            weight: ITM 权重 [0,1]
            normalize_itm: 是否先做集合内归一化。**默认 True**，
                理由见 `normalize()` —— ITM 绝对分跨查询不可比。

        Returns:
            融合分列表
        """
        w = min(max(weight, 0.0), 1.0)
        scores = cls.normalize(itm_scores) if normalize_itm else list(itm_scores)
        fused: List[float] = []
        for vec, itm in zip(vector_scores, scores):
            if itm is None:
                fused.append(float(vec))
            else:
                fused.append((1.0 - w) * float(vec) + w * float(itm))
        return fused


# ============================================================
# 模块级单例（沿用项目惯例：懒加载 + 进程内共享）
# ============================================================

_reranker: Optional[BlipItmReranker] = None


def get_blip_reranker(
    model_name: str = DEFAULT_MODEL_NAME,
    device: Optional[str] = None,
) -> BlipItmReranker:
    """获取全局 BLIP 精排器单例"""
    global _reranker
    if _reranker is None:
        _reranker = BlipItmReranker(model_name=model_name, device=device)
    return _reranker


def reset_blip_reranker() -> None:
    """重置单例（主要用于测试）"""
    global _reranker
    _reranker = None


def reranker_status(model_name: str = DEFAULT_MODEL_NAME) -> Dict[str, Any]:
    """
    精排器状态（供 /health 或排障用）

    Returns:
        available / device / model / error
    """
    r = get_blip_reranker(model_name=model_name)
    return {
        "available": r.available,
        "device": r.device,
        "model": r.model_name,
        "error": r.load_error,
    }
