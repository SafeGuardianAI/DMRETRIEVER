"""DMRetriever-based cross-source re-ranker.

Uses transformers + mean-pooled, L2-normalized embeddings so it works
across the entire DMRetriever family (33M MiniLM, 109M/335M BERT,
596M Qwen3, etc.) without source-model-specific glue.
"""

from typing import List, Optional

import torch
from transformers import AutoModel, AutoTokenizer

from .sources.base import SourceResult


def _mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


class DMRetrieverRanker:
    def __init__(
        self,
        model_id: str = "DMIR01/DMRetriever-33M",
        device: Optional[str] = None,
        max_length: int = 512,
    ):
        self.model_id = model_id
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()

    @torch.no_grad()
    def embed(self, texts: List[str], batch_size: int = 16) -> torch.Tensor:
        if not texts:
            hidden = getattr(self.model.config, "hidden_size", 384)
            return torch.zeros(0, hidden, device=self.device)
        chunks: List[torch.Tensor] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            out = self.model(**enc)
            pooled = _mean_pool(out.last_hidden_state, enc["attention_mask"])
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            chunks.append(pooled)
        return torch.cat(chunks, dim=0)

    def rerank(
        self,
        query: str,
        results: List[SourceResult],
        top_k: Optional[int] = None,
        batch_size: int = 16,
    ) -> List[SourceResult]:
        if not results or not query.strip():
            return results
        q_emb = self.embed([query], batch_size=1)
        docs = [(r.title + "\n" + r.snippet).strip() or r.url for r in results]
        d_emb = self.embed(docs, batch_size=batch_size)
        scores = (d_emb @ q_emb.T).squeeze(-1).cpu().tolist()
        for r, s in zip(results, scores):
            r.score = float(s)
        results.sort(key=lambda r: (r.score if r.score is not None else 0.0), reverse=True)
        if top_k is not None:
            results = results[:top_k]
        return results
