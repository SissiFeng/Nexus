"""Cross-model consistency analysis — check if different models agree."""

from __future__ import annotations


class CrossModelConsistency:
    """Check whether different models agree on conclusions.

    Computes pairwise rank correlations, ensemble confidence, and
    identifies regions of disagreement across model predictions.
    """

    def kendall_tau(self, ranking1: list, ranking2: list) -> float:
        """Compute Kendall rank correlation coefficient.

        ``tau = (concordant - discordant) / (n * (n-1) / 2)``

        Runs in O(n^2), acceptable for n < 100.

        Parameters
        ----------
        ranking1 : list
            First ranking (list of item names in rank order).
        ranking2 : list
            Second ranking (list of item names in rank order).

        Returns
        -------
        float
            Kendall tau in [-1, 1]. 1 means perfect agreement,
            -1 means perfectly reversed.
        """
        # Build rank maps: item -> position
        if not ranking1 or not ranking2:
            return 0.0

        items1 = {item: i for i, item in enumerate(ranking1)}
        items2 = {item: i for i, item in enumerate(ranking2)}

        # Use only items present in both rankings
        common = [item for item in ranking1 if item in items2]
        n = len(common)

        if n < 2:
            return 0.0

        # Assign ranks in each ranking (0-based position among common items)
        rank1 = [items1[item] for item in common]
        rank2 = [items2[item] for item in common]

        concordant = 0
        discordant = 0

        for i in range(n):
            for j in range(i + 1, n):
                diff1 = rank1[i] - rank1[j]
                diff2 = rank2[i] - rank2[j]
                product = diff1 * diff2
                if product > 0:
                    concordant += 1
                elif product < 0:
                    discordant += 1
                # ties (product == 0) are neither concordant nor discordant

        total_pairs = n * (n - 1) / 2
        if total_pairs == 0:
            return 0.0

        return (concordant - discordant) / total_pairs

    def model_agreement(
        self,
        model_rankings: dict[str, list],
        metric: str = "kendall",
    ) -> dict:
        """Compute pairwise agreement between model rankings.

        Parameters
        ----------
        model_rankings : dict[str, list]
            Mapping from model name to a list of item names in rank order.
            E.g. ``{"GP": ["A", "B", "C"], "RF": ["B", "A", "C"]}``.
        metric : str
            Rank correlation metric. Currently only ``"kendall"`` is supported.

        Returns
        -------
        dict
            ``"pairwise"`` (dict): ``(model_a, model_b) -> tau`` for all pairs.
            ``"mean_agreement"`` (float): average pairwise tau.
        """
        model_names = list(model_rankings.keys())
        n_models = len(model_names)

        pairwise: dict[tuple[str, str], float] = {}
        tau_sum = 0.0
        n_pairs = 0

        for i in range(n_models):
            for j in range(i + 1, n_models):
                name_a = model_names[i]
                name_b = model_names[j]
                tau = self.kendall_tau(model_rankings[name_a], model_rankings[name_b])
                pairwise[(name_a, name_b)] = tau
                tau_sum += tau
                n_pairs += 1

        mean_agreement = tau_sum / n_pairs if n_pairs > 0 else 0.0

        return {
            "pairwise": pairwise,
            "mean_agreement": mean_agreement,
        }

    def ensemble_confidence(
        self,
        model_predictions: dict[str, list[float]],
        names: list[str],
    ) -> dict:
        """Compute ensemble statistics to identify where models agree or disagree.

        Parameters
        ----------
        model_predictions : dict[str, list[float]]
            Mapping from model name to a list of predicted values, one per item.
        names : list[str]
            Names corresponding to each item.

        Returns
        -------
        dict
            ``"per_item"`` (list[dict]): for each item: ``name``, ``mean``,
            ``std``, ``agreement_score``.
            ``"overall_agreement"`` (float): mean agreement score across items.
        """
        model_names = list(model_predictions.keys())
        n_models = len(model_names)
        n_items = len(names)

        if n_models == 0 or n_items == 0:
            return {"per_item": [], "overall_agreement": 0.0}

        per_item: list[dict] = []
        agreement_sum = 0.0

        for idx in range(n_items):
            preds = [model_predictions[m][idx] for m in model_names]
            mean_pred = sum(preds) / n_models
            var_pred = sum((p - mean_pred) ** 2 for p in preds) / max(n_models - 1, 1)
            std_pred = var_pred ** 0.5

            # Agreement score: 1 / (1 + std) — high when std is low
            agreement = 1.0 / (1.0 + std_pred)

            per_item.append({
                "name": names[idx],
                "mean": mean_pred,
                "std": std_pred,
                "agreement_score": agreement,
            })
            agreement_sum += agreement

        overall_agreement = agreement_sum / n_items

        return {
            "per_item": per_item,
            "overall_agreement": overall_agreement,
        }

    def disagreement_regions(
        self,
        model_predictions: dict[str, list[float]],
        X: list[list[float]],
        names: list[str] | None = None,
    ) -> list[dict]:
        """Identify items where models disagree the most.

        Returns items sorted by disagreement score (highest first).

        Parameters
        ----------
        model_predictions : dict[str, list[float]]
            Mapping from model name to predicted values.
        X : list[list[float]]
            Feature matrix for each item.
        names : list[str] or None
            Optional item names. Defaults to ``"item_0"``, ``"item_1"``, etc.

        Returns
        -------
        list[dict]
            Sorted list of dicts with ``"name"``, ``"X"``, ``"predictions"``,
            and ``"disagreement_score"`` (std across models).
        """
        model_names = list(model_predictions.keys())
        n_models = len(model_names)
        n_items = len(X)

        if names is None:
            names = [f"item_{i}" for i in range(n_items)]

        if n_models == 0 or n_items == 0:
            return []

        results: list[dict] = []

        for idx in range(n_items):
            preds = {m: model_predictions[m][idx] for m in model_names}
            pred_values = list(preds.values())
            mean_pred = sum(pred_values) / n_models
            var_pred = sum((p - mean_pred) ** 2 for p in pred_values) / max(n_models - 1, 1)
            std_pred = var_pred ** 0.5

            results.append({
                "name": names[idx],
                "X": X[idx],
                "predictions": preds,
                "disagreement_score": std_pred,
            })

        # Sort by disagreement (highest first)
        results.sort(key=lambda d: d["disagreement_score"], reverse=True)

        return results
