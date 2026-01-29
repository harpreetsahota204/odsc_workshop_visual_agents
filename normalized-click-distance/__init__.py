import numpy as np
import fiftyone as fo
import fiftyone.operators as foo
from fiftyone.operators import types


class NormalizedClickDistance(foo.EvaluationMetric):
    @property
    def config(self):
        return foo.EvaluationMetricConfig(
            name="normalized_click_distance",
            label="Normalized Click Distance",
            description="Normalized Euclidean distance between predicted and ground truth click points",
            aggregate_key="mean_norm_dist",
            unlisted=False,
        )

    def resolve_input(self, ctx, inputs):
        inputs = types.Object()
        inputs.str(
            "pred_field",
            label="Prediction field",
            description="Field containing predicted (x, y) coordinates [0-1]",
            default="pred_point",
            required=True,
        )
        inputs.str(
            "gt_field",
            label="Ground truth field",
            description="Field containing ground truth (x, y) coordinates [0-1]",
            default="gt_point",
            required=True,
        )
        inputs.list(
            "thresholds",
            types.Float(),
            label="Accuracy thresholds",
            description="Thresholds as fraction of diagonal for Acc@ metrics",
            default=[0.025, 0.05, 0.10],
        )
        return types.Property(inputs)

    def compute(
        self,
        samples,
        results,
        pred_field="pred_point",
        gt_field="gt_point",
        thresholds=None,
    ):
        if thresholds is None:
            thresholds = [0.025, 0.05, 0.10]

        dataset = samples._dataset
        eval_key = results.key
        metric_field = f"{eval_key}_{self.config.name}"

        # Add per-sample field
        if metric_field not in dataset.get_field_schema():
            dataset.add_sample_field(metric_field, fo.FloatField)

        distances = []

        for sample in samples:
            pred = sample[pred_field]
            gt = sample[gt_field]

            if pred is None or gt is None:
                sample[metric_field] = None
                sample.save()
                continue

            # Handle fo.Keypoint if needed
            if hasattr(pred, "points"):
                pred = pred.points[0]
            if hasattr(gt, "points"):
                gt = gt.points[0]

            pred_arr = np.array(pred[:2])
            gt_arr = np.array(gt[:2])

            # Normalize by diagonal of unit square
            dist = np.linalg.norm(pred_arr - gt_arr)
            norm_dist = dist / np.sqrt(2)

            distances.append(norm_dist)
            sample[metric_field] = float(norm_dist)
            sample.save()

        if not distances:
            return {"mean_norm_dist": None, "num_samples": 0}

        distances = np.array(distances)

        metrics = {
            "mean_norm_dist": float(np.mean(distances)),
            "median_norm_dist": float(np.median(distances)),
            "std_norm_dist": float(np.std(distances)),
            "num_samples": len(distances),
        }

        for thresh in thresholds:
            metrics[f"acc@{thresh*100:.1f}%"] = float(np.mean(distances <= thresh))

        return metrics

    def get_fields(self, samples, config, eval_key):
        return [f"{eval_key}_{self.config.name}"]