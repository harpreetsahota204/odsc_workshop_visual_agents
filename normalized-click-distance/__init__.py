import numpy as np
import fiftyone as fo
import fiftyone.operators as foo


class NormalizedClickDistance(foo.EvaluationMetric):
    @property
    def config(self):
        return foo.EvaluationMetricConfig(
            name="normalized_click_distance",
            label="Normalized Click Distance",
            description="Normalized Euclidean distance between predicted and ground truth click points (as % of diagonal)",
            aggregate_key="mean_norm_dist",
            unlisted=False,
        )

    def compute(
        self,
        samples,
        results,
        pred_field="predicted_keypoints",
        gt_field="keypoints",
        thresholds=None,
    ):
        """
        Args:
            samples: FiftyOne samples view
            results: Evaluation results object
            pred_field: Field with predicted (x, y) in [0-1] normalized coords
            gt_field: Field with ground truth (x, y) in [0-1] normalized coords
            thresholds: List of accuracy thresholds as fraction of diagonal
        """
        if thresholds is None:
            thresholds = [0.025, 0.05, 0.10]

        dataset = samples._dataset
        eval_key = results.key
        distance_field = f"{eval_key}_{self.config.name}"

        # Add per-sample distance field
        if distance_field not in dataset.get_field_schema():
            dataset.add_sample_field(distance_field, fo.FloatField)

        distances = []

        for sample in samples:
            pred = sample[pred_field]
            gt = sample[gt_field]

            if pred is None or gt is None:
                sample[distance_field] = None
                sample.save()
                continue

            # Handle if stored as list, tuple, or keypoint
            if hasattr(pred, "points"):
                pred = pred.points[0]  # fo.Keypoint
            if hasattr(gt, "points"):
                gt = gt.points[0]

            pred_arr = np.array(pred[:2])  # Take x, y only
            gt_arr = np.array(gt[:2])

            # Euclidean in normalized [0-1] space, divide by diagonal (sqrt(2))
            dist = np.linalg.norm(pred_arr - gt_arr)
            norm_dist = dist / np.sqrt(2)

            distances.append(norm_dist)
            sample[distance_field] = float(norm_dist)
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

        # Accuracy at thresholds
        for thresh in thresholds:
            key = f"acc@{thresh*100:.1f}pct"
            metrics[key] = float(np.mean(distances <= thresh))

        return metrics

    def get_fields(self, samples, config, eval_key):
        return [f"{eval_key}_{self.config.name}"]