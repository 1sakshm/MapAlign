import numpy as np
import pandas as pd
import pytest

from mapalign import extensions as ext
from mapalign import robustness as rb
from mapalign.picktool import pending_rows, record


def test_polynomial_and_tps_reproduce_a_known_mapping():
    rng = np.random.default_rng(0)
    src = rng.uniform(0, 1000, (200, 2))
    dst = np.column_stack([3 + 1.01 * src[:, 0] + 2e-5 * src[:, 0] * src[:, 1], -7 + 0.99 * src[:, 1] + 1e-5 * src[:, 0] ** 2])
    test = rng.uniform(100, 900, (50, 2))
    truth = np.column_stack([3 + 1.01 * test[:, 0] + 2e-5 * test[:, 0] * test[:, 1], -7 + 0.99 * test[:, 1] + 1e-5 * test[:, 0] ** 2])
    assert np.abs(ext._model("poly:2", src, dst)(test) - truth).max() < 1e-6
    assert np.abs(ext._model("tps:0", src, dst)(test) - truth).max() < 0.05
    assert np.abs(ext._model("tps:0", src, dst)(src) - dst).max() < 1e-6  # unsmoothed TPS interpolates
    with pytest.raises(ValueError):
        ext._model("poly:3", src[:5], dst[:5])


def test_block_cv_prefers_the_true_model_order():
    rng = np.random.default_rng(1)
    src = rng.uniform(0, 1000, (400, 2))
    dst = src * 1.02 + 5 + rng.normal(0, 1.0, src.shape)  # affine + noise
    cv = {m: ext._block_cv(src, dst, m, 10.0) for m in ("affine", "poly:3", "tps:0")}
    assert cv["affine"] < cv["poly:3"] and cv["affine"] < cv["tps:0"]


def test_picktool_records_and_resumes():
    df = pd.DataFrame({"point_id": ["A", "B", "C"], "picker": "", "historical_x_px": np.nan, "historical_y_px": np.nan,
                       "seconds_spent": np.nan, "confident_yes_no": "", "notes": ""})
    record(df, 0, (10.04, 20.06), 12.34, "me", True)
    record(df, 1, None, 5, "me", True, skipped=True)
    assert pending_rows(df) == [2]
    assert df.loc[0, ["historical_x_px", "historical_y_px", "seconds_spent"]].tolist() == [10.0, 20.1, 12.3]
    assert df.loc[1, "notes"].startswith("SKIPPED")


@pytest.fixture(scope="module")
def ctx():
    return rb.Ctx()


def test_picker_with_identical_picks_reproduces_the_frozen_results(ctx, tmp_path, monkeypatch):
    # Feeding the original picks back in as a "re-pick" must change nothing: zero picker difference, and the
    # refitted Method C (same QC rule) must reproduce the published held-out errors.
    monkeypatch.setattr(rb, "OUT", tmp_path)
    ev = ctx.ev.assign(seconds_spent=30)[["point_id", "historical_x_px", "historical_y_px", "seconds_spent"]]
    fit = ctx.fit.assign(seconds_spent=60)[["point_id", "historical_x_px", "historical_y_px", "seconds_spent"]]
    ev.to_csv(tmp_path / "ev.csv", index=False)
    fit.to_csv(tmp_path / "fit.csv", index=False)
    out = rb.picker(ctx, str(tmp_path / "ev.csv"), str(tmp_path / "fit.csv"))
    assert out["evaluation"]["picker_difference_px"]["max"] == 0
    assert sorted(out["fitting"]["used"]) == sorted(ctx.runs["manual"]["fitting_point_ids"])
    assert out["fitting"]["heldout_original_eval_picks"]["rmse_m"] == pytest.approx(
        rb._summ(ctx.errors("homography", "manual"))["rmse_m"], abs=0.01)
    assert out["fitting"]["human_seconds_total"] == 60 * len(ctx.fit)


def test_extra_points_on_the_original_points_match_the_evaluation(ctx, tmp_path, monkeypatch):
    monkeypatch.setattr(rb, "OUT", tmp_path)
    ctx.ev.assign(seconds_spent=1, notes="")[["point_id", "historical_x_px", "historical_y_px", "true_easting_m",
                                             "true_northing_m", "seconds_spent", "notes"]].to_csv(tmp_path / "x.csv", index=False)
    out = rb.extra_points(ctx, str(tmp_path / "x.csv"))
    assert out["loftr"]["supplementary"]["rmse_m"] == pytest.approx(rb._summ(ctx.errors("homography", "loftr"))["rmse_m"], abs=0.01)
    assert out["n_points"] == len(ctx.ev)


def test_pickweb_crop_pads_outside_image():
    import numpy as np

    from mapalign.pickweb import crop

    img = np.arange(5 * 5 * 3, dtype=np.uint8).reshape(5, 5, 3)
    out, x0, y0 = crop(img, 1, 1, 2)
    assert out.shape == (4, 4, 3) and (x0, y0) == (-1, -1)
    assert (out[0] == 255).all() and (out[:, 0] == 255).all()
    assert (out[1:, 1:] == img[:3, :3]).all()
