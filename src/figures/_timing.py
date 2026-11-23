"""Figure of a smoke run: forecast time and peak GPU memory of every model.

Drawn from ``tables/timing.csv`` by the analysis of the template study 00-template/01-rq1-smoke.
"""

# pyright: reportUnknownMemberType=false
# (kept on purpose: matplotlib's own stubs type the **kwargs of every Axes and Figure method as
# Unknown, which no code here can fix; nothing else is silenced)

from pathlib import Path

import pandas as pd

from src.figures._style import DOUBLE_WIDTH, save_figure, use_paper_style


def draw_timing(timing: pd.DataFrame, *, pdf_path: Path, grey_path: Path) -> None:
    """Draw forecast seconds and peak GPU memory per model as horizontal grey bars from zero.

    Two panels share the model names on the y axis, the first model on top. A model is named
    by the last part of its id (``"chronos:amazon/chronos-2"`` is drawn as ``chronos-2``). The
    figure is 6.80 in wide and 0.6 in plus 0.25 in per model high, in the paper style.

    Parameters
    ----------
    timing : pd.DataFrame
        One row per model: ``model``, ``predict_seconds`` and ``peak_gpu_gb``.
    pdf_path : Path
        The PDF to write.
    grey_path : Path
        The greyscale PNG preview to write.
    """
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    names = [str(model).split("/")[-1].split(":")[-1] for model in timing["model"]]
    height = 0.6 + 0.25 * len(names)
    # the paper style is set only inside this block, so the caller's settings stay unchanged
    with mpl.rc_context():
        use_paper_style()
        figure, (left, right) = plt.subplots(1, 2, figsize=(DOUBLE_WIDTH, height), sharey=True)
        left.barh(names, timing["predict_seconds"], color="#888888")
        left.set_xlabel("Forecast time of all contexts (s)")
        right.barh(names, timing["peak_gpu_gb"], color="#888888")
        right.set_xlabel("Peak GPU memory (GB)")
        left.invert_yaxis()
        save_figure(figure, pdf_path, grey_path)
