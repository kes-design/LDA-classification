"""
LDA Classifier - Streamlit web app
=====================================
Upload a "database" Excel file (each sheet = one class, rows = samples,
columns = feature measurements) to train a Linear Discriminant Analysis
model, then upload an "unknown samples" Excel file to classify each row.

Run with:
    streamlit run lda_app.py

Requires: streamlit, pandas, numpy, scikit-learn, plotly, openpyxl, xlsxwriter
Install with:
    pip install streamlit pandas numpy scikit-learn plotly openpyxl xlsxwriter
"""

import io

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

try:
    import xlsxwriter  # noqa: F401
    EXCEL_ENGINE = "xlsxwriter"
except ImportError:
    EXCEL_ENGINE = "openpyxl"

st.set_page_config(page_title="LDA Classifier", layout="wide")
st.title("LDA Classifier")

st.markdown(
    """
Upload a **database file** where each sheet is a class (e.g. `Flatglas`,
`Verpakkingsglas`, `Telefoonglas`) and rows are samples with feature
concentrations as columns. Then upload an **unknown samples file** with
the same feature columns to classify.
"""
)


# =================================================================
# Helpers
# =================================================================
def _clip_line_to_box(a, b, c, x_lo, x_hi, y_lo, y_hi):
    """Points where line a*x + b*y + c = 0 crosses the given box, or None
    if the line doesn't cross the visible area."""
    pts = []
    if abs(b) > 1e-12:
        for x in (x_lo, x_hi):
            y = -(a * x + c) / b
            if y_lo - 1e-9 <= y <= y_hi + 1e-9:
                pts.append((x, y))
    if abs(a) > 1e-12:
        for y in (y_lo, y_hi):
            x = -(b * y + c) / a
            if x_lo - 1e-9 <= x <= x_hi + 1e-9:
                pts.append((x, y))
    uniq = []
    for p in pts:
        if not any(np.hypot(p[0] - q[0], p[1] - q[1]) < 1e-9 for q in uniq):
            uniq.append(p)
    if len(uniq) < 2:
        return None
    best_pair, best_d = None, -1
    for ii in range(len(uniq)):
        for jj in range(ii + 1, len(uniq)):
            d = np.hypot(uniq[ii][0] - uniq[jj][0], uniq[ii][1] - uniq[jj][1])
            if d > best_d:
                best_d, best_pair = d, (uniq[ii], uniq[jj])
    return best_pair


def _write_sheet_with_notes(writer, df, sheet_name, notes, index=True, index_label=None):
    """Write a dataframe to an Excel sheet with explanatory text lines placed
    above it (one line per row), so the sheet is self-contained without
    needing the app for context. Works with both the xlsxwriter and openpyxl
    engines, whichever ended up available as EXCEL_ENGINE."""
    startrow = len(notes) + 1  # +1 blank row between the notes and the table
    df.to_excel(writer, sheet_name=sheet_name, startrow=startrow, index=index, index_label=index_label)
    worksheet = writer.sheets[sheet_name]
    if EXCEL_ENGINE == "xlsxwriter":
        wrap_fmt = writer.book.add_format({"text_wrap": True, "valign": "top"})
        for i, line in enumerate(notes):
            worksheet.write(i, 0, line, wrap_fmt)
        worksheet.set_column(0, 0, 100)
    else:  # openpyxl
        for i, line in enumerate(notes):
            worksheet.cell(row=i + 1, column=1, value=line)
        worksheet.column_dimensions["A"].width = 100


def fit_extra_axis(model, X_scaled, y):
    """Second plot axis for the 2-class case: PC1 of the within-class
    variation, made orthogonal to the LD1 direction."""
    if X_scaled.shape[1] < 2:
        return None
    w = model.scalings_[:, 0]
    w = w / np.linalg.norm(w)
    resid = X_scaled.copy()
    for c in np.unique(y):
        resid[y == c] -= resid[y == c].mean(axis=0)
    resid = resid - np.outer(resid @ w, w)
    if not np.any(resid):
        return None
    v = PCA(n_components=1).fit(resid).components_[0]
    return {"vec": v, "mean": X_scaled.mean(axis=0)}


def project_2d(model, X_scaled, extra_axis):
    """Return an (n, 2) array: LD1 and either LD2 or the extra axis."""
    ld = model.transform(X_scaled)
    if ld.shape[1] >= 2:
        return ld[:, :2]
    if extra_axis is None:
        return np.column_stack([ld[:, 0], np.zeros(len(ld))])
    return np.column_stack([ld[:, 0], (X_scaled - extra_axis["mean"]) @ extra_axis["vec"]])


def build_projection_figure(
    labels, class_colors, plot_df, class_means_ld, region_means, priors_arr,
    y_label="LD2", unknown_df=None, highlight_idx=None,
):
    """2D projection with shaded decision regions and pairwise boundaries.
    class_means_ld: centroid positions (for the markers).
    region_means: positions used to compute regions/boundaries (for the
        2-class case the 2nd coordinate is 0, so boundaries are vertical).
    plot_df / unknown_df use columns 'LD1' and 'LD2' (LD2 may hold the extra axis).
    unknown_df additionally needs 'Sample' and 'Predicted_class' and a plain
    0..n-1 RangeIndex. highlight_idx = positional rows of unknown_df to highlight."""
    boundary_formulas = []
    has_unknown = unknown_df is not None and len(unknown_df) > 0
    highlight_idx = set(highlight_idx) if highlight_idx else set()
    hover_tail = f"<br>LD1: %{{x:.3f}}<br>{y_label}: %{{y:.3f}}<extra></extra>"

    def _unknown_traces():
        n = len(unknown_df)
        hover = unknown_df["Sample"].to_numpy()
        pred = unknown_df["Predicted_class"].to_numpy()
        x_all, y_all = unknown_df["LD1"].to_numpy(), unknown_df["LD2"].to_numpy()
        is_hl = np.array([i in highlight_idx for i in range(n)])
        specs = [
            (~is_hl, "Unknown sample",
             dict(symbol="circle", size=10, color="white", line=dict(width=2, color="black"))),
            (is_hl, "Selected sample",
             dict(symbol="circle", size=18, color="#FFD700", line=dict(width=3, color="red"))),
        ]
        out = []
        for mask, name, marker in specs:
            if mask.any():
                out.append(go.Scatter(
                    x=x_all[mask], y=y_all[mask], mode="markers", marker=marker, name=name,
                    customdata=np.stack([hover[mask], pred[mask]], axis=-1),
                    hovertemplate="Sample: %{customdata[0]}<br>Predicted: %{customdata[1]}" + hover_tail,
                ))
        return out

    all_ld1, all_ld2 = list(plot_df["LD1"]), list(plot_df["LD2"])
    if has_unknown:
        all_ld1 += list(unknown_df["LD1"])
        all_ld2 += list(unknown_df["LD2"])
    ld1_lo, ld1_hi = min(all_ld1), max(all_ld1)
    ld2_lo, ld2_hi = min(all_ld2), max(all_ld2)
    pad1 = 0.15 * (ld1_hi - ld1_lo) if ld1_hi > ld1_lo else 1.0
    pad2 = 0.15 * (ld2_hi - ld2_lo) if ld2_hi > ld2_lo else 1.0
    ld1_lo, ld1_hi = ld1_lo - pad1, ld1_hi + pad1
    ld2_lo, ld2_hi = ld2_lo - pad2, ld2_hi + pad2

    # Decision regions: nearest centroid (with priors) on a grid
    grid_res = 300
    ld1_grid = np.linspace(ld1_lo, ld1_hi, grid_res)
    ld2_grid = np.linspace(ld2_lo, ld2_hi, grid_res)
    LD1_mesh, LD2_mesh = np.meshgrid(ld1_grid, ld2_grid)
    mesh_points = np.stack([LD1_mesh.ravel(), LD2_mesh.ravel()], axis=1)
    diffs = mesh_points[:, None, :] - region_means[None, :, :]
    scores = -0.5 * np.sum(diffs ** 2, axis=2) + np.log(priors_arr)[None, :]
    region_idx = np.argmax(scores, axis=1).reshape(LD1_mesh.shape)
    n_cls = len(region_means)

    colorscale = []
    for i, cls in enumerate(labels):
        colorscale.append([i / n_cls, class_colors[cls]])
        colorscale.append([(i + 1) / n_cls, class_colors[cls]])

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        x=ld1_grid, y=ld2_grid, z=region_idx, zmin=-0.5, zmax=n_cls - 0.5,
        colorscale=colorscale, showscale=False, opacity=0.30, zsmooth="best", hoverinfo="skip",
    ))
    for cls in labels:
        sub = plot_df[plot_df["Class"] == cls]
        fig.add_trace(go.Scatter(
            x=sub["LD1"], y=sub["LD2"], mode="markers",
            marker=dict(color=class_colors[cls], size=7, line=dict(width=0.5, color="white")),
            name=cls, hovertemplate=f"Class: {cls}<br>LD1: %{{x:.3f}}<br>{y_label}: %{{y:.3f}}<extra></extra>",
        ))
    for i, cls in enumerate(labels):
        m = class_means_ld[i]
        fig.add_trace(go.Scatter(
            x=[m[0]], y=[m[1]], mode="markers+text",
            marker=dict(color="black", size=11, line=dict(width=1.5, color="white")),
            text=[f"{cls} centroid"], textposition="top center",
            showlegend=False, hoverinfo="skip",
        ))

    # Only draw a boundary between classes whose regions actually touch
    adjacent = np.zeros((n_cls, n_cls), dtype=bool)
    for a, b in ((region_idx[:, :-1], region_idx[:, 1:]), (region_idx[:-1, :], region_idx[1:, :])):
        d = a != b
        for r1, r2 in zip(a[d], b[d]):
            adjacent[r1, r2] = adjacent[r2, r1] = True

    for bi in range(n_cls):
        for bj in range(bi + 1, n_cls):
            if not adjacent[bi, bj]:
                continue
            mi, mj = region_means[bi], region_means[bj]
            a_coef, b_coef = mi[0] - mj[0], mi[1] - mj[1]
            c_coef = -0.5 * (np.sum(mi ** 2) - np.sum(mj ** 2)) + np.log(priors_arr[bi] / priors_arr[bj])
            seg = _clip_line_to_box(a_coef, b_coef, c_coef, ld1_lo, ld1_hi, ld2_lo, ld2_hi)
            if seg is None:
                continue
            (x0, y0), (x1, y1) = seg
            fig.add_trace(go.Scatter(
                x=[x0, x1], y=[y0, y1], mode="lines",
                line=dict(color="black", width=1.5, dash="dash"),
                showlegend=False, hoverinfo="skip",
            ))
            fig.add_annotation(
                x=(x0 + x1) / 2, y=(y0 + y1) / 2, text=f"{labels[bi]} | {labels[bj]}",
                showarrow=False, font=dict(size=10, color="black"), bgcolor="rgba(255,255,255,0.7)",
            )
            if abs(b_coef) > 1e-9:
                slope, icpt = -a_coef / b_coef, -c_coef / b_coef
                formula = f"{y_label} = ({slope:.4f})\u00b7LD1 + ({icpt:.4f})"
            else:
                x_b = -c_coef / a_coef
                first, second = (labels[bi], labels[bj]) if a_coef > 0 else (labels[bj], labels[bi])
                formula = (f"LD1 = {x_b:.4f}  (vertical line; LD1 > {x_b:.4f} \u2192 {first}, "
                           f"LD1 < {x_b:.4f} \u2192 {second})")
            boundary_formulas.append({"Boundary": f"{labels[bi]} vs {labels[bj]}",
                                      "Formula (plot coordinates)": formula})

    if has_unknown:
        for tr in _unknown_traces():
            fig.add_trace(tr)

    fig.update_layout(title="LDA projection (shaded decision regions)",
                      xaxis_title="LD1", yaxis_title=y_label, height=550, legend_title="Class")
    return fig, boundary_formulas


# ---------------------------------------------------------------
# Session state so results survive re-runs / interactions
# ---------------------------------------------------------------
_STATE_KEYS = [
    "model", "scaler", "label_encoder", "feature_cols", "class_labels",
    "coef_raw", "intercept_raw", "class_order",
    "ld_formula_df", "formula_df", "boundary_df",
    "plot_df", "class_colors", "class_means_ld", "region_means", "priors_arr",
    "extra_axis", "y_label",
]
for _k in _STATE_KEYS:
    st.session_state.setdefault(_k, None)


# =================================================================
# STEP 1 — Upload database & train
# =================================================================
st.header("1. Train on your database")

db_file = st.file_uploader("Database Excel file (.xlsx)", type=["xlsx", "xls"], key="db")

if db_file is not None:
    try:
        xls = pd.ExcelFile(db_file)
    except Exception as e:
        st.error(f"Could not read Excel file: {e}")
        st.stop()

    sheet_names = xls.sheet_names
    st.write(f"Found {len(sheet_names)} sheets (= classes): {', '.join(sheet_names)}")

    selected_sheets = st.multiselect(
        "Sheets to use as classes", sheet_names, default=sheet_names
    )

    if len(selected_sheets) < 2:
        st.warning("Select at least 2 sheets/classes to run LDA.")
    else:
        # Peek at the columns of the first selected sheet to offer an ID-column choice
        preview_cols = list(pd.read_excel(xls, sheet_name=selected_sheets[0], nrows=0).columns)
        id_col_choice = st.selectbox(
            "Sample ID column to exclude from training (not a measurement)",
            options=["(none)"] + preview_cols,
            index=1 if preview_cols else 0,
            help="Pick the column that identifies each sample (e.g. Sample_ID), "
                 "so it isn't treated as a feature. Defaults to the first column.",
        )

        test_size = st.slider("Test set size (for evaluation)", 0.1, 0.5, 0.3, 0.05)
        random_state = st.number_input("Random state", value=42, step=1)

        if st.button("Train model", type="primary"):
            frames = []
            for sheet in selected_sheets:
                sheet_df = pd.read_excel(xls, sheet_name=sheet)
                sheet_df = sheet_df.dropna(how="all")
                sheet_df["label"] = sheet
                frames.append(sheet_df)
            data = pd.concat(frames, ignore_index=True)

            excluded_cols = {"label"}
            if id_col_choice != "(none)":
                excluded_cols.add(id_col_choice)
            feature_cols = [c for c in data.columns if c not in excluded_cols]

            X = data[feature_cols].apply(pd.to_numeric, errors="coerce")
            y_raw = data["label"]

            mask = X.notna().all(axis=1) & y_raw.notna()
            n_dropped = (~mask).sum()
            if n_dropped:
                st.warning(f"Dropped {n_dropped} row(s) with missing/non-numeric values.")
            X, y_raw = X[mask], y_raw[mask]

            st.write(f"Using {len(feature_cols)} feature columns: {', '.join(feature_cols)}")
            st.write("Class counts:")
            st.dataframe(y_raw.value_counts().rename("samples"))

            le = LabelEncoder()
            y = le.fit_transform(y_raw)
            labels = list(le.classes_)

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            n_classes = len(labels)
            max_components = min(len(feature_cols), n_classes - 1)
            n_components = max(1, min(2, max_components))

            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
                )
            except ValueError as e:
                st.error(
                    f"Could not create a stratified train/test split: {e}\n\n"
                    "This usually means a class has too few samples for the chosen test size."
                )
                st.stop()

            eval_lda = LinearDiscriminantAnalysis(n_components=n_components)
            eval_lda.fit(X_train, y_train)
            y_pred = eval_lda.predict(X_test)

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Test accuracy", f"{accuracy_score(y_test, y_pred):.1%}")
                st.text("Classification report:")
                st.text(classification_report(y_test, y_pred, target_names=labels))
            with col2:
                cm = confusion_matrix(y_test, y_pred)
                st.text("Confusion matrix:")
                st.dataframe(pd.DataFrame(cm, index=labels, columns=labels))

            st.subheader("Explained variance ratio")
            ev = pd.Series(
                eval_lda.explained_variance_ratio_,
                index=[f"LD{i+1}" for i in range(len(eval_lda.explained_variance_ratio_))],
            )
            st.bar_chart(ev)

            # Final model trained on ALL data (used for real classification)
            final_lda = LinearDiscriminantAnalysis(n_components=n_components)
            final_lda.fit(X_scaled, y)

            st.session_state.model = final_lda
            st.session_state.scaler = scaler
            st.session_state.label_encoder = le
            st.session_state.feature_cols = feature_cols
            st.session_state.class_labels = labels

            st.success(
                "Final model trained on all available data. "
                "You can now classify unknown samples below."
            )

            # ---------------------------------------------------
            # 2D projection plot. With 2 classes LDA has just one
            # axis (LD1), so a second, visualization-only axis is
            # added (PC1 of the within-class variation).
            # ---------------------------------------------------
            st.subheader("LDA projection of the training data")
            extra_axis = fit_extra_axis(final_lda, X_scaled, y) if n_components == 1 else None
            ld2_is_pc = extra_axis is not None
            y_label = "PC1 (within-class, extra axis)" if ld2_is_pc else "LD2"

            X_2d = project_2d(final_lda, X_scaled, extra_axis)
            plot_df = pd.DataFrame({"Class": y_raw.values, "LD1": X_2d[:, 0], "LD2": X_2d[:, 1]})

            palette = px.colors.qualitative.Set2
            class_colors = {cls: palette[i % len(palette)] for i, cls in enumerate(labels)}

            class_means_ld = project_2d(final_lda, final_lda.means_, extra_axis)
            region_means = class_means_ld.copy()
            if ld2_is_pc:
                region_means[:, 1] = 0.0  # classification depends on LD1 only
            priors_arr = final_lda.priors_

            fig, boundary_formulas = build_projection_figure(
                labels, class_colors, plot_df, class_means_ld, region_means, priors_arr,
                y_label=y_label,
            )
            st.plotly_chart(fig, use_container_width=True)

            if ld2_is_pc:
                st.caption(
                    "With 2 classes LDA has only one discriminant axis (LD1). The vertical axis "
                    "is the first principal component of the within-class variation, orthogonal "
                    "to LD1. It is for visualization only and does not affect classification, "
                    "so the decision boundary is a vertical line on LD1."
                )
            elif max_components > 2:
                st.caption(
                    f"The full model uses {max_components} discriminant axes; only LD1 and LD2 "
                    "are shown, so the shaded regions and boundaries here are approximate."
                )
            if boundary_formulas:
                st.dataframe(pd.DataFrame(boundary_formulas), use_container_width=True, hide_index=True)

            st.session_state.plot_df = plot_df
            st.session_state.class_colors = class_colors
            st.session_state.class_means_ld = class_means_ld
            st.session_state.region_means = region_means
            st.session_state.priors_arr = priors_arr
            st.session_state.extra_axis = extra_axis
            st.session_state.y_label = y_label

            # ---------------------------------------------------
            # Formulas in raw feature units
            # ---------------------------------------------------
            coef_std = final_lda.coef_
            intercept_std = final_lda.intercept_
            coef_raw = coef_std / scaler.scale_
            intercept_raw = intercept_std - (coef_std * scaler.mean_ / scaler.scale_).sum(axis=1)

            if coef_raw.shape[0] == 1 and n_classes == 2:
                coef_raw = np.vstack([-coef_raw, coef_raw])
                intercept_raw = np.array([-intercept_raw[0], intercept_raw[0]])
            class_order = labels

            formula_df = pd.DataFrame(coef_raw, index=class_order, columns=feature_cols)
            formula_df.insert(0, "Intercept", intercept_raw)

            boundary_rows, boundary_index = [], []
            for bi in range(len(class_order)):
                for bj in range(bi + 1, len(class_order)):
                    diff_coef = coef_raw[bi] - coef_raw[bj]
                    diff_intercept = intercept_raw[bi] - intercept_raw[bj]
                    boundary_index.append(f"{class_order[bi]} vs {class_order[bj]}")
                    boundary_rows.append(np.concatenate([[diff_intercept], diff_coef]))
            boundary_df = pd.DataFrame(
                boundary_rows, index=boundary_index, columns=["Intercept"] + feature_cols
            )

            st.session_state.coef_raw = coef_raw
            st.session_state.intercept_raw = intercept_raw
            st.session_state.class_order = class_order
            st.session_state.formula_df = formula_df
            st.session_state.boundary_df = boundary_df

            # ---------------------------------------------------
            # LD1 / second-axis formulas in raw feature units. Scaling +
            # projection is an affine map, so we recover the intercept and
            # coefficients numerically from the zero vector and unit vectors.
            # ---------------------------------------------------
            zeros_raw = pd.DataFrame(np.zeros((1, len(feature_cols))), columns=feature_cols)
            baseline_ld = project_2d(final_lda, scaler.transform(zeros_raw), extra_axis)[0]

            ld_coef = np.zeros((len(feature_cols), 2))
            for j in range(len(feature_cols)):
                unit_raw = zeros_raw.copy()
                unit_raw.iloc[0, j] = 1.0
                ld_coef[j, :] = project_2d(final_lda, scaler.transform(unit_raw), extra_axis)[0] - baseline_ld

            ld_formula_df = pd.DataFrame(ld_coef.T, index=["LD1", y_label], columns=feature_cols)
            ld_formula_df.insert(0, "Intercept", baseline_ld)
            st.session_state.ld_formula_df = ld_formula_df

            st.subheader("Formulas (in your raw feature units)")
            st.markdown(
                "**Class scores:** `score(class) = Intercept + \u03a3(coef \u00d7 feature)`. "
                "A sample goes to the class with the highest score."
            )
            st.dataframe(formula_df.round(5), use_container_width=True)

            st.markdown("**Decision boundaries:** a sample sits exactly on the boundary when "
                        "`Intercept + \u03a3(coef \u00d7 feature) = 0`. Positive favours the first-named class.")
            st.dataframe(boundary_df.round(5), use_container_width=True)

            if n_classes == 2:
                row = boundary_df.iloc[0]
                terms = " ".join(f"{row[c]:+.5f}\u00b7[{c}]" for c in feature_cols)
                st.code(f"{row['Intercept']:+.5f} {terms} = 0", language=None)

            st.markdown("**Projection axes:** `axis = Intercept + \u03a3(coef \u00d7 feature)`")
            st.dataframe(ld_formula_df.round(5), use_container_width=True)


# =================================================================
# STEP 2 — Upload unknown samples & classify
# =================================================================
st.header("2. Classify unknown samples")

if st.session_state.model is None:
    st.info("Train a model in step 1 first.")
else:
    unknown_file = st.file_uploader(
        "Unknown samples Excel file (.xlsx)", type=["xlsx", "xls"], key="unknown"
    )

    if unknown_file is not None:
        unknown_df = pd.read_excel(unknown_file)
        unknown_df = unknown_df.dropna(how="all")

        feature_cols = st.session_state.feature_cols
        missing_cols = [c for c in feature_cols if c not in unknown_df.columns]
        if missing_cols:
            st.error(f"Unknown samples file is missing required columns: {missing_cols}")
        else:
            X_unknown = unknown_df[feature_cols].apply(pd.to_numeric, errors="coerce")

            if X_unknown.isna().any().any():
                st.warning(
                    "Some values are missing or non-numeric in the unknown samples file. "
                    "Rows with missing values may give unreliable predictions."
                )

            valid_mask = X_unknown.notna().all(axis=1)
            X_unknown_valid = X_unknown[valid_mask]

            if len(X_unknown_valid) == 0:
                st.error("No valid rows to classify after cleaning.")
            else:
                model = st.session_state.model
                scaler = st.session_state.scaler
                le = st.session_state.label_encoder
                extra_axis = st.session_state.extra_axis
                y_label = st.session_state.y_label

                X_unknown_scaled = scaler.transform(X_unknown_valid)
                predictions = le.inverse_transform(model.predict(X_unknown_scaled))
                probabilities = model.predict_proba(X_unknown_scaled)

                # LD1 + second-axis position of each unknown sample
                X_unknown_2d = project_2d(model, X_unknown_scaled, extra_axis)

                results = unknown_df.loc[valid_mask].copy()
                results["Predicted_class"] = predictions
                for i, cls in enumerate(le.classes_):
                    results[f"P({cls})"] = probabilities[:, i]
                results["LD1"] = X_unknown_2d[:, 0]
                results[y_label] = X_unknown_2d[:, 1]
                results = results.reset_index(drop=True)

                st.subheader("Results")
                st.caption(
                    "Select a row below to highlight that sample in the projection plot. "
                    "The LD1 and second-axis columns are each sample's position on the projection "
                    "below, computed with axis = Intercept + sum(coefficient x raw feature "
                    "value) — see the 'LD1_LD2 formula' tab in the downloaded Excel file for "
                    "the exact intercept and coefficients used."
                )
                selection_event = st.dataframe(
                    results,
                    key="results_table",
                    on_select="rerun",
                    selection_mode="multi-row",
                    use_container_width=True,
                )
                selected_positions = []
                if selection_event is not None:
                    try:
                        selected_positions = list(selection_event["selection"]["rows"])
                    except (TypeError, KeyError):
                        selected_positions = []

                buffer = io.BytesIO()
                with pd.ExcelWriter(buffer, engine=EXCEL_ENGINE) as writer:
                    _write_sheet_with_notes(
                        writer, results, "Predictions",
                        notes=[
                            "Each row is an unknown sample with its predicted class, "
                            "per-class probabilities, and its position on the projection "
                            "plot (LD1 plus LD2, or the PC1 extra axis when there are 2 classes).",
                            "These are computed as axis = Intercept + sum(coefficient x "
                            "raw feature value) — see the 'LD1_LD2 formula' tab for the "
                            "exact intercept and per-feature coefficients used.",
                        ],
                        index=False,
                    )

                    _write_sheet_with_notes(
                        writer, st.session_state.formula_df.round(5), "Classification formula",
                        notes=[
                            "LDA assigns each class a linear discriminant score. A new sample is "
                            "classified into the class with the highest score:",
                            "score(class) = Intercept + sum(coefficient x feature value)",
                            "Coefficients below are expressed in your raw, unstandardized "
                            "measurements, so raw values can be plugged in directly.",
                        ],
                        index=True, index_label="Class",
                    )

                    _write_sheet_with_notes(
                        writer, st.session_state.boundary_df.round(5), "Pairwise boundaries",
                        notes=[
                            "The boundary between any two classes is where their scores are tied:",
                            "(coefficient A - coefficient B) x feature values + "
                            "(Intercept A - Intercept B) = 0",
                            "Positive favors the first-named class in the 'Boundary' column, "
                            "negative favors the second.",
                        ],
                        index=True, index_label="Boundary",
                    )

                    if st.session_state.ld_formula_df is not None:
                        _write_sheet_with_notes(
                            writer, st.session_state.ld_formula_df.round(5), "LD1_LD2 formula",
                            notes=[
                                "This is the formula behind the projection axes used in the "
                                "app's plots: each axis is a linear combination of your raw "
                                "feature values. With 2 classes the second axis is a "
                                "visualization-only PC1 and does not affect classification.",
                                "axis = Intercept + sum(coefficient x feature value)",
                            ],
                            index=True, index_label="Component",
                        )
                buffer.seek(0)

                st.download_button(
                    "Download results as Excel (predictions + formulas)",
                    data=buffer,
                    file_name="classified_samples.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                )
                st.caption(
                    "The Excel file includes separate tabs for the predictions, the "
                    "per-class classification formula, the pairwise decision boundary "
                    "formulas, and the projection axis formulas — each with an "
                    "explanation of how to read it, in your raw feature units."
                )

                # ---------------------------------------------------
                # Overlay the unknown samples onto the same projection
                # (and decision boundaries) shown in step 1, highlighting
                # any rows currently selected in the table above.
                # ---------------------------------------------------
                st.subheader("Projection with unknown samples")

                # Prefer a non-feature column (e.g. Sample_ID) as the hover label
                non_feature_cols = [c for c in unknown_df.columns if c not in feature_cols]
                if non_feature_cols:
                    sample_labels = unknown_df.loc[valid_mask, non_feature_cols[0]].astype(str).values
                else:
                    sample_labels = [f"Row {i}" for i in unknown_df.loc[valid_mask].index]

                unknown_plot_df = pd.DataFrame(
                    {"Sample": sample_labels, "Predicted_class": predictions,
                     "LD1": X_unknown_2d[:, 0], "LD2": X_unknown_2d[:, 1]}
                )
                fig2, boundary_formulas2 = build_projection_figure(
                    st.session_state.class_labels, st.session_state.class_colors,
                    st.session_state.plot_df, st.session_state.class_means_ld,
                    st.session_state.region_means, st.session_state.priors_arr,
                    y_label=y_label, unknown_df=unknown_plot_df, highlight_idx=selected_positions,
                )
                st.plotly_chart(fig2, use_container_width=True)
                if selected_positions:
                    st.caption(
                        "White dots are the unknown samples projected into the same space "
                        "as the training data. The gold-and-red marker(s) show the sample(s) "
                        "currently selected in the results table above."
                    )
                else:
                    st.caption(
                        "White dots are the unknown samples projected into the same "
                        "space as the training data — hover over one to see its predicted "
                        "class, or select a row in the results table above to highlight it here. "
                        "Where a dot falls relative to the dashed boundary lines shows how "
                        "confidently it was assigned."
                    )
                if boundary_formulas2:
                    with st.expander("Decision boundary formulas (plot coordinates)"):
                        st.dataframe(
                            pd.DataFrame(boundary_formulas2),
                            use_container_width=True,
                            hide_index=True,
                        )