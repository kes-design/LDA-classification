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
# Helpers for the LD1/LD2 projection figure (shared by step 1 and 2
# so unknown samples can be overlaid onto the same figure/boundaries
# once they've been classified).
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


def build_projection_figure(
    labels, class_colors, plot_df, n_components, max_components,
    class_means_ld=None, priors_arr=None, unknown_df=None, highlight_idx=None,
):
    """Build the LD projection figure (shaded decision regions + pairwise
    boundary lines when the plot contains the model's full discriminant
    space), optionally overlaying already-classified unknown samples as
    white dot markers.

    plot_df: DataFrame with columns 'Class', 'LD1', and 'LD2' if n_components >= 2.
    unknown_df: optional DataFrame with columns 'LD1' (+ 'LD2' if applicable),
                'Predicted_class', 'Sample' (a display label for hover text).
                Expected to have a plain 0..n-1 RangeIndex.
    highlight_idx: optional collection of positional row indices (into
        unknown_df) to draw as larger, highlighted markers — e.g. rows
        currently selected in a results table.

    Returns (fig, boundary_formulas, boundaries_are_exact).
    """
    boundary_formulas = []
    has_unknown = unknown_df is not None and len(unknown_df) > 0
    highlight_idx = set(highlight_idx) if highlight_idx else set()

    def _unknown_traces(x_col, y_col=None):
        n = len(unknown_df)
        hover_text = (
            unknown_df["Sample"].to_numpy() if "Sample" in unknown_df else np.array([""] * n)
        )
        pred_text = unknown_df["Predicted_class"].to_numpy()
        x_all = unknown_df[x_col].to_numpy()
        y_all = unknown_df[y_col].to_numpy() if y_col is not None else np.zeros(n)
        is_hl = np.array([i in highlight_idx for i in range(n)])

        traces = []
        base_mask = ~is_hl
        if base_mask.any():
            traces.append(
                go.Scatter(
                    x=x_all[base_mask], y=y_all[base_mask], mode="markers",
                    marker=dict(symbol="circle", size=10, color="white", line=dict(width=2, color="black")),
                    name="Unknown sample",
                    customdata=np.stack([hover_text[base_mask], pred_text[base_mask]], axis=-1),
                    hovertemplate="Sample: %{customdata[0]}<br>Predicted: %{customdata[1]}"
                                  "<br>LD1: %{x:.3f}" + ("<br>LD2: %{y:.3f}" if y_col is not None else "")
                                  + "<extra></extra>",
                )
            )
        if is_hl.any():
            traces.append(
                go.Scatter(
                    x=x_all[is_hl], y=y_all[is_hl], mode="markers",
                    marker=dict(symbol="circle", size=18, color="#FFD700",
                                line=dict(width=3, color="red")),
                    name="Selected sample",
                    customdata=np.stack([hover_text[is_hl], pred_text[is_hl]], axis=-1),
                    hovertemplate="Sample: %{customdata[0]}<br>Predicted: %{customdata[1]}"
                                  "<br>LD1: %{x:.3f}" + ("<br>LD2: %{y:.3f}" if y_col is not None else "")
                                  + "<extra></extra>",
                )
            )
        return traces

    if n_components >= 2:
        boundaries_are_exact = n_components == max_components == 2

        if boundaries_are_exact:
            all_ld1 = list(plot_df["LD1"])
            all_ld2 = list(plot_df["LD2"])
            if has_unknown:
                all_ld1 += list(unknown_df["LD1"])
                all_ld2 += list(unknown_df["LD2"])
            ld1_lo, ld1_hi = min(all_ld1), max(all_ld1)
            ld2_lo, ld2_hi = min(all_ld2), max(all_ld2)
            pad1 = 0.15 * (ld1_hi - ld1_lo) if ld1_hi > ld1_lo else 1.0
            pad2 = 0.15 * (ld2_hi - ld2_lo) if ld2_hi > ld2_lo else 1.0
            ld1_lo, ld1_hi = ld1_lo - pad1, ld1_hi + pad1
            ld2_lo, ld2_hi = ld2_lo - pad2, ld2_hi + pad2

            grid_res = 300
            ld1_grid = np.linspace(ld1_lo, ld1_hi, grid_res)
            ld2_grid = np.linspace(ld2_lo, ld2_hi, grid_res)
            LD1_mesh, LD2_mesh = np.meshgrid(ld1_grid, ld2_grid)
            mesh_points = np.stack([LD1_mesh.ravel(), LD2_mesh.ravel()], axis=1)
            diffs = mesh_points[:, None, :] - class_means_ld[None, :, :]
            sq_dists = np.sum(diffs ** 2, axis=2)
            scores = -0.5 * sq_dists + np.log(priors_arr)[None, :]
            region_idx = np.argmax(scores, axis=1).reshape(LD1_mesh.shape)
            n_region_classes = len(class_means_ld)

            colorscale = []
            for i, cls in enumerate(labels):
                color = class_colors[cls]
                colorscale.append([i / n_region_classes, color])
                colorscale.append([(i + 1) / n_region_classes, color])

            fig = go.Figure()
            fig.add_trace(
                go.Heatmap(
                    x=ld1_grid, y=ld2_grid, z=region_idx,
                    zmin=-0.5, zmax=n_region_classes - 0.5,
                    colorscale=colorscale, showscale=False,
                    opacity=0.30, zsmooth="best", hoverinfo="skip",
                )
            )
            for cls in labels:
                sub = plot_df[plot_df["Class"] == cls]
                fig.add_trace(
                    go.Scatter(
                        x=sub["LD1"], y=sub["LD2"], mode="markers",
                        marker=dict(color=class_colors[cls], size=7,
                                    line=dict(width=0.5, color="white")),
                        name=cls,
                        hovertemplate=f"Class: {cls}<br>LD1: %{{x:.3f}}<br>LD2: %{{y:.3f}}<extra></extra>",
                    )
                )
            for i, cls in enumerate(labels):
                m = class_means_ld[i]
                fig.add_trace(
                    go.Scatter(
                        x=[m[0]], y=[m[1]], mode="markers+text",
                        marker=dict(color="black", size=11, line=dict(width=1.5, color="white")),
                        text=[f"{cls} centroid"], textposition="top center",
                        showlegend=False, hoverinfo="skip",
                    )
                )

            # Only draw a boundary between two classes if their regions
            # actually touch on the grid — with >2 classes a third class can
            # sit entirely between a given pair.
            adjacent = np.zeros((n_region_classes, n_region_classes), dtype=bool)
            h_diff = region_idx[:, :-1] != region_idx[:, 1:]
            for r1, r2 in zip(region_idx[:, :-1][h_diff], region_idx[:, 1:][h_diff]):
                adjacent[r1, r2] = adjacent[r2, r1] = True
            v_diff = region_idx[:-1, :] != region_idx[1:, :]
            for r1, r2 in zip(region_idx[:-1, :][v_diff], region_idx[1:, :][v_diff]):
                adjacent[r1, r2] = adjacent[r2, r1] = True

            for bi in range(n_region_classes):
                for bj in range(bi + 1, n_region_classes):
                    if not adjacent[bi, bj]:
                        continue
                    mi, mj = class_means_ld[bi], class_means_ld[bj]
                    a_coef = mi[0] - mj[0]
                    b_coef = mi[1] - mj[1]
                    c_coef = -0.5 * (np.sum(mi ** 2) - np.sum(mj ** 2)) + np.log(
                        priors_arr[bi] / priors_arr[bj]
                    )
                    seg = _clip_line_to_box(a_coef, b_coef, c_coef, ld1_lo, ld1_hi, ld2_lo, ld2_hi)
                    if seg is None:
                        continue
                    (x0, y0), (x1, y1) = seg
                    fig.add_trace(
                        go.Scatter(
                            x=[x0, x1], y=[y0, y1], mode="lines",
                            line=dict(color="black", width=1.5, dash="dash"),
                            showlegend=False, hoverinfo="skip",
                        )
                    )
                    fig.add_annotation(
                        x=(x0 + x1) / 2, y=(y0 + y1) / 2,
                        text=f"{labels[bi]} | {labels[bj]}",
                        showarrow=False, font=dict(size=10, color="black"),
                        bgcolor="rgba(255,255,255,0.7)",
                    )
                    boundary_formulas.append(
                        {
                            "Boundary": f"{labels[bi]} vs {labels[bj]}",
                            "Formula (LD-space; > 0 favors first class)":
                                f"({a_coef:.4f})·LD1 + ({b_coef:.4f})·LD2 + ({c_coef:.4f}) = 0",
                        }
                    )

            if has_unknown:
                for tr in _unknown_traces("LD1", "LD2"):
                    fig.add_trace(tr)

            fig.update_layout(
                title="LDA projection (shaded decision regions)",
                xaxis_title="LD1", yaxis_title="LD2", height=550, legend_title="Class",
            )
        else:
            fig = px.scatter(
                plot_df, x="LD1", y="LD2", color="Class",
                title="LDA projection of the training data", opacity=0.75, height=500,
                color_discrete_map=class_colors,
            )
            if has_unknown:
                for tr in _unknown_traces("LD1", "LD2"):
                    fig.add_trace(tr)
    else:
        boundaries_are_exact = False
        fig = px.strip(
            plot_df, x="LD1", color="Class",
            title="LDA projection (1 component)", height=350,
            color_discrete_map=class_colors,
        )
        if has_unknown:
            for tr in _unknown_traces("LD1", None):
                fig.add_trace(tr)

    return fig, boundary_formulas, boundaries_are_exact


# ---------------------------------------------------------------
# Session state so results survive re-runs / interactions
# ---------------------------------------------------------------
if "model" not in st.session_state:
    st.session_state.model = None
    st.session_state.scaler = None
    st.session_state.label_encoder = None
    st.session_state.feature_cols = None
    st.session_state.class_labels = None
    st.session_state.coef_raw = None
    st.session_state.intercept_raw = None
    st.session_state.class_order = None
    st.session_state.ld_formula_df = None
    st.session_state.formula_df = None
    st.session_state.boundary_df = None
    st.session_state.plot_df = None
    st.session_state.n_components = None
    st.session_state.max_components = None
    st.session_state.class_colors = None
    st.session_state.class_means_ld = None
    st.session_state.priors_arr = None


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
            # Example projection plot (LD1 vs LD2), analogous to
            # the example-tree visualization in the Random Forest app
            # ---------------------------------------------------
            st.subheader("LDA projection of the training data")
            X_lda_all = final_lda.transform(X_scaled)
            plot_df = pd.DataFrame({"Class": y_raw.values})
            plot_df["LD1"] = X_lda_all[:, 0]
            if n_components >= 2:
                plot_df["LD2"] = X_lda_all[:, 1]

            palette = px.colors.qualitative.Set2
            class_colors = {cls: palette[i % len(palette)] for i, cls in enumerate(labels)}

            class_means_ld = final_lda.transform(final_lda.means_) if n_components >= 2 else None
            priors_arr = final_lda.priors_

            fig, boundary_formulas, boundaries_are_exact = build_projection_figure(
                labels, class_colors, plot_df, n_components, max_components,
                class_means_ld=class_means_ld, priors_arr=priors_arr,
            )
            st.plotly_chart(fig, use_container_width=True)

            if n_components >= 2 and not boundaries_are_exact and n_classes > 2:
                st.caption(
                    "Decision regions aren't shaded here because this plot doesn't "
                    "contain all of the model's discriminant directions — with "
                    f"{n_classes} classes the full model uses {max_components} "
                    "component(s), but only 2 are shown."
                )

            if boundary_formulas:
                st.caption(
                    "Dashed lines mark the linear decision boundary between each pair "
                    "of adjacent classes in this LD1/LD2 projection — where a sample "
                    "would score equally for both classes. These are expressed in the "
                    "transformed LD1/LD2 coordinates of the plot, not the raw feature "
                    "units (see the pairwise boundary formulas in section 3 for those). "
                    "Once you classify unknown samples in step 2, they'll be overlaid "
                    "on this same projection."
                )
                st.dataframe(
                    pd.DataFrame(boundary_formulas),
                    use_container_width=True,
                    hide_index=True,
                )

            # Store everything needed to redraw/extend this figure in step 2
            st.session_state.plot_df = plot_df
            st.session_state.n_components = n_components
            st.session_state.max_components = max_components
            st.session_state.class_colors = class_colors
            st.session_state.class_means_ld = class_means_ld
            st.session_state.priors_arr = priors_arr

            # ---------------------------------------------------
            # Formulas — stored for display in the section below
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
            # LD1 / LD2 formulas, i.e. the projection axes themselves,
            # expressed in raw feature units. Since scaling + LDA.transform
            # is an affine map (LD = A @ raw + b), we recover A and b
            # numerically by transforming the zero vector and each unit
            # vector — this works regardless of LDA solver internals.
            # ---------------------------------------------------
            zeros_raw = pd.DataFrame(np.zeros((1, len(feature_cols))), columns=feature_cols)
            baseline_ld = final_lda.transform(scaler.transform(zeros_raw))[0]

            ld_coef = np.zeros((len(feature_cols), n_components))
            for j, col in enumerate(feature_cols):
                unit_raw = zeros_raw.copy()
                unit_raw.iloc[0, j] = 1.0
                ld_at_unit = final_lda.transform(scaler.transform(unit_raw))[0]
                ld_coef[j, :] = ld_at_unit - baseline_ld

            ld_formula_df = pd.DataFrame(
                ld_coef.T, index=[f"LD{k+1}" for k in range(n_components)], columns=feature_cols
            )
            ld_formula_df.insert(0, "Intercept", baseline_ld)

            st.session_state.ld_formula_df = ld_formula_df


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

                X_unknown_scaled = scaler.transform(X_unknown_valid)
                predictions = le.inverse_transform(model.predict(X_unknown_scaled))
                probabilities = model.predict_proba(X_unknown_scaled)

                results = unknown_df.loc[valid_mask].copy()
                results["Predicted_class"] = predictions
                for i, cls in enumerate(le.classes_):
                    results[f"P({cls})"] = probabilities[:, i]
                results = results.reset_index(drop=True)

                st.subheader("Results")
                st.caption("Select a row below to highlight that sample in the projection plot.")
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
                    results.to_excel(writer, index=False, sheet_name="Predictions")

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
                                "app's LD1/LD2 plots: each LD score is a linear combination of "
                                "your raw feature values.",
                                "LDk = Intercept + sum(coefficient x feature value)",
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
                    "formulas, and the LD1/LD2 projection formula — each with an "
                    "explanation of how to read it, in your raw feature units."
                )

                # ---------------------------------------------------
                # Overlay the unknown samples onto the same LD1/LD2
                # projection (and decision boundaries) shown in step 1,
                # highlighting any rows currently selected in the table above
                # ---------------------------------------------------
                st.subheader("Projection with unknown samples")

                # Prefer a non-feature column (e.g. Sample_ID) as the hover label
                non_feature_cols = [c for c in unknown_df.columns if c not in feature_cols]
                if non_feature_cols:
                    sample_labels = unknown_df.loc[valid_mask, non_feature_cols[0]].astype(str).values
                else:
                    sample_labels = [f"Row {i}" for i in unknown_df.loc[valid_mask].index]

                X_unknown_lda = model.transform(X_unknown_scaled)
                unknown_plot_df = pd.DataFrame(
                    {"Sample": sample_labels, "Predicted_class": predictions}
                )
                unknown_plot_df["LD1"] = X_unknown_lda[:, 0]
                if st.session_state.n_components >= 2:
                    unknown_plot_df["LD2"] = X_unknown_lda[:, 1]

                fig2, boundary_formulas2, boundaries_are_exact2 = build_projection_figure(
                    st.session_state.class_labels, st.session_state.class_colors,
                    st.session_state.plot_df, st.session_state.n_components,
                    st.session_state.max_components,
                    class_means_ld=st.session_state.class_means_ld,
                    priors_arr=st.session_state.priors_arr,
                    unknown_df=unknown_plot_df,
                    highlight_idx=selected_positions,
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
                    with st.expander("Decision boundary formulas (LD-space)"):
                        st.dataframe(
                            pd.DataFrame(boundary_formulas2),
                            use_container_width=True,
                            hide_index=True,
                        )