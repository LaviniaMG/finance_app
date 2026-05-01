"""Reusable Plotly charts for the FinPlan app."""
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

COLORS = {
    "actual": "#2196F3",
    "forecast": "#FF9800",
    "budget": "#9E9E9E",
    "positive": "#4CAF50",
    "negative": "#F44336",
    "neutral": "#607D8B",
}

LAYOUT_DEFAULTS = dict(
    plot_bgcolor="white",
    paper_bgcolor="white",
    font=dict(family="Inter, sans-serif", size=12),
    margin=dict(l=10, r=10, t=40, b=10),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)


def monthly_bar_chart(
    df: pd.DataFrame,
    period_col: str = "period_id",
    actual_col: str = "actual_mtd",
    forecast_col: str = "forecast_mtd",
    budget_col: str = "budget_mtd",
    title: str = "Venit lunar — Actual vs Forecast vs Budget",
    unit: int = 1_000,
    currency_symbol: str = "€",
) -> go.Figure:
    """Grouped horizontal bar chart: Actual / Forecast / Budget by month."""
    fig = go.Figure()

    period_labels = [f"P{int(p)}" for p in df[period_col]]

    if actual_col in df.columns:
        fig.add_trace(go.Bar(
            name="Actual",
            x=df[actual_col] / unit,
            y=period_labels,
            orientation="h",
            marker_color=COLORS["actual"],
        ))

    if forecast_col in df.columns:
        fig.add_trace(go.Bar(
            name="Forecast",
            x=df[forecast_col] / unit,
            y=period_labels,
            orientation="h",
            marker_color=COLORS["forecast"],
        ))

    if budget_col in df.columns:
        fig.add_trace(go.Bar(
            name="Budget",
            x=df[budget_col] / unit,
            y=period_labels,
            orientation="h",
            marker_color=COLORS["budget"],
        ))

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=title,
        barmode="group",
        xaxis_title=f"Valoare ({currency_symbol}k)",
        yaxis=dict(autorange="reversed"),
        height=420,
    )
    return fig


def line_chart_actuals_vs_forecast(
    df: pd.DataFrame,
    period_col: str = "period_label",
    actual_col: str = "actual_ytd",
    forecast_col: str = "forecast_ytd",
    budget_col: str = "budget_ytd",
    title: str = "YTD Trend",
    unit: int = 1_000,
    currency_symbol: str = "€",
) -> go.Figure:
    fig = go.Figure()

    if actual_col in df.columns:
        fig.add_trace(go.Scatter(
            x=df[period_col], y=df[actual_col] / unit,
            mode="lines+markers", name="Actual",
            line=dict(color=COLORS["actual"], width=2),
        ))
    if forecast_col in df.columns:
        fig.add_trace(go.Scatter(
            x=df[period_col], y=df[forecast_col] / unit,
            mode="lines+markers", name="Forecast",
            line=dict(color=COLORS["forecast"], width=2, dash="dot"),
        ))
    if budget_col in df.columns:
        fig.add_trace(go.Scatter(
            x=df[period_col], y=df[budget_col] / unit,
            mode="lines+markers", name="Budget",
            line=dict(color=COLORS["budget"], width=1, dash="dash"),
        ))

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=title,
        yaxis_title=f"{currency_symbol}k",
        height=350,
    )
    return fig


def waterfall_chart(
    categories: list[str],
    values: list[float],
    title: str = "Bridge Actual → Forecast",
    unit: int = 1_000,
    currency_symbol: str = "€",
) -> go.Figure:
    """Waterfall / bridge chart for variance analysis."""
    measure = ["absolute"] + ["relative"] * (len(values) - 2) + ["total"]
    colors = []
    for i, v in enumerate(values):
        if i == 0 or i == len(values) - 1:
            colors.append(COLORS["neutral"])
        elif v >= 0:
            colors.append(COLORS["positive"])
        else:
            colors.append(COLORS["negative"])

    fig = go.Figure(go.Waterfall(
        name="Bridge",
        orientation="v",
        measure=measure,
        x=categories,
        y=[v / unit for v in values],
        connector=dict(line=dict(color="rgb(63, 63, 63)")),
        increasing=dict(marker_color=COLORS["positive"]),
        decreasing=dict(marker_color=COLORS["negative"]),
        totals=dict(marker_color=COLORS["neutral"]),
    ))

    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=title,
        yaxis_title=f"{currency_symbol}k",
        height=380,
    )
    return fig


def variance_heatmap(
    df: pd.DataFrame,
    row_col: str,
    col_col: str,
    value_col: str,
    title: str = "Variance Heatmap",
    colorscale: str = "RdYlGn",
) -> go.Figure:
    """Heatmap for confidence/variance by entity × account."""
    pivot = df.pivot_table(index=row_col, columns=col_col, values=value_col, aggfunc="mean")

    fig = px.imshow(
        pivot,
        color_continuous_scale=colorscale,
        title=title,
        aspect="auto",
    )
    fig.update_layout(**LAYOUT_DEFAULTS, height=400)
    return fig


def feature_importance_bar(
    importance_df: pd.DataFrame,
    feature_col: str = "feature_name",
    score_col: str = "importance_score",
    title: str = "Factori care influențează predicția",
) -> go.Figure:
    df = importance_df.sort_values(score_col, ascending=True).tail(10)

    colors = []
    for score in df[score_col]:
        if score > 0.3:
            colors.append(COLORS["positive"])
        elif score > 0.15:
            colors.append(COLORS["forecast"])
        else:
            colors.append(COLORS["neutral"])

    fig = go.Figure(go.Bar(
        x=df[score_col],
        y=df[feature_col],
        orientation="h",
        marker_color=colors,
    ))
    fig.update_layout(
        **LAYOUT_DEFAULTS,
        title=title,
        xaxis_title="Importanță",
        height=350,
    )
    return fig


def kpi_sparkline(values: list[float], color: str = "#2196F3") -> go.Figure:
    """Minimal sparkline for KPI cards."""
    fig = go.Figure(go.Scatter(
        y=values,
        mode="lines",
        line=dict(color=color, width=2),
        fill="tozeroy",
        fillcolor=f"rgba{tuple(list(bytes.fromhex(color.lstrip('#'))) + [30])}",
    ))
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=60,
        showlegend=False,
    )
    return fig
