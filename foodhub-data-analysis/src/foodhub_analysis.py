"""FoodHub order analysis.

This script converts the MIT Professional Education FoodHub notebook into a
small reproducible analysis workflow. It computes the core business metrics
used in the project and writes summary tables that can be reused in reports.

Author: Shivang Sharma
"""

from argparse import ArgumentParser
from pathlib import Path
from typing import Dict

import pandas as pd

RATING_NOT_GIVEN = "Not given"
HIGH_VALUE_ORDER_THRESHOLD = 20.0
LONG_FULFILLMENT_THRESHOLD = 60


def load_orders(input_path: str) -> pd.DataFrame:
    """Load FoodHub order data from CSV."""
    data_path = Path(input_path)
    if not data_path.exists():
        raise FileNotFoundError(f"Input file not found: {data_path}")
    return pd.read_csv(data_path)


def summarize_orders(data: pd.DataFrame) -> Dict[str, float]:
    """Compute core FoodHub business summary metrics."""
    total_time = data["food_preparation_time"] + data["delivery_time"]
    high_value_orders = data[data["cost_of_the_order"] > HIGH_VALUE_ORDER_THRESHOLD]

    return {
        "orders": float(len(data)),
        "columns": float(data.shape[1]),
        "unique_customers": float(data["customer_id"].nunique()),
        "unique_restaurants": float(data["restaurant_name"].nunique()),
        "unique_cuisines": float(data["cuisine_type"].nunique()),
        "orders_without_rating": float((data["rating"] == RATING_NOT_GIVEN).sum()),
        "mean_delivery_time": float(data["delivery_time"].mean()),
        "orders_above_20": float(len(high_value_orders)),
        "orders_above_20_pct": float(len(high_value_orders) / len(data) * 100),
        "long_fulfillment_pct": float((total_time > LONG_FULFILLMENT_THRESHOLD).mean() * 100),
    }


def calculate_platform_revenue(data: pd.DataFrame) -> pd.Series:
    """Calculate FoodHub commission revenue using the project business rules."""
    revenue = data["cost_of_the_order"].apply(
        lambda cost: cost * 0.25 if cost > 20 else cost * 0.15 if cost > 5 else 0
    )
    return revenue


def promotional_candidates(data: pd.DataFrame) -> pd.DataFrame:
    """Return restaurants with more than 50 ratings and average rating above 4."""
    rated_orders = data[data["rating"] != RATING_NOT_GIVEN].copy()
    rated_orders["rating"] = rated_orders["rating"].astype(float)

    rating_counts = rated_orders.groupby("restaurant_name")["rating"].count()
    average_ratings = rated_orders.groupby("restaurant_name")["rating"].mean()

    candidates = pd.DataFrame(
        {"rating_count": rating_counts, "average_rating": average_ratings}
    ).reset_index()
    return candidates[
        (candidates["rating_count"] > 50) & (candidates["average_rating"] > 4)
    ].sort_values("average_rating", ascending=False)


def write_outputs(data: pd.DataFrame, output_dir: str) -> None:
    """Write reusable summary outputs to CSV files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    summary = summarize_orders(data)
    pd.DataFrame([summary]).to_csv(output_path / "summary_metrics.csv", index=False)

    top_restaurants = data["restaurant_name"].value_counts().head(10)
    top_restaurants.to_csv(output_path / "top_restaurants_by_orders.csv")

    data_with_revenue = data.copy()
    data_with_revenue["platform_revenue"] = calculate_platform_revenue(data)
    revenue_by_restaurant = (
        data_with_revenue.groupby("restaurant_name")["platform_revenue"]
        .sum()
        .sort_values(ascending=False)
        .head(15)
    )
    revenue_by_restaurant.to_csv(output_path / "top_restaurants_by_revenue.csv")

    promotional_candidates(data).to_csv(
        output_path / "promotional_offer_candidates.csv", index=False
    )


def parse_args() -> ArgumentParser:
    """Parse command-line arguments."""
    parser = ArgumentParser(description="Run FoodHub order analysis.")
    parser.add_argument("--input", required=True, help="Path to foodhub_order.csv")
    parser.add_argument(
        "--output",
        default="results/metrics",
        help="Directory where summary outputs should be written",
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    orders = load_orders(arguments.input)
    write_outputs(orders, arguments.output)
    print("FoodHub analysis complete.")
