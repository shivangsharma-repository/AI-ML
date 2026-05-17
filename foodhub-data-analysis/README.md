# FoodHub Order Analysis

## Project Context

This project was completed as part of the MIT Professional Education Applied Data Science Program. It analyzes online food delivery orders for FoodHub, a food aggregator platform that connects customers, restaurants, and delivery partners through a mobile app.

The business objective is to understand order demand, cuisine preferences, customer behavior, delivery performance, restaurant revenue, and rating patterns so FoodHub can improve customer experience and support better operational decisions.

## Dataset

- Observations: 1,898 orders
- Columns: 9
- Customers: 1,200 unique customers
- Restaurants: 178 unique restaurants
- Cuisine types: 14
- Missing values: none detected

### Data Dictionary

| Column | Description |
|---|---|
| `order_id` | Unique order identifier |
| `customer_id` | Customer identifier |
| `restaurant_name` | Restaurant that fulfilled the order |
| `cuisine_type` | Cuisine ordered by the customer |
| `cost_of_the_order` | Cost paid for the food order |
| `day_of_the_week` | Weekday or weekend order indicator |
| `rating` | Customer rating from 1 to 5 or `Not given` |
| `food_preparation_time` | Minutes from restaurant confirmation to pickup |
| `delivery_time` | Minutes from pickup to customer drop-off |

## Analysis Questions

The notebook answers business questions related to:

- Data quality and missing values
- Order cost distribution
- Cuisine popularity
- Restaurant order volume
- Weekend versus weekday demand
- Customer rating behavior
- Delivery and food preparation times
- Promotional offer eligibility
- Revenue generation under FoodHub commission rules
- Total fulfillment time from order placement to delivery

## Key Findings

| Metric / Question | Result |
|---|---:|
| Dataset size | 1,898 orders, 9 columns |
| Orders without rating | 736 |
| Top restaurant by order count | Shake Shack, 219 orders |
| Top 5 restaurants by orders | Shake Shack, The Meatball Shop, Blue Ribbon Sushi, Blue Ribbon Fried Chicken, Parm |
| Most popular weekend cuisine | American, 415 weekend orders |
| Orders above $20 | 555 orders, 29.24% |
| Mean delivery time | 24.16 minutes |
| Top frequent customers | 52832, 47440, 83287 |
| Net revenue under commission rules | Approximately $6,166.30 |
| Orders taking more than 60 minutes total | 10.54% |
| Mean weekday delivery time | Approximately 28 minutes |
| Mean weekend delivery time | Approximately 22 minutes |

## Promotional Offer Candidates

Restaurants that met the promotion condition of more than 50 ratings and an average rating greater than 4:

| Restaurant | Average Rating |
|---|---:|
| The Meatball Shop | 4.51 |
| Blue Ribbon Fried Chicken | 4.33 |
| Shake Shack | 4.28 |
| Blue Ribbon Sushi | 4.22 |

## Business Recommendations

1. **Prioritize high-demand partners:** FoodHub should deepen partnerships with high-order-volume restaurants such as Shake Shack, The Meatball Shop, Blue Ribbon Sushi, and Blue Ribbon Fried Chicken.
2. **Promote high-performing restaurants:** The restaurants that satisfy the promotional criteria should be prioritized in advertisements because they combine strong volume with high customer satisfaction.
3. **Use cuisine demand for marketing:** American, Italian, and Japanese cuisines should be highlighted in customer acquisition campaigns, especially for weekend demand.
4. **Improve rating collection:** A large share of orders are not rated, so FoodHub should introduce post-delivery prompts or small incentives to increase feedback coverage.
5. **Monitor long fulfillment times:** Around 10.54% of orders take more than 60 minutes, making fulfillment time a clear opportunity for operational improvement.
6. **Investigate weekday delivery delays:** Weekday delivery times are longer than weekend delivery times, suggesting weekday logistics, staffing, or traffic patterns need further review.

## Technical Skills Demonstrated

- Exploratory data analysis
- Data quality checks
- Summary statistics
- Univariate and multivariate analysis
- Grouped aggregation
- Revenue calculation
- Business rule implementation
- Data visualization with Matplotlib and Seaborn
- Business recommendation development

## How to Run

```bash
cd foodhub-data-analysis
pip install -r requirements.txt
python src/foodhub_analysis.py --input data/raw/foodhub_order.csv
```

The raw dataset is not included in this repository. Place `foodhub_order.csv` in `data/raw/` before running the script locally.

## Author

Shivang Sharma  
MS Data Science Candidate, Purdue University  
MIT Applied Data Science Program Graduate
