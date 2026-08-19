"""Generated tests for the migrated customer revenue pipeline.

Each test pins one semantic difference the migration plan declared. They use
small hand-built frames rather than the sample data, because the traps only
appear for specific inputs the sample may not contain.
"""

from pyspark.sql import Row
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    StringType,
    StructField,
    StructType,
)

CUSTOMERS_SCHEMA = StructType(
    [
        StructField("customer_id", IntegerType(), True),
        StructField("name", StringType(), True),
        StructField("age", DoubleType(), True),
        StructField("country", StringType(), True),
        StructField("signup_date", StringType(), True),
    ]
)

ORDERS_SCHEMA = StructType(
    [
        StructField("order_id", IntegerType(), True),
        StructField("customer_id", IntegerType(), True),
        StructField("product", StringType(), True),
        StructField("quantity", IntegerType(), True),
        StructField("price", DoubleType(), True),
        StructField("order_date", StringType(), True),
    ]
)


def _stage(spark, tmp_path, customers, orders):
    """Write frames where the pipeline expects to read them.

    Spark writes a directory of part files and its reader accepts a directory,
    so the pipeline sees exactly the layout it would in production.
    """
    base = str(tmp_path)
    spark.createDataFrame(customers, CUSTOMERS_SCHEMA).write.mode("overwrite").option(
        "header", True
    ).csv(base + "/customers.csv")
    spark.createDataFrame(orders, ORDERS_SCHEMA).write.mode("overwrite").option(
        "header", True
    ).csv(base + "/orders.csv")
    return base


def _run(spark, pipeline, tmp_path, customers, orders):
    base = _stage(spark, tmp_path, customers, orders)
    out = base + "/out"
    pipeline.run(spark, base, out)
    rows = spark.read.option("header", True).csv(out + "/revenue_by_country").collect()
    return {r["country"]: (None if r["revenue"] is None else float(r["revenue"])) for r in rows}


def test_null_country_rows_are_excluded(spark, pipeline, tmp_path):
    """pandas groupby drops null keys; a naive Spark port keeps them as a group."""
    customers = [
        Row(customer_id=1, name="a", age=30.0, country="FR", signup_date="2024-01-01"),
        Row(customer_id=2, name="b", age=40.0, country=None, signup_date="2024-01-02"),
    ]
    orders = [
        Row(order_id=1, customer_id=1, product="p", quantity=2, price=10.0, order_date="2024-01-03"),
        Row(order_id=2, customer_id=2, product="p", quantity=5, price=10.0, order_date="2024-01-04"),
    ]
    result = _run(spark, pipeline, tmp_path, customers, orders)
    assert set(result) == {"FR"}
    assert result["FR"] == 20.0


def test_group_with_no_orders_sums_to_zero_not_null(spark, pipeline, tmp_path):
    """pandas sum() of an all-NaN group is 0.0; Spark sum() returns null."""
    customers = [
        Row(customer_id=1, name="a", age=30.0, country="DE", signup_date="2024-01-01"),
    ]
    orders = [
        Row(order_id=1, customer_id=99, product="p", quantity=1, price=5.0, order_date="2024-01-02"),
    ]
    result = _run(spark, pipeline, tmp_path, customers, orders)
    assert result == {"DE": 0.0}


def test_minors_and_unknown_age_are_excluded(spark, pipeline, tmp_path):
    """age >= 18 drops both under-18s and nulls, matching NaN >= 18 being False."""
    customers = [
        Row(customer_id=1, name="adult", age=18.0, country="ES", signup_date="2024-01-01"),
        Row(customer_id=2, name="minor", age=17.0, country="ES", signup_date="2024-01-01"),
        Row(customer_id=3, name="unknown", age=None, country="ES", signup_date="2024-01-01"),
    ]
    orders = [
        Row(order_id=1, customer_id=1, product="p", quantity=1, price=7.0, order_date="2024-01-02"),
        Row(order_id=2, customer_id=2, product="p", quantity=1, price=100.0, order_date="2024-01-02"),
        Row(order_id=3, customer_id=3, product="p", quantity=1, price=100.0, order_date="2024-01-02"),
    ]
    result = _run(spark, pipeline, tmp_path, customers, orders)
    assert result == {"ES": 7.0}


def test_revenue_is_quantity_times_price_summed_per_country(spark, pipeline, tmp_path):
    customers = [
        Row(customer_id=1, name="a", age=30.0, country="MA", signup_date="2024-01-01"),
        Row(customer_id=2, name="b", age=30.0, country="MA", signup_date="2024-01-01"),
        Row(customer_id=3, name="c", age=30.0, country="US", signup_date="2024-01-01"),
    ]
    orders = [
        Row(order_id=1, customer_id=1, product="p", quantity=3, price=2.5, order_date="2024-01-02"),
        Row(order_id=2, customer_id=2, product="p", quantity=4, price=0.5, order_date="2024-01-02"),
        Row(order_id=3, customer_id=3, product="p", quantity=2, price=1.25, order_date="2024-01-02"),
    ]
    result = _run(spark, pipeline, tmp_path, customers, orders)
    assert result["MA"] == 9.5
    assert result["US"] == 2.5


def test_duplicate_orders_multiply_revenue_as_pandas_does(spark, pipeline, tmp_path):
    """A repeated customer_id in orders is a legitimate fan-out, not a bug."""
    customers = [
        Row(customer_id=1, name="a", age=30.0, country="FR", signup_date="2024-01-01"),
    ]
    orders = [
        Row(order_id=1, customer_id=1, product="p", quantity=1, price=10.0, order_date="2024-01-02"),
        Row(order_id=2, customer_id=1, product="q", quantity=1, price=10.0, order_date="2024-01-03"),
    ]
    result = _run(spark, pipeline, tmp_path, customers, orders)
    assert result == {"FR": 20.0}


def test_output_schema_has_no_synthesised_index_column(spark, pipeline, tmp_path):
    """reset_index() must emit nothing; an extra id column breaks schema parity."""
    customers = [
        Row(customer_id=1, name="a", age=30.0, country="FR", signup_date="2024-01-01"),
    ]
    orders = [
        Row(order_id=1, customer_id=1, product="p", quantity=1, price=1.0, order_date="2024-01-02"),
    ]
    base = _stage(spark, tmp_path, customers, orders)
    out = base + "/out"
    pipeline.run(spark, base, out)
    written = spark.read.option("header", True).csv(out + "/revenue_by_country")
    assert written.columns == ["country", "revenue"]


def test_empty_input_produces_no_rows(spark, pipeline, tmp_path):
    customers = spark.createDataFrame([], CUSTOMERS_SCHEMA)
    orders = spark.createDataFrame([], ORDERS_SCHEMA)
    base = str(tmp_path)
    customers.write.mode("overwrite").option("header", True).csv(base + "/customers.csv")
    orders.write.mode("overwrite").option("header", True).csv(base + "/orders.csv")
    out = base + "/out"
    pipeline.run(spark, base, out)
    written = spark.read.option("header", True).csv(out + "/revenue_by_country")
    assert written.count() == 0
