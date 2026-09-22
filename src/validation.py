import great_expectations as gx
import great_expectations.expectations as gxe
import pandas as pd


def validate_order(order: dict) -> list:
    df = pd.DataFrame([order])

    context = gx.get_context()

    suite = gx.ExpectationSuite(name="order_validation")

    suite.add_expectation(gxe.ExpectColumnValuesToBeBetween(
        column="num_items", min_value=1, max_value=50
    ))
    suite.add_expectation(gxe.ExpectColumnValuesToBeBetween(
        column="total_freight", min_value=0, max_value=10000
    ))
    suite.add_expectation(gxe.ExpectColumnValuesToBeBetween(
        column="total_payment_value", min_value=0, max_value=100000
    ))
    suite.add_expectation(gxe.ExpectColumnValuesToBeInSet(
        column="customer_state",
        value_set=["AC","AL","AM","AP","BA","CE","DF","ES","GO","MA","MG",
                   "MS","MT","PA","PB","PE","PI","PR","RJ","RN","RO","RR",
                   "RS","SC","SE","SP","TO"]
    ))

    data_source = context.data_sources.add_pandas("pandas_source")
    data_asset = data_source.add_dataframe_asset(name="order_asset")
    batch_definition = data_asset.add_batch_definition_whole_dataframe("batch")
    batch = batch_definition.get_batch(batch_parameters={"dataframe": df})

    result = batch.validate(suite)

    failed_expectations = [
        r["expectation_config"]["kwargs"]["column"]
        for r in result["results"] if not r["success"]
    ]

    return failed_expectations