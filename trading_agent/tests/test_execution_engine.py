import unittest
from datetime import datetime

import pytz

from execution.execution_engine import SimulatedExecutor
from utils.types import Order, OrderType, SignalDirection


class TestExecutionEngine(unittest.TestCase):
    def test_filled_orders_queue_is_drained(self):
        executor = SimulatedExecutor(fill_probability=1.0)

        order1 = Order(
            symbol="TEST",
            quantity=5,
            order_type=OrderType.MARKET,
            direction=SignalDirection.LONG,
            limit_price=100.0,
            timestamp=datetime.now(pytz.UTC),
        )
        order2 = Order(
            symbol="TEST",
            quantity=3,
            order_type=OrderType.MARKET,
            direction=SignalDirection.SHORT,
            limit_price=102.0,
            timestamp=datetime.now(pytz.UTC),
        )

        executor.submit_order(order1)
        executor.submit_order(order2)

        first_read = executor.get_filled_orders()
        second_read = executor.get_filled_orders()

        self.assertEqual(len(first_read), 2)
        self.assertEqual(len(second_read), 0)


if __name__ == "__main__":
    unittest.main()
