import unittest
import cutlass
from cutlass.utils import datatypes

class CupyTypeTest(unittest.TestCase):
    def test_cupy_type_initialized_after_availability_check(self):
        if not datatypes.is_cupy_available():
            self.skipTest("cupy not available")
        import cupy as cp
        dt = datatypes.cupy_type(cutlass.DataType.f32)
        self.assertEqual(dt, cp.float32)

if __name__ == "__main__":
    unittest.main()
