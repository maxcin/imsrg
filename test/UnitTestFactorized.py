#!/usr/bin/env python3

import pyIMSRG

ms = pyIMSRG.ModelSpace(2,'He8','He8')
ut = pyIMSRG.UnitTest(ms)
passed = ut.TestFactorizedDoubleCommutators()

print('passed? ',passed)

exit(not passed)
