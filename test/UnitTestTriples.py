#!/usr/bin/env python3

import pyIMSRG

ms = pyIMSRG.ModelSpace(3,'He8','He8')
ut = pyIMSRG.UnitTest(ms)
passed = ut.TestPerturbativeTriples()

print('passed? ',passed)

