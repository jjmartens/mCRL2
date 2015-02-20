#!/usr/bin/env python

#~ Copyright 2015 Wieger Wesselink.
#~ Distributed under the Boost Software License, Version 1.0.
#~ (See accompanying file LICENSE_1_0.txt or http://www.boost.org/LICENSE_1_0.txt)

import os
import sys
sys.path += [os.path.join(os.path.dirname(__file__), '..', 'python')]
import testing
from regression_testing import *
from testcommand import YmlTest

class TestRunner(testing.TestRunner):
    def __init__(self):
        super(TestRunner, self).__init__()
        ymlfile = self.ymlfile
        mcrl2file = self.mcrl2file
        abspath = self.abspath
        pbessolve_files = sorted(os.listdir(abspath('pbessolve')))
        self.tests = [
                       YmlTest('ticket_283',  ymlfile('mcrl22lps'),         [abspath('tickets/283/1.mcrl2')]),
                       YmlTest('ticket_325' , ymlfile('alphabet'),          [abspath('tickets/325/1.mcrl2')]),
                       YmlTest('ticket_1090', ymlfile('ticket_1090'),       [abspath('tickets/1090/1.mcrl2'), abspath('tickets/1090/1.mcf')]),
                       #YmlTest('ticket_1093', ymlfile('alphabet'),          [abspath('tickets/1093/1.mcrl2')], { 'timeout': 300, 'memlimit': 500000000 }),
                       # YmlTest('ticket_1122', ymlfile('ticket_1122'),       [abspath('tickets/1122/1.mcrl2')]), # specification is not well typed
                       YmlTest('ticket_1127', ymlfile('mcrl22lps'),         [abspath('tickets/1127/1.mcrl2')]),
                       PbesrewrTest('ticket_1143', [abspath('tickets/1143/1.txt')], 'quantifier-one-point'),
                       YmlTest('ticket_1144', ymlfile('lpsbisim2pbes'),     [abspath('tickets/1144/test1.txt'), abspath('tickets/1144/test2.txt')]),
                       CountStatesTest('ticket_1167', [mcrl2file('/examples/academic/abp/abp.mcrl2')], 74),
                       YmlTest('ticket_1218', ymlfile('alphabet'),          [abspath('tickets/1218/1.mcrl2')]),
                       YmlTest('ticket_1234', ymlfile('lpsbinary'),         [mcrl2file('/examples/academic/cabp/cabp.mcrl2')]),
                       YmlTest('ticket_1241', ymlfile('alphabet'),          [abspath('tickets/1241/1.mcrl2')]),
                       YmlTest('ticket_1249', ymlfile('alphabet'),          [abspath('tickets/1249/1.mcrl2')]),
                       YmlTest('ticket_1315', ymlfile('alphabet'),          [abspath('tickets/1315/1.mcrl2')]),
                       YmlTest('ticket_1316', ymlfile('alphabet'),          [abspath('tickets/1316/1.mcrl2')]),
                       YmlTest('ticket_1317', ymlfile('alphabet'),          [abspath('tickets/1317/1.mcrl2')]),
                       YmlTest('ticket_1318', ymlfile('alphabet'),          [abspath('tickets/1318/1.mcrl2')]),
                       YmlTest('ticket_1319', ymlfile('alphabet'),          [abspath('tickets/1319/1.mcrl2')]),
                       YmlTest('ticket_1322', ymlfile('pbesstategraph'),    [abspath('tickets/1322/1.txt')]),
                       LpsconfcheckCtauTest('lpsconfcheck_1', [mcrl2file('/examples/academic/cabp/cabp.mcrl2')], 'T', (0, 18)),
                       LpsconfcheckCtauTest('lpsconfcheck_2', [mcrl2file('/examples/academic/trains/trains.mcrl2')], 'T', (0, 9)),
                       #LpsconfcheckCtauTest('lpsconfcheck_3', [mcrl2file('/examples/industrial/chatbox/chatbox.mcrl2')], 'x', (40, 72), { 'timeout': 300, 'memlimit': 500000000 } ),
                     ] + \
                     [
                       YmlTest('regression_pbessolve_{}'.format(file[:-4]), ymlfile('pbessolve'), [abspath('pbessolve/{}'.format(file))]) for file in pbessolve_files
                     ]
        for test in self.tests:
            test.name = 'regression_' + test.name

    def abspath(self, file):
        return os.path.abspath(os.path.join(os.path.dirname(__file__), file))

if __name__ == "__main__":
    sys.exit(TestRunner().main())
