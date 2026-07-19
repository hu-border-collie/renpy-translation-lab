testsuite global:
    teardown:
        exit

testcase smoke:
    $ _test.timeout = 5.0
    $ _test.transition_timeout = 0.05
    run Jump("start")
    advance until "Translation fixture complete." raw
    advance
