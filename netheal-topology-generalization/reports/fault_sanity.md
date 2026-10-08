# Fault Injection Sanity

## star
- link_failure: {'type': 'link_failure', 'location': 'center->edge_3'}
- device_failure: {'type': 'device_failure', 'location': 'edge_0'}
- performance_degradation: {'type': 'performance_degradation', 'location': 'center->edge_3'}
- misconfiguration: {'type': 'misconfiguration', 'location': 'edge_0'}

## linear
- link_failure: {'type': 'link_failure', 'location': 'device_2->device_1'}
- device_failure: {'type': 'device_failure', 'location': 'device_1'}
- performance_degradation: {'type': 'performance_degradation', 'location': 'device_2->device_1'}
- misconfiguration: {'type': 'misconfiguration', 'location': 'device_1'}

## mesh
- link_failure: {'type': 'link_failure', 'location': 'device_3->device_0'}
- device_failure: {'type': 'device_failure', 'location': 'device_4'}
- performance_degradation: {'type': 'performance_degradation', 'location': 'device_3->device_0'}
- misconfiguration: {'type': 'misconfiguration', 'location': 'device_4'}
