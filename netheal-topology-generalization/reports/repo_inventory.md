# NetHeal repo inventory (Step 2)

- **Environment**: `netheal/environment/env.py` — `NetworkTroubleshootingEnv`
- **Topology**: `netheal/network/topology.py` — star/linear/mesh/hierarchical generators
- **Faults**: `netheal/faults/injector.py` — `FaultType`, `FaultInjector`
- **Metrics**: `netheal/evaluation/wrapper.py` — `MetricsCollectorWrapper`, `CompetitionEvaluator`
- **PPO example**: `docs/guides/training-sb3.md` — `MultiInputPolicy`
- **Baseline note**: use Gym `get_valid_actions()` (not MCP `dummy_agent.py`)

Primary experiment code: `netheal-topology-generalization/src/`
