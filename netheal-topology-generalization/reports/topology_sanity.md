# Topology Sanity Report

## Observation shape check

- **star**: {'discovery_matrix': (6, 6), 'device_status': (6, 10), 'recent_diagnostics': (10, 6), 'episode_metadata': (4,)}
- **linear**: {'discovery_matrix': (6, 6), 'device_status': (6, 10), 'recent_diagnostics': (10, 6), 'episode_metadata': (4,)}
- **mesh**: {'discovery_matrix': (6, 6), 'device_status': (6, 10), 'recent_diagnostics': (10, 6), 'episode_metadata': (4,)}

## Fixed 6-device check (10 samples each)

- **star**: devices=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6] (unique={6}), edges mean=10.0
- **linear**: devices=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6] (unique={6}), edges mean=10.0
- **mesh**: devices=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6] (unique={6}), edges mean=11.4

## Mesh instances M1/M2/M3

- **M1** (seed=90001): devices=[6, 6, 6, 6, 6], edges=[14, 10, 12, 12, 18]
- **M2** (seed=90002): devices=[6, 6, 6, 6, 6], edges=[10, 12, 12, 18, 10]
- **M3** (seed=90003): devices=[6, 6, 6, 6, 6], edges=[12, 12, 18, 10, 10]

## Hierarchical (record only — excluded from Primary)

Default hierarchical generator ignores min/max_devices; not used in Primary.
