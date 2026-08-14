# Cloud Targets

This project is built to support Arm64 cloud environments. The architecture must be portable across Arm64 Linux cloud environments wherever the selected inference runtime supports them.

## Tested Platforms

- **None initially** (Current development is on x86_64)

## Potential / Untested Targets

- **AWS Graviton**: Amazon's custom Arm-based processors (Graviton2, Graviton3, Graviton4).
- **Google Axion**: Google's custom Arm-based processors.
- **Microsoft Cobalt**: Microsoft's custom Arm-based processors.

## Deployment Requirements

- **CPU Architecture**: `aarch64` / `arm64`
- **OS**: Linux
- **Docker**: Optional but recommended for reproducible environments.
