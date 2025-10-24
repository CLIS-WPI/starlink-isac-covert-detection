![Header](header.png)

# Covert Channel Detection in LEO Satellite ISAC Systems

**ML-based detector for covert signals in LEO satellites using 3GPP TR38.811 NTN channel models.**


### Run with Docker

```bash
# Build image
docker build -t covert_l .

# Run container
docker run --gpus all --user root -it \
  -v "$(pwd)":/workspace \
  -w /workspace \
  covert_l:latest

# Execute pipeline
python3 main.py
```

