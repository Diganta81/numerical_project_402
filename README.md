# Simplified Newton-Raphson Power Flow

CSE 402 (Numerical Analysis, Simulation and Modeling) project.

Reproduction and extension of T. Kulworawanichpong, *"Simplified Newton-Raphson
power-flow solution method"*, Int. J. Electrical Power & Energy Systems 32
(2010) 551-558.

### Commands:

1. Initial setup:
```bash
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

2. Run tests for Paper example and Jacobian:
```bash
python -m pytest tests/test_paper_example.py
python -m pytest tests/test_jacobians.py
```

3. Run the three bus example:
```bash
python scripts/01_worked_example_3bus.py
```

