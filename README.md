# Anonymous artifact: Threat-Aware Edge Agents

Code and result logs for the double-blind submission.

- `main_fisco_real_v11.py`: main experiment (20 agents, permissioned PBFT ledger in Docker, detector, adaptive audit policy, DQN agents).
- `byzantine_engine.py`: reference-batch detector and attack models.
- `docker-compose.yml`, `setup_docker_chain.sh`: ledger setup (4 PBFT nodes).
- `results/`, `results_reportlog/`: logs behind the tables and figures.
- `requirements.txt`: Python dependencies.

Run the ledger with `docker compose up -d`, then start the experiment script. Seeds are command-line arguments.
