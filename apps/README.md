# Applications

- `research_ui`: factor, model, and backtest exploration;
- `operations_ui`: workflow health, reconciliation, approvals, and kill switch;
- `execution_agent_windows`: isolated Windows broker process.

Application code is added only after its underlying `src/aquant` service is tested. UI files must not
become the sole implementation of production logic.

