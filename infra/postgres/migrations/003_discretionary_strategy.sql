BEGIN;

CREATE TABLE aquant.knowledge_candidates (
    candidate_id text PRIMARY KEY,
    title text NOT NULL CHECK (length(trim(title)) > 0),
    source_kind text NOT NULL CHECK (source_kind IN (
        'ACADEMIC_PAPER', 'BROKER_RESEARCH', 'GITHUB_TOOL', 'REGULATION',
        'INVESTOR_PRACTICE', 'OVERSEAS_RESEARCH', 'ALTERNATIVE_DATA', 'FAILURE_CASE'
    )),
    source_uri text NOT NULL CHECK (length(trim(source_uri)) > 0),
    hypothesis text NOT NULL CHECK (length(trim(hypothesis)) > 0),
    applicability text NOT NULL CHECK (length(trim(applicability)) > 0),
    status text NOT NULL CHECK (status IN (
        'DISCOVERED', 'REPRODUCED', 'VALIDATED', 'SHADOW', 'APPROVED',
        'PRODUCTION', 'REJECTED', 'DEPRECATED'
    )),
    evidence_hash text CHECK (evidence_hash IS NULL OR evidence_hash ~ '^[0-9a-f]{64}$'),
    failure_reason text,
    approved_by text,
    acquired_at timestamptz NOT NULL,
    updated_at timestamptz NOT NULL DEFAULT now(),
    CHECK (status <> 'REJECTED' OR COALESCE(length(trim(failure_reason)), 0) > 0),
    CHECK (status IN ('DISCOVERED', 'REJECTED') OR evidence_hash IS NOT NULL),
    CHECK (
        status NOT IN ('APPROVED', 'PRODUCTION')
        OR COALESCE(length(trim(approved_by)), 0) > 0
    )
);

CREATE TABLE aquant.discretionary_radar_signals (
    signal_id text PRIMARY KEY,
    symbol text NOT NULL REFERENCES aquant.instruments(symbol),
    signal_type text NOT NULL,
    description text NOT NULL,
    model_version text NOT NULL,
    observed_at timestamptz NOT NULL,
    available_at timestamptz NOT NULL,
    data_release_id text NOT NULL REFERENCES aquant.data_releases(release_id),
    created_at timestamptz NOT NULL DEFAULT now(),
    CHECK (available_at >= observed_at)
);

CREATE TABLE aquant.discretionary_research_cases (
    case_id text PRIMARY KEY,
    symbol text NOT NULL REFERENCES aquant.instruments(symbol),
    radar_signal_id text NOT NULL REFERENCES aquant.discretionary_radar_signals(signal_id),
    title text NOT NULL CHECK (length(trim(title)) > 0),
    market_question text NOT NULL CHECK (length(trim(market_question)) > 0),
    created_at timestamptz NOT NULL
);

CREATE TABLE aquant.discretionary_hypotheses (
    case_id text NOT NULL REFERENCES aquant.discretionary_research_cases(case_id),
    hypothesis_id text NOT NULL,
    kind text NOT NULL CHECK (kind IN ('FUNDAMENTAL', 'FLOW_TECHNICAL', 'NOISE')),
    statement text NOT NULL CHECK (length(trim(statement)) > 0),
    prior_probability numeric(18,12) NOT NULL CHECK (prior_probability > 0 AND prior_probability <= 1),
    falsification_conditions jsonb NOT NULL CHECK (jsonb_typeof(falsification_conditions) = 'array'),
    PRIMARY KEY (case_id, hypothesis_id),
    UNIQUE (case_id, kind)
);

CREATE TABLE aquant.discretionary_causal_nodes (
    case_id text NOT NULL REFERENCES aquant.discretionary_research_cases(case_id),
    stage text NOT NULL CHECK (stage IN (
        'INDUSTRY_CHANGE', 'CUSTOMER_BEHAVIOR', 'COMPANY_ORDERS', 'REVENUE',
        'MARGIN', 'CASH_FLOW', 'SHAREHOLDER_RETURN'
    )),
    stage_order smallint NOT NULL CHECK (stage_order BETWEEN 1 AND 7),
    claim text NOT NULL CHECK (length(trim(claim)) > 0),
    known_facts jsonb NOT NULL CHECK (jsonb_typeof(known_facts) = 'array'),
    unknowns jsonb NOT NULL CHECK (jsonb_typeof(unknowns) = 'array'),
    indicator_ids jsonb NOT NULL CHECK (jsonb_typeof(indicator_ids) = 'array'),
    next_validation_at timestamptz NOT NULL,
    falsification_condition text NOT NULL CHECK (length(trim(falsification_condition)) > 0),
    revision_no integer NOT NULL CHECK (revision_no >= 0),
    available_at timestamptz NOT NULL,
    recorded_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (case_id, stage, revision_no),
    UNIQUE (case_id, stage_order, revision_no),
    CHECK (recorded_at >= available_at)
);

CREATE TABLE aquant.discretionary_evidence (
    evidence_id text PRIMARY KEY,
    case_id text NOT NULL REFERENCES aquant.discretionary_research_cases(case_id),
    variable_id text NOT NULL,
    source_id text NOT NULL,
    source_record_id text NOT NULL,
    source_uri text NOT NULL,
    family text NOT NULL CHECK (family IN (
        'PRICE', 'CUSTOMER', 'SUPPLIER', 'COMPANY_ACTION', 'CHANNEL', 'COMPETITOR',
        'OFFICIAL_DISCLOSURE', 'FINANCIAL_RESULT'
    )),
    dependency_cluster_id text NOT NULL,
    acquisition_basis text NOT NULL CHECK (acquisition_basis IN (
        'PUBLIC', 'LICENSED', 'COMPLIANT_RESEARCH'
    )),
    method_version text NOT NULL,
    summary text NOT NULL CHECK (length(trim(summary)) > 0),
    observed_at timestamptz NOT NULL,
    published_at timestamptz NOT NULL,
    collected_at timestamptz NOT NULL,
    available_at timestamptz NOT NULL,
    raw_content_sha256 text NOT NULL CHECK (raw_content_sha256 ~ '^[0-9a-f]{64}$'),
    maturity text NOT NULL CHECK (maturity IN (
        'LEADING', 'OPERATING_CONFIRMED', 'FINANCIAL_CONFIRMED'
    )),
    validation_domains jsonb NOT NULL CHECK (jsonb_typeof(validation_domains) = 'array'),
    quality jsonb NOT NULL CHECK (jsonb_typeof(quality) = 'object'),
    likelihood_ratios jsonb NOT NULL CHECK (jsonb_typeof(likelihood_ratios) = 'object'),
    falsifies_fundamental boolean NOT NULL DEFAULT false,
    supersedes_evidence_id text REFERENCES aquant.discretionary_evidence(evidence_id),
    created_at timestamptz NOT NULL DEFAULT now(),
    CHECK (observed_at <= published_at),
    CHECK (published_at <= collected_at),
    CHECK (collected_at <= available_at),
    UNIQUE (source_id, source_record_id, method_version)
);

CREATE FUNCTION aquant.prevent_discretionary_evidence_mutation()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION 'discretionary evidence is append-only; insert a superseding record';
END;
$$;

CREATE TRIGGER discretionary_evidence_append_only
BEFORE UPDATE OR DELETE ON aquant.discretionary_evidence
FOR EACH ROW EXECUTE FUNCTION aquant.prevent_discretionary_evidence_mutation();

CREATE TABLE aquant.discretionary_belief_snapshots (
    snapshot_id text PRIMARY KEY CHECK (snapshot_id ~ '^[0-9a-f]{64}$'),
    case_id text NOT NULL REFERENCES aquant.discretionary_research_cases(case_id),
    asof_time timestamptz NOT NULL,
    data_release_id text NOT NULL REFERENCES aquant.data_releases(release_id),
    vintage text NOT NULL CHECK (vintage IN ('REAL_TIME', 'RETROSPECTIVE')),
    probabilities jsonb NOT NULL CHECK (jsonb_typeof(probabilities) = 'object'),
    evidence_ids jsonb NOT NULL CHECK (jsonb_typeof(evidence_ids) = 'array'),
    evidence_hash text NOT NULL CHECK (evidence_hash ~ '^[0-9a-f]{64}$'),
    research_stage text NOT NULL CHECK (research_stage IN (
        'RADAR', 'HYPOTHESIS', 'WATCHLIST', 'OBSERVATION', 'EVIDENCE', 'CORE', 'INVALIDATED'
    )),
    evidence_strength numeric(18,12) NOT NULL CHECK (evidence_strength BETWEEN 0 AND 1),
    independent_clusters jsonb NOT NULL CHECK (jsonb_typeof(independent_clusters) = 'array'),
    non_price_clusters jsonb NOT NULL CHECK (jsonb_typeof(non_price_clusters) = 'array'),
    validation_domains jsonb NOT NULL CHECK (jsonb_typeof(validation_domains) = 'array'),
    falsifier_seen boolean NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (case_id, asof_time, data_release_id, vintage, evidence_hash)
);

CREATE TABLE aquant.discretionary_market_expectations (
    expectation_id text PRIMARY KEY CHECK (expectation_id ~ '^[0-9a-f]{64}$'),
    case_id text NOT NULL REFERENCES aquant.discretionary_research_cases(case_id),
    fundamental_hypothesis_id text NOT NULL,
    implied_probability numeric(18,12) NOT NULL CHECK (implied_probability BETWEEN 0 AND 1),
    current_price numeric(20,8) NOT NULL CHECK (current_price > 0),
    rationale text NOT NULL CHECK (length(trim(rationale)) > 0),
    source_uri text NOT NULL,
    observed_at timestamptz NOT NULL,
    available_at timestamptz NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    CHECK (available_at >= observed_at),
    FOREIGN KEY (case_id, fundamental_hypothesis_id)
        REFERENCES aquant.discretionary_hypotheses(case_id, hypothesis_id)
);

CREATE TABLE aquant.discretionary_calibration_outcomes (
    outcome_id text PRIMARY KEY,
    evidence_id text NOT NULL REFERENCES aquant.discretionary_evidence(evidence_id),
    source_id text NOT NULL,
    method_version text NOT NULL,
    layer text NOT NULL CHECK (layer IN ('MEASUREMENT', 'TRANSMISSION', 'INVESTMENT')),
    predicted_probability numeric(18,12) NOT NULL CHECK (predicted_probability BETWEEN 0 AND 1),
    realized_outcome numeric(18,12) NOT NULL CHECK (realized_outcome BETWEEN 0 AND 1),
    prediction_available_at timestamptz NOT NULL,
    outcome_available_at timestamptz NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    CHECK (outcome_available_at > prediction_available_at),
    UNIQUE (evidence_id, layer, outcome_available_at)
);

CREATE TABLE aquant.discretionary_allocation_proposals (
    proposal_id text PRIMARY KEY CHECK (proposal_id ~ '^[0-9a-f]{64}$'),
    case_id text NOT NULL REFERENCES aquant.discretionary_research_cases(case_id),
    symbol text NOT NULL REFERENCES aquant.instruments(symbol),
    asof_time timestamptz NOT NULL,
    data_release_id text NOT NULL REFERENCES aquant.data_releases(release_id),
    belief_snapshot_id text NOT NULL REFERENCES aquant.discretionary_belief_snapshots(snapshot_id),
    market_expectation_id text NOT NULL REFERENCES aquant.discretionary_market_expectations(expectation_id),
    payoff_profile jsonb NOT NULL CHECK (jsonb_typeof(payoff_profile) = 'object'),
    policy_hash text NOT NULL CHECK (policy_hash ~ '^[0-9a-f]{64}$'),
    action text NOT NULL CHECK (action IN ('WATCH', 'OPEN', 'ADD', 'HOLD', 'REDUCE', 'EXIT')),
    tier text NOT NULL CHECK (tier IN ('NONE', 'OBSERVATION', 'EVIDENCE', 'CORE')),
    target_weight numeric(18,10) NOT NULL CHECK (target_weight BETWEEN 0 AND 1),
    current_weight numeric(18,10) NOT NULL CHECK (current_weight BETWEEN 0 AND 1),
    posterior_probability numeric(18,12) NOT NULL CHECK (posterior_probability BETWEEN 0 AND 1),
    market_probability numeric(18,12) NOT NULL CHECK (market_probability BETWEEN 0 AND 1),
    expected_return numeric(18,12) NOT NULL,
    evidence_strength numeric(18,12) NOT NULL CHECK (evidence_strength BETWEEN 0 AND 1),
    evidence_hash text NOT NULL CHECK (evidence_hash ~ '^[0-9a-f]{64}$'),
    reasons jsonb NOT NULL CHECK (jsonb_typeof(reasons) = 'array'),
    requires_manual_approval boolean NOT NULL DEFAULT true CHECK (requires_manual_approval),
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE FUNCTION aquant.validate_discretionary_proposal_lineage()
RETURNS trigger
LANGUAGE plpgsql
AS $$
DECLARE
    snapshot_case text;
    snapshot_release text;
    snapshot_vintage text;
    snapshot_evidence_hash text;
    expectation_case text;
BEGIN
    SELECT case_id, data_release_id, vintage, evidence_hash
    INTO snapshot_case, snapshot_release, snapshot_vintage, snapshot_evidence_hash
    FROM aquant.discretionary_belief_snapshots
    WHERE snapshot_id = NEW.belief_snapshot_id;

    SELECT case_id
    INTO expectation_case
    FROM aquant.discretionary_market_expectations
    WHERE expectation_id = NEW.market_expectation_id;

    IF snapshot_vintage <> 'REAL_TIME' THEN
        RAISE EXCEPTION 'retrospective belief snapshots cannot generate allocation proposals';
    END IF;
    IF snapshot_case <> NEW.case_id
       OR snapshot_release <> NEW.data_release_id
       OR snapshot_evidence_hash <> NEW.evidence_hash THEN
        RAISE EXCEPTION 'allocation proposal belief lineage mismatch';
    END IF;
    IF expectation_case <> NEW.case_id THEN
        RAISE EXCEPTION 'allocation proposal market expectation lineage mismatch';
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER discretionary_proposal_lineage_gate
BEFORE INSERT ON aquant.discretionary_allocation_proposals
FOR EACH ROW EXECUTE FUNCTION aquant.validate_discretionary_proposal_lineage();

CREATE TABLE aquant.discretionary_allocation_approvals (
    approval_id text PRIMARY KEY CHECK (approval_id ~ '^[0-9a-f]{64}$'),
    proposal_id text NOT NULL UNIQUE REFERENCES aquant.discretionary_allocation_proposals(proposal_id),
    approved_weight numeric(18,10) NOT NULL CHECK (approved_weight BETWEEN 0 AND 1),
    approved_by text NOT NULL CHECK (length(trim(approved_by)) > 0),
    rationale text NOT NULL CHECK (length(trim(rationale)) > 0),
    approved_at timestamptz NOT NULL,
    manual_confirmation boolean NOT NULL CHECK (manual_confirmation),
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE FUNCTION aquant.validate_discretionary_approval()
RETURNS trigger
LANGUAGE plpgsql
AS $$
DECLARE
    proposal_weight numeric(18,10);
    proposal_asof timestamptz;
BEGIN
    SELECT target_weight, asof_time
    INTO proposal_weight, proposal_asof
    FROM aquant.discretionary_allocation_proposals
    WHERE proposal_id = NEW.proposal_id;

    IF NEW.approved_weight > proposal_weight THEN
        RAISE EXCEPTION 'approved weight cannot exceed proposed weight';
    END IF;
    IF NEW.approved_at < proposal_asof THEN
        RAISE EXCEPTION 'approval cannot predate proposal';
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER discretionary_approval_gate
BEFORE INSERT ON aquant.discretionary_allocation_approvals
FOR EACH ROW EXECUTE FUNCTION aquant.validate_discretionary_approval();

CREATE FUNCTION aquant.prevent_discretionary_decision_mutation()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION '% is append-only', TG_TABLE_NAME;
END;
$$;

CREATE TRIGGER discretionary_proposals_append_only
BEFORE UPDATE OR DELETE ON aquant.discretionary_allocation_proposals
FOR EACH ROW EXECUTE FUNCTION aquant.prevent_discretionary_decision_mutation();

CREATE TRIGGER discretionary_approvals_append_only
BEFORE UPDATE OR DELETE ON aquant.discretionary_allocation_approvals
FOR EACH ROW EXECUTE FUNCTION aquant.prevent_discretionary_decision_mutation();

CREATE TRIGGER discretionary_radar_append_only
BEFORE UPDATE OR DELETE ON aquant.discretionary_radar_signals
FOR EACH ROW EXECUTE FUNCTION aquant.prevent_discretionary_decision_mutation();

CREATE TRIGGER discretionary_hypotheses_append_only
BEFORE UPDATE OR DELETE ON aquant.discretionary_hypotheses
FOR EACH ROW EXECUTE FUNCTION aquant.prevent_discretionary_decision_mutation();

CREATE TRIGGER discretionary_causal_nodes_append_only
BEFORE UPDATE OR DELETE ON aquant.discretionary_causal_nodes
FOR EACH ROW EXECUTE FUNCTION aquant.prevent_discretionary_decision_mutation();

CREATE TRIGGER discretionary_beliefs_append_only
BEFORE UPDATE OR DELETE ON aquant.discretionary_belief_snapshots
FOR EACH ROW EXECUTE FUNCTION aquant.prevent_discretionary_decision_mutation();

CREATE TRIGGER discretionary_expectations_append_only
BEFORE UPDATE OR DELETE ON aquant.discretionary_market_expectations
FOR EACH ROW EXECUTE FUNCTION aquant.prevent_discretionary_decision_mutation();

CREATE TRIGGER discretionary_calibration_append_only
BEFORE UPDATE OR DELETE ON aquant.discretionary_calibration_outcomes
FOR EACH ROW EXECUTE FUNCTION aquant.prevent_discretionary_decision_mutation();

CREATE TABLE aquant.discretionary_postmortems (
    postmortem_id text PRIMARY KEY,
    case_id text NOT NULL REFERENCES aquant.discretionary_research_cases(case_id),
    asof_time timestamptz NOT NULL,
    measurement_ledger jsonb NOT NULL CHECK (jsonb_typeof(measurement_ledger) = 'object'),
    transmission_ledger jsonb NOT NULL CHECK (jsonb_typeof(transmission_ledger) = 'object'),
    investment_ledger jsonb NOT NULL CHECK (jsonb_typeof(investment_ledger) = 'object'),
    error_patterns jsonb NOT NULL CHECK (jsonb_typeof(error_patterns) = 'array'),
    reusable_lessons jsonb NOT NULL CHECK (jsonb_typeof(reusable_lessons) = 'array'),
    available_at timestamptz NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    CHECK (created_at >= available_at)
);

CREATE TRIGGER discretionary_postmortems_append_only
BEFORE UPDATE OR DELETE ON aquant.discretionary_postmortems
FOR EACH ROW EXECUTE FUNCTION aquant.prevent_discretionary_decision_mutation();

CREATE INDEX discretionary_evidence_case_available_idx
    ON aquant.discretionary_evidence(case_id, available_at);
CREATE INDEX discretionary_calibration_source_available_idx
    ON aquant.discretionary_calibration_outcomes(source_id, method_version, layer, outcome_available_at);
CREATE INDEX discretionary_cases_symbol_idx
    ON aquant.discretionary_research_cases(symbol);

COMMIT;
