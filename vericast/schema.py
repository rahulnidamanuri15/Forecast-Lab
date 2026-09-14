"""Database schema DDL and idempotent migrations for VeriCast tables.

Idempotent DDL plus a versioned provenance correction. The nowcast migration
relabels legacy late daily rows and their single-day scores without changing
prediction values or issuance timestamps. Stop old workers before first rollout:
source-aware conflict keys are incompatible with their old INSERT statements.

Usage:
    python -m vericast.schema
"""
import os
import psycopg
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

# Note demand_lag_6, not lag_7. Features at as_of = t predict t+1, so the
# same-weekday-last-week value for the *target* is y(t-6), not y(t-7). The same
# column feeds LightGBM's weekly signal and the seasonal_naive baseline.
TABLES = {
    "observations": """
        CREATE TABLE IF NOT EXISTS observations (
            id SERIAL PRIMARY KEY,
            city VARCHAR(100) NOT NULL,
            as_of DATE NOT NULL,
            pm2_5 FLOAT,
            pm10 FLOAT,
            temperature_2m_mean FLOAT,
            wind_speed_10m_max FLOAT,
            precipitation_sum FLOAT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(city, as_of)
        );
    """,
    "features": """
        CREATE TABLE IF NOT EXISTS features (
            id SERIAL PRIMARY KEY,
            city VARCHAR(100) NOT NULL,
            as_of DATE NOT NULL,

            pm2_5_lag_1 FLOAT,
            pm10_lag_1 FLOAT,
            temperature_lag_1 FLOAT,
            wind_speed_lag_1 FLOAT,
            precipitation_lag_1 FLOAT,

            pm2_5_roll_7 FLOAT,
            pm2_5_roll_30 FLOAT,
            pm10_roll_7 FLOAT,
            pm10_roll_30 FLOAT,

            day_of_week INTEGER,  -- 0=Monday, 6=Sunday
            month INTEGER,        -- 1-12
            -- BOOLEAN here vs INT on electricity_features.is_weekend: intentional,
            -- each matches its target's training frame (bool feeds float(True)=1.0
            -- identically). Unified only with a retrain + backtest delta; the
            -- leakage tests normalize via bool() so both read the same.
            is_weekend BOOLEAN,

            temperature_2m_mean FLOAT,
            wind_speed_10m_max FLOAT,
            precipitation_sum FLOAT,

            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(city, as_of)
        );
    """,
    # `source` splits the record: a 'daily' row was published before its
    # actual_pm2_5 existed, a 'backtest' row was written by
    # experiments/save_backtest_results.py with the actual already in hand.
    # /evaluation aggregates these rows directly and is the headline accuracy
    # claim, so averaging the two answers a different question than the one this
    # project exists to answer.
    "predictions": """
        CREATE TABLE IF NOT EXISTS predictions (
            id SERIAL PRIMARY KEY,
            city VARCHAR(100) NOT NULL,
            forecast_date DATE NOT NULL,
            predicted_pm2_5 FLOAT,
            actual_pm2_5 FLOAT,
            model VARCHAR(50) NOT NULL DEFAULT 'naive_baseline',
            source TEXT NOT NULL DEFAULT 'daily',
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(city, forecast_date, model, source)
        );
    """,
    # No city column, unlike its electricity counterpart: this table predates the
    # second target, so changing CITY needs a migration, not just an env var -
    # two cities' rows would land on the same (score_date, model) key.
    #
    # Rows come from vericast/pm25/score.py (one scored day, sample_size normally
    # 1) and, once at launch, from experiments/save_backtest_results.py (a whole
    # backtest, sample_size in the hundreds). /leaderboard reads the latest
    # score_date per model, so without `source` a backtest re-run today would
    # become the published leaderboard.
    "model_performance": """
        CREATE TABLE IF NOT EXISTS model_performance (
            id SERIAL PRIMARY KEY,
            city VARCHAR(100) NOT NULL DEFAULT 'Nagpur',
            score_date DATE NOT NULL,
            model VARCHAR(50) NOT NULL,
            mae FLOAT,
            rmse FLOAT,
            sample_size INTEGER,
            source TEXT NOT NULL DEFAULT 'daily',
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(city, score_date, model, source)
        );
    """,
    "electricity_observations": """
        CREATE TABLE IF NOT EXISTS electricity_observations (
            id SERIAL PRIMARY KEY,
            state VARCHAR(100) NOT NULL,
            as_of DATE NOT NULL,
            peak_demand_mw FLOAT NOT NULL,
            energy_met_mu FLOAT,
            temperature_2m_mean FLOAT,
            temperature_2m_max FLOAT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(state, as_of)
        );
    """,
    "electricity_features": """
        CREATE TABLE IF NOT EXISTS electricity_features (
            id SERIAL PRIMARY KEY,
            state VARCHAR(100) NOT NULL,
            as_of DATE NOT NULL,

            demand_lag_1 FLOAT,
            demand_lag_2 FLOAT,
            demand_lag_6 FLOAT,

            demand_roll_7_mean FLOAT,
            demand_roll_7_max FLOAT,
            demand_roll_30_mean FLOAT,

            temp_lag_1 FLOAT,
            temp_roll_7 FLOAT,
            cooling_degree_days FLOAT,

            day_of_week INT,
            month INT,
            is_weekend INT,

            temperature_2m_mean FLOAT,
            temperature_2m_max FLOAT,

            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(state, as_of)
        );
    """,
    # `source` as on predictions above: 'daily' means published before the actual
    # existed, 'backtest' means seeded by experiments/save_elec_backtest_results.py.
    "electricity_predictions": """
        CREATE TABLE IF NOT EXISTS electricity_predictions (
            id SERIAL PRIMARY KEY,
            state VARCHAR(100) NOT NULL,
            forecast_date DATE NOT NULL,
            predicted_demand_mw FLOAT NOT NULL,
            actual_demand_mw FLOAT,
            model VARCHAR(50) NOT NULL,
            source TEXT NOT NULL DEFAULT 'daily',
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(state, forecast_date, model, source)
        );
    """,
    # mape lives here and not on model_performance: adding a column to a live table
    # the published record reads from is unrequested risk. `source` is the one
    # exception, and it went on both tables together.
    "electricity_model_performance": """
        CREATE TABLE IF NOT EXISTS electricity_model_performance (
            id SERIAL PRIMARY KEY,
            state VARCHAR(100) NOT NULL,
            score_date DATE NOT NULL,
            model VARCHAR(50) NOT NULL,
            mae FLOAT NOT NULL,
            rmse FLOAT NOT NULL,
            mape FLOAT,
            sample_size INT NOT NULL,
            source TEXT NOT NULL DEFAULT 'daily',
            created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(state, score_date, model, source)
        );
    """,
}

# Columns added after the tables were already live. ADD COLUMN IF NOT EXISTS keeps
# this as idempotent as the CREATEs above, so `python -m vericast.schema` stays safe
# to run against production - which is what makes this the migration rather than a
# psql snippet only one machine ever ran.
#
# The backfill is the half that is easy to miss: ADD COLUMN ... DEFAULT 'daily'
# marks the existing backtest rows 'daily' too.
#
# All four backfills are bounded by created_at, because an unbounded one is a
# standing rule rather than a migration. Every writer now labels its own rows, so
# only rows predating the column need fixing - and those are exactly what the bound
# selects. Unbounded, a legitimate multi-day daily score would be relabelled
# 'backtest' and drop off /leaderboard permanently, with no down-migration to undo
# it. The other available bound (MAX(score_date) WHERE source = 'backtest') is a
# value the seeders *advance*, so "re-run the seeder, then run this module" was two
# routine operations that together relabelled verified rows.
#
# The date is in the past and must stay there: a future cutoff leaves the window
# open on rows being written today, which is the whole failure mode.
MIGRATION_CUTOFF = "2026-08-29"  # `source` landed this day

MIGRATIONS = (
    "ALTER TABLE model_performance "
    "ADD COLUMN IF NOT EXISTS source TEXT NOT NULL DEFAULT 'daily';",
    "ALTER TABLE electricity_model_performance "
    "ADD COLUMN IF NOT EXISTS source TEXT NOT NULL DEFAULT 'daily';",
    # Multi-city unlock: model_performance predates the second target and has no
    # city column, so require_city_of_record() pins the PM2.5 pipeline to Nagpur.
    # This migration is additive and idempotent: existing deploys gain a backfilled
    # city, new deploys get it from the CREATE TABLE above. The source-aware
    # UNIQUE(city, score_date, model, source) index is what writers use;
    # migrate_nowcasts() below replaces legacy source-blind keys with
    # provenance-aware ones (stop old workers before first rollout).
    "ALTER TABLE model_performance "
    "ADD COLUMN IF NOT EXISTS city VARCHAR(100) NOT NULL DEFAULT 'Nagpur';",
    "UPDATE model_performance SET city = 'Nagpur' WHERE city IS NULL;",
    # Row-level source columns first: every index below references `source`,
    # and legacy tables predate it — creating the index first fails there.
    "ALTER TABLE predictions "
    "ADD COLUMN IF NOT EXISTS source TEXT NOT NULL DEFAULT 'daily';",
    "ALTER TABLE electricity_predictions "
    "ADD COLUMN IF NOT EXISTS source TEXT NOT NULL DEFAULT 'daily';",
    # Backfill NULL models (pre-NOT NULL rows): NULLs are distinct in a UNIQUE
    # constraint, so a NULL model permits unlimited duplicates for one date.
    "UPDATE predictions SET model = 'naive_baseline' WHERE model IS NULL;",
    "ALTER TABLE predictions ALTER COLUMN model SET NOT NULL;",
    # NOTE (audit 3.4): a daily and a nowcast row may coexist for one
    # (city, forecast_date, model) by design — the provenance split requires
    # overlapping records (see test_schema_upgrade...: "former source-blind keys
    # must no longer prevent overlapping records"). A same-day re-run crossing
    # the cutoff therefore inserts a second row rather than conflicting. The
    # fix is at read time, not with a forbidding index: /evaluation never
    # merges provenances (separate verified/nowcast/backtest blocks, sorted on
    # verified MAE only) and /forecast?source=latest prefers daily for a tied
    # date (see app.py). No partial unique index is created here on purpose.
)


def migrate_backtest_labels(cur):
    """One-time backtest relabelling, version-gated so re-runs never move rows.

    Previously these UPDATEs lived in the unconditional MIGRATIONS tuple run on
    every create_tables(): the row-level bound is a LIVE subquery
    (MAX(score_date) WHERE source='backtest') that the seeders advance, so
    "re-seed, then run schema" relabelled additional genuine daily rows as
    backtest and dropped them off /evaluation. Version-gating evaluates the
    bound exactly once; the created_at < MIGRATION_CUTOFF bound additionally
    confines it to legacy rows.
    """
    cur.execute("""
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version TEXT PRIMARY KEY,
            applied_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)
    version = "2026-08-backtest-source-labels"
    cur.execute("SELECT 1 FROM schema_migrations WHERE version = %s", (version,))
    if cur.fetchone():
        return
    # Aggregate tables: a daily row scores one date (sample_size 1 by
    # construction) — anything larger came from a backtest.
    cur.execute(
        "UPDATE model_performance SET source = 'backtest' "
        "WHERE sample_size > 1 AND source = 'daily' "
        f"AND created_at < '{MIGRATION_CUTOFF}';")
    cur.execute(
        "UPDATE electricity_model_performance SET source = 'backtest' "
        "WHERE sample_size > 1 AND source = 'daily' "
        f"AND created_at < '{MIGRATION_CUTOFF}';")
    # Row-level tables: the backtest's own aggregate row records score_date =
    # its last evaluated date, so every prediction row it wrote has
    # forecast_date <= that date. A NULL subquery (no backtest seeded) updates
    # nothing. Evaluated once under this version gate — never again.
    cur.execute(
        "UPDATE predictions SET source = 'backtest' "
        "WHERE source = 'daily' AND forecast_date <= "
        "(SELECT MAX(score_date) FROM model_performance WHERE source = 'backtest') "
        f"AND created_at < '{MIGRATION_CUTOFF}';")
    cur.execute(
        "UPDATE electricity_predictions p SET source = 'backtest' "
        "WHERE p.source = 'daily' AND p.forecast_date <= "
        "(SELECT MAX(score_date) FROM electricity_model_performance "
        " WHERE source = 'backtest' AND state = p.state) "
        f"AND p.created_at < '{MIGRATION_CUTOFF}';")
    cur.execute("INSERT INTO schema_migrations (version) VALUES (%s)", (version,))


def migrate_nowcasts(cur):
    """One-time provenance correction and source-aware keys, in the DDL transaction.

    Stop old workers before the first rollout: their conflict targets no longer
    match the schema. Prediction values and issuance timestamps are never changed.
    """
    cur.execute("""
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version TEXT PRIMARY KEY,
            applied_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
    """)
    version = "2026-09-nowcast-provenance"
    cur.execute("SELECT 1 FROM schema_migrations WHERE version = %s", (version,))
    if cur.fetchone():
        return
    # Pre-flight BEFORE any DDL: relabelling a daily row to nowcast collides
    # with an already-labelled nowcast for the same (key, date, model), which
    # would raise on the new unique index, roll back the whole transaction
    # (version insert included), and wedge every subsequent run identically.
    # Fail here with the offending rows instead of wedging the migration.
    for prefix, key, zone in (("", "city", "UTC"),
                              ("electricity_", "state", "Asia/Kolkata")):
        predictions = prefix + "predictions"
        cur.execute(f"""
            SELECT d.{key}, d.forecast_date, d.model
            FROM {predictions} d
            JOIN {predictions} n ON n.{key} = d.{key}
              AND n.forecast_date = d.forecast_date AND n.model = d.model
              AND n.source = 'nowcast'
            WHERE d.source = 'daily' AND (d.created_at IS NULL OR
                d.created_at >= (d.forecast_date::timestamp AT TIME ZONE %s))
            LIMIT 10
        """, (zone,))
        conflicts = cur.fetchall()
        if conflicts:
            detail = "; ".join(
                f"{k} {fd} {m}" for k, fd, m in conflicts)
            raise RuntimeError(
                f"Provenance migration blocked: {len(conflicts)}+ daily row(s) in "
                f"{predictions} would collide with an existing nowcast row "
                f"({detail}). Resolve by deleting or re-sourcing the duplicate "
                f"row(s) before re-running python -m vericast.schema.")
    for prefix, key, zone in (("", "city", "UTC"),
                              ("electricity_", "state", "Asia/Kolkata")):
        predictions = prefix + "predictions"
        performance = prefix + "model_performance"
        for table, old_keys, new_keys in (
            (predictions, f"{key}_forecast_date_model", f"{key}, forecast_date, model, source"),
            (performance, "score_date_model" if not prefix else "state_score_date_model",
             f"{key}, score_date, model, source"),
        ):
            cur.execute(f"ALTER TABLE {table} DROP CONSTRAINT IF EXISTS {table}_{old_keys}_key")
            cur.execute(f"CREATE UNIQUE INDEX IF NOT EXISTS {table}_provenance_uidx "
                        f"ON {table} ({new_keys})")
        if not prefix:
            cur.execute("DROP INDEX IF EXISTS model_performance_city_score_model_uidx")

        # Old daily records issued during/after the target period were estimates,
        # not advance forecasts. Unknown issuance cannot prove advance status.
        # Move only scores belonging to rows corrected by this migration, not
        # an advance score sharing a date/model with an already-labelled nowcast.
        # Conflicting provenance rows fail the transaction rather than discard data.
        cur.execute(f"""
            WITH corrected AS (
                UPDATE {predictions} SET source = 'nowcast'
                WHERE source = 'daily' AND (created_at IS NULL OR
                    created_at >= (forecast_date::timestamp AT TIME ZONE %s))
                RETURNING {key}, forecast_date, model
            )
            UPDATE {performance} m SET source = 'nowcast'
            FROM corrected p
            WHERE m.{key} = p.{key} AND m.score_date = p.forecast_date
              AND m.model = p.model AND m.source = 'daily'
              AND m.sample_size = 1
        """, (zone,))
    cur.execute("INSERT INTO schema_migrations (version) VALUES (%s)", (version,))


def create_tables():
    # Runs unattended in ci.yml; without this a missing DSN reaches psycopg as None
    # and surfaces as a connection-string parse error rather than the real problem.
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL environment variable is not set")

    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            # Serialize schema jobs before any DDL, including concurrent CI/manual runs.
            cur.execute("SET LOCAL lock_timeout = '30s'")
            cur.execute("SELECT pg_advisory_xact_lock(830100)")
            for name, ddl in TABLES.items():
                cur.execute(ddl)
                print(f"[OK] {name}")

            for sql in MIGRATIONS:
                cur.execute(sql)
                print(f"[OK] {sql[:58]}... ({cur.rowcount} rows)")
            migrate_backtest_labels(cur)
            migrate_nowcasts(cur)
            conn.commit()

            for name in TABLES:
                cur.execute("""
                    SELECT column_name, data_type
                    FROM information_schema.columns
                    WHERE table_name = %s
                    ORDER BY ordinal_position;
                """, (name,))
                cols = cur.fetchall()
                print(f"\n{name} ({len(cols)} columns):")
                for col_name, dtype in cols:
                    print(f"  {col_name}: {dtype}")


if __name__ == "__main__":
    create_tables()
