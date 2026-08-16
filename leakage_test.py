import os
import psycopg
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

def run_leakage_test():
    """Run leakage test to ensure no future data leakage in features"""
    try:
        with psycopg.connect(DATABASE_URL) as conn:
            with conn.cursor() as cur:
                # 1. Get all observations for Nagpur, ordered by as_of
                cur.execute("""
                    SELECT as_of, pm2_5, pm10, temperature_2m_mean, wind_speed_10m_max, precipitation_sum
                    FROM observations
                    WHERE city = 'Nagpur'
                    ORDER BY as_of
                """)
                obs_rows = cur.fetchall()
                print(f"Fetched {len(obs_rows)} observation records")

                # Build a list of dates and a dict for quick lookup
                obs_dates = [row[0] for row in obs_rows]
                obs_dict = {}
                for row in obs_rows:
                    date = row[0]
                    obs_dict[date] = {
                        'pm2_5': row[1],
                        'pm10': row[2],
                        'temperature': row[3],
                        'wind': row[4],
                        'precip': row[5]
                    }

                # 2. Get all features for Nagpur, ordered by as_of
                cur.execute("""
                    SELECT as_of,
                           pm2_5_lag_1, pm10_lag_1, temperature_lag_1, wind_speed_lag_1, precipitation_lag_1,
                           pm2_5_roll_7, pm2_5_roll_30, pm10_roll_7, pm10_roll_30,
                           day_of_week, month, is_weekend,
                           temperature_2m_mean, wind_speed_10m_max, precipitation_sum
                    FROM features
                    WHERE city = 'Nagpur'
                    ORDER BY as_of
                """)
                feat_rows = cur.fetchall()
                print(f"Fetched {len(feat_rows)} feature records")

                # 3. Check each feature row
                tolerance = 1e-9  # for floating point comparison
                errors = []

                for i, feat in enumerate(feat_rows):
                    as_of = feat[0]
                    # Same-day weather features (indices 13,14,15 in feat tuple)
                    feat_temp = feat[13]
                    feat_wind = feat[14]
                    feat_precip = feat[15]

                    # Get corresponding observation
                    obs = obs_dict.get(as_of)
                    if obs is None:
                        errors.append(f"Date {as_of}: missing in observations")
                        continue

                    # Check same-day weather
                    if abs(feat_temp - obs['temperature']) > tolerance:
                        errors.append(f"Date {as_of}: temperature mismatch: feature={feat_temp}, obs={obs['temperature']}")
                    if abs(feat_wind - obs['wind']) > tolerance:
                        errors.append(f"Date {as_of}: wind mismatch: feature={feat_wind}, obs={obs['wind']}")
                    if abs(feat_precip - obs['precip']) > tolerance:
                        errors.append(f"Date {as_of}: precipitation mismatch: feature={feat_precip}, obs={obs['precip']}")

                    # Check lagged features (indices 1-5 in feat tuple)
                    if i > 0:
                        prev_date = obs_dates[i-1]
                        prev_obs = obs_dict[prev_date]
                        # pm2_5_lag_1
                        if feat[1] is not None:
                            if abs(feat[1] - prev_obs['pm2_5']) > tolerance:
                                errors.append(f"Date {as_of}: pm2_5_lag_1 mismatch: feature={feat[1]}, obs_previous={prev_obs['pm2_5']}")
                        else:
                            errors.append(f"Date {as_of}: pm2_5_lag_1 is None but previous day exists")
                        # pm10_lag_1
                        if feat[2] is not None:
                            if abs(feat[2] - prev_obs['pm10']) > tolerance:
                                errors.append(f"Date {as_of}: pm10_lag_1 mismatch: feature={feat[2]}, obs_previous={prev_obs['pm10']}")
                        else:
                            errors.append(f"Date {as_of}: pm10_lag_1 is None but previous day exists")
                        # temperature_lag_1
                        if feat[3] is not None:
                            if abs(feat[3] - prev_obs['temperature']) > tolerance:
                                errors.append(f"Date {as_of}: temperature_lag_1 mismatch: feature={feat[3]}, obs_previous={prev_obs['temperature']}")
                        else:
                            errors.append(f"Date {as_of}: temperature_lag_1 is None but previous day exists")
                        # wind_speed_lag_1
                        if feat[4] is not None:
                            if abs(feat[4] - prev_obs['wind']) > tolerance:
                                errors.append(f"Date {as_of}: wind_speed_lag_1 mismatch: feature={feat[4]}, obs_previous={prev_obs['wind']}")
                        else:
                            errors.append(f"Date {as_of}: wind_speed_lag_1 is None but previous day exists")
                        # precipitation_lag_1
                        if feat[5] is not None:
                            if abs(feat[5] - prev_obs['precip']) > tolerance:
                                errors.append(f"Date {as_of}: precipitation_lag_1 mismatch: feature={feat[5]}, obs_previous={prev_obs['precip']}")
                        else:
                            errors.append(f"Date {as_of}: precipitation_lag_1 is None but previous day exists")
                    else:
                        # First row should have null lagged features
                        if feat[1] is not None or feat[2] is not None or feat[3] is not None or feat[4] is not None or feat[5] is not None:
                            errors.append(f"Date {as_of}: first row should have null lagged features")

                    # Check rolling averages for PM2.5 (indices 6,7)
                    # 7-day rolling average (index 6)
                    if i >= 6:
                        # Get the last 7 days including current day: indices i-6 to i
                        pm2_5_values = [obs_dict[obs_dates[j]]['pm2_5'] for j in range(i-6, i+1) if obs_dict[obs_dates[j]]['pm2_5'] is not None]
                        if pm2_5_values:
                            expected = sum(pm2_5_values) / len(pm2_5_values)
                            if feat[6] is not None:
                                if abs(feat[6] - expected) > tolerance:
                                    errors.append(f"Date {as_of}: pm2_5_roll_7 mismatch: feature={feat[6]}, expected={expected}")
                            else:
                                errors.append(f"Date {as_of}: pm2_5_roll_7 is None but enough data")
                        else:
                            if feat[6] is not None:
                                errors.append(f"Date {as_of}: pm2_5_roll_7 is not None but all values are None")
                    else:
                        # Should be None for first 6 days
                        if feat[6] is not None:
                            errors.append(f"Date {as_of}: pm2_5_roll_7 should be None for first 6 days")

                    # 30-day rolling average (index 7)
                    if i >= 29:
                        pm2_5_values = [obs_dict[obs_dates[j]]['pm2_5'] for j in range(i-29, i+1) if obs_dict[obs_dates[j]]['pm2_5'] is not None]
                        if pm2_5_values:
                            expected = sum(pm2_5_values) / len(pm2_5_values)
                            if feat[7] is not None:
                                if abs(feat[7] - expected) > tolerance:
                                    errors.append(f"Date {as_of}: pm2_5_roll_30 mismatch: feature={feat[7]}, expected={expected}")
                            else:
                                errors.append(f"Date {as_of}: pm2_5_roll_30 is None but enough data")
                        else:
                            if feat[7] is not None:
                                errors.append(f"Date {as_of}: pm2_5_roll_30 is not None but all values are None")
                    else:
                        if feat[7] is not None:
                            errors.append(f"Date {as_of}: pm2_5_roll_30 should be None for first 29 days")

                    # Check rolling averages for PM10 (indices 8,9)
                    # 7-day rolling average (index 8)
                    if i >= 6:
                        pm10_values = [obs_dict[obs_dates[j]]['pm10'] for j in range(i-6, i+1) if obs_dict[obs_dates[j]]['pm10'] is not None]
                        if pm10_values:
                            expected = sum(pm10_values) / len(pm10_values)
                            if feat[8] is not None:
                                if abs(feat[8] - expected) > tolerance:
                                    errors.append(f"Date {as_of}: pm10_roll_7 mismatch: feature={feat[8]}, expected={expected}")
                            else:
                                errors.append(f"Date {as_of}: pm10_roll_7 is None but enough data")
                        else:
                            if feat[8] is not None:
                                errors.append(f"Date {as_of}: pm10_roll_7 is not None but all values are None")
                    else:
                        if feat[8] is not None:
                            errors.append(f"Date {as_of}: pm10_roll_7 should be None for first 6 days")

                    # 30-day rolling average (index 9)
                    if i >= 29:
                        pm10_values = [obs_dict[obs_dates[j]]['pm10'] for j in range(i-29, i+1) if obs_dict[obs_dates[j]]['pm10'] is not None]
                        if pm10_values:
                            expected = sum(pm10_values) / len(pm10_values)
                            if feat[9] is not None:
                                if abs(feat[9] - expected) > tolerance:
                                    errors.append(f"Date {as_of}: pm10_roll_30 mismatch: feature={feat[9]}, expected={expected}")
                            else:
                                errors.append(f"Date {as_of}: pm10_roll_30 is None but enough data")
                        else:
                            if feat[9] is not None:
                                errors.append(f"Date {as_of}: pm10_roll_30 is not None but all values are None")
                    else:
                        if feat[9] is not None:
                            errors.append(f"Date {as_of}: pm10_roll_30 should be None for first 29 days")

                # 4. Report results
                if errors:
                    print("Leakage test FAILED")
                    print(f"Found {len(errors)} error(s):")
                    for err in errors[:10]:  # Show first 10 errors
                        print(f"  - {err}")
                    if len(errors) > 10:
                        print(f"  ... and {len(errors) - 10} more")
                    return False
                else:
                    print("Leakage test PASSED")
                    print(f"All {len(feat_rows)} feature records checked successfully.")
                    return True

    except Exception as e:
        print(f"Error running leakage test: {e}")
        raise

if __name__ == "__main__":
    success = run_leakage_test()
    exit(0 if success else 1)