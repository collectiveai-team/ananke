-- Create additional databases for MLflow and Dagster
CREATE DATABASE mlflow;
CREATE DATABASE dagster;

-- Grant permissions
GRANT ALL PRIVILEGES ON DATABASE mlflow TO ananke;
GRANT ALL PRIVILEGES ON DATABASE dagster TO ananke;
