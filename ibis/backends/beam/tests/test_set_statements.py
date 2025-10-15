"""Tests for SET statement functionality in Beam backend."""

import pytest

import ibis
from ibis.backends.beam import Backend


def test_set_runner():
    """Test setting the runner using SET statement."""
    import apache_beam as beam
    
    pipeline = beam.Pipeline()
    backend = Backend()
    backend.do_connect(pipeline)
    
    # Test setting DataflowRunner
    result = backend.raw_sql("SET runner = 'dataflow'")
    assert backend.get_runner_config()['runner'] == 'DataflowRunner'
    
    # Test setting FlinkRunner
    result = backend.raw_sql("SET runner = 'flink'")
    assert backend.get_runner_config()['runner'] == 'FlinkRunner'
    
    # Test setting DirectRunner
    result = backend.raw_sql("SET runner = 'direct'")
    assert backend.get_runner_config()['runner'] == 'DirectRunner'


def test_set_dataflow_options():
    """Test setting Dataflow-specific options."""
    import apache_beam as beam
    
    pipeline = beam.Pipeline()
    backend = Backend()
    backend.do_connect(pipeline)
    
    # Set DataflowRunner
    backend.raw_sql("SET runner = 'dataflow'")
    
    # Set Dataflow options
    backend.raw_sql("SET dataflow.project = 'my-gcp-project'")
    backend.raw_sql("SET dataflow.region = 'us-central1'")
    backend.raw_sql("SET dataflow.staging_location = 'gs://my-bucket/staging'")
    backend.raw_sql("SET dataflow.temp_location = 'gs://my-bucket/temp'")
    backend.raw_sql("SET dataflow.num_workers = '5'")
    backend.raw_sql("SET dataflow.max_num_workers = '10'")
    backend.raw_sql("SET dataflow.machine_type = 'n1-standard-4'")
    backend.raw_sql("SET dataflow.use_public_ips = 'false'")
    backend.raw_sql("SET dataflow.enable_streaming_engine = 'true'")
    
    config = backend.get_runner_config()
    assert config['project'] == 'my-gcp-project'
    assert config['region'] == 'us-central1'
    assert config['staging_location'] == 'gs://my-bucket/staging'
    assert config['temp_location'] == 'gs://my-bucket/temp'
    assert config['num_workers'] == '5'
    assert config['max_num_workers'] == '10'
    assert config['machine_type'] == 'n1-standard-4'
    assert config['use_public_ips'] == 'false'
    assert config['enable_streaming_engine'] == 'true'


def test_set_pipeline_options():
    """Test setting pipeline options."""
    import apache_beam as beam
    
    pipeline = beam.Pipeline()
    backend = Backend()
    backend.do_connect(pipeline)
    
    # Set pipeline options
    backend.raw_sql("SET pipeline.streaming = 'true'")
    backend.raw_sql("SET pipeline.save_main_session = 'true'")
    backend.raw_sql("SET pipeline.setup_file = '/path/to/setup.py'")
    
    options = backend.get_pipeline_options()
    assert options['streaming'] == 'true'
    assert options['save_main_session'] == 'true'
    assert options['setup_file'] == '/path/to/setup.py'


def test_set_generic_options():
    """Test setting generic options."""
    import apache_beam as beam
    
    pipeline = beam.Pipeline()
    backend = Backend()
    backend.do_connect(pipeline)
    
    # Set generic options
    backend.raw_sql("SET streaming = 'true'")
    backend.raw_sql("SET job_name = 'my-beam-job'")
    
    options = backend.get_pipeline_options()
    assert options['streaming'] == 'true'
    assert options['job_name'] == 'my-beam-job'


def test_set_with_quotes():
    """Test SET statements with quoted values."""
    import apache_beam as beam
    
    pipeline = beam.Pipeline()
    backend = Backend()
    backend.do_connect(pipeline)
    
    # Test with single quotes
    backend.raw_sql("SET runner = 'dataflow'")
    assert backend.get_runner_config()['runner'] == 'DataflowRunner'
    
    # Test with double quotes
    backend.raw_sql('SET runner = "flink"')
    assert backend.get_runner_config()['runner'] == 'FlinkRunner'
    
    # Test with spaces and quotes
    backend.raw_sql("SET dataflow.project = 'my-gcp-project'")
    assert backend.get_runner_config()['project'] == 'my-gcp-project'


def test_set_invalid_statement():
    """Test invalid SET statements."""
    import apache_beam as beam
    
    pipeline = beam.Pipeline()
    backend = Backend()
    backend.do_connect(pipeline)
    
    # Test invalid SET statement
    with pytest.raises(Exception):
        backend.raw_sql("SET invalid statement")
    
    # Test unknown runner
    with pytest.raises(Exception):
        backend.raw_sql("SET runner = 'unknown'")


def test_create_configured_pipeline():
    """Test creating a pipeline with configured options."""
    import apache_beam as beam
    
    pipeline = beam.Pipeline()
    backend = Backend()
    backend.do_connect(pipeline)
    
    # Configure for Dataflow
    backend.raw_sql("SET runner = 'dataflow'")
    backend.raw_sql("SET dataflow.project = 'my-gcp-project'")
    backend.raw_sql("SET dataflow.region = 'us-central1'")
    backend.raw_sql("SET dataflow.staging_location = 'gs://my-bucket/staging'")
    backend.raw_sql("SET dataflow.temp_location = 'gs://my-bucket/temp'")
    backend.raw_sql("SET dataflow.num_workers = '5'")
    
    # Create configured pipeline
    configured_pipeline = backend.create_configured_pipeline()
    assert configured_pipeline is not None
    
    # Verify the pipeline has options
    options = configured_pipeline._options
    assert options is not None


def test_set_labels():
    """Test setting Dataflow labels."""
    import apache_beam as beam
    
    pipeline = beam.Pipeline()
    backend = Backend()
    backend.do_connect(pipeline)
    
    # Set DataflowRunner
    backend.raw_sql("SET runner = 'dataflow'")
    
    # Set labels
    backend.raw_sql("SET dataflow.labels = 'env=prod,team=data,version=1.0'")
    
    config = backend.get_runner_config()
    assert config['labels'] == 'env=prod,team=data,version=1.0'


def test_complete_dataflow_setup():
    """Test complete Dataflow setup with multiple SET statements."""
    import apache_beam as beam
    
    pipeline = beam.Pipeline()
    backend = Backend()
    backend.do_connect(pipeline)
    
    # Complete Dataflow configuration
    setup_statements = [
        "SET runner = 'dataflow'",
        "SET dataflow.project = 'my-gcp-project'",
        "SET dataflow.region = 'us-central1'",
        "SET dataflow.staging_location = 'gs://my-bucket/staging'",
        "SET dataflow.temp_location = 'gs://my-bucket/temp'",
        "SET dataflow.service_account = 'my-service-account@my-gcp-project.iam.gserviceaccount.com'",
        "SET dataflow.network = 'my-network'",
        "SET dataflow.subnetwork = 'my-subnetwork'",
        "SET dataflow.use_public_ips = 'false'",
        "SET dataflow.num_workers = '3'",
        "SET dataflow.max_num_workers = '10'",
        "SET dataflow.machine_type = 'n1-standard-4'",
        "SET dataflow.disk_size_gb = '100'",
        "SET dataflow.disk_type = 'pd-ssd'",
        "SET dataflow.autoscaling_algorithm = 'THROUGHPUT_BASED'",
        "SET dataflow.enable_streaming_engine = 'true'",
        "SET dataflow.flexrs_goal = 'COST_OPTIMIZED'",
        "SET dataflow.labels = 'env=prod,team=data'",
        "SET pipeline.streaming = 'true'",
        "SET pipeline.save_main_session = 'true'"
    ]
    
    # Execute all SET statements
    for statement in setup_statements:
        result = backend.raw_sql(statement)
    
    # Verify configuration
    runner_config = backend.get_runner_config()
    pipeline_options = backend.get_pipeline_options()
    
    assert runner_config['runner'] == 'DataflowRunner'
    assert runner_config['project'] == 'my-gcp-project'
    assert runner_config['region'] == 'us-central1'
    assert runner_config['staging_location'] == 'gs://my-bucket/staging'
    assert runner_config['temp_location'] == 'gs://my-bucket/temp'
    assert runner_config['service_account'] == 'my-service-account@my-gcp-project.iam.gserviceaccount.com'
    assert runner_config['network'] == 'my-network'
    assert runner_config['subnetwork'] == 'my-subnetwork'
    assert runner_config['use_public_ips'] == 'false'
    assert runner_config['num_workers'] == '3'
    assert runner_config['max_num_workers'] == '10'
    assert runner_config['machine_type'] == 'n1-standard-4'
    assert runner_config['disk_size_gb'] == '100'
    assert runner_config['disk_type'] == 'pd-ssd'
    assert runner_config['autoscaling_algorithm'] == 'THROUGHPUT_BASED'
    assert runner_config['enable_streaming_engine'] == 'true'
    assert runner_config['flexrs_goal'] == 'COST_OPTIMIZED'
    assert runner_config['labels'] == 'env=prod,team=data'
    
    assert pipeline_options['streaming'] == 'true'
    assert pipeline_options['save_main_session'] == 'true'
    
    # Create configured pipeline
    configured_pipeline = backend.create_configured_pipeline()
    assert configured_pipeline is not None
