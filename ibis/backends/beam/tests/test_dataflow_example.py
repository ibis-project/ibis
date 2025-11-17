"""Example of using SET statements to configure DataflowRunner."""

import pytest

import ibis
from ibis.backends.beam import Backend


def test_dataflow_runner_setup_example():
    """Example of setting up DataflowRunner with SET statements."""
    import apache_beam as beam
    
    # Create initial pipeline
    pipeline = beam.Pipeline()
    backend = Backend()
    backend.do_connect(pipeline)
    
    # Configure DataflowRunner using SET statements
    setup_commands = [
        # Set the runner
        "SET runner = 'dataflow'",
        
        # Required Dataflow options
        "SET dataflow.project = 'my-gcp-project'",
        "SET dataflow.staging_location = 'gs://my-bucket/staging'",
        "SET dataflow.temp_location = 'gs://my-bucket/temp'",
        
        # Optional Dataflow options
        "SET dataflow.region = 'us-central1'",
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
        "SET dataflow.labels = 'env=prod,team=data,version=1.0'",
        
        # Pipeline options
        "SET pipeline.streaming = 'true'",
        "SET pipeline.save_main_session = 'true'",
        "SET job_name = 'my-dataflow-job'"
    ]
    
    # Execute all SET statements
    for command in setup_commands:
        result = backend.raw_sql(command)
        # Verify the command was processed
        assert result is not None
    
    # Verify configuration
    runner_config = backend.get_runner_config()
    pipeline_options = backend.get_pipeline_options()
    
    # Check runner configuration
    assert runner_config['runner'] == 'DataflowRunner'
    assert runner_config['project'] == 'my-gcp-project'
    assert runner_config['staging_location'] == 'gs://my-bucket/staging'
    assert runner_config['temp_location'] == 'gs://my-bucket/temp'
    assert runner_config['region'] == 'us-central1'
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
    assert runner_config['labels'] == 'env=prod,team=data,version=1.0'
    
    # Check pipeline options
    assert pipeline_options['streaming'] == 'true'
    assert pipeline_options['save_main_session'] == 'true'
    assert pipeline_options['job_name'] == 'my-dataflow-job'
    
    # Create configured pipeline
    configured_pipeline = backend.create_configured_pipeline()
    assert configured_pipeline is not None
    
    # The configured pipeline can now be used for actual data processing
    return configured_pipeline


def test_dataflow_with_ibis_operations():
    """Example of using DataflowRunner with Ibis operations."""
    import apache_beam as beam
    import pandas as pd
    
    # Create initial pipeline
    pipeline = beam.Pipeline()
    backend = Backend()
    backend.do_connect(pipeline)
    
    # Configure for Dataflow
    backend.raw_sql("SET runner = 'dataflow'")
    backend.raw_sql("SET dataflow.project = 'my-gcp-project'")
    backend.raw_sql("SET dataflow.staging_location = 'gs://my-bucket/staging'")
    backend.raw_sql("SET dataflow.temp_location = 'gs://my-bucket/temp'")
    backend.raw_sql("SET dataflow.region = 'us-central1'")
    backend.raw_sql("SET dataflow.num_workers = '5'")
    
    # Create sample data
    data = pd.DataFrame({
        'user_id': [1, 2, 3, 4, 5],
        'event_type': ['click', 'view', 'click', 'purchase', 'view'],
        'timestamp': ['2023-01-01 10:00:00', '2023-01-01 10:01:00', 
                     '2023-01-01 10:02:00', '2023-01-01 10:03:00', '2023-01-01 10:04:00'],
        'value': [10.0, 0.0, 15.0, 100.0, 0.0]
    })
    
    # Create table
    events = backend.create_view('events', data, temp=True)
    
    # Perform analytics
    click_events = events.filter(events.event_type == 'click')
    total_clicks = click_events.count()
    avg_click_value = click_events.value.mean()
    
    # Compile to SQL (this would be executed on Dataflow)
    click_sql = backend.compile(click_events)
    count_sql = backend.compile(total_clicks)
    avg_sql = backend.compile(avg_click_value)
    
    # Verify SQL compilation
    assert click_sql is not None
    assert count_sql is not None
    assert avg_sql is not None
    
    # Create configured pipeline for actual execution
    configured_pipeline = backend.create_configured_pipeline()
    assert configured_pipeline is not None
    
    # In a real scenario, you would run the pipeline:
    # with configured_pipeline as pipeline:
    #     con = ibis.beam.connect(pipeline)
    #     result = con.execute(avg_click_value)
    #     print(result)


def test_dataflow_streaming_example():
    """Example of configuring DataflowRunner for streaming."""
    import apache_beam as beam
    
    # Create initial pipeline
    pipeline = beam.Pipeline()
    backend = Backend()
    backend.do_connect(pipeline)
    
    # Configure for streaming Dataflow
    streaming_setup = [
        "SET runner = 'dataflow'",
        "SET dataflow.project = 'my-gcp-project'",
        "SET dataflow.staging_location = 'gs://my-bucket/staging'",
        "SET dataflow.temp_location = 'gs://my-bucket/temp'",
        "SET dataflow.region = 'us-central1'",
        "SET dataflow.enable_streaming_engine = 'true'",
        "SET dataflow.num_workers = '3'",
        "SET dataflow.max_num_workers = '20'",
        "SET dataflow.autoscaling_algorithm = 'THROUGHPUT_BASED'",
        "SET pipeline.streaming = 'true'",
        "SET pipeline.save_main_session = 'true'"
    ]
    
    # Execute setup
    for command in streaming_setup:
        backend.raw_sql(command)
    
    # Verify streaming configuration
    runner_config = backend.get_runner_config()
    pipeline_options = backend.get_pipeline_options()
    
    assert runner_config['enable_streaming_engine'] == 'true'
    assert runner_config['autoscaling_algorithm'] == 'THROUGHPUT_BASED'
    assert pipeline_options['streaming'] == 'true'
    assert pipeline_options['save_main_session'] == 'true'
    
    # Create configured pipeline
    configured_pipeline = backend.create_configured_pipeline()
    assert configured_pipeline is not None


def test_dataflow_cost_optimization():
    """Example of configuring DataflowRunner for cost optimization."""
    import apache_beam as beam
    
    # Create initial pipeline
    pipeline = beam.Pipeline()
    backend = Backend()
    backend.do_connect(pipeline)
    
    # Configure for cost-optimized Dataflow
    cost_optimized_setup = [
        "SET runner = 'dataflow'",
        "SET dataflow.project = 'my-gcp-project'",
        "SET dataflow.staging_location = 'gs://my-bucket/staging'",
        "SET dataflow.temp_location = 'gs://my-bucket/temp'",
        "SET dataflow.region = 'us-central1'",
        "SET dataflow.flexrs_goal = 'COST_OPTIMIZED'",
        "SET dataflow.machine_type = 'n1-standard-1'",
        "SET dataflow.disk_type = 'pd-standard'",
        "SET dataflow.disk_size_gb = '50'",
        "SET dataflow.num_workers = '1'",
        "SET dataflow.max_num_workers = '5'",
        "SET dataflow.autoscaling_algorithm = 'NONE'"
    ]
    
    # Execute setup
    for command in cost_optimized_setup:
        backend.raw_sql(command)
    
    # Verify cost optimization configuration
    runner_config = backend.get_runner_config()
    
    assert runner_config['flexrs_goal'] == 'COST_OPTIMIZED'
    assert runner_config['machine_type'] == 'n1-standard-1'
    assert runner_config['disk_type'] == 'pd-standard'
    assert runner_config['disk_size_gb'] == '50'
    assert runner_config['num_workers'] == '1'
    assert runner_config['max_num_workers'] == '5'
    assert runner_config['autoscaling_algorithm'] == 'NONE'
    
    # Create configured pipeline
    configured_pipeline = backend.create_configured_pipeline()
    assert configured_pipeline is not None
