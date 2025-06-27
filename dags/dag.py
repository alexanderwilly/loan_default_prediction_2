from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.dummy import DummyOperator
from datetime import datetime, timedelta

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'dag',
    default_args=default_args,
    description='data pipeline run once a month',
    schedule_interval='0 0 1 * *',  # At 00:00 on day-of-month 1
    start_date=datetime(2023, 1, 1),
    # end_date=datetime(2024, 12, 1),
    end_date=datetime(2023, 1, 3),
    catchup=True,
) as dag:

    # data pipeline

    # --- label store ---
    dep_check_source_label_data = BashOperator(
        task_id="dep_check_source_label_data",
        bash_command=(
            'cd /opt/airflow/scripts &&'
            'python3 dep_check_source_label_data.py'
        )
    )
    
    bronze_label_store = BashOperator(
        task_id="bronze_label_store",
        bash_command=(
            'cd /opt/airflow/scripts &&'
            'python3 bronze_label_store.py'
        )    
    )
    silver_label_store = BashOperator(
        task_id="silver_label_store",
        bash_command=(
            'cd /opt/airflow/scripts &&'
            'python3 silver_label_store.py'
        )    
    )

    # Define task dependencies to run scripts sequentially
    dep_check_source_label_data >> bronze_label_store >> silver_label_store
 
    # --- feature store ---
    dep_check_source_feature_data = BashOperator(
        task_id="dep_check_source_feature_data",
        bash_command=(
            'cd /opt/airflow/scripts &&'
            'python3 dep_check_source_feature_data.py'
        )
    )
    
    bronze_clickstream = BashOperator(
        task_id="bronze_clickstream",
        bash_command=(
            'cd /opt/airflow/scripts &&'
            'python3 bronze_clickstream.py'
        )    
    )
    bronze_attributes = BashOperator(
        task_id="bronze_attributes",
        bash_command=(
            'cd /opt/airflow/scripts &&'
            'python3 bronze_attributes.py'
        )
    )
    bronze_financials = BashOperator(
        task_id="bronze_financials",
        bash_command=(
            'cd /opt/airflow/scripts &&'
            'python3 bronze_financials.py'
        )
    )

    silver_clickstream = BashOperator(
        task_id="silver_clickstream",
        bash_command=(
            'cd /opt/airflow/scripts &&'
            'python3 silver_clickstream.py'
        )
    )
    silver_attributes = BashOperator(
        task_id="silver_attributes",
        bash_command=(
            'cd /opt/airflow/scripts &&'
            'python3 silver_attributes.py'
            )
    )
    silver_financials = BashOperator(
        task_id="silver_financials",
        bash_command=(
            'cd /opt/airflow/scripts &&'
            'python3 silver_financials.py'   
        )
    )


    # Define task dependencies to run scripts sequentially
    dep_check_source_feature_data >> bronze_clickstream >> silver_clickstream
    dep_check_source_feature_data >> bronze_attributes >> silver_attributes
    dep_check_source_feature_data >> bronze_financials >> silver_financials


    # gold tables
    gold_feature_label_store = BashOperator(
        task_id="gold_feature_label_store",
        bash_command=(
            'cd /opt/airflow/scripts &&'
            'python3 gold_feature_label_store.py'   
        )    
    )
    feature_label_store_completed = BashOperator(
        task_id="feature_label_store_completed", 
        bash_command=(
            'cd /opt/airflow/scripts &&'
            'python3 feature_label_store_completed.py'
        )
    )

    # Define task dependencies to run scripts sequentially
    silver_label_store >> gold_feature_label_store
    silver_clickstream >> gold_feature_label_store
    silver_attributes >> gold_feature_label_store
    silver_financials >> gold_feature_label_store

    gold_feature_label_store >> feature_label_store_completed


    # --- model train, inference, and monitor ---
    model_training = BashOperator(
        task_id="model_training",
        bash_command=(
            'cd /opt/airflow/scripts &&'
            'python3 model_training.py'
        )
    )
    model_inference = BashOperator(
        task_id="model_inference",
        bash_command=(
            'cd /opt/airflow/scripts &&'
            'python3 model_inference.py'
        )       
    )
    model_monitoring = DummyOperator(task_id="model_monitoring")

    feature_label_store_completed >> model_training >> model_inference >> model_monitoring

    
    # Define task dependencies to run scripts sequentially
    # feature_store_completed >> model_inference_start
    # model_inference_start >> model_1_inference >> model_inference_completed
    # model_inference_start >> model_2_inference >> model_inference_completed


    # --- model monitoring ---
    # model_monitor_start = DummyOperator(task_id="model_monitor_start")

    # model_1_monitor = DummyOperator(task_id="model_1_monitor")

    # model_2_monitor = DummyOperator(task_id="model_2_monitor")

    # model_monitor_completed = DummyOperator(task_id="model_monitor_completed")
    
    # Define task dependencies to run scripts sequentially
    # model_inference_completed >> model_monitor_start
    # model_monitor_start >> model_1_monitor >> model_monitor_completed
    # model_monitor_start >> model_2_monitor >> model_monitor_completed


    # --- model auto training ---

    # model_automl_start = DummyOperator(task_id="model_automl_start")
    
    # model_1_automl = DummyOperator(task_id="model_1_automl")

    # model_2_automl = DummyOperator(task_id="model_2_automl")

    # model_automl_completed = DummyOperator(task_id="model_automl_completed")
    
    # Define task dependencies to run scripts sequentially
    # feature_store_completed >> model_automl_start
    # label_store_completed >> model_automl_start
    # model_automl_start >> model_1_automl >> model_automl_completed
    # model_automl_start >> model_2_automl >> model_automl_completed