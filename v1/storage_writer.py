import json
import os
import sys
from datetime import datetime
from kafka import KafkaConsumer
import pyarrow as pa
import pyarrow.parquet as pq
from schema_registry import registry, load_default_schemas

# --- Configuration ---
DATA_LAKE_PATH = "manifold_data_lake"
KAFKA_BOOTSTRAP_SERVERS = 'localhost:9092'
KAFKA_GROUP_ID = 'storage_writer_group_v1'
BATCH_SIZE = 100 # Write to disk every 100 messages

# --- Pre-load schemas ---
# The writer needs to know the schemas to write to Parquet correctly.
load_default_schemas()

# --- Type Mapping ---
# A simple mapper from JSON Schema types to PyArrow types
JSON_TO_PYARROW_TYPE_MAP = {
    "string": pa.string(),
    "number": pa.float64(),
    "integer": pa.int64(),
    "boolean": pa.bool_()
    # Note: This doesn't handle nested objects or arrays (V2 problem)
}

def get_pyarrow_schema(entity_id, stream_name):
    """
    Converts our JSON Schema into a PyArrow Schema.
    This is what makes the Parquet files strongly-typed.
    """
    json_schema = registry.get_schema(entity_id, stream_name)
    if not json_schema:
        print(f"Warning: No schema found for {entity_id}/{stream_name}. Cannot write.", file=sys.stderr)
        return None
    
    fields = [
        # Add our standard metadata fields
        pa.field("entity_id", pa.string(), nullable=False),
        pa.field("stream_name", pa.string(), nullable=False),
        pa.field("timestamp_utc", pa.timestamp('us', tz='UTC'), nullable=False),
    ]
    
    # Add fields from the "data" payload
    for name, props in json_schema.get('properties', {}).items():
        pa_type = JSON_TO_PYARROW_TYPE_MAP.get(props.get('type'), pa.string())
        is_nullable = name not in json_schema.get('required', [])
        fields.append(pa.field(name, pa_type, nullable=is_nullable))
        
    return pa.schema(fields)

def get_kafka_consumer():
    """Initializes and subscribes the Kafka Consumer."""
    try:
        # This consumer will subscribe to any new topic that appears
        consumer = KafkaConsumer(
            bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
            auto_offset_reset='earliest', # Start from the beginning
            consumer_timeout_ms=5000,     # Stop iterating if no new messages for 5s
            value_deserializer=lambda m: json.loads(m.decode('utf-8')),
            group_id=KAFKA_GROUP_ID # So it tracks its own progress
        )
        consumer.subscribe(pattern=".*") # Subscribe to EVERYTHING
        print(f"Kafka Consumer connected. Subscribed to all topics as group '{KAFKA_GROUP_ID}'.")
        return consumer
    except Exception as e:
        print(f"CRITICAL: Error connecting to Kafka consumer: {e}", file=sys.stderr)
        return None

def write_batch_to_parquet(batch_data):
    """
    Converts a batch of messages into a single Parquet file.
    This function handles partitioning by entity, stream, and date.
    """
    if not batch_data:
        return

    # We must write one Parquet file per schema.
    # Group the batch by entity and stream.
    grouped_batch = {}
    for msg in batch_data:
        key = (msg['entity_id'], msg['stream_name'])
        if key not in grouped_batch:
            grouped_batch[key] = []
        grouped_batch[key].append(msg)

    # Process each group
    for (entity_id, stream_name), messages in grouped_batch.items():
        try:
            # 1. Get the official schema for this stream
            arrow_schema = get_pyarrow_schema(entity_id, stream_name)
            if not arrow_schema:
                print(f"Skipping write for {entity_id}/{stream_name}: No schema.", file=sys.stderr)
                continue

            # 2. Prepare the data, flattening it to match the schema
            flattened_batch = []
            for msg in messages:
                row = {
                    "entity_id": msg["entity_id"],
                    "stream_name": msg["stream_name"],
                    "timestamp_utc": datetime.fromisoformat(msg["timestamp_utc"].replace('Z', '+00:00')),
                }
                # Unpack the arbitrary data
                row.update(msg['data'])
                flattened_batch.append(row)

            # 3. Create the table using the EXPLICIT schema
            table = pa.Table.from_pylist(flattened_batch, schema=arrow_schema)
            
            # 4. Determine path and write to Parquet
            # We partition by date and hour for efficient querying
            dt = datetime.utcnow()
            date_path = dt.strftime('%Y-%m-%d')
            hour_path = dt.strftime('%H')
            
            dir_path = os.path.join(DATA_LAKE_PATH, entity_id, stream_name, date_path, hour_path)
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)

            file_name = f"{dt.strftime('%Y%m%d_%H%M%S')}_{len(flattened_batch)}_records.parquet"
            file_path = os.path.join(dir_path, file_name)
            
            pq.write_table(table, file_path)
            print(f"Wrote batch of {len(flattened_batch)} records to {file_path}")

        except Exception as e:
            print(f"Error writing batch to Parquet for {entity_id}/{stream_name}: {e}", file=sys.stderr)

def main_loop():
    """The main consumer loop."""
    consumer = get_kafka_consumer()
    if not consumer:
        print("Aborting.")
        sys.exit(1)

    print("Starting V1 Storage Writer loop... (Batch size: %s)" % BATCH_SIZE)
    print("Waiting for messages from Kafka. (Ctrl+C to stop)")
    
    batch = []
    
    try:
        while True:
            for message in consumer:
                batch.append(message.value)
                
                if len(batch) >= BATCH_SIZE:
                    print(f"Batch size reached ({len(batch)}). Writing to Parquet...")
                    write_batch_to_parquet(batch)
                    consumer.commit() # Commit offsets *after* successful write
                    batch = []
            
            # Handle messages at the end of the timeout
            if batch:
                print(f"Consumer timeout. Writing remaining {len(batch)} messages...")
                write_batch_to_parquet(batch)
                consumer.commit()
                batch = []

    except KeyboardInterrupt:
        print("\nShutdown signal received...")
    except Exception as e:
        print(f"FATAL: Error in consumer loop: {e}", file=sys.stderr)
    finally:
        # Final write before shutting down
        if batch:
            print(f"Writing final batch of {len(batch)} messages...")
            write_batch_to_parquet(batch)
        
        consumer.close()
        print("Kafka consumer closed. Storage writer shut down.")

if __name__ == "__main__":
    if not os.path.exists(DATA_LAKE_PATH):
        os.makedirs(DATA_LAKE_PATH)
        print(f"Created data lake directory: {DATA_LAKE_PATH}")
    main_loop()
