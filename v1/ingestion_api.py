from flask import Flask, request, jsonify
from kafka import KafkaProducer
import json
import sys
from schema_registry import registry, load_default_schemas

# --- Flask & Kafka Setup ---
app = Flask(__name__)

# Configure Kafka Producer
# Assumes Kafka is running on localhost:9092
try:
    producer = KafkaProducer(
        bootstrap_servers='localhost:9092',
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
    print("Kafka Producer connected.")
except Exception as e:
    print(f"CRITICAL: Error connecting to Kafka: {e}", file=sys.stderr)
    print("Please ensure Kafka is running.", file=sys.stderr)
    producer = None # Will cause 503 errors

# --- Entity & Security Setup ---

# In a real app, you'd load this from the EntityService or a secure config
VALID_API_KEYS = {
    # api_key: (entity_id, stream_name)
    "key_fusion_01": ("fusion_reactor_01", "plasma_sensors"),
    "key_company_01": ("acme_corp", "sales_data")
}

# Pre-load the schemas for our valid streams
load_default_schemas()


# --- API Endpoints ---

@app.route('/health', methods=['GET'])
def health_check():
    """Simple health check endpoint."""
    if producer and producer.bootstrap_connected():
        return jsonify({"status": "ok", "kafka_connected": True}), 200
    return jsonify({"status": "error", "kafka_connected": False}), 503

@app.route('/ingest', methods=['POST'])
def ingest_data():
    """
    The main arbitrary data ingestion endpoint.
    It validates, standardizes, and forwards data to Kafka.
    """
    
    # 1. Authentication
    auth_key = request.headers.get('X-API-Key')
    if not auth_key or auth_key not in VALID_API_KEYS:
        return jsonify({"error": "Unauthorized"}), 401

    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    # 2. Identify Entity & Stream
    entity_id, stream_name = VALID_API_KEYS[auth_key]
    
    # 3. Get Payload
    data_payload = request.get_json()
    raw_data = data_payload.get('data')
    timestamp = data_payload.get('timestamp') # Expects ISO 8601 format

    if not raw_data or not timestamp:
        return jsonify({"error": "Payload must contain 'timestamp' and 'data' keys"}), 400

    # 4. Schema Validation
    validation = registry.validate_payload(entity_id, stream_name, raw_data)
    
    if not validation["valid"]:
        # Data is structured, but it's the *wrong* structure. Reject it.
        return jsonify({
            "error": "Invalid data structure",
            "details": validation["error"]
        }), 400

    # 5. Create Standardized Message
    message = {
        "entity_id": entity_id,
        "stream_name": stream_name,
        "timestamp_utc": timestamp,
        "data": raw_data 
    }

    # 6. Publish to Kafka
    if not producer:
        return jsonify({"error": "Kafka service unavailable"}), 503

    try:
        # We use the entity_id as the Kafka "topic" name.
        # This groups all data for one entity into its own stream.
        producer.send(entity_id, value=message)
        producer.flush() # Ensure it sends immediately (for V1)
        
        return jsonify({"status": "received and validated"}), 202
        
    except Exception as e:
        print(f"Error sending to Kafka: {e}", file=sys.stderr)
        return jsonify({"error": "Internal server error writing to stream"}), 500

if __name__ == '__main__':
    if not producer:
        print("CRITICAL: Kafka producer is not connected. Aborting API start.", file=sys.stderr)
        sys.exit(1)
        
    print("Starting V1 Ingestion API on http://localhost:5000")
    app.run(port=5000, debug=True)
