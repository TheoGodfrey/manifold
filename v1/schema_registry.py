import jsonschema

class SchemaRegistry:
    """
    Manages and validates JSON Schemas for incoming data streams.
    In a production system, this would be backed by a database.
    """
    def __init__(self):
        # (entity_id, stream_name) -> JSON Schema
        self.schemas = {} 

    def register_schema(self, entity_id, stream_name, schema):
        """
        Registers a new JSON Schema for a specific data stream.
        """
        try:
            # First, check if the schema itself is a valid JSON Schema
            jsonschema.Draft7Validator.check_schema(schema)
        except jsonschema.SchemaError as e:
            return {"status": "error", "message": f"Invalid JSON Schema: {e}"}
            
        key = (entity_id, stream_name)
        self.schemas[key] = schema
        print(f"Registered schema for: {key}")
        return {"status": "success", "schema": schema}

    def get_schema(self, entity_id, stream_name):
        """Retrieves the schema for a stream."""
        key = (entity_id, stream_name)
        return self.schemas.get(key)

    def validate_payload(self, entity_id, stream_name, payload_data):
        """
        Validates a data payload against its registered schema.
        """
        schema = self.get_schema(entity_id, stream_name)
        if not schema:
            # For a robust system, you should deny un-schematized data.
            return {"valid": False, "error": "No schema registered for this stream"}
        
        try:
            # This is the core validation step!
            jsonschema.validate(instance=payload_data, schema=schema)
            return {"valid": True, "error": None}
        except jsonschema.ValidationError as e:
            return {"valid": False, "error": str(e)}

# --- Create a singleton instance to be imported by other services ---
# This allows the API and the Writer to share the same schema definitions.
registry = SchemaRegistry()

# --- Example of how to pre-load schemas ---
def load_default_schemas():
    """A helper function to pre-populate the registry with examples."""
    
    # Define the schema for our fusion reactor's plasma sensors
    plasma_schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "sensor_id": {"type": "string"},
            "value_kelvin": {"type": "number"},
            "density": {"type": "number"},
            "status_code": {"type": "integer"}
        },
        "required": ["sensor_id", "value_kelvin", "density"]
    }
    
    registry.register_schema(
        "fusion_reactor_01", 
        "plasma_sensors", 
        plasma_schema
    )

    # Define a schema for a different "form" of data
    sales_schema = {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "transaction_id": {"type": "string"},
            "amount_usd": {"type": "number"},
            "customer_id": {"type": "string"}
        },
        "required": ["transaction_id", "amount_usd"]
    }

    registry.register_schema(
        "acme_corp",
        "sales_data",
        sales_schema
    )
    
    print("--- Default schemas loaded ---")

if __name__ == "__main__":
    # This file isn't meant to be run directly,
    # but you can run `python schema_registry.py` to test it.
    load_default_schemas()
    print("\nRegistry contains:")
    print(registry.schemas)
    
    test_data_valid = {"sensor_id": "t-100", "value_kelvin": 1.5e8, "density": 1.0e20}
    print("\nTesting valid data:")
    print(registry.validate_payload("fusion_reactor_01", "plasma_sensors", test_data_valid))
    
    test_data_invalid = {"sensor_id": "t-101", "value_kelvin": "hot"} # Invalid type
    print("\nTesting invalid data:")
    print(registry.validate_payload("fusion_reactor_01", "plasma_sensors", test_data_invalid))
