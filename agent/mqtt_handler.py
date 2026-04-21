"""
MQTT message handling for communication with Unity/RTD.
"""
import json
import random
import ssl
import time
import paho.mqtt.client as mqtt_client

from .context import agent_context, update_dataframe_from_layer
from .orchestrator import process_user_request
from .config import (
    MQTT_HOST,
    MQTT_PORT,
    MQTT_USERNAME,
    MQTT_PASSWORD,
    MQTT_TOPIC_IN,
    MQTT_TOPIC_OUT,
)

# Global client reference (set after connection)
_mqtt_client = None


def on_message(client, userdata, msg):
    """Handle inbound MQTT messages."""
    payload = msg.payload.decode('utf-8', errors='ignore').strip()
    print(f"\nReceived: {payload[:200]}...")

    # Quick exit hook
    if payload.strip().lower() in ('exit.', 'stop.', 'quit.'):
        print("Exiting...")
        client.disconnect()
        return

    # Parse JSON
    try:
        data = json.loads(payload)
    except json.JSONDecodeError:
        print("Invalid JSON format in message.")
        return

    # Handle chart metadata boot message
    if "chart_metadata_index" in data:
        agent_context["chart_metadata_index"] = data["chart_metadata_index"]
        print("Chart metadata index registered")
        return

    # Handle layer data update
    if data.get("message_type") == "layer_data_update":
        update_dataframe_from_layer(data)
        return

    # Handle RTD data (chart metadata + image from renderer)
    if "rtd_data_for_agent" in data:
        rtd_data = data["rtd_data_for_agent"]
        agent_context["chart_type"] = rtd_data.get("chart_type")
        agent_context["data_name"] = rtd_data.get("data_name")
        agent_context["image_data"] = rtd_data.get("image_data")
        agent_context["image_format"] = rtd_data.get("image_format")

        # Extract schema-level color field and pre-built overview if present
        schema = rtd_data.get("schema") or {}
        encoding = schema.get("encoding") or {}
        color_encoding = encoding.get("color") or {}
        agent_context["color_field"] = color_encoding.get("field") or None
        overview = schema.get("overview")
        if overview:
            agent_context["chart_overview"] = overview

        print(f"RTD data registered: chart_type={rtd_data.get('chart_type')}, data_name={rtd_data.get('data_name')}")
        return

    # Handle user request
    if "user_request_for_agent" in data:
        try:
            result = process_user_request(payload)
            publish_message(
                response_text=result.get("response", ""),
                rtd_command=result.get("rtd_command"),
                nodes=result.get("nodes"),
                followup_stage=result.get("followup_stage", False),
                referents=result.get("referents"),
            )
        except Exception as e:
            print(f"Error processing request: {e}")
            import traceback
            traceback.print_exc()
            publish_message(
                response_text="I encountered an error processing your request.",
                followup_stage=False,
            )


def publish_message(
    response_text: str,
    rtd_command: dict = None,
    nodes: dict = None,
    followup_stage: bool = False,
    referents: dict = None
):
    """Publish agent response back to Unity over MQTT."""
    global _mqtt_client

    if _mqtt_client is None:
        print("MQTT client not connected")
        return

    payload = {
        "agent_response_for_user": {
            "response_text": response_text,
            "followup_stage": followup_stage
        }
    }

    if nodes is not None:
        payload["agent_response_for_user"]["nodes"] = nodes
    if rtd_command:
        payload["agent_response_for_user"]["rtd_command"] = rtd_command
    if referents:
        payload["agent_response_for_user"]["referents"] = referents

    response_json = json.dumps(payload)

    info = _mqtt_client.publish(MQTT_TOPIC_OUT, response_json, qos=1, retain=False)
    status = getattr(info, "rc", None)

    if status == mqtt_client.MQTT_ERR_SUCCESS:
        print(f"Sent response to '{MQTT_TOPIC_OUT}'")
    else:
        print(f"Error: Failed to send response. rc={status}")


def on_connect(client, userdata, flags, reason_code, properties):
    """Handle MQTT connection."""
    if reason_code == 0 or str(reason_code) == "Success":
        print("Connected to MQTT broker")
        client.subscribe(MQTT_TOPIC_IN)
        print(f"Subscribed to '{MQTT_TOPIC_IN}'")
    else:
        print(f"Error: Failed to connect. Reason: {reason_code}")


def on_disconnect(client, userdata, disconnect_flags, reason_code, properties):
    """Handle MQTT disconnection."""
    if reason_code == 0:
        print("Disconnected from MQTT broker cleanly.")
    else:
        print(f"Warning: Unexpected disconnect from MQTT broker (rc={reason_code}). Will attempt to reconnect...")


def create_mqtt_client() -> mqtt_client.Client:
    """Create and configure MQTT client."""
    global _mqtt_client

    client_id = f'python-agent-{random.randint(0, 1000)}'
    client = mqtt_client.Client(
        client_id=client_id,
        callback_api_version=mqtt_client.CallbackAPIVersion.VERSION2
    )

    client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)
    client.tls_set(tls_version=ssl.PROTOCOL_TLS)
    # Exponential backoff: wait 1s after first disconnect, up to 60s between retries
    client.reconnect_delay_set(min_delay=1, max_delay=60)

    client.on_connect = on_connect
    client.on_message = on_message
    client.on_disconnect = on_disconnect

    _mqtt_client = client
    return client


def run():
    """Start the MQTT client loop."""
    client = create_mqtt_client()

    # Retry initial connection with backoff — broker may not be ready yet
    retry_delay = 1
    while True:
        try:
            print(f"Connecting to {MQTT_HOST}:{MQTT_PORT}...")
            client.connect(MQTT_HOST, MQTT_PORT)
            break
        except Exception as e:
            print(f"Warning: Connection failed: {e}. Retrying in {retry_delay}s...")
            time.sleep(retry_delay)
            retry_delay = min(retry_delay * 2, 60)

    print("Starting message loop. Press Ctrl+C to exit.")
    try:
        # loop_forever() handles mid-session reconnects automatically
        # using the reconnect_delay_set backoff configured above
        client.loop_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        client.disconnect()
