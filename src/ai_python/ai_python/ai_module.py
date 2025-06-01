from rclpy.node import Node
import torch
import torch.nn as nn
import numpy as np
from collections import deque
from iot_interfaces.msg import TemperatureHumidity
import rclpy
from ament_index_python.packages import get_package_share_directory

import paho.mqtt.client as mqttclient
import time
import json


BROKER_ADDRESS = "app.coreiot.io"
PORT = 1883
ACCESS_TOKEN = "Edge"
ACCESS_USERNAME = "Edge"

def connected(client, usedata, flags, rc):
    if rc == 0:
        print("Connected successfully!!")
        client.subscribe("v1/devices/me/rpc/request/+")
    else:
        print("Connection is failed")

# Load the same model class definition used in training
class FireLSTM(nn.Module):
    def __init__(self, input_size=2, hidden_size=32, num_layers=1):
        super(FireLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

class AINode(Node):
    def __init__(self):
        super().__init__('ai_node')
        self.get_logger().info('AI Node for fire detection has been initialized.')

        # Load the model
        self.model = FireLSTM()
        state_dict_dir = get_package_share_directory('ai_python') + '/fire_lstm_checkpoint.pth'
        self.model.load_state_dict(torch.load(state_dict_dir, map_location=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'), weights_only=True))
        self.model.eval()

        # Buffer to store last 5 sensor readings
        self.history = deque(maxlen=5)

        self.temperature_humidity_subscriber = self.create_subscription(
            TemperatureHumidity,
            'temperature_humidity_data',
            self.temperature_humidity_callback,
            10
        )

        self.client = mqttclient.Client("Edge")
        self.client.username_pw_set(ACCESS_USERNAME, ACCESS_TOKEN)

        self.client.on_connect = connected
        self.client.connect(BROKER_ADDRESS, PORT)
        self.client.loop_start()

    def temperature_humidity_callback(self, msg):
        temp_hum = (msg.temperature, msg.humidity)
        self.history.append(temp_hum)
        self.get_logger().info(f'Received temperature: {msg.temperature}, humidity: {msg.humidity}')

        if len(self.history) == 5:
            result = self.process_data()
            label = "🔥 FIRE" if result == 1 else "✅ Normal"
            self.get_logger().info(f'Model Prediction: {label}')
            alarm_message = {
                "alarm": result,
            }
            self.client.publish("v1/devices/me/telemetry", json.dumps(alarm_message))
        else:
            self.get_logger().info(f'Waiting for 5 readings... ({len(self.history)}/5)')

    def process_data(self):
        # Convert list to tensor of shape (1, 5, 2)
        input_seq = np.array(self.history, dtype=np.float32)
        input_tensor = torch.tensor(input_seq).unsqueeze(0)  # shape: [1, 5, 2]

        with torch.no_grad():
            output = self.model(input_tensor)
            predicted = torch.argmax(output, dim=1).item()
        return predicted

def main(args=None):

    rclpy.init(args=args)
    ai_node = AINode()
    rclpy.spin(ai_node)
    ai_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
